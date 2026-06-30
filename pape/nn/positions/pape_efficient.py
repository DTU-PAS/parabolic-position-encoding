import math

import torch
import triton
import triton.language as tl

from pape.configs import Config
from pape.nn.positions.base import PositionEncoder


class ParabolicPositionEncoder(PositionEncoder):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.model.hidden_size
        self.num_heads = config.model.num_heads
        self.head_size = config.model.head_size
        self.num_parabolas = config.pape.parabolas
        self.num_positions = config.num_positions

        # The attention kernel is faster and more memory efficient for head sizes that are multiples of 8.
        total_head_size = self.head_size + self.num_positions**2 + 2 * self.num_positions + 2
        self.num_pad = 8 * math.ceil(total_head_size / 8) - total_head_size

        a_eval_size = self.num_heads * self.num_parabolas
        b_eval_size = self.num_heads * self.num_positions
        self.ab_eval_split_sizes = [a_eval_size, b_eval_size]

        self._registered = False

    def train(self, mode: bool = True):
        if not mode:
            self._setup_eval_encode_query_key()
        return super().train(mode)

    def eval(self):
        self._setup_eval_encode_query_key()
        return super().eval()

    def register_model_weights(self):
        pass

    def register_layer_weights(self):
        position = torch.nn.Linear(self.num_positions, self.num_heads * self.num_parabolas, bias=False)
        self.register_module("position", position)

        num_features = self.num_heads * self.num_parabolas
        ab = torch.nn.Linear(self.hidden_size, 2 * num_features, bias=False)

        self.register_module("ab", ab)

        self._registered = True

    def prepare_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.float()

        batch_size, seq_length, _ = positions.size()
        positions_matrix = positions.unsqueeze(-1) * positions.unsqueeze(-2)  # (b, 1, s, p, p)
        positions_flat = positions_matrix.view(batch_size, 1, seq_length, self.num_positions * self.num_positions)

        return positions, positions_flat

    def encode_absolute(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            return self._train_encode_query_key(hidden_state, query, key, prepared_positions)
        else:
            return self._eval_encode_query_key(hidden_state, query, key, prepared_positions)

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")

    def _setup_eval_encode_query_key(self):
        if not self._registered:
            return

        W_p = self.position.weight.data  # (h*m, p)
        W_p = W_p.reshape(self.num_heads, self.num_parabolas, self.num_positions)  # (h, m, p)
        self.W_p = W_p

        ab = self.ab.weight.data
        ab = ab.reshape(2, self.num_heads * self.num_parabolas, self.hidden_size)
        ab = ab.permute(0, 2, 1)  # (2, d, h*m)
        a, b = ab.unbind(0)

        b = b.reshape(self.hidden_size, self.num_heads, self.num_parabolas)
        b = b.permute(1, 0, 2)  # (h, d, m)
        b = b @ W_p  # (h, d, p)
        b = b.permute(1, 0, 2)  # (d, h, p)
        b = b.reshape(self.hidden_size, self.num_heads * self.num_positions)

        self.ab_eval = torch.cat([a, b], dim=1)  # (d, h*m + h*p)

    def _train_encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, _ = prepared_positions

        B = key.size(0)
        H = self.num_heads
        S = key.size(2)
        M = self.num_parabolas
        P = self.num_positions

        ab_out = self.ab(hidden_state)  # (B, S, 2*H*M)
        a_pre, b_pre = ab_out.chunk(2, dim=-1)
        a_pre = a_pre.reshape(B, S, H, M)
        b_pre = b_pre.reshape(B, S, H, M)

        W_p_3d = self.position.weight.view(H, M, P)

        return _PaPETrainAugmentation.apply(query, key, a_pre, b_pre, W_p_3d, positions, self.num_pad)

    def _eval_encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, _ = prepared_positions

        B = key.size(0)
        H = self.num_heads
        S = key.size(2)
        D = self.head_size
        M = self.num_parabolas
        P = self.num_positions
        D_OUT = D + 2 + P * P + 2 * P + self.num_pad

        ab = hidden_state @ self.ab_eval  # (B, S, H*M + H*P)
        a_size, _ = self.ab_eval_split_sizes
        a_pre = ab[..., :a_size].reshape(B, S, H, M)
        b_part = ab[..., a_size:].reshape(B, S, H, P)
        pos_contig = positions.contiguous()

        Q_out = torch.zeros(B, H, S, D_OUT, dtype=query.dtype, device=query.device)
        K_out = torch.zeros(B, H, S, D_OUT, dtype=key.dtype, device=key.device)

        BLOCK_D = max(triton.next_power_of_2(D), 16)
        BLOCK_M = max(triton.next_power_of_2(M), 16)
        BLOCK_P = max(triton.next_power_of_2(P), 4)

        grid = (B * H, S)
        _pape_qk_eval_kernel[grid](
            query,
            key,
            a_pre,
            b_part,
            self.W_p,
            pos_contig,
            Q_out,
            K_out,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            a_pre.stride(0),
            a_pre.stride(1),
            a_pre.stride(2),
            b_part.stride(0),
            b_part.stride(1),
            b_part.stride(2),
            self.W_p.stride(0),
            self.W_p.stride(1),
            pos_contig.stride(0),
            pos_contig.stride(1),
            Q_out.stride(0),
            Q_out.stride(1),
            Q_out.stride(2),
            K_out.stride(0),
            K_out.stride(1),
            K_out.stride(2),
            H=H,
            D=D,
            M=M,
            P=P,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK_M,
            BLOCK_P=BLOCK_P,
        )

        return Q_out, K_out


@triton.jit
def _pape_qk_eval_kernel(
    Q_ptr,
    K_ptr,
    A_PRE_ptr,
    B_ptr,
    WP_ptr,
    POS_ptr,
    QO_ptr,
    KO_ptr,
    sQb,
    sQh,
    sQs,
    sKb,
    sKh,
    sKs,
    sAb,
    sAs,
    sAh,
    sBb,
    sBs,
    sBh,
    sWPh,
    sWPm,
    sPOSb,
    sPOSs,
    sQOb,
    sQOh,
    sQOs,
    sKOb,
    sKOh,
    sKOs,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    out_dtype = QO_ptr.dtype.element_ty

    q_base = pid_b * sQb + pid_h * sQh + pid_s * sQs
    k_base = pid_b * sKb + pid_h * sKh + pid_s * sKs
    a_base = pid_b * sAb + pid_s * sAs + pid_h * sAh
    b_base = pid_b * sBb + pid_s * sBs + pid_h * sBh
    pos_base = pid_b * sPOSb + pid_s * sPOSs
    qo_base = pid_b * sQOb + pid_h * sQOh + pid_s * sQOs
    ko_base = pid_b * sKOb + pid_h * sKOh + pid_s * sKOs

    m_idx = tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    p_idx = tl.arange(0, BLOCK_P)
    p_mask = p_idx < P

    a_pre = tl.load(A_PRE_ptr + a_base + m_idx, mask=m_mask, other=0.0).to(tl.float32)
    a_softplus = tl.where(a_pre > 20.0, a_pre, tl.log(1.0 + tl.exp(a_pre)))
    a_softplus = tl.where(m_mask, a_softplus, 0.0)

    b_vec = tl.load(B_ptr + b_base + p_idx, mask=p_mask, other=0.0).to(tl.float32)

    wp_off = m_idx[:, None] * sWPm + p_idx[None, :]
    wp_mask = m_mask[:, None] & p_mask[None, :]
    wp = tl.load(WP_ptr + pid_h * sWPh + wp_off, mask=wp_mask, other=0.0).to(tl.float32)

    positions = tl.load(POS_ptr + pos_base + p_idx, mask=p_mask, other=0.0).to(tl.float32)

    # a_mat[p1, p2] = sum_m wp[m, p1] * wp[m, p2] * softplus(a_pre[m])
    weighted_wp = a_softplus[:, None] * wp
    a_mat = tl.sum(wp[:, :, None] * weighted_wp[:, None, :], axis=0)

    right_side = tl.sum(a_mat * positions[None, :], axis=1)
    squares = tl.sum(positions * right_side, axis=0)
    dot_b_pos = tl.sum(b_vec * positions, axis=0)

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    pp_2d = p_idx[:, None] * P + p_idx[None, :]
    pp_2d_mask = (p_idx[:, None] < P) & (p_idx[None, :] < P)

    one_scalar = tl.full([], 1.0, dtype=out_dtype)

    # ----- Q_out -----
    q_in = tl.load(Q_ptr + q_base + d_idx, mask=d_mask)
    tl.store(QO_ptr + qo_base + d_idx, q_in, mask=d_mask)
    tl.store(QO_ptr + qo_base + D, (-squares).to(out_dtype))
    tl.store(QO_ptr + qo_base + (D + 1) + pp_2d, (-a_mat).to(out_dtype), mask=pp_2d_mask)
    tl.store(QO_ptr + qo_base + (D + 1 + P * P) + p_idx, right_side.to(out_dtype), mask=p_mask)
    tl.store(QO_ptr + qo_base + (D + 1 + P * P + P), dot_b_pos.to(out_dtype))
    tl.store(QO_ptr + qo_base + (D + 2 + P * P + P) + p_idx, b_vec.to(out_dtype), mask=p_mask)

    # ----- K_out -----
    k_in = tl.load(K_ptr + k_base + d_idx, mask=d_mask)
    tl.store(KO_ptr + ko_base + d_idx, k_in, mask=d_mask)
    tl.store(KO_ptr + ko_base + D, one_scalar)
    pos_outer = positions[:, None] * positions[None, :]
    tl.store(KO_ptr + ko_base + (D + 1) + pp_2d, pos_outer.to(out_dtype), mask=pp_2d_mask)
    tl.store(KO_ptr + ko_base + (D + 1 + P * P) + p_idx, (2.0 * positions).to(out_dtype), mask=p_mask)
    tl.store(KO_ptr + ko_base + (D + 1 + P * P + P), one_scalar)
    tl.store(KO_ptr + ko_base + (D + 2 + P * P + P) + p_idx, (-positions).to(out_dtype), mask=p_mask)


class _PaPETrainAugmentation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, a_pre, b_pre, W_p_3d, positions, num_pad):
        with torch.no_grad():
            aug_q, aug_k = _pape_train_triton_forward(query, key, a_pre, b_pre, W_p_3d, positions, num_pad)
        ctx.save_for_backward(query, key, a_pre, b_pre, W_p_3d, positions)
        ctx.num_pad = num_pad
        return aug_q, aug_k

    @staticmethod
    def backward(ctx, daug_q, daug_k):
        query, key, a_pre, b_pre, W_p_3d, positions = ctx.saved_tensors

        in_dtypes = (query.dtype, key.dtype, a_pre.dtype, b_pre.dtype, W_p_3d.dtype)

        # Run entirely in fp32; cast each grad back to its input's original dtype.
        with torch.no_grad():
            grads = _pape_train_triton_backward(
                query.to(torch.float32),
                key.to(torch.float32),
                a_pre.to(torch.float32),
                b_pre.to(torch.float32),
                W_p_3d.to(torch.float32),
                positions.to(torch.float32),
                daug_q.to(torch.float32),
                daug_k.to(torch.float32),
            )

        casted = tuple(g.to(d) for g, d in zip(grads, in_dtypes, strict=True))
        return casted[0], casted[1], casted[2], casted[3], casted[4], None, None


@triton.jit
def _pape_qk_train_kernel(
    Q_ptr,
    K_ptr,
    A_PRE_ptr,
    B_PRE_ptr,
    WP_ptr,
    POS_ptr,
    QO_ptr,
    KO_ptr,
    sQb,
    sQh,
    sQs,
    sKb,
    sKh,
    sKs,
    sAb,
    sAs,
    sAh,
    sBb,
    sBs,
    sBh,
    sWPh,
    sWPm,
    sPOSb,
    sPOSs,
    sQOb,
    sQOh,
    sQOs,
    sKOb,
    sKOh,
    sKOs,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    out_dtype = QO_ptr.dtype.element_ty

    q_base = pid_b * sQb + pid_h * sQh + pid_s * sQs
    k_base = pid_b * sKb + pid_h * sKh + pid_s * sKs
    a_base = pid_b * sAb + pid_s * sAs + pid_h * sAh
    b_base = pid_b * sBb + pid_s * sBs + pid_h * sBh
    pos_base = pid_b * sPOSb + pid_s * sPOSs
    qo_base = pid_b * sQOb + pid_h * sQOh + pid_s * sQOs
    ko_base = pid_b * sKOb + pid_h * sKOh + pid_s * sKOs

    m_idx = tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    p_idx = tl.arange(0, BLOCK_P)
    p_mask = p_idx < P

    a_pre = tl.load(A_PRE_ptr + a_base + m_idx, mask=m_mask, other=0.0).to(tl.float32)
    a_softplus = tl.where(a_pre > 20.0, a_pre, tl.log(1.0 + tl.exp(a_pre)))
    a_softplus = tl.where(m_mask, a_softplus, 0.0)

    b_pre = tl.load(B_PRE_ptr + b_base + m_idx, mask=m_mask, other=0.0).to(tl.float32)

    wp_off = m_idx[:, None] * sWPm + p_idx[None, :]
    wp_mask = m_mask[:, None] & p_mask[None, :]
    wp = tl.load(WP_ptr + pid_h * sWPh + wp_off, mask=wp_mask, other=0.0).to(tl.float32)

    positions = tl.load(POS_ptr + pos_base + p_idx, mask=p_mask, other=0.0).to(tl.float32)

    # b_p[p] = sum_m b_pre[m] * W_p[m, p]
    b_p_vec = tl.sum(b_pre[:, None] * wp, axis=0)

    # a_mat[p1, p2] = sum_m softplus(a_pre[m]) * W_p[m, p1] * W_p[m, p2]
    weighted_wp = a_softplus[:, None] * wp
    a_mat = tl.sum(wp[:, :, None] * weighted_wp[:, None, :], axis=0)

    right_side = tl.sum(a_mat * positions[None, :], axis=1)
    squares = tl.sum(positions * right_side, axis=0)
    dot_b_pos = tl.sum(b_p_vec * positions, axis=0)

    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    pp_2d = p_idx[:, None] * P + p_idx[None, :]
    pp_2d_mask = (p_idx[:, None] < P) & (p_idx[None, :] < P)

    one_scalar = tl.full([], 1.0, dtype=out_dtype)

    # ----- Q_out -----
    q_in = tl.load(Q_ptr + q_base + d_idx, mask=d_mask)
    tl.store(QO_ptr + qo_base + d_idx, q_in, mask=d_mask)
    tl.store(QO_ptr + qo_base + D, (-squares).to(out_dtype))
    tl.store(QO_ptr + qo_base + (D + 1) + pp_2d, (-a_mat).to(out_dtype), mask=pp_2d_mask)
    tl.store(QO_ptr + qo_base + (D + 1 + P * P) + p_idx, right_side.to(out_dtype), mask=p_mask)
    tl.store(QO_ptr + qo_base + (D + 1 + P * P + P), dot_b_pos.to(out_dtype))
    tl.store(QO_ptr + qo_base + (D + 2 + P * P + P) + p_idx, b_p_vec.to(out_dtype), mask=p_mask)

    # ----- K_out -----
    k_in = tl.load(K_ptr + k_base + d_idx, mask=d_mask)
    tl.store(KO_ptr + ko_base + d_idx, k_in, mask=d_mask)
    tl.store(KO_ptr + ko_base + D, one_scalar)
    pos_outer = positions[:, None] * positions[None, :]
    tl.store(KO_ptr + ko_base + (D + 1) + pp_2d, pos_outer.to(out_dtype), mask=pp_2d_mask)
    tl.store(KO_ptr + ko_base + (D + 1 + P * P) + p_idx, (2.0 * positions).to(out_dtype), mask=p_mask)
    tl.store(KO_ptr + ko_base + (D + 1 + P * P + P), one_scalar)
    tl.store(KO_ptr + ko_base + (D + 2 + P * P + P) + p_idx, (-positions).to(out_dtype), mask=p_mask)


@triton.jit
def _pape_backward_kernel(
    # Saved forward inputs
    A_PRE_ptr,
    B_PRE_ptr,
    WP_ptr,
    POS_ptr,
    # Incoming gradients
    DAQ_ptr,
    DAK_ptr,
    # Output gradients
    DQ_ptr,
    DK_ptr,
    DA_PRE_ptr,
    DB_PRE_ptr,
    DWP_ptr,
    # Strides for a_pre [B, S, H, M]
    sAb,
    sAs,
    sAh,
    # Strides for b_pre [B, S, H, M]
    sBb,
    sBs,
    sBh,
    # Strides for W_p [H, M, P]
    sWPh,
    sWPm,
    # Strides for positions [B, S, P]
    sPb,
    sPs,
    # Strides for daug_q [B, H, S, D_OUT]
    sDAQb,
    sDAQh,
    sDAQs,
    # Strides for daug_k [B, H, S, D_OUT]
    sDAKb,
    sDAKh,
    sDAKs,
    # Strides for dq [B, H, S, D]
    sDQb,
    sDQh,
    sDQs,
    # Strides for dk [B, H, S, D]
    sDKb,
    sDKh,
    sDKs,
    # Strides for d_a_pre [B, S, H, M]
    sDAb,
    sDAs,
    sDAh,
    # Strides for d_b_pre [B, S, H, M]
    sDBb,
    sDBs,
    sDBh,
    # Strides for dW_p [H, M, P]
    sDWPh,
    sDWPm,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    m_idx = tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    p_idx = tl.arange(0, BLOCK_P)
    p_mask = p_idx < P
    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D
    pp_2d = p_idx[:, None] * P + p_idx[None, :]
    pp_2d_mask = (p_idx[:, None] < P) & (p_idx[None, :] < P)

    # ---- Load saved forward inputs ----
    a_base = pid_b * sAb + pid_s * sAs + pid_h * sAh
    b_base = pid_b * sBb + pid_s * sBs + pid_h * sBh
    pos_base = pid_b * sPb + pid_s * sPs

    a_pre = tl.load(A_PRE_ptr + a_base + m_idx, mask=m_mask, other=0.0).to(tl.float32)
    b_pre_vals = tl.load(B_PRE_ptr + b_base + m_idx, mask=m_mask, other=0.0).to(tl.float32)
    positions = tl.load(POS_ptr + pos_base + p_idx, mask=p_mask, other=0.0).to(tl.float32)

    wp_off = m_idx[:, None] * sWPm + p_idx[None, :]
    wp_mask = m_mask[:, None] & p_mask[None, :]
    wp = tl.load(WP_ptr + pid_h * sWPh + wp_off, mask=wp_mask, other=0.0).to(tl.float32)

    # ---- Recompute a_softplus ----
    a_softplus = tl.where(a_pre > 20.0, a_pre, tl.log(1.0 + tl.exp(a_pre)))
    a_softplus = tl.where(m_mask, a_softplus, 0.0)

    # ---- Load gradient signals from daug_q ----
    daq_base = pid_b * sDAQb + pid_h * sDAQh + pid_s * sDAQs

    # d_loss / d(-squares)
    g_sq_neg = tl.load(DAQ_ptr + daq_base + D).to(tl.float32)
    # d_loss / d(-a_mat[p1,p2])
    g_neg_amat = tl.load(DAQ_ptr + daq_base + (D + 1) + pp_2d, mask=pp_2d_mask, other=0.0).to(tl.float32)
    # d_loss / d(right_side[p])
    g_rs = tl.load(DAQ_ptr + daq_base + (D + 1 + P * P) + p_idx, mask=p_mask, other=0.0).to(tl.float32)
    # d_loss / d(dot_b_pos)
    g_dbp = tl.load(DAQ_ptr + daq_base + (D + 1 + P * P + P)).to(tl.float32)
    # d_loss / d(b_p[p])
    g_bp = tl.load(DAQ_ptr + daq_base + (D + 2 + P * P + P) + p_idx, mask=p_mask, other=0.0).to(tl.float32)

    dak_base = pid_b * sDAKb + pid_h * sDAKh + pid_s * sDAKs

    # ---- dq / dk: direct passthrough ----
    dq_vals = tl.load(DAQ_ptr + daq_base + d_idx, mask=d_mask)
    dk_vals = tl.load(DAK_ptr + dak_base + d_idx, mask=d_mask)

    # ---- Backprop: squares -> right_side -> a_mat ----
    # aug_q[D] = -squares  =>  d(squares) = -g_sq_neg
    g_rs_total = g_rs - g_sq_neg * positions  # [BLOCK_P]
    g_rs_total = tl.where(p_mask, g_rs_total, 0.0)

    # aug_q[D+1:D+1+P*P] = -a_mat  =>  d(a_mat) = -g_neg_amat + outer(g_rs_total, positions)
    g_amat = -g_neg_amat + g_rs_total[:, None] * positions[None, :]  # [BLOCK_P, BLOCK_P]
    g_amat = tl.where(pp_2d_mask, g_amat, 0.0)

    # ---- Backprop through a_mat[p1,p2] = sum_m a_softplus[m]*wp[m,p1]*wp[m,p2] ----
    # term1[m,p1] = sum_p2 g_amat[p1,p2] * wp[m,p2]
    term1 = tl.sum(wp[:, None, :] * g_amat[None, :, :], axis=2)  # [BLOCK_M, BLOCK_P]
    # term2[m,p2] = sum_p1 g_amat[p1,p2] * wp[m,p1]  (transposed path)
    term2 = tl.sum(wp[:, :, None] * g_amat[None, :, :], axis=1)  # [BLOCK_M, BLOCK_P]

    # d_a_softplus[m] = sum_p1 term1[m,p1] * wp[m,p1]
    d_a_softplus = tl.sum(term1 * wp, axis=1)  # [BLOCK_M]
    d_a_softplus = tl.where(m_mask, d_a_softplus, 0.0)

    sigmoid_a = tl.sigmoid(a_pre)
    d_a_pre_vals = tl.where(m_mask, d_a_softplus * sigmoid_a, 0.0)

    dW_p_amat = a_softplus[:, None] * (term1 + term2)  # [BLOCK_M, BLOCK_P]

    # ---- Backprop through b_p[p] = sum_m b_pre[m]*wp[m,p] and dot_b_pos ----
    d_b_p = tl.where(p_mask, g_bp + g_dbp * positions, 0.0)  # [BLOCK_P]

    d_b_pre_vals = tl.where(m_mask, tl.sum(d_b_p[None, :] * wp, axis=1), 0.0)  # [BLOCK_M]

    dW_p_bpre = b_pre_vals[:, None] * d_b_p[None, :]  # [BLOCK_M, BLOCK_P]

    dW_p = tl.where(wp_mask, dW_p_amat + dW_p_bpre, 0.0)  # [BLOCK_M, BLOCK_P]

    # ---- Store outputs ----
    out_dtype = DA_PRE_ptr.dtype.element_ty

    dq_base_out = pid_b * sDQb + pid_h * sDQh + pid_s * sDQs
    dk_base_out = pid_b * sDKb + pid_h * sDKh + pid_s * sDKs
    tl.store(DQ_ptr + dq_base_out + d_idx, dq_vals, mask=d_mask)
    tl.store(DK_ptr + dk_base_out + d_idx, dk_vals, mask=d_mask)

    da_base = pid_b * sDAb + pid_s * sDAs + pid_h * sDAh
    db_base = pid_b * sDBb + pid_s * sDBs + pid_h * sDBh
    tl.store(DA_PRE_ptr + da_base + m_idx, d_a_pre_vals.to(out_dtype), mask=m_mask)
    tl.store(DB_PRE_ptr + db_base + m_idx, d_b_pre_vals.to(out_dtype), mask=m_mask)

    dwp_off = m_idx[:, None] * sDWPm + p_idx[None, :]
    tl.atomic_add(DWP_ptr + pid_h * sDWPh + dwp_off, dW_p.to(DWP_ptr.dtype.element_ty), mask=wp_mask)


def _pape_train_triton_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    a_pre: torch.Tensor,
    b_pre: torch.Tensor,
    W_p_3d: torch.Tensor,
    positions: torch.Tensor,
    daug_q: torch.Tensor,
    daug_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = query.shape
    M = a_pre.shape[-1]
    P = positions.shape[-1]

    a_pre_c = a_pre.contiguous()
    b_pre_c = b_pre.contiguous()
    pos_c = positions.contiguous()
    W_p_c = W_p_3d.contiguous()
    daug_q_c = daug_q.contiguous()
    daug_k_c = daug_k.contiguous()

    dq = torch.empty_like(query)
    dk = torch.empty_like(key)
    d_a_pre = torch.empty_like(a_pre_c)
    d_b_pre = torch.empty_like(b_pre_c)
    dW_p = torch.zeros_like(W_p_c)

    BLOCK_D = max(triton.next_power_of_2(D), 16)
    BLOCK_M = max(triton.next_power_of_2(M), 16)
    BLOCK_P = max(triton.next_power_of_2(P), 4)

    grid = (B * H, S)
    _pape_backward_kernel[grid](
        a_pre_c,
        b_pre_c,
        W_p_c,
        pos_c,
        daug_q_c,
        daug_k_c,
        dq,
        dk,
        d_a_pre,
        d_b_pre,
        dW_p,
        a_pre_c.stride(0),
        a_pre_c.stride(1),
        a_pre_c.stride(2),
        b_pre_c.stride(0),
        b_pre_c.stride(1),
        b_pre_c.stride(2),
        W_p_c.stride(0),
        W_p_c.stride(1),
        pos_c.stride(0),
        pos_c.stride(1),
        daug_q_c.stride(0),
        daug_q_c.stride(1),
        daug_q_c.stride(2),
        daug_k_c.stride(0),
        daug_k_c.stride(1),
        daug_k_c.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        d_a_pre.stride(0),
        d_a_pre.stride(1),
        d_a_pre.stride(2),
        d_b_pre.stride(0),
        d_b_pre.stride(1),
        d_b_pre.stride(2),
        dW_p.stride(0),
        dW_p.stride(1),
        H=H,
        D=D,
        M=M,
        P=P,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
    )

    return dq, dk, d_a_pre, d_b_pre, dW_p


def _pape_train_triton_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    a_pre: torch.Tensor,
    b_pre: torch.Tensor,
    W_p_3d: torch.Tensor,
    positions: torch.Tensor,
    num_pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, D = query.shape
    M = a_pre.shape[-1]
    P = positions.shape[-1]
    D_OUT = D + 2 + P * P + 2 * P + num_pad

    a_pre_c = a_pre.contiguous()
    b_pre_c = b_pre.contiguous()
    pos_c = positions.contiguous()
    W_p_c = W_p_3d.contiguous()

    Q_out = torch.zeros(B, H, S, D_OUT, dtype=query.dtype, device=query.device)
    K_out = torch.zeros(B, H, S, D_OUT, dtype=key.dtype, device=key.device)

    BLOCK_D = max(triton.next_power_of_2(D), 16)
    BLOCK_M = max(triton.next_power_of_2(M), 16)
    BLOCK_P = max(triton.next_power_of_2(P), 4)

    grid = (B * H, S)
    _pape_qk_train_kernel[grid](
        query,
        key,
        a_pre_c,
        b_pre_c,
        W_p_c,
        pos_c,
        Q_out,
        K_out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        a_pre_c.stride(0),
        a_pre_c.stride(1),
        a_pre_c.stride(2),
        b_pre_c.stride(0),
        b_pre_c.stride(1),
        b_pre_c.stride(2),
        W_p_c.stride(0),
        W_p_c.stride(1),
        pos_c.stride(0),
        pos_c.stride(1),
        Q_out.stride(0),
        Q_out.stride(1),
        Q_out.stride(2),
        K_out.stride(0),
        K_out.stride(1),
        K_out.stride(2),
        H=H,
        D=D,
        M=M,
        P=P,
        BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
    )

    return Q_out, K_out
