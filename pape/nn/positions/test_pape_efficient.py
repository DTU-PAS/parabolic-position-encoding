import pytest
import torch

from pape.configs import Config
from pape.configs import Dataset
from pape.configs import ModelConfig
from pape.configs import PaPEConfig
from pape.configs import Size
from pape.nn.positions.pape_efficient import ParabolicPositionEncoder as ParabolicPositionEncoderEfficient
from pape.nn.positions.pape_naive import ParabolicPositionEncoder as ParabolicPositionEncoderNaive

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton requires CUDA")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    c = Config(dataset=Dataset.imagenet)
    c.model = ModelConfig(size=Size.nano)
    c.pape = PaPEConfig(parabolas=18)
    return c


def make_paired(cfg, seed=0):
    """Return (naive, triton) encoders with identical weights on CUDA."""
    torch.manual_seed(seed)
    naive = ParabolicPositionEncoderNaive(cfg)
    naive.register_layer_weights()

    tri = ParabolicPositionEncoderEfficient(cfg)
    tri.register_layer_weights()
    tri.position.weight.data.copy_(naive.position.weight.data)
    tri.ab.weight.data.copy_(naive.ab.weight.data)

    return naive.cuda(), tri.cuda()


def make_inputs(cfg, batch_size=2, seq_length=10, seed=1, dtype=torch.float32):
    """Return (hidden_state, query, key, positions) on CUDA."""
    torch.manual_seed(seed)
    device = "cuda"
    H, D, C, P = cfg.model.num_heads, cfg.model.head_size, cfg.model.hidden_size, cfg.num_positions
    hs = torch.randn(batch_size, seq_length, C, device=device, dtype=dtype)
    q = torch.randn(batch_size, H, seq_length, D, device=device, dtype=dtype)
    k = torch.randn(batch_size, H, seq_length, D, device=device, dtype=dtype)
    pos = torch.randn(batch_size, seq_length, P, device=device)
    return hs, q, k, pos


def unscaled_attn(q, k):
    """Unscaled attention logits Q·Kᵀ, shape (B, H, S, S)."""
    return torch.matmul(q, k.transpose(-2, -1))


def encode_and_attn_loss(encoder, hs, q, k, pos):
    """Augment Q/K and return the sum of unscaled Q·Kᵀ as a scalar loss.

    Using unscaled logits makes the loss identical between naive (larger
    augmented dim) and Triton (smaller augmented dim), since both produce the
    same Q·Kᵀ values regardless of head-size padding.
    """
    q_aug, k_aug = encoder.encode_query_key(hs, q, k, encoder.prepare_positions(pos))
    return unscaled_attn(q_aug, k_aug).sum()


# ---------------------------------------------------------------------------
# Forward: unscaled attention scores match naive
# ---------------------------------------------------------------------------


def test_train_forward_attn_matches_naive(cfg):
    naive, tri = make_paired(cfg)
    naive.train()
    tri.train()
    hs, q, k, pos = make_inputs(cfg)

    with torch.no_grad():
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    torch.testing.assert_close(unscaled_attn(q_n, k_n), unscaled_attn(q_t, k_t), atol=1e-4, rtol=1e-4)


def test_eval_forward_attn_matches_naive(cfg):
    naive, tri = make_paired(cfg)
    naive.eval()
    tri.eval()
    hs, q, k, pos = make_inputs(cfg)

    with torch.no_grad():
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    torch.testing.assert_close(unscaled_attn(q_n, k_n), unscaled_attn(q_t, k_t), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Backward: parameter and activation gradients match naive
# ---------------------------------------------------------------------------


def test_train_backward_grads_match_naive(cfg):
    """Gradients w.r.t. all learnable parameters and activations match naive."""
    naive, tri = make_paired(cfg)
    naive.train()
    tri.train()
    hs, q, k, pos = make_inputs(cfg)

    q_n = q.detach().clone().requires_grad_(True)
    k_n = k.detach().clone().requires_grad_(True)
    hs_n = hs.detach().clone().requires_grad_(True)
    encode_and_attn_loss(naive, hs_n, q_n, k_n, pos).backward()

    q_t = q.detach().clone().requires_grad_(True)
    k_t = k.detach().clone().requires_grad_(True)
    hs_t = hs.detach().clone().requires_grad_(True)
    encode_and_attn_loss(tri, hs_t, q_t, k_t, pos).backward()

    tol = dict(atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(naive.position.weight.grad, tri.position.weight.grad, **tol)
    torch.testing.assert_close(naive.ab.weight.grad, tri.ab.weight.grad, **tol)
    torch.testing.assert_close(q_n.grad, q_t.grad, **tol)
    torch.testing.assert_close(k_n.grad, k_t.grad, **tol)
    torch.testing.assert_close(hs_n.grad, hs_t.grad, **tol)


def test_train_backward_accumulated_grads(cfg):
    """Gradient accumulation over two batches is consistent with naive."""
    naive, tri = make_paired(cfg)
    naive.train()
    tri.train()

    tol = dict(atol=1e-3, rtol=1e-3)
    for seed in (10, 20):
        hs, q, k, pos = make_inputs(cfg, seed=seed)

        q_n = q.detach().clone().requires_grad_(True)
        k_n = k.detach().clone().requires_grad_(True)
        hs_n = hs.detach().clone().requires_grad_(True)
        encode_and_attn_loss(naive, hs_n, q_n, k_n, pos).backward()

        q_t = q.detach().clone().requires_grad_(True)
        k_t = k.detach().clone().requires_grad_(True)
        hs_t = hs.detach().clone().requires_grad_(True)
        encode_and_attn_loss(tri, hs_t, q_t, k_t, pos).backward()

    torch.testing.assert_close(naive.position.weight.grad, tri.position.weight.grad, **tol)
    torch.testing.assert_close(naive.ab.weight.grad, tri.ab.weight.grad, **tol)


# ---------------------------------------------------------------------------
# Forward + backward across shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [Size.nano, Size.tiny])
@pytest.mark.parametrize("parabolas", [4, 18])
@pytest.mark.parametrize("seq_length", [8, 64, 129])
def test_train_shapes(size, parabolas, seq_length):
    cfg = Config(dataset=Dataset.imagenet)
    cfg.model = ModelConfig(size=size)
    cfg.pape = PaPEConfig(parabolas=parabolas)

    naive, tri = make_paired(cfg)
    naive.train()
    tri.train()
    hs, q, k, pos = make_inputs(cfg, seq_length=seq_length)

    with torch.no_grad():
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    torch.testing.assert_close(unscaled_attn(q_n, k_n), unscaled_attn(q_t, k_t), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("size", [Size.nano, Size.tiny])
@pytest.mark.parametrize("parabolas", [4, 18])
@pytest.mark.parametrize("seq_length", [8, 64, 129])
def test_eval_shapes(size, parabolas, seq_length):
    cfg = Config(dataset=Dataset.imagenet)
    cfg.model = ModelConfig(size=size)
    cfg.pape = PaPEConfig(parabolas=parabolas)

    naive, tri = make_paired(cfg)
    naive.eval()
    tri.eval()
    hs, q, k, pos = make_inputs(cfg, seq_length=seq_length)

    with torch.no_grad():
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    torch.testing.assert_close(unscaled_attn(q_n, k_n), unscaled_attn(q_t, k_t), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Autocast: forward attention scores still match naive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_train_autocast_attn_matches_naive(cfg, dtype):
    naive, tri = make_paired(cfg)
    naive.train()
    tri.train()
    hs, q, k, pos = make_inputs(cfg, dtype=dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    # bfloat16 has fewer mantissa bits than float16, so needs a looser tolerance.
    tol = 5e-2 if dtype == torch.float16 else 1e-1
    torch.testing.assert_close(
        unscaled_attn(q_n.float(), k_n.float()),
        unscaled_attn(q_t.float(), k_t.float()),
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eval_autocast_attn_matches_naive(cfg, dtype):
    naive, tri = make_paired(cfg)
    naive.eval()
    tri.eval()
    hs, q, k, pos = make_inputs(cfg, dtype=dtype)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        q_n, k_n = naive.encode_query_key(hs, q.clone(), k.clone(), naive.prepare_positions(pos))
        q_t, k_t = tri.encode_query_key(hs, q.clone(), k.clone(), tri.prepare_positions(pos))

    tol = 5e-2 if dtype == torch.float16 else 1e-1
    torch.testing.assert_close(
        unscaled_attn(q_n.float(), k_n.float()),
        unscaled_attn(q_t.float(), k_t.float()),
        atol=tol,
        rtol=tol,
    )


# ---------------------------------------------------------------------------
# End-to-end gradient flow
# ---------------------------------------------------------------------------


def test_end_to_end_grads_non_zero(cfg):
    """Parameters receive non-zero gradients in train mode."""
    _, tri = make_paired(cfg)
    tri.train()
    hs, q, k, pos = make_inputs(cfg)
    hs = hs.requires_grad_(True)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)

    encode_and_attn_loss(tri, hs, q, k, pos).backward()

    assert tri.position.weight.grad is not None and tri.position.weight.grad.abs().sum() > 0
    assert tri.ab.weight.grad is not None and tri.ab.weight.grad.abs().sum() > 0
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert hs.grad is not None and hs.grad.abs().sum() > 0


def test_train_eval_attn_consistent(cfg):
    """Train-mode and eval-mode Triton encoders produce the same attention scores."""
    torch.manual_seed(0)
    tri_train = ParabolicPositionEncoderEfficient(cfg)
    tri_train.register_layer_weights()
    tri_train = tri_train.cuda().train()

    tri_eval = ParabolicPositionEncoderEfficient(cfg)
    tri_eval.register_layer_weights()
    tri_eval.position.weight.data.copy_(tri_train.position.weight.data)
    tri_eval.ab.weight.data.copy_(tri_train.ab.weight.data)
    tri_eval = tri_eval.cuda().eval()

    hs, q, k, pos = make_inputs(cfg)

    with torch.no_grad():
        q_tr, k_tr = tri_train.encode_query_key(hs, q.clone(), k.clone(), tri_train.prepare_positions(pos))
        q_ev, k_ev = tri_eval.encode_query_key(hs, q.clone(), k.clone(), tri_eval.prepare_positions(pos))

    torch.testing.assert_close(unscaled_attn(q_tr, k_tr), unscaled_attn(q_ev, k_ev), atol=1e-4, rtol=1e-4)
