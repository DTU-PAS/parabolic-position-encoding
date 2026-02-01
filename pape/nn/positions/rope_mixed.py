import torch
import torch.nn as nn

from pape.configs import Config
from pape.nn.positions.base import PositionEncoder


class RoPEMixedPositionEncoder(PositionEncoder):
    """
    Encode positions using RoPE-Mixed.

    Paper: Rotary Position Embedding for Vision Transformer
    arXiv: https://arxiv.org/abs/2403.13298
    """

    def __init__(self, config: Config):
        super().__init__()

        if config.model.head_size % 2 != 0:
            raise ValueError(f"Head size {config.model.head_size} must be even.")

        self.num_heads = config.model.num_heads
        self.head_size = config.model.head_size
        self.head_size_half = self.head_size // 2
        self.num_positions = config.num_positions
        self.frequency_base = config.rope_mixed.base

    def register_model_weights(self):
        pass

    def register_layer_weights(self):
        dim_series = torch.linspace(0, self.head_size - 2, self.head_size_half)
        dim_series = dim_series / self.head_size
        frequencies = torch.pow(self.frequency_base, dim_series)

        # Clone frequencies for each position
        frequencies = [frequencies.clone() for _ in range(self.num_positions)]
        frequencies = torch.stack(frequencies, dim=0)

        # Clone frequencies for each head
        frequencies = [frequencies.clone() for _ in range(self.num_heads)]
        frequencies = torch.stack(frequencies, dim=0)

        frequencies = nn.Parameter(frequencies, requires_grad=True)
        self.register_parameter("frequencies", frequencies)

    def prepare_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return positions

    def encode_absolute(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, num_heads seq_length, head_size).
            key: Tensor of shape (batch_size, num_heads, seq_length, head_size).
            prepared_positions: Tuple containing sine and cosine tensors.
                Shape of each: (batch_size, seq_length, head_size // 2).
        """
        positions = prepared_positions.unsqueeze(-2).unsqueeze(-1)  # (batch_size, seq_length, 1, num_positions, 1)
        positions = positions / self.frequencies  # (batch_size, seq_length, num_heads, num_positions, head_size // 2)
        positions = positions.sum(-2)  # (batch_size, seq_length, num_heads, head_size // 2)
        positions = positions.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size // 2)
        sin = torch.sin(positions)
        cos = torch.cos(positions)

        q1 = query[..., : self.head_size_half]
        q2 = query[..., self.head_size_half :]
        query_rope = torch.cat(
            (
                q1 * cos - q2 * sin,
                q1 * sin + q2 * cos,
            ),
            dim=-1,
        )
        k1 = key[..., : self.head_size_half]
        k2 = key[..., self.head_size_half :]
        key_rope = torch.cat(
            (
                k1 * cos - k2 * sin,
                k1 * sin + k2 * cos,
            ),
            dim=-1,
        )

        return query_rope, key_rope

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")
