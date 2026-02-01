import math

import torch

from pape.configs import Config
from pape.nn.positions.base import PositionEncoder


class RoPEPositionEncoder(PositionEncoder):
    """
    Encode positions using Rotary Position Embeddings (RoPE).
    """

    def __init__(self, config: Config):
        super().__init__()

        divisor = 2 * config.num_positions
        head_size = config.model.head_size
        self.num_pad = divisor * math.ceil(head_size / divisor) - head_size
        self.should_pad = self.num_pad != 0

        self.num_heads = config.model.num_heads
        self.head_size = config.model.head_size + self.num_pad
        self.head_size_half = self.head_size // 2
        self.head_size_div = self.head_size_half // config.num_positions
        self.num_positions = config.num_positions
        self.frequency_base = config.rope.base

    def register_model_weights(self):
        dim_series = torch.linspace(0, self.head_size - 2 * self.num_positions, self.head_size_div)
        dim_series = dim_series / self.head_size
        frequencies = torch.pow(self.frequency_base, dim_series)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def register_layer_weights(self):
        pass

    def prepare_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.unsqueeze(-1)  # Shape: (batch_size, seq_length, num_positions, 1)
        positions = positions / self.frequencies
        positions = positions.flatten(-2)
        sin = torch.sin(positions)
        cos = torch.cos(positions)
        return sin, cos

    def encode_absolute(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, num_heads seq_length, head_size).
            key: Tensor of shape (batch_size, num_heads, seq_length, head_size).
            prepared_positions: Tuple containing sine and cosine tensors.
                Shape of each: (batch_size, seq_length, head_size // 2).
        """
        sin, cos = prepared_positions

        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

        if self.should_pad:
            query = torch.nn.functional.pad(query, (0, self.num_pad))
            key = torch.nn.functional.pad(key, (0, self.num_pad))

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

        if self.should_pad:
            query_rope = query_rope[..., : -self.num_pad]
            key_rope = key_rope[..., : -self.num_pad]

        return query_rope, key_rope

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")
