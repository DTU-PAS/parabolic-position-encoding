import math

import torch
import torch.nn.functional as F

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

        # The attention kernel is faster and more memory efficient
        # for head sizes that are multiples of 8.
        total_head_size = self.head_size + 3 * self.num_parabolas + 2
        self.num_pad = 8 * math.ceil(total_head_size / 8) - total_head_size
        self.should_pad = self.num_pad != 0

    def register_model_weights(self):
        pass

    def register_layer_weights(self):
        position = torch.nn.Linear(self.num_positions, self.num_heads * self.num_parabolas, bias=False)
        self.register_module("position", position)

        num_features = self.num_heads * self.num_parabolas
        ab = torch.nn.Linear(self.hidden_size, 2 * num_features, bias=False)

        self.register_module("ab", ab)

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.float()

    def encode_absolute(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        prepared_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = prepared_positions
        batch_size, seq_length, _ = positions.size()
        positions: torch.Tensor = self.position(positions)
        positions = positions.view(batch_size, seq_length, self.num_heads, self.num_parabolas)
        positions = positions.permute(0, 2, 1, 3)
        squared_positions = positions.pow(2)

        batch_size = key.size(0)
        seq_length = key.size(2)

        positions = positions.expand(batch_size, -1, -1, -1)
        squared_positions = squared_positions.expand(batch_size, -1, -1, -1)

        ab: torch.Tensor = self.ab(hidden_state)
        ab = ab.view(batch_size, seq_length, 2, self.num_heads, self.num_parabolas)
        ab = ab.permute(2, 0, 3, 1, 4)

        a = F.softplus(ab[0])
        b = ab[1]

        neg_squared_positions = -squared_positions

        ones = torch.ones((batch_size, self.num_heads, seq_length, 1), device=positions.device, dtype=positions.dtype)

        query = torch.cat(
            [
                query,
                self.dot(a, neg_squared_positions),
                a,
                a * 2 * positions,
                self.dot(b, positions),
                b,
            ],
            dim=-1,
        )

        key = torch.cat(
            [
                key,
                ones,
                neg_squared_positions,
                positions,
                ones,
                -positions,
            ],
            dim=-1,
        )

        if self.should_pad:
            pad_shape = (0, self.num_pad)
            query = F.pad(query, pad_shape, "constant", 0)
            key = F.pad(key, pad_shape, "constant", 0)

        return query, key

    def dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x * y).sum(dim=-1, keepdim=True)

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")
