import math

import torch

from pape.configs import Config
from pape.nn.positions.base import PositionEncoder


class ALiBiPositionEncoder(PositionEncoder):
    """
    nD-ALiBi.

    This is a n-dimensional generalization of the 2D-ALiBi method.

    Paper: CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders
    Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
    """

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.model.num_heads

    def register_model_weights(self):
        slopes = self.get_slopes(self.num_heads)
        slopes = torch.tensor(slopes).view(1, self.num_heads, 1, 1)  # (1, num_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)

    def register_layer_weights(self):
        pass

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (batch_size, seq_length, num_positions)

        # (batch_size, seq_length, seq_length, num_positions)
        differences = positions[:, :, None, :] - positions[:, None, :, :]
        differences = differences.float()
        distances = torch.norm(differences, dim=-1, p=2)  # (batch_size, seq_length, seq_length)

        # (batch_size, 1, seq_length, seq_length)
        distances = distances.unsqueeze(1)

        neg_distances = -distances

        sloped_neg_distances = self.slopes * neg_distances  # (batch_size, num_heads, seq_length, seq_length)

        return sloped_neg_distances

    def encode_absolute(self, x: torch.Tensor, prepared_positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self, hidden_state: torch.Tensor, query: torch.Tensor, key: torch.Tensor, prepared_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def has_bias(self) -> bool:
        return True

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        return prepared_positions

    def get_slopes(self, num_heads: int) -> list[float]:
        """Source: https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742"""

        def get_slopes_power_of_2(n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self.get_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
            )
