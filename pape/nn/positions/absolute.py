import torch

from pape.configs import Config
from pape.nn.positions.base import PositionEncoder


class AbsolutePositionEncoder(PositionEncoder):
    """
    Position encoder that encodes absolute trigonometric positions.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.model.hidden_size
        self.num_positions = config.num_positions
        self.frequency_base = config.absolute.base

        self.pos_size = self.hidden_size // self.num_positions
        self.last_pos_size = self.pos_size + (self.hidden_size % self.num_positions)

    def register_model_weights(self):
        dim_series = torch.arange(self.pos_size, dtype=torch.int64).float()
        dim_series = torch.div(dim_series, 2, rounding_mode="floor")
        pos_frequencies = self.frequency_base ** (2 * dim_series / self.pos_size)
        self.register_buffer("pos_frequencies", pos_frequencies, persistent=False)

        dim_series = torch.arange(self.last_pos_size, dtype=torch.int64).float()
        dim_series = torch.div(dim_series, 2, rounding_mode="floor")
        last_pos_frequencies = self.frequency_base ** (2 * dim_series / self.last_pos_size)
        self.register_buffer("last_pos_frequencies", last_pos_frequencies, persistent=False)

    def register_layer_weights(self):
        pass

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions

    def encode_absolute(self, x: torch.Tensor, prepared_positions: torch.Tensor) -> torch.Tensor:
        # create sine/cosine positional encodings for each position
        encoded_positions = []
        for i in range(self.num_positions - 1):
            angles = prepared_positions[..., i : i + 1] / self.pos_frequencies
            sine = torch.sin(angles[..., 0::2])
            cosine = torch.cos(angles[..., 1::2])
            encoded_position = torch.cat((sine, cosine), dim=-1)
            encoded_positions.append(encoded_position)

        angles = prepared_positions[..., -1:] / self.last_pos_frequencies
        sine = torch.sin(angles[..., 0::2])
        cosine = torch.cos(angles[..., 1::2])
        encoded_position = torch.cat((sine, cosine), dim=-1)
        encoded_positions.append(encoded_position)

        encoded_positions = torch.concat(encoded_positions, dim=-1)

        return x + encoded_positions

    def encode_query_key(
        self, hidden_state: torch.Tensor, query: torch.Tensor, key: torch.Tensor, prepared_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")
