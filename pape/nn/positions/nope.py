import torch

from pape.nn.positions.base import PositionEncoder


class NoPositionEncoder(PositionEncoder):
    """
    Does not encode any positions.
    """

    def register_model_weights(self):
        pass

    def register_layer_weights(self):
        pass

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions

    def encode_absolute(self, x: torch.Tensor, prepared_positions: torch.Tensor) -> torch.Tensor:
        return x

    def encode_query_key(
        self, hidden_state: torch.Tensor, query: torch.Tensor, key: torch.Tensor, prepared_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def has_bias(self) -> bool:
        return False

    def get_bias(self, prepared_positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This position encoder does not support bias.")
