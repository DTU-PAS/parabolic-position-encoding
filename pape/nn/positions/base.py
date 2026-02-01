import abc
from typing import Any

import torch
import torch.nn as nn


class PositionEncoder(nn.Module, abc.ABC):
    """
    Abstract base class for position encoders.
    """

    @abc.abstractmethod
    def register_model_weights(self):
        """Register model-wide weights for the position encoder."""
        pass

    @abc.abstractmethod
    def register_layer_weights(self):
        """Register layer weights for the position encoder."""
        pass

    @abc.abstractmethod
    def prepare_positions(self, positions: torch.Tensor) -> Any:
        """
        Prepare positions for encoding. Will be called before the first transformer layer.
        """
        pass

    @abc.abstractmethod
    def encode_absolute(self, x: torch.Tensor, prepared_positions: torch.Tensor) -> torch.Tensor:
        """
        Encode absolute positions.
        """
        pass

    @abc.abstractmethod
    def encode_query_key(
        self, hidden_state: torch.Tensor, query: torch.Tensor, key: torch.Tensor, prepared_positions: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode query positions.
        """
        pass

    @abc.abstractmethod
    def has_bias(self) -> bool:
        pass

    @abc.abstractmethod
    def get_bias(self, prepared_positions: Any) -> torch.Tensor:
        """
        Get bias for attention scores.
        """
        pass
