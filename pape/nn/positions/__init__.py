from pape.configs import Config
from pape.configs import Positional
from pape.nn.positions.base import PositionEncoder


def get_position_encoder(config: Config) -> PositionEncoder:
    match config.positional:
        case Positional.absolute:
            from .absolute import AbsolutePositionEncoder

            return AbsolutePositionEncoder(config)
        case Positional.alibi:
            from .alibi import ALiBiPositionEncoder

            return ALiBiPositionEncoder(config)
        case Positional.lookhere:
            from .lookhere import LookHerePositionEncoder

            return LookHerePositionEncoder(config)
        case Positional.nope:
            from .nope import NoPositionEncoder

            return NoPositionEncoder()
        case Positional.pape:
            from .pape import ParabolicPositionEncoder

            return ParabolicPositionEncoder(config)
        case Positional.pape_ri:
            from .pape_ri import ParabolicRotationInvariantPositionEncoder

            return ParabolicRotationInvariantPositionEncoder(config)
        case Positional.rope:
            from .rope import RoPEPositionEncoder

            return RoPEPositionEncoder(config)
        case Positional.rope_mixed:
            from .rope_mixed import RoPEMixedPositionEncoder

            return RoPEMixedPositionEncoder(config)
        case _:
            raise ValueError(f"Unsupported positional encoding: {config.positional}")
