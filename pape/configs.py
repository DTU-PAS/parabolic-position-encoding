import dataclasses
import enum
from typing import Annotated
from typing import Literal

import dataclasses_json
import tyro


class Checkpoint(enum.StrEnum):
    best = "best"
    last = "last"


class Dataset(enum.StrEnum):
    coco = "coco"
    dvsgesture = "dvsgesture"
    gen1 = "gen1"
    imagenet = "imagenet"
    ucf101 = "ucf101"


class Positional(enum.StrEnum):
    absolute = "absolute"
    alibi = "alibi"
    lookhere = "lookhere"
    nope = "nope"
    rope = "rope"
    rope_mixed = "rope_mixed"
    pape = "pape"
    pape_ri = "pape_ri"


class Size(enum.StrEnum):
    nano = "nano"
    tiny = "tiny"
    small = "small"
    base = "base"


class Split(enum.StrEnum):
    train = "train"
    val = "val"
    test = "test"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class AbsoluteConfig:
    base: float = 10_000.0


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class EventsConfig:
    buckets: int = 10
    max_detection_events: int = 500_000
    ref_ms: int = 0
    threshold: int = 256
    time_scale: int = 50_000

    @property
    def ref_us(self) -> int:
        return self.ref_ms * 1000


MODEL_SIZES = {
    Size.nano: {
        "intermediate_size": 256,
        "head_size": 32,
        "hidden_size": 128,
        "num_heads": 4,
        "num_layers": 4,
        "size": Size.nano,
    },
    Size.tiny: {
        "intermediate_size": 512,
        "head_size": 64,
        "hidden_size": 256,
        "num_heads": 4,
        "num_layers": 7,
        "size": Size.tiny,
    },
    Size.small: {
        "intermediate_size": 2048,
        "head_size": 64,
        "hidden_size": 512,
        "num_heads": 8,
        "num_layers": 10,
        "size": Size.small,
    },
    Size.base: {
        "intermediate_size": 3072,
        "head_size": 64,
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "size": Size.base,
    },
}


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class DetectionConfig:
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    fpn_size: int = 256
    max_det: int = 100
    min_conf: float = 0.001
    reg_max: int = 16


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelConfig:
    _head_size: Annotated[int, tyro.conf.arg(aliases=("--model.head-size",))] = -1
    _hidden_size: Annotated[int, tyro.conf.arg(aliases=("--model.hidden-size",))] = -1
    _intermediate_size: Annotated[int, tyro.conf.arg(aliases=("--model.intermediate-size",))] = -1
    _num_heads: Annotated[int, tyro.conf.arg(aliases=("--model.num-heads",))] = -1
    _num_layers: Annotated[int, tyro.conf.arg(aliases=("--model.num-layers",))] = -1

    drop_path: float = 0.1
    dropout: float = 0.0
    cutmix: float = 1.0
    label_smoothing: float = 0.1
    mixup: float = 0.8
    size: Size = Size.base

    @property
    def head_size(self) -> int:
        if self._head_size == -1:
            return MODEL_SIZES[self.size]["head_size"]
        return self._head_size

    @property
    def hidden_size(self) -> int:
        if self._hidden_size == -1:
            return MODEL_SIZES[self.size]["hidden_size"]
        return self._hidden_size

    @property
    def intermediate_size(self) -> int:
        if self._intermediate_size == -1:
            return MODEL_SIZES[self.size]["intermediate_size"]
        return self._intermediate_size

    @property
    def num_heads(self) -> int:
        if self._num_heads == -1:
            return MODEL_SIZES[self.size]["num_heads"]
        return self._num_heads

    @property
    def num_layers(self) -> int:
        if self._num_layers == -1:
            return MODEL_SIZES[self.size]["num_layers"]
        return self._num_layers


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class OptimizerConfig:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    layer_decay: float = 0.75
    final_lr: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.05


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class PaPEConfig:
    parabolas: int = 50


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class RoPEConfig:
    base: float = 10_000.0


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class RoPEMixedConfig:
    base: float = 100.0


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TrainConfig:
    acc_gradients: int = 1
    check_val_every_n_epoch: int = 1
    ckpt_n_hour: int | None = None
    gradient_clip_algorithm: Literal["norm"] | Literal["value"] = "value"
    grad_clip_value: float | None = None
    log_every_n_step: int = 1000
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "16-mixed"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class VideoConfig:
    frame_length: int = 2  # Number of frames per sample.
    frame_step: int = 10  # Number of frames to skip between sampled frames.
    max_samples: int = 5  # Maximum number of samples from a video.


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Config:
    dataset: Dataset
    absolute: AbsoluteConfig = dataclasses.field(default_factory=AbsoluteConfig)
    batch_size: int = 1
    compile: bool = False
    debug: bool = False
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    fold: int = 1  # For cross-validation.
    epochs: int = 100
    events: EventsConfig = dataclasses.field(default_factory=EventsConfig)
    group: str | None = None
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    name: str | None = None
    normalize: bool = False  # Whether to normalize positions to [0, 1].
    num_workers: int = 4
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    pape: PaPEConfig = dataclasses.field(default_factory=PaPEConfig)
    patch_size: int = 16
    pin_memory: bool = False
    positional: Positional = Positional.nope
    rope: RoPEConfig = dataclasses.field(default_factory=RoPEConfig)
    rope_mixed: RoPEMixedConfig = dataclasses.field(default_factory=RoPEMixedConfig)
    seed: int | None = None
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    valid_ratio: float = 0.2  # For datasets without a dedicated validation set.
    validate: bool = True
    video: VideoConfig = dataclasses.field(default_factory=VideoConfig)

    @property
    def height(self) -> int:
        height, _ = load_dimensions(self.dataset)
        return height

    @property
    def num_classes(self) -> int:
        return load_num_classes(self.dataset)

    @property
    def num_positions(self) -> int:
        return load_num_positions(self.dataset)

    @property
    def width(self) -> int:
        _, width = load_dimensions(self.dataset)
        return width


def load_dimensions(dataset: Dataset) -> tuple[int, int]:
    match dataset:
        case Dataset.coco:
            from pape.coco import HEIGHT
            from pape.coco import WIDTH
        case Dataset.dvsgesture:
            from pape.dvs_gesture import HEIGHT
            from pape.dvs_gesture import WIDTH
        case Dataset.gen1:
            from pape.gen1 import HEIGHT
            from pape.gen1 import WIDTH
        case Dataset.imagenet:
            from pape.imagenet import HEIGHT
            from pape.imagenet import WIDTH
        case Dataset.ucf101:
            from pape.ucf101 import HEIGHT
            from pape.ucf101 import WIDTH
        case _:
            raise ValueError(f"Dataset '{dataset}' is not supported.")

    return HEIGHT, WIDTH


def load_num_classes(dataset: Dataset) -> int:
    match dataset:
        case Dataset.coco:
            from pape.coco import NUM_CLASSES
        case Dataset.dvsgesture:
            from pape.dvs_gesture import NUM_CLASSES
        case Dataset.gen1:
            from pape.gen1 import NUM_CLASSES
        case Dataset.imagenet:
            from pape.imagenet import NUM_CLASSES
        case Dataset.ucf101:
            from pape.ucf101 import NUM_CLASSES
        case _:
            raise ValueError(f"Dataset '{dataset}' is not supported.")
    return NUM_CLASSES


def load_num_positions(dataset: Dataset) -> int:
    match dataset:
        case Dataset.coco:
            return 2
        case Dataset.dvsgesture:
            return 3
        case Dataset.gen1:
            return 3
        case Dataset.imagenet:
            return 2
        case Dataset.ucf101:
            return 3
        case _:
            raise ValueError(f"Dataset '{dataset}' is not supported.")
