from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    url: str
    hash_prefix: str
    channels: tuple[int, int, int, int, int]
    builder: str


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "resnet34": EncoderSpec(
        name="resnet34",
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        hash_prefix="b627a593",
        channels=(64, 64, 128, 256, 512),
        builder="resnet34",
    ),
    "resnet50": EncoderSpec(
        name="resnet50",
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        hash_prefix="0676ba61",
        channels=(64, 256, 512, 1024, 2048),
        builder="resnet50",
    ),
}

SUPPORTED_ENCODERS = tuple(ENCODER_SPECS.keys())
DEFAULT_ENCODER_NAME = "resnet34"


def get_encoder_spec(name: str) -> EncoderSpec:
    key = name.lower()
    if key not in ENCODER_SPECS:
        options = ", ".join(SUPPORTED_ENCODERS)
        raise ValueError(f"Unsupported encoder '{name}'. Available: {options}.")
    return ENCODER_SPECS[key]


def default_weights_dir() -> Path:
    return Path(__file__).resolve().parent / "weights"


def default_weights_path(name: str) -> Path:
    spec = get_encoder_spec(name)
    filename = Path(spec.url).name
    return default_weights_dir() / filename
