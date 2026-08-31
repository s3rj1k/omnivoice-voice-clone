# SPDX-License-Identifier: Unlicense
"""Settings loaded from a TOML file, plus input path validation.

No field carries a default, so missing and unknown keys are both errors.
Must not import torch, even transitively.
"""

from __future__ import annotations

import difflib
import logging
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomli is the same parser under its pre-3.11 name.
    import tomli as tomllib

logger = logging.getLogger(__name__)

FILENAME = "voiceclone.toml"
ENV_VAR = "VOICECLONE_CONFIG"

DEVICES = ("mps", "cpu")
DTYPES = ("auto", "float32", "float16", "bfloat16")

_MAX_SEED = 2**32  # numpy's seeding limit, reached via fix_random_seed.


def resolve_input(path: str, description: str = "File") -> Path:
    """Expand and validate an input path."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return resolved


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be above 0, got {value}")


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be 0 or above, got {value}")


def _one_of(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, got {value!r}")


@dataclass(frozen=True)
class EngineOptions:
    """Which checkpoint to run, and where."""

    model: str
    device: str
    dtype: str
    # Unset leaves the choice to transformers, which validates it itself.
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        _one_of("device", self.device, DEVICES)
        _one_of("dtype", self.dtype, DTYPES)


@dataclass(frozen=True)
class VoiceOptions:
    """Output language and voice attributes. Unset lets OmniVoice decide."""

    language: str | None = None
    instruct: str | None = None


@dataclass(frozen=True)
class PathOptions:
    """Where voice files are written and looked up."""

    voice_dir: str


@dataclass(frozen=True)
class ReferenceOptions:
    """How a reference recording is conditioned before encoding."""

    trim_threshold: float
    trim_max_duration: float
    trim_min_duration: float
    preprocess: bool
    asr_model: str | None = None
    asr_device: str | None = None
    # The transcript's language, unrelated to the [voice] language we speak in.
    asr_language: str | None = None

    def __post_init__(self) -> None:
        _positive("trim_threshold", self.trim_threshold)
        _positive("trim_max_duration", self.trim_max_duration)
        _positive("trim_min_duration", self.trim_min_duration)
        # min > max contradicts upstream's clamp, and max > threshold never trims.
        if not self.trim_min_duration <= self.trim_max_duration <= self.trim_threshold:
            raise ValueError(
                "trim durations must satisfy "
                "trim_min_duration <= trim_max_duration <= trim_threshold, got "
                f"{self.trim_min_duration} <= {self.trim_max_duration} "
                f"<= {self.trim_threshold}"
            )
        if self.asr_device is not None:
            _one_of("asr_device", self.asr_device, DEVICES)


@dataclass(frozen=True)
class GenerationOptions:
    """Per-utterance generation settings."""

    num_step: int
    guidance_scale: float
    speed: float
    position_temperature: float
    class_temperature: float
    t_shift: float
    layer_penalty_factor: float
    denoise: bool
    postprocess_output: bool
    audio_chunk_duration: float
    audio_chunk_threshold: float
    pad_duration: float
    fade_duration: float
    # Optional, so last, since a dataclass cannot default before a required field.
    duration: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_step < 1:
            raise ValueError(f"num_step must be 1 or above, got {self.num_step}")
        _non_negative("guidance_scale", self.guidance_scale)
        _positive("speed", self.speed)
        if self.duration is not None:
            _positive("duration", self.duration)
        if self.seed is not None and not 0 <= self.seed < _MAX_SEED:
            raise ValueError(
                f"seed must be between 0 and {_MAX_SEED - 1}, got {self.seed}"
            )
        # Upstream gates sampling on > 0, so a negative temperature is silently greedy.
        _non_negative("position_temperature", self.position_temperature)
        _non_negative("class_temperature", self.class_temperature)
        # At 0 the schedule collapses and below 0 it divides by zero, unraised.
        _positive("t_shift", self.t_shift)
        _non_negative("layer_penalty_factor", self.layer_penalty_factor)
        _positive("audio_chunk_duration", self.audio_chunk_duration)
        _positive("audio_chunk_threshold", self.audio_chunk_threshold)
        # 0 disables, and negative is a silent no-op upstream.
        _non_negative("pad_duration", self.pad_duration)
        _non_negative("fade_duration", self.fade_duration)


@dataclass(frozen=True)
class Settings:
    """Everything a run is configured with."""

    path: Path
    engine: EngineOptions
    paths: PathOptions
    voice: VoiceOptions
    clone: ReferenceOptions
    tts: GenerationOptions

    def voice_path(self, name: str) -> Path:
        """A bare name resolves under ``voice_dir``, anything else is a path."""
        candidate = Path(name).expanduser()
        if candidate.suffix == ".voice" or len(candidate.parts) > 1:
            return candidate
        return Path(self.paths.voice_dir).expanduser() / f"{name}.voice"


_TABLES = {
    "engine": EngineOptions,
    "paths": PathOptions,
    "voice": VoiceOptions,
    "clone": ReferenceOptions,
    "tts": GenerationOptions,
}

_KINDS = {"int": int, "float": float, "bool": bool, "str": str}


def _kind(item):
    """Target type from the annotation, which PEP 563 leaves a str."""
    return _KINDS[item.type.removesuffix(" | None")]


def _optional(item) -> bool:
    return item.type.endswith(" | None")


def find() -> Path | None:
    """The project root config, meaning the working directory only."""
    path = Path.cwd() / FILENAME
    return path if path.is_file() else None


def load(explicit: str | None = None) -> Settings:
    """Find, parse and validate the required config file."""
    # A wrong explicit path must fail, not fall through to discovery.
    environ = os.environ.get(ENV_VAR)
    if explicit:
        path = resolve_input(explicit, "Config file")
    elif environ:
        path = resolve_input(environ, f"Config file from {ENV_VAR}")
    else:
        path = find()

    if path is None:
        raise FileNotFoundError(
            f"No {FILENAME} in {Path.cwd()}. Run from the project root, set "
            f"{ENV_VAR}, or pass --config PATH."
        )

    path = path.resolve()
    # INFO, since a stray file changing your output must be visible by default.
    logger.info("Config: %s", path)

    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{path}: {error}") from error

    unknown = sorted(set(data) - set(_TABLES))
    if unknown:
        raise ValueError(
            f"{path}: unknown table [{unknown[0]}]. Valid tables: {', '.join(_TABLES)}."
        )

    return Settings(
        path=path,
        **{
            name: _build(cls, data.get(name, {}), name, path)
            for name, cls in _TABLES.items()
        },
    )


def _build(cls, table, name: str, path: Path):
    if not isinstance(table, dict):
        # ValueError so cli.main logs it instead of printing a traceback.
        raise ValueError(f"{path}: [{name}] must be a table.")  # noqa: TRY004

    known = {item.name: item for item in fields(cls)}
    values = {}
    for key, value in table.items():
        item = known.get(key)
        if item is None:
            matches = difflib.get_close_matches(key, known, n=1)
            hint = f" Did you mean {matches[0]!r}?" if matches else ""
            raise ValueError(f"{path}: [{name}] unknown key {key!r}.{hint}")
        values[key] = _coerce(value, _kind(item), key, name, path)

    missing = sorted(k for k, i in known.items() if not _optional(i) and k not in table)
    if missing:
        raise ValueError(f"{path}: [{name}] missing keys: {', '.join(missing)}")

    try:
        return cls(**values)
    except (TypeError, ValueError) as error:
        # Only the loader knows which file and table the value came from.
        raise ValueError(f"{path}: [{name}] {error}") from error


def _coerce(value, kind, key: str, table: str, path: Path):
    def fail():
        raise ValueError(
            f"{path}: [{table}] {key}: expected {kind.__name__}, "
            f"got {type(value).__name__}."
        )

    # bool is an int subclass, so match it first. 'denoise = 1' is a mistake.
    if isinstance(value, bool) != (kind is bool):
        fail()
    if kind is float:
        if not isinstance(value, (int, float)):  # TOML reads 'speed = 1' as an int.
            fail()
        return float(value)
    if not isinstance(value, kind):
        fail()
    return value
