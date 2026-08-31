# SPDX-License-Identifier: Unlicense
"""Input checks, device selection, model loading, and synthesis.

WAV only, so ffmpeg is never needed. Metal is the target, CPU the fallback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.utils.audio import load_audio, trim_long_audio
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES

from voiceclone.config import (
    GenerationOptions,
    ReferenceOptions,
    VoiceOptions,
    resolve_input,
)

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def check_reference(ref_audio: str) -> Path:
    """Validate the reference exists and is a WAV."""
    path = resolve_input(ref_audio, "Reference audio")
    if path.suffix.lower() != ".wav":
        raise ValueError(
            f"Reference audio must be .wav, got {path.suffix or 'no extension'}: "
            f"{path}. Convert first, e.g. afconvert -f WAVE -d LEI16 in.m4a ref.wav"
        )
    return path


def check_voice(options: VoiceOptions) -> None:
    """Reject an unknown language, which upstream would silently ignore."""
    language = options.language
    if language is None or language in LANG_IDS or language.lower() in LANG_NAMES:
        return
    raise ValueError(
        f"Unknown language {language!r}. Use a code like en, a name like "
        "English, or leave it unset. Codes are case-sensitive."
    )


def resolve_device(requested: str = "mps") -> str:
    """Confirm Metal is usable, or honour a CPU fallback."""
    if requested == "cpu":
        logger.warning("Running on CPU; generation is far slower than on Metal.")
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"

    reason = (
        "MPS needs Apple Silicon and macOS 12.3 or newer"
        if torch.backends.mps.is_built()
        else "this PyTorch build was compiled without MPS support"
    )
    raise RuntimeError(
        f'Metal (MPS) is unavailable: {reason}. Set device = "cpu" under [engine] '
        "in the config file to continue."
    )


def resolve_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """fp16 is what OmniVoice documents for Apple Silicon. CPU fp16 is slow."""
    if requested == "auto":
        return torch.float32 if device == "cpu" else torch.float16

    dtype = _DTYPES.get(requested)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype {requested!r}; expected auto or one of "
            f"{', '.join(_DTYPES)}."
        )
    if requested == "float16" and device == "cpu":
        logger.warning("fp16 is unaccelerated on CPU; float32 is usually faster.")
    return dtype


@dataclass
class Result:
    audio: np.ndarray
    sampling_rate: int
    elapsed: float

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sampling_rate

    @property
    def rtf(self) -> float:
        """Compute seconds per second of audio produced."""
        return self.elapsed / self.seconds if self.seconds > 0 else float("nan")


class VoiceCloner:
    """Loads OmniVoice once and clones a voice across many utterances."""

    def __init__(self, model: OmniVoice):
        self.model = model

    @classmethod
    def load(
        cls,
        model_id: str,
        device: str,
        dtype: str = "auto",
        attn_implementation: str | None = None,
        asr_model: str | None = None,
        asr_device: str | None = None,
    ) -> VoiceCloner:
        resolved = resolve_dtype(device, dtype)
        logger.info(
            "Loading %s on %s (%s) ...",
            model_id,
            device,
            str(resolved).removeprefix("torch."),
        )
        started = time.perf_counter()
        if device == "mps":
            # transformers reads weights straight into Metal buffers from a four
            # thread pool, which deadlocks before the first tensor lands. Loading
            # them one at a time costs nothing, the checkpoint still reads in ~2s.
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        # OmniVoice puts the audio tokenizer on CPU itself when on MPS.
        extra = {}
        if attn_implementation is not None:
            extra["attn_implementation"] = attn_implementation
        if asr_model is not None:
            extra["asr_model_name"] = asr_model
        if asr_device is not None:
            extra["asr_device"] = asr_device
        model = OmniVoice.from_pretrained(
            model_id, device_map=device, dtype=resolved, **extra
        )
        logger.info("Model ready in %.1fs", time.perf_counter() - started)
        return cls(model)

    @property
    def sampling_rate(self) -> int:
        return self.model.sampling_rate

    def _prepare_asr(self, language: str | None) -> None:
        """Settle Whisper's decoding prefix before upstream calls it.

        `OmniVoice.transcribe` forwards no generate kwargs, so anything we want
        to say to Whisper has to be set on the pipeline it hides.
        """
        if self.model._asr_pipe is None:
            self.model.load_asr_model()
        pipe = self.model._asr_pipe

        # transformers skips this for BPE tokenizers and warns once per run.
        # False is what it already does, minus the warning.
        pipe.tokenizer.clean_up_tokenization_spaces = False

        if language is not None:
            # Detection is a coin flip on a few seconds of speech. Naming a
            # language also keeps _retrieve_language off the deprecated
            # forced_decoder_ids branch (generation_whisper.py:1504-1512).
            pipe.model.generation_config.language = language
            pipe.model.generation_config.task = "transcribe"

    def prompt_from_audio(
        self,
        ref_audio: str,
        options: ReferenceOptions,
        ref_text: str = "",
        transcribe: bool = False,
    ) -> VoiceClonePrompt:
        """Encode an example recording into a voice clone prompt."""
        path = check_reference(ref_audio)
        started = time.perf_counter()

        waveform = load_audio(str(path), self.sampling_rate)
        duration = waveform.shape[-1] / self.sampling_rate
        if options.preprocess and duration > options.trim_threshold:
            # Upstream only trims on the transcribe path, so do it for both.
            waveform = trim_long_audio(
                waveform,
                self.sampling_rate,
                max_duration=options.trim_max_duration,
                min_duration=options.trim_min_duration,
                trim_threshold=options.trim_threshold,
            )
            logger.info(
                "Trimmed reference from %.1fs to %.1fs at the largest silence gap",
                duration,
                waveform.shape[-1] / self.sampling_rate,
            )

        if transcribe:
            logger.info("Transcribing reference with Whisper ...")
            self._prepare_asr(options.asr_language)

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(torch.from_numpy(waveform), self.sampling_rate),
            # None rather than a string is what makes OmniVoice run Whisper.
            ref_text=None if transcribe else ref_text,
            preprocess_prompt=options.preprocess,
        )
        logger.info("Encoded reference in %.1fs", time.perf_counter() - started)
        return prompt

    def synthesize(
        self,
        text: str,
        prompt: VoiceClonePrompt,
        options: GenerationOptions,
        voice: VoiceOptions,
    ) -> Result:
        started = time.perf_counter()
        audios = self.model.generate(
            text=text,
            voice_clone_prompt=prompt,
            language=voice.language,
            instruct=voice.instruct,
            # These three are generate() parameters, not config fields.
            duration=options.duration,
            speed=options.speed,
            # Off for good. It needs WeTextProcessing, whose pynini dependency
            # has no arm64 wheel and no working source build on Apple Silicon.
            normalize_text=False,
            generation_config=_generation_config(options),
        )
        return Result(
            audio=audios[0],
            sampling_rate=self.sampling_rate,
            elapsed=time.perf_counter() - started,
        )


def _generation_config(options: GenerationOptions) -> OmniVoiceGenerationConfig:
    """Every field OmniVoice accepts, since anything omitted silently reverts."""
    return OmniVoiceGenerationConfig(
        num_step=options.num_step,
        guidance_scale=options.guidance_scale,
        t_shift=options.t_shift,
        layer_penalty_factor=options.layer_penalty_factor,
        position_temperature=options.position_temperature,
        class_temperature=options.class_temperature,
        denoise=options.denoise,
        postprocess_output=options.postprocess_output,
        audio_chunk_duration=options.audio_chunk_duration,
        audio_chunk_threshold=options.audio_chunk_threshold,
        pad_duration=options.pad_duration,
        fade_duration=options.fade_duration,
    )
