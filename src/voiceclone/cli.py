# SPDX-License-Identifier: Unlicense
"""Command-line interface.

    clone   example recording -> reusable voice file
    tts     voice file + text -> audio

Only paths and text are arguments, everything else is config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import soundfile as sf
from omnivoice import VoiceClonePrompt
from omnivoice.utils.common import fix_random_seed

from voiceclone import __version__, config
from voiceclone.config import resolve_input
from voiceclone.engine import (
    VoiceCloner,
    check_reference,
    check_voice,
    resolve_device,
)

logger = logging.getLogger("voiceclone")


def _loader(args: argparse.Namespace, **extra) -> VoiceCloner:
    engine = args.settings.engine
    return VoiceCloner.load(
        engine.model,
        resolve_device(engine.device),
        dtype=engine.dtype,
        attn_implementation=engine.attn_implementation,
        **extra,
    )


def add_clone(subparsers, parents: list[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "clone",
        parents=parents,
        help="Clone a voice from an example recording into a reusable file.",
        description="Clone a voice from an example recording. The voice file is "
        "named after the recording and written to [paths] voice_dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ref_audio",
        metavar="REF_AUDIO",
        help="Example WAV recording of the voice. 3-10s of clean speech works "
        "best. igor.wav becomes igor.voice.",
    )
    # Both describe this recording, not tool behaviour, so they stay arguments.
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument(
        "--ref-text",
        default="",
        help="Transcript of the recording, used instead of running Whisper.",
    )
    transcript.add_argument(
        "--transcribe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill the transcript by running Whisper on the recording once, "
        "downloading an ASR model. The result is stored in the voice file.",
    )
    parser.set_defaults(handler=run_clone)
    return parser


def run_clone(args: argparse.Namespace) -> int:
    options = args.settings.clone
    check_reference(args.ref_audio)  # before pulling a multi-GB checkpoint

    cloner = _loader(args, asr_model=options.asr_model, asr_device=options.asr_device)
    prompt = cloner.prompt_from_audio(
        args.ref_audio,
        options,
        ref_text=args.ref_text,
        # A supplied transcript wins, so the default only fills an empty one.
        transcribe=args.transcribe and not args.ref_text,
    )

    # Always a plain join, since the stem is a name rather than a path.
    voice_dir = Path(args.settings.paths.voice_dir).expanduser()
    out = voice_dir / f"{Path(args.ref_audio).stem}.voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    prompt.save(str(out))
    logger.info(
        "Wrote %s with %d reference frames from %s",
        out,
        prompt.ref_audio_tokens.size(-1),
        Path(args.ref_audio).name,
    )
    if prompt.ref_text:
        logger.info("Reference text %r", prompt.ref_text)
    else:
        logger.warning(
            "No reference transcript, so pacing and text conditioning fall back "
            "to a generic speaker. Drop --no-transcribe or pass --ref-text."
        )
    return 0


def add_tts(subparsers, parents: list[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "tts",
        parents=parents,
        help="Synthesize text in a voice created by 'clone'.",
        description="Synthesize text in a voice created by 'clone'. Language "
        "and accent are [voice] settings, decoding is [tts].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "voice",
        metavar="VOICE",
        help="Voice made by 'clone'. A bare name like igor is looked up in "
        "[paths] voice_dir, anything else is used as a path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output WAV path. An index is appended per line with --text-file "
        "(out_001.wav).",
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--text", help="Text to synthesize.")
    what.add_argument(
        "--text-file",
        help="UTF-8 file with one utterance per line. Blank lines are skipped.",
    )
    parser.set_defaults(handler=run_tts)
    return parser


def read_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]

    path = resolve_input(args.text_file, "Text file")
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError(f"No non-empty lines in {args.text_file}")
    return texts


def output_path(output: str, index: int, count: int) -> Path:
    """Suffix the stem per utterance when there are several."""
    path = Path(output).expanduser()
    if count == 1:
        return path
    return path.with_name(f"{path.stem}_{index + 1:03d}{path.suffix}")


def run_tts(args: argparse.Namespace) -> int:
    options = args.settings.tts
    voice = args.settings.voice

    # Before pulling a multi-GB checkpoint.
    check_voice(voice)
    voice_file = resolve_input(str(args.settings.voice_path(args.voice)), "Voice file")
    prompt = VoiceClonePrompt.load(str(voice_file))
    texts = read_texts(args)

    Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Voice %s (%d reference frames), speaking %s as %s",
        voice_file,
        prompt.ref_audio_tokens.size(-1),
        voice.language or "any language",
        voice.instruct or "the reference voice",
    )
    if not prompt.ref_text:
        logger.warning(
            "Voice file has no reference transcript, so pacing and text "
            "conditioning fall back to a generic speaker. Reclone to fill it in."
        )
    logger.debug("Generation settings: %s", options)

    cloner = _loader(args)

    total = len(texts)
    for index, text in enumerate(texts):
        logger.info("[%d/%d] Generating: %s", index + 1, total, text[:80])
        if options.seed is not None:
            fix_random_seed(options.seed + index)

        result = cloner.synthesize(text, prompt, options, voice)
        out_path = output_path(args.output, index, total)
        sf.write(str(out_path), result.audio, result.sampling_rate)
        logger.info(
            "[%d/%d] Wrote %s | %.1fs audio in %.1fs (RTF %.3f)",
            index + 1,
            total,
            out_path,
            result.seconds,
            result.elapsed,
            result.rtf,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone",
        description="Zero-shot voice cloning with OmniVoice, on Apple Silicon.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Shared by every command, declared once and inherited via parents=.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        help=f"Settings file, required. Falls back to ${config.ENV_VAR}, then "
        f"./{config.FILENAME} in the project root.",
    )
    common.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    add_clone(subparsers, [common])
    add_tts(subparsers, [common])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        force=True,
    )

    try:
        # Before dispatch, so a bad setting fails in milliseconds.
        args.settings = config.load(args.config)
        return args.handler(args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
