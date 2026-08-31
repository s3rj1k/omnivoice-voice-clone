# omnivoice-voice-clone

Zero-shot voice cloning on Apple Silicon, built on [OmniVoice](https://github.com/k2-fsa/OmniVoice). English with a Russian accent by default, any of 646 languages if you ask. WAV in, WAV out.

> [!WARNING]
> **Only clone voices you have permission to use.**

## Usage

```bash
uv sync
uv run voice-clone clone igor.wav
uv run voice-clone tts igor --text "Hello, this is my cloned voice." -o out.wav
```

`igor.wav` becomes `voices/igor.voice`, which is gitignored. Reference audio wants 3-10s of clean speech. Whisper transcribes it on the first clone, which calibrates pacing and conditions generation on the reference. Pass `--ref-text` when you already know the transcript, or `--no-transcribe` to skip the extra model.

## Tuning

Every setting lives in `voiceclone.toml` in the project root, or wherever `--config PATH` or `$VOICECLONE_CONFIG` points.

## License

[The Unlicense](https://unlicense.org). Public domain.
