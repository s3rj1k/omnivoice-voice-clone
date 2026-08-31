# SPDX-License-Identifier: Unlicense
"""Zero-shot voice cloning on macOS, built on OmniVoice.

Must not import torch, even transitively. Torch reads the fallback below once,
at import time.
"""

import os

# Metal lacks some ATen ops. Without this they raise mid-generation.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
