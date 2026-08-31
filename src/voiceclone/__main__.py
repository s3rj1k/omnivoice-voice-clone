# SPDX-License-Identifier: Unlicense
"""Entry point for ``python -m voiceclone``."""

import sys

from voiceclone.cli import main

if __name__ == "__main__":
    sys.exit(main())
