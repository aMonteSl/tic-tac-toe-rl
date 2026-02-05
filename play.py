#!/usr/bin/env python
"""Main entry point for Tic-Tac-Toe pygame menu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ttt.play.main_menu_pygame import main

if __name__ == "__main__":
    main()
