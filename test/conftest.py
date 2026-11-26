"""Ensure the CleanEM project root is importable when running tests."""
from pathlib import Path
import sys

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
