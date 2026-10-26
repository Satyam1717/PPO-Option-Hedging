"""Allow direct script execution without requiring an editable install."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".runtime").exists():
    sys.path.insert(0, str(ROOT / ".runtime"))
