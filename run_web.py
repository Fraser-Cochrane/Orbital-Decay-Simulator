"""Launch the interactive website from a source checkout."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from velox_decay.web import main  # noqa: E402


if __name__ == "__main__":
    main()
