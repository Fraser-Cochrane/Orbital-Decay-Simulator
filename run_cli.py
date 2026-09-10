"""Launch the command-line interface from a source checkout."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from velox_decay.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
