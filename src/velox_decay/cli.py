"""Expose the numerical model's command-line entry point."""

from .model import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
