"""Module entry-point that proxies to :mod:`devcrew.cli`."""
from __future__ import annotations

from .cli import main


def run() -> int:
    return main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
