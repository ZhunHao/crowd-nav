"""`crowdnav-preflight` — verify runtime deps and exit non-zero on first problem."""

from __future__ import annotations

import sys

from crowd_nav.utils.preflight import run_all


def main() -> int:
    rc = run_all()
    sys.exit(rc)


if __name__ == "__main__":
    main()
