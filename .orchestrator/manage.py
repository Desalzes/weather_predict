from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from project_orchestrator.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
