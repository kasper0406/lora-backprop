from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples" / "compare_local_learning.py"


def run(label: str, extra: list[str], passthrough: list[str]) -> None:
    print(f"\n=== {label} ===")
    cmd = [sys.executable, str(BASE), *passthrough, *extra]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ordinary vs power-of-two depth skips under the same training methods."
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    passthrough = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    run("sequential-depth baseline", [], passthrough)
    run("power-of-two depth skips", ["--use-log-skip"], passthrough)


if __name__ == "__main__":
    main()
