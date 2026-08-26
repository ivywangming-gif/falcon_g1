#!/usr/bin/env python3
"""Render independent PNGs from the JSON-only EE gap audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_ee_gap_and_render import plot_variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.audit.read_text())
    root = args.audit.parent
    for variant, report in payload["variants"].items():
        for kind in ("visual", "collider", "overlay"):
            plot_variant(report, root / variant / f"{kind}_closeup.png", variant, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
