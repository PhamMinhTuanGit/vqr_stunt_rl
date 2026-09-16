#!/usr/bin/env python3
"""Ensure inconsistent saved WORLD/BODY contacts are rejected."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--yaml", required=True, type=Path)
    arguments = parser.parse_args()

    with arguments.yaml.open("r", encoding="utf-8") as stream:
        saved = yaml.safe_load(stream)
    fl_y = float(saved["physical_tread_contacts_world"]["FL"][1])
    if fl_y <= 0.0:
        raise RuntimeError(
            "YAML sequence item '- 0.200...' must parse as a positive value"
        )
    saved["physical_tread_contacts_world"]["FL"][1] = -fl_y

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", delete=False
        ) as stream:
            yaml.safe_dump(saved, stream, sort_keys=False)
            temporary_path = Path(stream.name)
        result = subprocess.run(
            [str(arguments.validator), str(temporary_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise RuntimeError("Validator accepted inconsistent WORLD/BODY contacts")
        diagnostic = result.stdout + result.stderr
        if "saved FL WORLD-to-BODY representation mismatch" not in diagnostic:
            raise RuntimeError(
                "Validator failed without the expected frame-consistency diagnostic"
            )
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
