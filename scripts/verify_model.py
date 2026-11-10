"""Verify local weights against their recorded SHA-256 manifest, with no network."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("models/Vemana-Base"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / "model_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Invalid manifest path")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified {manifest['repo_id']} at {manifest['revision']}")


if __name__ == "__main__":
    main()
