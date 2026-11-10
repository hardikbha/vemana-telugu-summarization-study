"""Connected staging only: download the exact Vemana checkpoint and hash its files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "udgamlabs/Vemana-Base"
REVISION = "93cefe2594c5157e4aa8eb9908639c253652fed9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models/Vemana-Base")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download, HfApi
    revision = HfApi().model_info(REPO, revision=args.revision).sha
    snapshot_download(REPO, revision=revision, local_dir=args.directory, max_workers=4,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "README.md", "LICENSE*", "NOTICE*"])
    hashes = {}
    for path in sorted(args.directory.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name == "model_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(args.directory))] = digest.hexdigest()
    (args.directory / "model_manifest.json").write_text(json.dumps({"repo_id": REPO, "revision": revision, "sha256": hashes}, indent=2) + "\n")
    print(f"Downloaded {REPO} at {revision} to {args.directory}")


if __name__ == "__main__":
    main()
