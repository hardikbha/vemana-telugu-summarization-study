"""Run metadata and file handling without loading model dependencies."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fresh_directory(path):
    path = Path(path).resolve()
    # A fresh directory is required; refuse even new subdirectories of fixtures.
    repo = Path(__file__).resolve().parents[1]
    for name in ("results", "benchmark_100", "input_notes", "model_metadata", "repro"):
        if path.is_relative_to(repo / name):
            raise ValueError(f"Output must be outside {name}/")
    path.mkdir(parents=True, exist_ok=False)
    return path


def versions():
    result = {}
    for name in ("torch", "transformers", "vemana-summarization-study"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def model_metadata(path):
    path = Path(path)
    result = {"directory_name": path.name}
    for name in ("model_manifest.json", "summarization_training.json", "config.json"):
        file = path / name
        if file.exists():
            content = file.read_text(encoding="utf-8")
            result[name] = json.loads(content)
            result[name + "_sha256"] = digest(content)
    return result
