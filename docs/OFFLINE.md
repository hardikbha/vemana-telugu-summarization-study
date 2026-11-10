# Connected staging and offline execution

GitHub hosts source and recorded synthetic results. It is not a runtime dependency
once the files, weights and Python packages have been transferred.

## Stage on a connected Linux x86_64 machine

Use the same Python minor version as the target (the recorded run used 3.11).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model,reports,test]'
python scripts/download_model.py
python scripts/verify_model.py
python -m pip download --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu \
  --dest wheelhouse -r requirements-tested.txt
```

Transfer the source repository, `models/Vemana-Base/` and `wheelhouse/` through your
approved mechanism. Do not transfer credentials or a virtual environment whose
paths belong to the staging machine. Prepare/patch the target OS separately.

## Install on the compatible offline target

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse -r requirements-tested.txt
python scripts/verify_model.py
python scripts/benchmark_vemana.py --limit 3 --skip-pilots --output-dir outputs/offline-test
```

The runner uses local model paths, `local_files_only=True`, `trust_remote_code=False`,
offline Hugging Face mode, and an outbound-socket guard during evaluation. It does
not call a model API. Host/container network rules must enforce actual runtime
isolation; an application flag alone is not a physical air gap.

For CUDA inference, stage a matching driver and CUDA PyTorch distribution rather
than the CPU wheel listed here. The Vemana result in this repository is a CPU result,
so it cannot establish GPU VRAM or AWS latency. The earlier Qwen reference GPU
measurement is for a different model/configuration and is not Vemana sizing data.

## Public sharing

This public repository contains code, publisher model metadata, synthetic inputs
and generated outputs. It contains no real helpline complaints or citizen records.
Keep private training/evaluation data outside the repository and use an ignored
output directory for experiments on such data.
