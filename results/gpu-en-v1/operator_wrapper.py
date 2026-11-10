"""Run the requested CLI unchanged and record external resource observations."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "operator_logs"
OUTPUT = ROOT / "outputs/gpu-en-v1"
EXPECTED_COMMIT = "82bedc160dc80d573249ed8e8d015f6a3b59de2a"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def capture(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    os.chdir(ROOT)
    # Compute nodes may omit git; this checkout uses a pinned, detached HEAD.
    git_binary = shutil.which("git")
    commit = (subprocess.check_output([git_binary, "rev-parse", "HEAD"], text=True).strip()
              if git_binary else (ROOT / ".git/HEAD").read_text().strip())
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Checkout changed: expected {EXPECTED_COMMIT}, found {commit}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    command = [str(ROOT / ".venv/bin/vemana-summarize"), "--device", "cuda",
               "--input-dir", "benchmark_100/input_notes", "--target-language", "en",
               "--examples", "3", "--output-dir", "outputs/gpu-en-v1"]
    metadata = {"status": "starting", "commit": commit, "command": command,
                "started_utc": utc_now(), "pbs_job_id": os.environ.get("PBS_JOBID"),
                "execution_mode": "pbs" if os.environ.get("PBS_JOBID") else "direct_ssh",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "hostname": os.uname().nodename, "factual_accuracy": None,
                "gpu_hardware": capture(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
                                         "--format=csv"]),
                "memory_measurement": "Process GPU memory and process-tree RSS sampled about once per second; observed peaks may miss shorter spikes. These are not PyTorch allocator measurements."}
    started = time.monotonic()
    peak_gpu_mib = 0
    peak_rss_bytes = 0
    samples = []
    with (LOGS / "benchmark.stdout.log").open("w") as stdout, \
            (LOGS / "benchmark.stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        metadata["pid"] = process.pid
        try:
            while process.poll() is None:
                sample = {"elapsed_seconds": round(time.monotonic() - started, 3)}
                try:
                    parent = psutil.Process(process.pid)
                    processes = [parent] + parent.children(recursive=True)
                    pids = {p.pid for p in processes}
                    rss = 0
                    for child in processes:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    sample["process_tree_rss_bytes"] = rss
                    peak_rss_bytes = max(peak_rss_bytes, rss)
                    gpu = capture(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits"])
                    gpu_mib = 0
                    if gpu["returncode"] == 0:
                        for line in gpu["stdout"].splitlines():
                            columns = [value.strip() for value in line.split(",")]
                            if len(columns) == 2 and columns[0].isdigit() and int(columns[0]) in pids:
                                try:
                                    gpu_mib += int(columns[1])
                                except ValueError:
                                    pass
                        sample["gpu_process_memory_mib"] = gpu_mib
                        peak_gpu_mib = max(peak_gpu_mib, gpu_mib)
                    else:
                        sample["gpu_query_error"] = gpu
                except psutil.NoSuchProcess:
                    pass
                samples.append(sample)
                time.sleep(1)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            returncode = process.poll()
            metadata.update(status="completed" if returncode == 0 else "failed",
                            returncode=returncode, finished_utc=utc_now(),
                            wall_seconds=round(time.monotonic() - started, 3),
                            observed_peak_gpu_process_memory_mib=peak_gpu_mib,
                            observed_peak_process_tree_rss_bytes=peak_rss_bytes,
                            resource_samples=samples)
            (LOGS / "operator_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if OUTPUT.is_dir():
        for name in ("operator_run.json", "benchmark.stdout.log", "benchmark.stderr.log",
                     "model-verify.log", "context_preflight.json"):
            shutil.copy2(LOGS / name, OUTPUT / name)
        records_path = OUTPUT / "results.jsonl"
        if records_path.exists():
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            diagnostics = {"notes_attempted": len(records),
                           "status_counts": dict(Counter(r["status"] for r in records)),
                           "notes_with_example_overlap": sum(bool(r.get("example_overlap_ids")) for r in records),
                           "source_truncations": sum(r.get("source_truncated", False) for r in records),
                           "factual_accuracy": None,
                           "note": "Mechanical checks are not factual accuracy. Human source/output review is required."}
            (OUTPUT / "operator_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        archive = shutil.make_archive(str(ROOT / "outputs/gpu-en-v1"), "zip",
                                      root_dir=ROOT, base_dir="outputs/gpu-en-v1")
        print(f"Output archive: {archive}", flush=True)
    print(json.dumps({key: value for key, value in metadata.items() if key != "resource_samples"}, indent=2), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
