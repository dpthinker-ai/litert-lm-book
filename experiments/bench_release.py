#!/usr/bin/env python3
"""Serial v0.17.0 benchmarks; preserve raw metrics, failures and provenance.

Run with the isolated v0.17.0 Python interpreter. OUT must not exist.
Each case uses one discarded warmup and three measured fresh processes.
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


METRICS = {
    "prefill_tps": "last_prefill_tokens_per_second",
    "decode_tps": "last_decode_tokens_per_second",
    "init_s": "init_time_in_second",
    "ttft_s": "time_to_first_token_in_second",
}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    return {"argv": list(args), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def cases():
    result = []

    def add(group, backend, prefill, decode=128, capacity=8192, mtp=False, ring=False):
        result.append(dict(group=group, backend=backend, prefill=prefill,
                           decode=decode, capacity=capacity, mtp=mtp, ring=ring))

    for backend in ("cpu", "gpu"):
        for prefill in (256, 1024, 4096):
            add("baseline", backend, prefill)
    add("ringbuffer", "gpu", 1024, ring=True)
    for backend in ("cpu", "gpu"):
        add("mtp", backend, 1024, mtp=True)
    for prefill in (100, 250, 500, 1000, 2000, 3000, 4000):
        add("prefill_sweep", "cpu", prefill, decode=32)
    for capacity in (1024, 2048, 4096):
        add("capacity_sweep", "cpu", 256, capacity=capacity)
    # The baseline cpu/256 case supplies the 8192-capacity comparison.
    return result


def worker(model, case):
    import litert_lm
    from litert_lm.interfaces import CPU, GPU

    backend = CPU(thread_count=8) if case["backend"] == "cpu" else GPU()
    benchmark = litert_lm.Benchmark(
        model_path=str(Path(model).resolve()), backend=backend,
        prefill_tokens=case["prefill"], decode_tokens=case["decode"],
        max_num_tokens=case["capacity"], cache_dir="",
        enable_speculative_decoding=case["mtp"], prompt="benchmark",
        # This v0.17.0 C setter applies only to GpuArtisanConfig.
        use_ringbuffers_local_attention=case["ring"] if case["backend"] == "gpu" else None,
        enable_ynnpack=False,
    )
    print("BENCH_RESULT " + json.dumps(dataclasses.asdict(benchmark.run())), flush=True)


def validate(stdout, stderr, returncode, case):
    matches = re.findall(r"^BENCH_RESULT (.+)$", stdout, re.MULTILINE)
    errors = []
    result = None
    if returncode != 0:
        errors.append(f"process exit {returncode}")
    # Native generation errors may leave a zero process exit code.
    if re.search(r"^(?:ERROR:|E\d{4}\s|F\d{4}\s|Traceback)", stderr, re.MULTILINE):
        errors.append("native/Python error in stderr")
    if len(matches) != 1:
        errors.append("missing or duplicate metric record")
    else:
        try:
            result = json.loads(matches[0])
            for metric in METRICS.values():
                if not math.isfinite(result[metric]) or result[metric] <= 0:
                    errors.append(f"invalid {metric}")
            for field, expected in (("last_prefill_token_count", case["prefill"]),
                                    ("last_decode_token_count", case["decode"])):
                if result[field] != expected:
                    errors.append(f"{field}: {result[field]} != {expected}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid metric record: {exc}")
    return result, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for package in ("litert-lm", "litert-lm-api"):
        if importlib.metadata.version(package) != "0.17.0":
            parser.error(f"{package} must be exactly 0.17.0")
    if args.worker:
        worker(args.model, json.loads(args.worker))
        return
    if args.out is None:
        parser.error("--out is required")
    model = args.model.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import litert_lm

    manifest = {
        "started_at_utc": utc(), "status": "running",
        "source_tag": "v0.17.0", "source_commit": "e9fd8c53ff968071774206163027dd84bedfe925",
        "runtime_kind": "PyPI prebuilt packages; not a local source build",
        "python": sys.version, "platform": platform.platform(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "libraries": [{"name": p.name, "sha256": sha256(p)}
                      for p in Path(litert_lm.__file__).parent.glob("*.dylib")],
        "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": sha256(model)},
        "runner_sha256": sha256(__file__),
        "host": command("sysctl", "machdep.cpu.brand_string", "hw.model", "hw.memsize"),
        "power_start": command("pmset", "-g", "batt"),
        "thermal_start": command("pmset", "-g", "therm"),
        "protocol": {
            "repetitions": 3, "warmups_per_case": 1, "order": "listed cases, serial fresh processes",
            "cpu_threads": 8, "gpu_decode_steps_per_sync": "runtime default",
            "activation_data_type": "model/runtime default", "ynnpack": False,
            "prompt": "benchmark", "prompt_padding": "runtime truncates or zero-pads token IDs to requested prefill",
            "cache": "disk, files alongside model; prior smoke/pilot caches may exist",
            "ringbuffer": "explicit GPU switch; CPU setter omitted because it only supports GpuArtisanConfig",
            "sampler": "default Benchmark session; synthetic token-count workload",
            "validation": "exit code, native error logs, finite positive metrics and exact token counts",
            "limits": "desktop background load/temperature not controlled; no peak memory, power, quality or Android measurements; historical v0.13.1 settings differ",
        },
        "cases": cases(), "runs": [],
    }
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    rows = []
    for index, case in enumerate(manifest["cases"]):
        for repeat in range(4):
            name = f"{index:02d}-{case['group']}-{case['backend']}-p{case['prefill']}-k{case['capacity']}-r{repeat}"
            argv = [sys.executable, str(Path(__file__).resolve()), "--model", str(model),
                    "--worker", json.dumps(case, separators=(",", ":"))]
            started, clock = utc(), time.monotonic()
            with (out / (name + ".stdout.log")).open("w") as stdout, (out / (name + ".stderr.log")).open("w") as stderr:
                try:
                    proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=300)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    returncode = 124
            stdout_path, stderr_path = out / (name + ".stdout.log"), out / (name + ".stderr.log")
            metrics, errors = validate(stdout_path.read_text(), stderr_path.read_text(), returncode, case)
            record = dict(case_index=index, repeat=repeat, warmup=repeat == 0,
                          started_at_utc=started, wall_s=time.monotonic() - clock,
                          argv=argv, returncode=returncode, metrics=metrics, errors=errors,
                          logs=[dict(file=p.name, sha256=sha256(p)) for p in (stdout_path, stderr_path)])
            manifest["runs"].append(record)
            save()
            print(name, "FAIL " + "; ".join(errors) if errors else f"OK decode={metrics['last_decode_tokens_per_second']:.2f}", flush=True)
            if errors and repeat == 0:
                # Keep the failed warmup; do not produce summary values for this case.
                break
        measured = [r for r in manifest["runs"] if r["case_index"] == index and not r["warmup"]]
        row = dict(case_index=index, **case, valid_runs=sum(not r["errors"] for r in measured))
        if len(measured) == 3 and all(not r["errors"] for r in measured):
            for label, field in METRICS.items():
                values = [r["metrics"][field] for r in measured]
                row.update({label: statistics.median(values), label + "_min": min(values), label + "_max": max(values)})
        rows.append(row)
    fields = list(rows[0]) + [key for key in rows[-1] if key not in rows[0]]
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(finished_at_utc=utc(), status="completed" if all(r["valid_runs"] == 3 for r in rows) else "completed_with_failures",
                    power_end=command("pmset", "-g", "batt"), thermal_end=command("pmset", "-g", "therm"))
    save()
    if manifest["status"] != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
