#!/usr/bin/env python3
"""Verify archived v0.17.0 logs and recompute metrics without running a model."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

from bench_release import METRICS, sha256, validate


def report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Collection has not finished")
    with (directory / "summary.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != len(manifest["cases"]):
        raise ValueError("CSV case count mismatch")
    rows = []
    for index, case in enumerate(manifest["cases"]):
        records = [r for r in manifest["runs"] if r["case_index"] == index]
        if len({r["repeat"] for r in records}) != len(records):
            raise ValueError(f"Duplicate repeat in case {index}")
        successful, errors, observations, success_rates = [], [], set(), []
        for record in records:
            if record["warmup"] != (record["repeat"] == 0):
                raise ValueError(f"Warmup label mismatch in case {index}")
            contents = []
            for log in record["logs"]:
                path = directory / log["file"]
                if sha256(path) != log["sha256"]:
                    raise ValueError(f"Hash mismatch: {path}")
                contents.append(path.read_text())
            metrics, failures = validate(*contents, record["returncode"], case)
            if metrics != record["metrics"] or failures != record["errors"]:
                raise ValueError(f"Metric/validation mismatch in case {index}")
            stderr = contents[1]
            if "Failed to get GpuArtisanConfig to set use_ringbuffers_local_attention" in stderr:
                observations.add("ringbuffer_request_ignored")
            if "WebGPU sampler not available, falling back to statically linked C API" in stderr:
                observations.add("webgpu_sampler_fell_back_to_static_c_api")
            if "Selected adapter:" in stderr and "backend=Metal" in stderr:
                observations.add("webgpu_metal_adapter_selected")
            rates = re.findall(r"MTP Drafter - Success rate: ([\d.eE+-]+)", stderr)
            if rates:
                observations.add("mtp_acceptance_log_present")
                if not record["warmup"]:
                    success_rates.append(dict(repeat=record["repeat"], logged_rates=rates))
            if failures:
                errors.append(dict(repeat=record["repeat"], errors=failures))
            elif not record["warmup"]:
                successful.append(metrics)
        valid = ({r["repeat"] for r in records} == {0, 1, 2, 3}
                 and len(successful) == 3 and not errors)
        row = dict(case_index=index, **case, valid_runs=len(successful),
                   status="valid" if valid else "failed",
                   observations=sorted(observations), failures=errors,
                   mtp_logged_success_rates=success_rates)
        if valid and case["group"] == "ringbuffer" and "ringbuffer_request_ignored" in observations:
            row["status"] = "unsupported_control"
        for field in ("case_index", *case, "valid_runs"):
            if str(row[field]) != raw[index][field]:
                raise ValueError(f"CSV condition/count mismatch for case {index}: {field}")
        if valid:
            row["metrics"] = {}
            for name, field in METRICS.items():
                values = [r[field] for r in successful]
                median = statistics.median(values)
                row["metrics"][name] = dict(median=median, min=min(values), max=max(values))
                for suffix, expected in (("", median), ("_min", min(values)), ("_max", max(values))):
                    if expected != float(raw[index][name + suffix]):
                        raise ValueError(f"CSV metric mismatch for case {index}: {name + suffix}")
        elif any(raw[index].get(name + suffix) for name in METRICS for suffix in ("", "_min", "_max")):
            raise ValueError(f"CSV contains metrics for failed case {index}")
        rows.append(row)
    return dict(manifest_sha256=sha256(directory / "manifest.json"),
                summary_sha256=sha256(directory / "summary.csv"),
                verified_log_count=sum(len(r["logs"]) for r in manifest["runs"]),
                cases=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.directory), indent=2))
