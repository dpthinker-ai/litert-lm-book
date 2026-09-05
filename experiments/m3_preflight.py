#!/usr/bin/env python3
"""Collect M3 device prerequisites without running inference or changing settings."""

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import subprocess


SOURCE_COMMIT = "a0afb5a56acd106b23a2b2385b8469834dc268c0"
MODEL_SHA256 = "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0"


def devices_from_output(output):
    devices = []
    for line in output.splitlines():
        if not line.strip() or line.startswith(("List of devices", "*")):
            continue
        fields = line.split()
        if len(fields) >= 2:
            devices.append({"serial": fields[0], "state": fields[1]})
    return devices


def select_device(devices, serial):
    if serial:
        matches = [device for device in devices if device["serial"] == serial]
        if not matches:
            raise ValueError("requested_device_absent")
    else:
        if not devices:
            raise ValueError("no_device")
        if len(devices) != 1:
            raise ValueError("multiple_devices_require_serial")
        matches = devices
    if matches[0]["state"] != "device":
        raise ValueError("device_not_authorized_or_offline")
    return matches[0]["serial"]


def sha256_from_output(output):
    match = re.match(r"^([0-9a-fA-F]{64})\s", output)
    return match[1].lower() if match else None


class Recorder:
    def __init__(self, directory):
        self.directory = directory

    def run(self, name, argv, timeout=30):
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=timeout, check=False)
            record = {"returncode": result.returncode, "stdout": result.stdout,
                      "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            record = {"returncode": None, "stdout": "", "stderr": str(error)}
        record.update(name=name, argv=argv, started_utc=started)
        with (self.directory / "commands.jsonl").open("a") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def collect(args, recorder):
    summary = {"status": "blocked", "inference_run": False,
               "source_expected_commit": SOURCE_COMMIT,
               "model_expected_sha256": args.expected_model_sha256,
               "missing": [], "observations": {}}
    source = recorder.run("source_commit", ["git", "-C", str(args.source),
                                           "rev-parse", "HEAD"])
    summary["source_observed_commit"] = source["stdout"].strip()
    if source["returncode"] != 0 or source["stdout"].strip() != SOURCE_COMMIT:
        summary["missing"].append("frozen_source_commit")
    dirty = recorder.run("source_status", ["git", "-C", str(args.source),
                                           "status", "--porcelain"])
    if dirty["returncode"] != 0 or dirty["stdout"].strip():
        summary["missing"].append("clean_frozen_source")
    recorder.run("adb_version", [args.adb, "version"])
    result = recorder.run("devices", [args.adb, "devices", "-l"])
    if result["returncode"] != 0:
        summary["missing"].append("adb_device_listing_failed")
        return summary
    devices = devices_from_output(result["stdout"])
    summary["devices"] = devices
    try:
        serial = select_device(devices, args.serial)
    except ValueError as error:
        summary["missing"].append(str(error))
        return summary
    summary["selected_serial"] = serial

    def shell(name, words, timeout=30):
        return recorder.run(name, [args.adb, "-s", serial, "shell",
                                   shlex.join(words)], timeout)

    for key in ("ro.product.manufacturer", "ro.product.model", "ro.hardware",
                "ro.soc.model", "ro.build.version.release", "ro.build.version.sdk",
                "ro.build.fingerprint", "ro.product.cpu.abi"):
        result = shell(key, ["getprop", key])
        summary["observations"][key] = (result["stdout"].strip()
                                         if result["returncode"] == 0 else None)
    for name, words in (
        ("system_memory", ["cat", "/proc/meminfo"]),
        ("battery", ["dumpsys", "battery"]),
        ("thermal", ["dumpsys", "thermalservice"]),
        ("low_power_setting", ["settings", "get", "global", "low_power"]),
        ("storage", ["df", "-k", args.device_dir]),
        ("shell_process_memory_probe", ["cat", "/proc/self/smaps_rollup"]),
    ):
        result = shell(name, words)
        summary["observations"][name] = {"returncode": result["returncode"],
                                            "has_output": bool(result["stdout"].strip())}

    binary = args.device_dir.rstrip("/") + "/litert_lm_advanced_main"
    model = args.device_dir.rstrip("/") + "/model.litertlm"
    legacy = shell("legacy_binary_executable", ["test", "-x", binary])
    summary["observations"]["legacy_binary_executable"] = legacy["returncode"] == 0
    if shell("model_readable", ["test", "-r", model])["returncode"] != 0:
        summary["missing"].append("model_readable")
    if "model_readable" not in summary["missing"]:
        result = shell("model_sha256", ["sha256sum", model], timeout=180)
        observed = sha256_from_output(result["stdout"])
        summary["model_observed_sha256"] = observed
        if result["returncode"] != 0 or observed != args.expected_model_sha256:
            summary["missing"].append("model_sha256_mismatch_or_unavailable")
        shell("model_size_bytes", ["stat", "-c", "%s", model])
    if legacy["returncode"] == 0:
        shell("binary_sha256", ["sha256sum", binary])
    shell("runtime_files", ["ls", "-l", args.device_dir])
    # Readability and hashes do not establish runtime provenance or successful execution.
    summary["status"] = "blocked" if summary["missing"] else "inventory_collected"
    summary["next_required"] = ["verify_binary_and_library_build_provenance",
                                 "validate_process_memory_and_thermal_observation",
                                 "implement_and_verify_device_clock_event_capture",
                                 "complete_generation_smoke_test"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial")
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[2] / "LiteRT-LM-v0.13.1")
    parser.add_argument("--device-dir", default="/data/local/tmp/litertlm")
    parser.add_argument("--expected-model-sha256", default=MODEL_SHA256)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_model_sha256):
        parser.error("--expected-model-sha256 must contain 64 hexadecimal digits")
    args.expected_model_sha256 = args.expected_model_sha256.lower()
    now = dt.datetime.now(dt.timezone.utc)
    out = args.out or (Path(__file__).parent / "data" / now.strftime("%Y-%m-%d") /
                       ("m3-preflight-" + now.strftime("%H%M%S-%fZ")))
    out.mkdir(parents=True, exist_ok=False)
    summary = collect(args, Recorder(out))
    summary["recorded_utc"] = now.isoformat()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "missing": summary["missing"],
                      "output": str(out)}, ensure_ascii=False))
    return 2 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
