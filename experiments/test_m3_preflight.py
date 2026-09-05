"""Selection and evidence checks that do not require a connected phone."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from m3_preflight import (MODEL_SHA256, SOURCE_COMMIT, Recorder, collect,
                          devices_from_output, select_device, sha256_from_output)


class DeviceSelectionTests(unittest.TestCase):
    def test_adb_startup_messages_are_not_devices(self):
        output = "* daemon started successfully\nList of devices attached\n\n"
        self.assertEqual(devices_from_output(output), [])
        with self.assertRaisesRegex(ValueError, "no_device"):
            select_device(devices_from_output(output), None)

    def test_multiple_devices_are_never_selected_implicitly(self):
        devices = devices_from_output("List of devices attached\n"
                                      "phone-a device product:test model:Example\n"
                                      "phone-b unauthorized\n")
        with self.assertRaisesRegex(ValueError, "multiple_devices"):
            select_device(devices, None)
        self.assertEqual(select_device(devices, "phone-a"), "phone-a")
        with self.assertRaisesRegex(ValueError, "not_authorized"):
            select_device(devices, "phone-b")
        with self.assertRaisesRegex(ValueError, "absent"):
            select_device(devices, "phone-c")

    def test_offline_device_is_not_ready(self):
        with self.assertRaisesRegex(ValueError, "offline"):
            select_device([{"serial": "phone", "state": "offline"}], None)

    def test_sha256_errors_are_not_measurements(self):
        self.assertIsNone(sha256_from_output("sha256sum: Permission denied"))
        self.assertIsNone(sha256_from_output("0000"))
        self.assertEqual(sha256_from_output(MODEL_SHA256 + "  /data/model.litertlm\n"),
                         MODEL_SHA256)


class EvidenceTests(unittest.TestCase):
    def test_disconnected_collection_never_issues_device_shell_commands(self):
        class FakeRecorder:
            calls = []

            def run(self, name, argv, timeout=30):
                self.calls.append(argv)
                output = {"source_commit": SOURCE_COMMIT + "\n",
                          "devices": "List of devices attached\n\n"}.get(name, "")
                return {"returncode": 0, "stdout": output, "stderr": ""}

        recorder = FakeRecorder()
        args = SimpleNamespace(source=Path("unused"), adb="adb", serial=None,
                               expected_model_sha256=MODEL_SHA256)
        result = collect(args, recorder)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["missing"], ["no_device"])
        self.assertFalse(any("shell" in call for call in recorder.calls))
        self.assertNotIn("model_observed_sha256", result)

    def test_fresh_c_api_device_does_not_require_legacy_cli(self):
        class FakeRecorder:
            def run(self, name, argv, timeout=30):
                output = {"source_commit": SOURCE_COMMIT + "\n",
                          "devices": "List of devices attached\nphone device\n",
                          "model_sha256": MODEL_SHA256 + "  model.litertlm\n"}.get(name, "")
                return {"returncode": 1 if name == "legacy_binary_executable" else 0,
                        "stdout": output, "stderr": ""}
        args = SimpleNamespace(source=Path("unused"), adb="adb", serial="phone",
                               device_dir="/data/local/tmp/litertlm",
                               expected_model_sha256=MODEL_SHA256)
        result = collect(args, FakeRecorder())
        self.assertEqual(result["status"], "inventory_collected")
        self.assertEqual(result["missing"], [])
        self.assertFalse(result["observations"]["legacy_binary_executable"])

    def test_missing_executable_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Recorder(Path(directory)).run("missing", [directory + "/absent"])
            self.assertIsNone(result["returncode"])
            self.assertTrue(result["stderr"])
            self.assertTrue((Path(directory) / "commands.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
