"""Exercise archive tamper checks and unsupported-control classification."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from bench_release import METRICS, sha256
from bench_release_report import report


class ArchiveReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metrics = dict(last_prefill_token_count=256, last_decode_token_count=128,
                            last_prefill_tokens_per_second=120.0,
                            last_decode_tokens_per_second=25.0,
                            init_time_in_second=0.5, time_to_first_token_in_second=2.2)
        self.case = dict(group="ringbuffer", backend="gpu", prefill=256, decode=128,
                         capacity=8192, mtp=False, ring=True)
        records = []
        for repeat in range(4):
            stdout = self.root / f"{repeat}.stdout.log"
            stderr = self.root / f"{repeat}.stderr.log"
            stdout.write_text("BENCH_RESULT " + json.dumps(self.metrics) + "\n")
            stderr.write_text("I0000 Failed to get GpuArtisanConfig to set use_ringbuffers_local_attention\n")
            records.append(dict(case_index=0, repeat=repeat, warmup=repeat == 0,
                                returncode=0, metrics=self.metrics, errors=[],
                                logs=[dict(file=p.name, sha256=sha256(p)) for p in (stdout, stderr)]))
        (self.root / "manifest.json").write_text(json.dumps(dict(status="completed", cases=[self.case], runs=records)))
        row = dict(case_index=0, **self.case, valid_runs=3)
        for name, field in METRICS.items():
            row.update({name + suffix: self.metrics[field] for suffix in ("", "_min", "_max")})
        with (self.root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_ignored_control_is_not_a_valid_switch_comparison(self):
        result = report(self.root)
        self.assertEqual(result["verified_log_count"], 8)
        self.assertEqual(result["cases"][0]["status"], "unsupported_control")

    def test_changed_log_is_rejected(self):
        (self.root / "1.stdout.log").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            report(self.root)

    def test_wrong_csv_median_is_rejected(self):
        path = self.root / "summary.csv"
        path.write_text(path.read_text().replace("25.0", "50.0"))
        with self.assertRaisesRegex(ValueError, "CSV metric mismatch"):
            report(self.root)

    def test_wrong_range_count_or_condition_is_rejected(self):
        path = self.root / "summary.csv"
        original = path.read_text()
        for field, value in (("decode_tps_min", "0"), ("decode_tps_max", "999"),
                             ("valid_runs", "2"), ("capacity", "4096"),
                             ("case_index", "1"), ("backend", "cpu")):
            with self.subTest(field=field):
                rows = list(csv.DictReader(original.splitlines()))
                rows[0][field] = value
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    report(self.root)
        path.write_text(original)


if __name__ == "__main__":
    unittest.main()
