"""Guard against publishing failed native runs as successful benchmarks."""

import json
import csv
from pathlib import Path
import tempfile
import unittest

from bench_release import validate, write_summary


class ResultValidationTest(unittest.TestCase):
    case = {"prefill": 256, "decode": 128}
    metrics = {
        "last_prefill_token_count": 256, "last_decode_token_count": 128,
        "last_prefill_tokens_per_second": 120.0,
        "last_decode_tokens_per_second": 25.0,
        "init_time_in_second": 0.5, "time_to_first_token_in_second": 2.2,
    }

    def result(self, **changes):
        return "BENCH_RESULT " + json.dumps(self.metrics | changes) + "\n"

    def test_valid_metrics_with_benign_backend_warning(self):
        result, errors = validate(self.result(), "WARNING: NPU unavailable\n", 0, self.case)
        self.assertEqual(errors, [])
        self.assertEqual(result, self.metrics)

    def test_native_error_despite_zero_exit_and_positive_metrics(self):
        _, errors = validate(self.result(), "ERROR: Node 1830 failed to invoke.\n", 0, self.case)
        self.assertTrue(errors)

    def test_missing_metrics_despite_zero_exit(self):
        self.assertTrue(validate("", "", 0, self.case)[1])

    def test_short_generation(self):
        self.assertTrue(validate(self.result(last_decode_token_count=127), "", 0, self.case)[1])

    def test_invalid_throughput(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertTrue(validate(self.result(last_decode_tokens_per_second=value), "", 0, self.case)[1])

    def test_crash_after_metrics(self):
        self.assertTrue(validate(self.result(), "", -6, self.case)[1])

    def test_first_and_last_failure_preserve_middle_case_metrics(self):
        rows = [dict(case_index=0, valid_runs=0),
                dict(case_index=1, valid_runs=3, decode_tps=25.0),
                dict(case_index=2, valid_runs=0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_summary(path, rows)
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual(saved[1]["decode_tps"], "25.0")
        self.assertEqual(saved[0]["decode_tps"], "")


if __name__ == "__main__":
    unittest.main()
