"""Regression checks for release-source quotation validation."""

import tempfile
import unittest
from pathlib import Path

from check_code_references import check_source_excerpts, source_target


class SourceQuotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        (self.source / "sample.cc").write_text(
            "void Run() {\n"
            "  ABSL_RETURN_IF_ERROR(Prefill());\n"
            "  // Wait before reading the output.\n"
            "  Wait();\n"
            "  Decode();\n"
            "}\n",
            encoding="utf-8",
        )

    def check(self, body):
        return check_source_excerpts(f"```cpp\n{body}\n```\n", self.source)

    def test_source_excerpt_with_omission_and_annotation(self):
        count, errors = self.check(
            "// sample.cc:1\nvoid Run() {\n// ...\n"
            "  // Wait before reading the output.\n"
            "  Wait();  // (1)\n// ...\n}"
        )
        self.assertEqual(count, 1)
        self.assertEqual(errors, [])

    def test_existing_anchor_does_not_validate_old_macro(self):
        _, errors = self.check("// sample.cc:2\n  RETURN_IF_ERROR(Prefill());")
        self.assertTrue(errors)

    def test_reordered_valid_lines_fail(self):
        _, errors = self.check("// sample.cc:1\n  Decode();\n  Wait();")
        self.assertTrue(errors)

    def test_anchor_after_quoted_code_fails(self):
        _, errors = self.check("// sample.cc:5\n  Wait();")
        self.assertTrue(errors)

    def test_changed_source_comment_or_indentation_fails(self):
        for line in ("  // Wait after reading the output.", "Wait();"):
            with self.subTest(line=line):
                self.assertTrue(self.check("// sample.cc:1\n" + line)[1])

    def test_external_and_unanchored_examples_are_excluded(self):
        self.assertEqual(self.check("// llama.cpp/sample.cc:1\nother();"), (0, []))
        self.assertEqual(self.check("illustrative_example();"), (0, []))

    def test_dependency_quotation_uses_dependency_tree(self):
        dep = self.source / "dependency"
        dep.mkdir()
        (dep / "sample.cc").write_text("  Expert();\n")
        text = "```cpp\n// LiteRT/sample.cc:1\n  Expert();\n```\n"
        self.assertEqual(check_source_excerpts(text, self.source, dep), (1, []))
        changed = text.replace("  Expert();", "  Other();")
        self.assertTrue(check_source_excerpts(changed, self.source, dep)[1])

    def test_dependency_never_falls_back_to_lm_tree(self):
        with self.assertRaises(ValueError):
            source_target("LiteRT/sample.cc", self.source)
        self.assertEqual(source_target("sample.cc", self.source), self.source / "sample.cc")

    def test_length_includes_reference_and_omission_lines(self):
        _, errors = self.check("// sample.cc:1\n" + "// ...\n" * 30)
        self.assertTrue(any("30 lines" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
