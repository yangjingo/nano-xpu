"""Fault injection for the report gate; independent of runtime dependencies."""
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from check_junit import check_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.report = Path(folder.name) / "junit.xml"

    def write(self, content):
        self.report.write_text(content, encoding="utf-8")
        return self.report

    def test_success(self):
        path = self.write('<testsuites><testsuite><testcase name="ok"/></testsuite></testsuites>')
        self.assertEqual(check_report(path), 1)

    def test_empty(self):
        with self.assertRaisesRegex(ValueError, "no test cases"):
            check_report(self.write("<testsuites><testsuite/></testsuites>"))

    def test_failure_error_and_skip(self):
        for tag in ("failure", "error", "skipped"):
            with self.subTest(tag=tag):
                path = self.write(f'<testsuite><testcase name="bad"><{tag}/></testcase></testsuite>')
                with self.assertRaisesRegex(ValueError, tag):
                    check_report(path)

    def test_partial_skip(self):
        path = self.write('<testsuite><testcase name="ok"/><testcase name="skip"><skipped/></testcase></testsuite>')
        with self.assertRaisesRegex(ValueError, "skipped"):
            check_report(path)

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            check_report(self.report)

    def test_malformed(self):
        with self.assertRaises(ElementTree.ParseError):
            check_report(self.write("<testsuites>"))


if __name__ == "__main__":
    unittest.main()
