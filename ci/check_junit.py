"""Reject empty, failed or skipped pytest JUnit reports for the required CPU suite."""
import sys
from xml.etree import ElementTree


def check_report(path):
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("CPU suite collected no test cases")
    for tag in ("failure", "error", "skipped"):
        count = sum(1 for _ in root.iter(tag))
        if count:
            raise ValueError(f"CPU report contains {count} {tag} result(s)")
    return len(cases)


if __name__ == "__main__":
    try:
        count = check_report(sys.argv[1])
    except (OSError, ElementTree.ParseError, ValueError) as exc:
        sys.exit(f"CPU report rejected: {exc}")
    print(f"CPU report accepted: {count} passed, no failures/errors/skips")
