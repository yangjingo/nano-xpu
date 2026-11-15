"""Materialize the documented examples for tests and CI, never in the source tree."""
from pathlib import Path
import re
import sys


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def materialize_examples(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(EXAMPLES):
        raise ValueError("generated examples must stay outside the documentation directory")
    for document_name, folder, count in (("server.md", "serve", 4), ("deploy.md", "deployment", 3)):
        document = EXAMPLES / document_name
        blocks = re.findall(r"^```(?:yaml|bash|powershell) filename=([\w.-]+)\n(.*?)^```$",
                            document.read_text(encoding="utf-8"), flags=re.M | re.S)
        if len(blocks) != count or len({name for name, _ in blocks}) != count:
            raise ValueError(f"expected {count} distinct named example blocks in {document}")
        for name, source in blocks:
            if Path(name).suffix not in (".yaml", ".sh", ".ps1"):
                raise ValueError(f"invalid example filename in {document}")
            target = destination / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
    return destination


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/helpers/examples.py OUTPUT_DIRECTORY")
    print(materialize_examples(sys.argv[1]))
