"""An isolated committed checkout for process provenance tests; no remote or user checkout edits."""
from pathlib import Path
import shutil
import subprocess


def committed_checkout(destination):
    source = Path(__file__).parents[2]
    destination.mkdir(parents=True)
    for path in (source / "nanoxpu").rglob("*.py"):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    support_files = [source / "tests/__init__.py", *(source / "tests/helpers").rglob("*.py")]
    for path in support_files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(source / "pyproject.toml", destination / "pyproject.toml")
    (destination / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(destination), *args], stderr=subprocess.STDOUT, text=True).strip()
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=CPU Test Fixture", "-c", "user.email=cpu-fixture@example.invalid",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "CPU runtime fixture")
    return destination, git("rev-parse", "HEAD")
