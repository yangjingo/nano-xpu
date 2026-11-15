"""Inspect the actual imported checkout without importing accelerator dependencies."""
from pathlib import Path
import subprocess


def runtime_identity():
    package = Path(__file__).resolve().parent
    root = package.parent
    result = {"package_root": str(package), "source_root": str(root), "revision": None, "clean": False}
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return result
        result["revision"] = git("rev-parse", "HEAD")
        # Runtime code or packaging changes invalidate the source revision claim; docs do not.
        result["clean"] = not git("status", "--porcelain", "--untracked-files=all", "--", "nanoxpu", "pyproject.toml")
    except (OSError, subprocess.SubprocessError):
        pass  # Installed wheels without source provenance do not pretend to be a clean checkout.
    return result
