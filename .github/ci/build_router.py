"""Build the reviewed upstream router source on a Linux CPU runner and bind its provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "07c20da66c255ba65a97ab531f9fb5f60413a7d4"
REPOSITORY = "https://github.com/vllm-project/router.git"
RUST = "1.95.0"  # Upstream .buildkite/test-ready.yml at REVISION.
FEATURES = "serde_json/float_roundtrip"


def run(*command, cwd=None, capture=False):
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("the real-router integration build requires Linux")
    source = args.source.resolve()
    if source.exists() and any(source.iterdir()):
        parser.error("source directory must be empty; do not reuse an unverified checkout")
    source.mkdir(parents=True, exist_ok=True)
    run("git", "init", "-q", str(source))
    run("git", "remote", "add", "origin", REPOSITORY, cwd=source)
    run("git", "fetch", "--depth=1", "origin", REVISION, cwd=source)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
    assert run("git", "rev-parse", "HEAD", cwd=source, capture=True).stdout.strip() == REVISION
    assert not run("git", "status", "--porcelain", cwd=source, capture=True).stdout.strip()
    run("rustup", "toolchain", "install", RUST, "--profile", "minimal")
    os.environ.update(CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="0", CARGO_BUILD_JOBS="2")
    # HTTP KV metadata includes an f64 deadline. Preserve its exact value through
    # serde_json parsing so the decode worker can verify the registered export.
    run("cargo", "+" + RUST, "build", "--locked", "--features", FEATURES, "--bin", "vllm-router", cwd=source)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "target/debug/vllm-router", output)
    help_text = run(str(output), "--help", capture=True).stdout
    for flag in ("--worker-urls", "--prefill", "--decode", "--vllm-pd-disaggregation", "--kv-connector",
                 "--health-check-endpoint", "--disable-retries"):
        if flag not in help_text:
            raise RuntimeError(f"built router is missing required flag {flag}")
    with output.open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {"revision": REVISION, "repository": REPOSITORY, "sha256": binary_hash,
               "cargo_lock_sha256": hashlib.sha256((source / "Cargo.lock").read_bytes()).hexdigest(),
               "rustc": run("rustc", "+" + RUST, "--version", capture=True).stdout.strip(),
               "version": run(str(output), "--version", capture=True).stdout.strip(), "profile": "dev, debug=0",
               "features": FEATURES.split(",")}
    Path(str(output) + ".source.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".help.txt").write_text(help_text, encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
