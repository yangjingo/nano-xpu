"""Real CPU process for supervisor failure tests; not installed as a worker backend."""
import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sleep", "fail", "spawn"), default="sleep")
    parser.add_argument("--ignore-term", action="store_true")
    args = parser.parse_args()
    if args.mode == "fail":
        print("injected worker startup failure", flush=True)
        return 7
    if args.ignore_term and os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args.mode == "spawn":
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        child = subprocess.Popen([sys.executable, __file__, "--ignore-term"], **options)
        print(f"CHILD_PID={child.pid}", flush=True)
    print("READY " + os.environ.get("NANO_XPU_INTERNAL_TOKEN", ""), flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
