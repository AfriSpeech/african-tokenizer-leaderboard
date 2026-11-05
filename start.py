"""Launcher that starts the FastAPI backend which also serves the React frontend."""

import subprocess
import sys
import signal
import os

import env

API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", os.environ.get("API_PORT", "7860")))

_SUBPROCESS_ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def main():
    print(f"start.py: launching API server on port {API_PORT} ...", flush=True)
    print(f"start.py: HF_HOME: {os.environ.get('HF_HOME')}", flush=True)
    print(f"start.py: HF_REPO_RESULTS: {os.environ.get('HF_REPO_RESULTS')}", flush=True)
    print(f"start.py: HF_REPO_BENCHMARK: {os.environ.get('HF_REPO_BENCHMARK')}", flush=True)
    print(f"start.py: DATA_DIR: {os.environ.get('DATA_DIR')}", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"start.py: HF_TOKEN: {hf_token[:5]}...{hf_token[-4:]}", flush=True)

    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "uvicorn",
            "api:app",
            "--host", API_HOST,
            "--port", str(API_PORT),
        ],
        env=_SUBPROCESS_ENV,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    def _cleanup(signum=None, frame=None):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    try:
        proc.wait()
    except KeyboardInterrupt:
        _cleanup()


if __name__ == "__main__":
    main()
