#!/usr/bin/env python3
# Linux process-level smoke test for Gunicorn's post-fork analysis worker.

import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main():
    if os.name != "posix":
        print("SKIP: Gunicorn fork lifecycle smoke test requires Linux/POSIX.")
        return 0

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="classmind-gunicorn-") as temp_dir:
        temp = Path(temp_dir)
        port = free_port()
        env = os.environ.copy()
        env.update({
            "PORT": str(port),
            "UPLOAD_FOLDER": str(temp / "uploads"),
            "PIPELINE_OUTPUT_FOLDER": str(temp / "pipeline_output"),
            "DATABASE_PATH": str(temp / "database.db"),
            "SECRET_KEY": "gunicorn-lifecycle-test-key",
            "PYTHONUNBUFFERED": "1",
        })
        command = [
            sys.executable, "-m", "gunicorn",
            "--config", str(root / "gunicorn.conf.py"),
            "--bind", f"127.0.0.1:{port}",
            "app:app",
        ]
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = ""
        ready = False
        poll_elapsed = None
        deadline = time.monotonic() + 25
        try:
            status_url = f"http://127.0.0.1:{port}/api/analyze/jobs/lifecycle-probe"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output, _ = process.communicate(timeout=2)
                    raise AssertionError(
                        f"Gunicorn exited during startup ({process.returncode}).\n{output}"
                    )
                started = time.perf_counter()
                try:
                    urlopen(status_url, timeout=0.5)
                except HTTPError as exc:
                    poll_elapsed = time.perf_counter() - started
                    if exc.code == 404:
                        ready = True
                        break
                    raise AssertionError(f"Unexpected job poll status: HTTP {exc.code}") from exc
                except (URLError, TimeoutError, OSError):
                    time.sleep(0.15)
                    continue
                else:
                    raise AssertionError("Unknown job ID unexpectedly returned HTTP 200.")

            if not ready:
                raise TimeoutError("Gunicorn did not serve the job-poll route within 25 seconds.")

            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait(timeout=5)
                output, _ = process.communicate(timeout=2)
                raise AssertionError(f"Gunicorn did not shut down gracefully.\n{output}") from exc
            output, _ = process.communicate(timeout=2)

            start_lines = [
                line for line in output.splitlines()
                if "[AnalysisJobs] Starting the single job worker in pid=" in line
            ]
            if len(start_lines) != 1:
                raise AssertionError(
                    f"Expected one analysis worker startup in the child process; "
                    f"found {len(start_lines)}.\n{output}"
                )

            start_match = re.search(
                r"pid=(\d+) \(parent pid=(\d+)\)",
                start_lines[0],
            )
            init_match = re.search(
                r"Analysis job manager pid=(\d+) initialized for Gunicorn worker pid=(\d+)",
                output,
            )
            if not start_match or not init_match:
                raise AssertionError(f"Could not verify worker/master PIDs.\n{output}")

            analysis_pid, parent_pid = map(int, start_match.groups())
            manager_pid, gunicorn_worker_pid = map(int, init_match.groups())
            if parent_pid != process.pid:
                raise AssertionError(
                    f"Analysis worker parent PID {parent_pid} did not match Gunicorn master "
                    f"PID {process.pid}.\n{output}"
                )
            if analysis_pid != gunicorn_worker_pid or manager_pid != gunicorn_worker_pid:
                raise AssertionError(f"Job manager did not start inside the Gunicorn child.\n{output}")
            if process.returncode != 0:
                raise AssertionError(
                    f"Gunicorn exited with {process.returncode} after SIGTERM.\n{output}"
                )
            if "Worker exiting (pid:" not in output or "Shutting down: Master" not in output:
                raise AssertionError(f"Expected graceful worker/master shutdown logs.\n{output}")
            if "Analysis job worker stopped during worker exit pid=" not in output:
                raise AssertionError(
                    f"Gunicorn worker exit did not stop the process-local job thread.\n{output}"
                )

            print(
                "PASS: Ubuntu Gunicorn preload/post_fork starts exactly one analysis worker "
                f"in child pid {gunicorn_worker_pid} (master {process.pid}); "
                f"job polling returned 404 in {poll_elapsed:.4f}s and SIGTERM shut down cleanly."
            )
            return 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
