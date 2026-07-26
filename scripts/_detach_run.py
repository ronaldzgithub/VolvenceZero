"""Detach a long ecology run into its own session so terminal teardown
cannot reap it. Usage: python scripts/_detach_run.py <logfile> <cmd...>"""

from __future__ import annotations

import os
import sys


def main() -> int:
    log_path = sys.argv[1]
    argv = sys.argv[2:]
    if not argv:
        raise SystemExit("no command given")
    pid = os.fork()
    if pid != 0:
        print(f"detached pid={pid} log={log_path}")
        return 0
    os.setsid()
    fd_in = os.open(os.devnull, os.O_RDONLY)
    fd_out = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd_in, 0)
    os.dup2(fd_out, 1)
    os.dup2(fd_out, 2)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
