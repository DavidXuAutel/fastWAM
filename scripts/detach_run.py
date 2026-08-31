#!/usr/bin/env python3
"""Detach a command into its own session (double-fork + setsid) so it survives
the parent shell's exit. Re-parented to launchd (PID 1).

Usage: python3 detach_run.py <logfile> <cmd> [args...]
"""
import os
import sys

if len(sys.argv) < 3:
    print("usage: detach_run.py <logfile> <cmd> [args...]", file=sys.stderr)
    sys.exit(2)

logfile = sys.argv[1]
cmd = sys.argv[2]
cmd_args = sys.argv[2:]

# First fork
if os.fork() != 0:
    # Parent exits immediately so the tool shell's child returns fast.
    sys.stdout.flush()
    sys.exit(0)

# Child: new session, detach from controlling terminal.
os.setsid()

# Second fork: prevent re-acquiring a controlling terminal.
if os.fork() != 0:
    os._exit(0)

# Grandchild: redirect stdio to logfile, /dev/null for stdin.
os.chdir("/")
try:
    os.umask(0o022)
    fd_in = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd_in, 0)
    os.close(fd_in)
    fd_out = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd_out, 1)
    os.dup2(fd_out, 2)
    os.close(fd_out)
except Exception:
    pass

os.execvp(cmd, cmd_args)
