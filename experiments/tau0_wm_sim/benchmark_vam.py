#!/usr/bin/env python3
"""Benchmark tau0-WM VAM-only inference loop latency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vam_only_loop import MockPolicyClient, MockSimBridge, OfficialWebsocketPolicyClient, run_vam_only_loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--execution-step", type=int, default=10)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--output", type=Path, default=Path("tau0_vam_runtime_report.json"))
    args = parser.parse_args()

    if args.mock:
        policy = MockPolicyClient()
    else:
        if args.tau_repo is None:
            raise SystemExit("--tau-repo is required unless --mock is set")
        policy = OfficialWebsocketPolicyClient(tau_repo=args.tau_repo, host=args.host, port=args.port)

    report = run_vam_only_loop(
        policy=policy,
        sim=MockSimBridge(),
        prompt=args.prompt,
        iterations=args.iterations,
        execution_step=args.execution_step,
        output_report=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
