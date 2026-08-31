#!/usr/bin/env python3
"""Download a HuggingFace dataset with adaptive parallelism.

- Few files: huggingface_hub.snapshot_download (max_workers scales with file count).
- Many files (>= threshold): aria2c batch mode (parallel files + connections).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path


def _ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23"]
        )


def _match_patterns(relpath: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(relpath, pat) for pat in patterns)


def list_target_files(repo: str, patterns: list[str]) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    out: list[str] = []
    for rel in files:
        if rel in (".gitattributes",):
            continue
        if _match_patterns(rel, patterns):
            out.append(rel)
    return sorted(out)


def missing_files(out_dir: Path, relfiles: list[str]) -> list[str]:
    missing: list[str] = []
    for rel in relfiles:
        target = out_dir / rel
        if not target.is_file() or target.stat().st_size == 0:
            partial = Path(str(target) + ".aria2")
            if partial.exists():
                partial.unlink(missing_ok=True)
            missing.append(rel)
    return missing


def fetch_via_snapshot(repo: str, out_dir: Path, patterns: list[str], max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    kwargs: dict = {
        "repo_id": repo,
        "repo_type": "dataset",
        "local_dir": str(out_dir),
        "max_workers": max_workers,
    }
    if patterns:
        kwargs["allow_patterns"] = patterns
    snapshot_download(**kwargs)
    print(f"done {repo} -> {out_dir} (snapshot workers={max_workers})")


def write_aria2_list(repo: str, out_dir: Path, relfiles: list[str], list_path: Path) -> int:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as f:
        for rel in relfiles:
            url = (
                f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}?download=true"
            )
            f.write(f"{url}\n")
            f.write(f"  out={rel}\n")
    return len(relfiles)


def aria2_download(
    repo: str,
    out_dir: Path,
    relfiles: list[str],
    list_path: Path,
    *,
    concurrent: int,
    connections: int,
    log_path: Path | None,
) -> None:
    if not relfiles:
        print(f"skip aria2 {repo}: no missing files")
        return
    write_aria2_list(repo, out_dir, relfiles, list_path)
    cmd = [
        "aria2c",
        f"--input-file={list_path}",
        f"--dir={out_dir}",
        "--continue=true",
        f"--max-concurrent-downloads={concurrent}",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=1M",
        "--max-tries=5",
        "--retry-wait=3",
        "--console-log-level=warn",
        "--summary-interval=30",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
    ]
    print(
        f"aria2 {repo} -> {out_dir} files={len(relfiles)} "
        f"j={concurrent} conn={connections}",
        flush=True,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as logf:
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)
    print(f"done {repo} -> {out_dir} (aria2 files={len(relfiles)})")


def adaptive_workers(file_count: int, cap: int) -> int:
    return max(8, min(cap, max(8, file_count // 5)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--patterns",
        default="",
        help="Pipe-separated allow_patterns (fnmatch), e.g. 'libero_90/**'",
    )
    parser.add_argument(
        "--file-threshold",
        type=int,
        default=int(os.environ.get("HF_PARALLEL_FILE_THRESHOLD", "150")),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "32")),
    )
    parser.add_argument(
        "--aria2-j",
        type=int,
        default=int(os.environ.get("ARIA2_MAX_CONCURRENT", "16")),
    )
    parser.add_argument(
        "--aria2-x",
        type=int,
        default=int(os.environ.get("ARIA2_CONNECTIONS_PER_SERVER", "8")),
    )
    parser.add_argument(
        "--aria2-list",
        default="",
        help="Persist aria2 input list path (default: <out>/.aria2_input.txt)",
    )
    parser.add_argument(
        "--aria2-log",
        default="",
        help="Optional aria2 log file path",
    )
    parser.add_argument(
        "--skip-if-info",
        action="store_true",
        help="Skip when out/meta/info.json already exists and no files missing",
    )
    args = parser.parse_args()

    _ensure_hf()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = [p for p in args.patterns.split("|") if p]

    relfiles = list_target_files(args.repo, patterns)
    missing = missing_files(out_dir, relfiles)
    print(f"repo={args.repo} total_files={len(relfiles)} missing={len(missing)}")

    if args.skip_if_info and (out_dir / "meta" / "info.json").is_file() and not missing:
        print(f"skip {args.repo}: complete ({out_dir})")
        return 0

    if not missing:
        print(f"skip {args.repo}: all files present")
        return 0

    use_aria2 = len(relfiles) >= args.file_threshold
    aria2_ok = subprocess.call(["bash", "-lc", "command -v aria2c >/dev/null"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if use_aria2 and aria2_ok:
        list_path = Path(args.aria2_list) if args.aria2_list else out_dir / ".aria2_input.txt"
        log_path = Path(args.aria2_log) if args.aria2_log else None
        aria2_download(
            args.repo,
            out_dir,
            missing,
            list_path,
            concurrent=args.aria2_j,
            connections=args.aria2_x,
            log_path=log_path,
        )
    else:
        if use_aria2 and not aria2_ok:
            print(
                f"WARN aria2c missing; falling back to snapshot_download "
                f"(files={len(relfiles)})",
                flush=True,
            )
        workers = adaptive_workers(len(relfiles), args.max_workers)
        fetch_via_snapshot(args.repo, out_dir, patterns, workers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
