# Remote package patches applied 2026-08-04

These were applied on `a25689@10.239.121.11` under
`/home/a25689/FastWAM/scoutxwam_droid100_inference` so `infer.py` / `serve.py` can import.

1. `infer.py` — insert `package/src` on `sys.path` before X-WAM.
2. `third_party/X-WAM/runners/xwam_runner.py` — lazy-import `Wan21VAEAdapter`
   only when `vae_family == "wan21"` (default path is wan22).
3. `src/fastwam/models/wan22/helpers/loader.py` — optional `_load_video_vae`
   shim for the Wan21 adapter.

Bridge code lives in this directory; `serve.py` is also copied to the package root.
