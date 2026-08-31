# tau0-WM Sim-First Deployment Kit

This directory implements the staged tau0-WM deployment plan without modifying
the official `sii-research/tau-0-wm` repository.

## Stage 0: VAM Preflight

Check whether the released VAM policy can be deployed on the target host:

```bash
python3 preflight.py \
  --tau-repo /path/to/tau-0-wm \
  --tau-checkpoint /path/to/tau-0-wm/checkpoint \
  --wan-root /path/to/Wan2.2-TI2V-5B \
  --output reports/preflight.json
```

Use `--no-require-cuda` for a local dry-run on non-CUDA machines.

Expected official artifacts:

- `configs/deployment/wan_pretrain_rela_eef6d.yaml`
- `run_infer_server.sh`
- `web_infer_utils/simple_client.py`
- tau0-WM VAM checkpoint from Hugging Face
- Wan2.2 VAE, T5 text encoder, and tokenizer

## Stage 1: VAM-Only Loop

Local smoke test without the official server:

```bash
python3 vam_only_loop.py --mock --iterations 3 --output-report reports/mock_loop.json
```

Connect to a running official tau0-WM server:

```bash
bash /path/to/tau-0-wm/run_infer_server.sh 0.0.0.0 8001

python3 vam_only_loop.py \
  --tau-repo /path/to/tau-0-wm \
  --host 127.0.0.1 \
  --port 8001 \
  --prompt "pick up the object" \
  --iterations 10 \
  --execution-step 10 \
  --output-report reports/vam_loop.json
```

Replace `MockSimBridge` in `vam_only_loop.py` with an Isaac bridge that reads:

- RGB views as `[V, H, W, 3]` uint8 or `[V, 3, H, W]` normalized tensors
- left/right EEF poses in each arm base frame
- gripper openness fractions

The released tau0-WM action contract is `[T,16]`:

```text
left xyz + left quat xyzw + left gripper + right xyz + right quat xyzw + right gripper
```

## Stage 2: Candidate Filtering

`candidate_filter.py` provides a deployment-safe RCS-lite layer:

- penalize large EEF jumps
- penalize quaternion jumps
- reject gripper values outside `[0, 120]`
- rank candidates before execution

This can be used before official ACVS/TTC code is available.

## Stage 3: Runtime Measurement

Run:

```bash
python3 benchmark_vam.py \
  --tau-repo /path/to/tau-0-wm \
  --host 127.0.0.1 \
  --port 8001 \
  --iterations 20 \
  --output reports/runtime.json
```

The JSON report includes sample count, latency min/mean/max, and optional GPU
memory fields.

## Stage 4: ACVS Upgrade

See `acvs_upgrade_checklist.md`. The full paper pipeline requires official
simulator weights and test-time computation code, which were not released in
the official README at the time this kit was written.

## Tests

```bash
python3 -m pytest experiments/tau0_wm_sim/test_tau0_wm_sim.py -q
```
