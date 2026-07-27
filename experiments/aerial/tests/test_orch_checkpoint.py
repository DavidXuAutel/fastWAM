from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint


def test_complete_requires_sha_and_stable_size(tmp_path):
    pt = tmp_path / "step_001000.pt"
    pt.write_bytes(b"abc")
    assert is_complete_checkpoint(pt, settle_s=0.0, min_bytes=1) is False
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")
    assert is_complete_checkpoint(pt, settle_s=0.0, min_bytes=1) is True
