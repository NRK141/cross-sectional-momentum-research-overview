from pathlib import Path
from src.features.build_features import _status, _files_for_base, _files_for_eligible


def test_stale_manifest_is_not_equivalent_to_store(tmp_path: Path):
    out = tmp_path / "features"
    out.mkdir()
    manifest = tmp_path / "features_success.json"
    manifest.write_text("{}")
    s = _status(out, manifest)
    assert s["manifest_exists"] is True
    assert s["base_ready"] is False
    assert s["eligible_ready"] is False


def test_detects_base_without_eligible(tmp_path: Path):
    out = tmp_path / "features"
    base = out / "year=2025"
    base.mkdir(parents=True)
    (base / "data_0.parquet").write_bytes(b"x")
    manifest = tmp_path / "features_success.json"
    s = _status(out, manifest)
    assert s["base_ready"] is True
    assert s["eligible_ready"] is False
    assert len(_files_for_base(out)) == 1
    assert len(_files_for_eligible(out)) == 0
