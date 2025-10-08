from pathlib import Path
from recollex.encoder.splade import prefetch

def test_prefetch_skips_when_exists(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    target = models_dir / "org__model"
    target.mkdir(parents=True, exist_ok=True)

    called = {"v": False}
    def fake_snapshot_download(**kwargs):
        called["v"] = True

    monkeypatch.setattr("recollex.encoder.splade.snapshot_download", fake_snapshot_download)
    out = prefetch("org/model", str(models_dir))
    assert out == str(target)
    assert called["v"] is False

def test_prefetch_downloads_when_missing(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    called = {"v": False}

    def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks):
        called["v"] = True
        # simulate download by creating the directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("recollex.encoder.splade.snapshot_download", fake_snapshot_download)
    out = prefetch("org/model", str(models_dir))
    assert called["v"] is True
    assert Path(out).exists()
