from pathlib import Path

from app.assets.storage import AssetStorage
from app.config import Settings


def test_purge_all_content_preserves_gitkeep(tmp_path: Path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / ".gitkeep").touch()
    staged = root / ".staging" / "doc"
    staged.mkdir(parents=True)
    (staged / "image.png").write_bytes(b"test")
    promoted = root / "doc"
    promoted.mkdir()
    (promoted / "image.png").write_bytes(b"test")

    settings = Settings.model_construct(
        asset_storage_dir=root,
        asset_public_base_url="/assets",
    )
    removed = AssetStorage(settings).purge_all_content()

    assert removed == 2
    assert (root / ".gitkeep").is_file()
    assert not (root / ".staging").exists()
    assert not promoted.exists()
