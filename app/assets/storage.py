import hashlib
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import Settings


@dataclass(frozen=True)
class StoredAsset:
    asset_id: uuid.UUID
    stable_asset_key: str
    token: str
    staged_path: str
    local_path: str
    public_url: str
    checksum: str


class AssetStorage:
    def __init__(self, settings: Settings):
        self.root = settings.asset_storage_dir
        self.staging_root = self.root / ".staging"
        self.public_base = settings.asset_public_base_url.rstrip("/")

    def stage_bytes(
        self,
        *,
        doc_id: uuid.UUID,
        document_checksum: str,
        data: bytes,
        extension: str,
        index: int,
        source_key: str | None = None,
        token_override: str | None = None,
    ) -> StoredAsset:
        extension = extension.lower().lstrip(".") or "bin"
        asset_checksum = hashlib.sha256(data).hexdigest()
        identity_source = (
            f"{document_checksum}:{source_key or f'asset-{index:04d}'}:{asset_checksum}"
        )
        stable_asset_key = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
        token = token_override or f"[asset:{stable_asset_key[:32]}]"
        file_name = f"{stable_asset_key[:24]}.{extension}"
        staging_directory = self.staging_root / str(doc_id)
        final_directory = self.root / str(doc_id)
        staging_directory.mkdir(parents=True, exist_ok=True)
        staged_path = staging_directory / file_name
        final_path = final_directory / file_name
        staged_path.write_bytes(data)
        self._validate_image_if_applicable(staged_path)
        return StoredAsset(
            asset_id=uuid.uuid4(),
            stable_asset_key=stable_asset_key,
            token=token,
            staged_path=str(staged_path),
            local_path=str(final_path),
            public_url=f"{self.public_base}/{doc_id}/{file_name}",
            checksum=asset_checksum,
        )

    def promote(self, doc_id: uuid.UUID) -> None:
        staged_directory = self.staging_root / str(doc_id)
        if not staged_directory.exists():
            return
        final_directory = self.root / str(doc_id)
        if final_directory.exists():
            shutil.rmtree(final_directory)
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        staged_directory.replace(final_directory)

    def rollback(self, doc_id: uuid.UUID) -> None:
        shutil.rmtree(self.staging_root / str(doc_id), ignore_errors=True)
        shutil.rmtree(self.root / str(doc_id), ignore_errors=True)

    def purge_all_content(self) -> int:
        """Remove all promoted and staged assets while preserving the storage root."""
        if not self.root.exists():
            return 0
        removed = 0
        for path in list(self.root.iterdir()):
            if path.name == ".gitkeep":
                continue
            if path.is_dir():
                removed += sum(1 for item in path.rglob("*") if item.is_file())
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def staged_file_exists(self, stored: StoredAsset) -> bool:
        return Path(stored.staged_path).is_file()

    def cleanup_staging(self, max_age_seconds: int = 86_400) -> int:
        if not self.staging_root.exists():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in self.staging_root.iterdir():
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def cleanup_untracked_files(
        self,
        tracked_paths: set[str],
        max_age_seconds: int = 86_400,
    ) -> int:
        if not self.root.exists():
            return 0
        tracked = {str(Path(path).resolve()) for path in tracked_paths if path}
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in self.root.rglob("*"):
            if (
                not path.is_file()
                or self.staging_root in path.parents
                or path.name == ".gitkeep"
                or path.stat().st_mtime >= cutoff
                or str(path.resolve()) in tracked
            ):
                continue
            path.unlink()
            removed += 1
        for directory in sorted(
            (path for path in self.root.iterdir() if path.is_dir() and path != self.staging_root),
            reverse=True,
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        return removed

    @staticmethod
    def _validate_image_if_applicable(path: Path) -> None:
        try:
            with Image.open(path) as image:
                image.verify()
        except Image.UnidentifiedImageError:
            return
