import shutil
from pathlib import Path
from typing import Protocol

from finsight_core.config import get_settings


class ArtifactStore(Protocol):
    def save_artifact(self, local_src: Path | str, target_name: str) -> str: ...

    def load_artifact(self, target_name: str, local_dst: Path | str) -> Path: ...


class LocalArtifactStore:
    def __init__(self, base_dir: Path | str | None = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.ARTIFACTS_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, local_src: Path | str, target_name: str) -> str:
        src = Path(local_src)
        dst = self.base_dir / target_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)

    def load_artifact(self, target_name: str, local_dst: Path | str) -> Path:
        src = self.base_dir / target_name
        dst = Path(local_dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst


def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    if settings.STORAGE_BACKEND == "local":
        return LocalArtifactStore()
    # MinIO / S3 store will be enabled in stage D0
    return LocalArtifactStore()
