"""Manage immutable snapshots and shared assets for labeled image datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from annotation_storage import (
    ROI_DATABASE_FILENAME,
    load_image_annotations,
    save_image_annotation,
    write_png,
)
from decision import LABELS
from image_io import IMAGE_SUFFIXES, ImageData, decode_image_bytes
from thermal_roi import extract_thermal_roi

MANIFEST_FILENAME = "manifest.json"


class DatasetVersionStore:
    """Keep immutable annotation databases and shared, append-only image assets."""

    def __init__(self, output_dir: str | Path, source_workspace: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.source_workspace = Path(source_workspace).expanduser().resolve()
        self.root = self.output_dir / "dataset_versions"
        self.versions_dir = self.root / "versions"
        self.assets_dir = self.root / "assets"
        self.manifest_path = self.root / MANIFEST_FILENAME

    def initialized(self) -> bool:
        return self.manifest_path.is_file()

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read dataset version manifest: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("versions"), list)
            or not isinstance(manifest.get("active_version"), int)
            or not isinstance(manifest.get("latest_version"), int)
        ):
            raise ValueError(f"invalid dataset version manifest: {self.manifest_path}")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.root,
                prefix=".manifest-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _version_dir(versions_dir: Path, version: int) -> Path:
        return versions_dir / f"v{version:04d}"

    def _version_record(self, manifest: dict[str, Any], version: int) -> dict[str, Any]:
        records = manifest["versions"]
        assert isinstance(records, list)
        for record in records:
            if isinstance(record, dict) and record.get("version") == version:
                return record
        raise ValueError(f"unknown dataset version: {version}")

    def ensure_initialized(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return self._read_manifest()
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(
                f"dataset version directory exists without a manifest: {self.root}"
            )

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        (self.assets_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.assets_dir / "sources").mkdir(parents=True, exist_ok=True)
        version = 1
        final_dir = self._version_dir(self.versions_dir, version)
        staging_dir = self.versions_dir / f".{final_dir.name}-{uuid.uuid4().hex}.tmp"
        staging_dir.mkdir()
        try:
            database = staging_dir / ROI_DATABASE_FILENAME
            sample_hashes: set[str] = set()
            sample_count = 0
            database_sources = sorted(
                path
                for path in self.source_workspace.rglob(ROI_DATABASE_FILENAME)
                if path.is_file()
                and not path.is_symlink()
                and self.root not in path.resolve().parents
            )
            for source_database in database_sources:
                location = source_database.parent.relative_to(
                    self.source_workspace
                ).as_posix() or "root"
                for row in load_image_annotations(source_database):
                    relative = Path(row["image_path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            f"unsafe image path in baseline dataset: {relative}"
                        )
                    source_image = (source_database.parent / relative).resolve()
                    if not source_image.is_file():
                        raise FileNotFoundError(
                            f"baseline training image does not exist: {source_image}"
                        )
                    asset_relative = (
                        Path("legacy") / location / relative
                    ).as_posix()
                    destination = self.assets_dir / asset_relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(source_image, destination)
                    original_path = Path(row["source_path"])
                    if not original_path.is_file():
                        source_folder = source_database.parent / "images_source"
                        original_path = source_folder / Path(row["source_path"]).name
                    source_exists = original_path.is_file()
                    if source_exists:
                        sample_hashes.add(self._sha256_file(original_path))
                    save_image_annotation(
                        database,
                        {
                            "image_path": asset_relative,
                            "label": row["label"],
                            "source_path": str(
                                original_path.resolve()
                                if source_exists
                                else destination.resolve()
                            ),
                        },
                    )
                    sample_count += 1

            if sample_count == 0:
                load_image_annotations(database)
            with closing(sqlite3.connect(database)) as connection, connection:
                sample_count = int(
                    connection.execute("SELECT COUNT(*) FROM image_annotations").fetchone()[0]
                )

            os.replace(staging_dir, final_dir)
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "active_version": version,
                "latest_version": version,
                "versions": [
                    {
                        "version": version,
                        "parent": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "sample_count": sample_count,
                        "source_hashes": sorted(sample_hashes),
                    }
                ],
            }
            self._write_manifest(manifest)
            return manifest
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def status(self) -> dict[str, Any]:
        manifest = self.ensure_initialized()
        records = manifest["versions"]
        assert isinstance(records, list)
        active = int(manifest["active_version"])
        return {
            "active_version": active,
            "latest_version": int(manifest["latest_version"]),
            "versions": [dict(record) for record in records if isinstance(record, dict)],
        }

    def active_paths(self) -> tuple[Path, Path, int]:
        manifest = self.ensure_initialized()
        active = int(manifest["active_version"])
        database = self._version_dir(self.versions_dir, active) / ROI_DATABASE_FILENAME
        if not database.is_file():
            raise FileNotFoundError(f"active dataset database does not exist: {database}")
        return database, self.assets_dir, active

    def activate(self, version: int) -> dict[str, Any]:
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("dataset version must be an integer")
        manifest = self.ensure_initialized()
        self._version_record(manifest, version)
        manifest["active_version"] = version
        self._write_manifest(manifest)
        return self.status()

    def delete(self, version: int) -> dict[str, Any]:
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("dataset version must be an integer")
        manifest = self.ensure_initialized()
        self._version_record(manifest, version)
        if int(manifest["active_version"]) == version:
            records = manifest["versions"]
            candidates = sorted(
                (
                    int(record["version"])
                    for record in records
                    if isinstance(record, dict)
                    and isinstance(record.get("version"), int)
                    and not isinstance(record.get("version"), bool)
                    and record["version"] != version
                ),
                reverse=True,
            )
            replacement = None
            for candidate in candidates:
                candidate_dir = self._version_dir(self.versions_dir, candidate)
                candidate_database = candidate_dir / ROI_DATABASE_FILENAME
                if (
                    candidate_dir.is_dir()
                    and not candidate_dir.is_symlink()
                    and candidate_database.is_file()
                    and not candidate_database.is_symlink()
                ):
                    replacement = candidate
                    break
            if replacement is None:
                raise ValueError(
                    "cannot delete the active dataset version because no other usable version exists"
                )
            manifest["active_version"] = replacement

        version_dir = self._version_dir(self.versions_dir, version)
        if version_dir.is_symlink() or not version_dir.is_dir():
            raise FileNotFoundError(
                f"dataset version directory does not exist: {version_dir}"
            )
        temporary_dir = self.versions_dir / (
            f".{version_dir.name}-{uuid.uuid4().hex}.deleting"
        )
        os.replace(version_dir, temporary_dir)
        manifest["versions"] = [
            record
            for record in manifest["versions"]
            if not isinstance(record, dict) or record.get("version") != version
        ]
        try:
            self._write_manifest(manifest)
        except OSError:
            try:
                os.replace(temporary_dir, version_dir)
            except OSError as rollback_error:
                raise RuntimeError(
                    f"could not restore dataset version directory: {version_dir}"
                ) from rollback_error
            raise

        try:
            shutil.rmtree(temporary_dir)
        except OSError as exc:
            raise RuntimeError(
                f"dataset version {version} was removed from history, "
                f"but its files could not be fully deleted: {exc}"
            ) from exc
        return self.status()

    def import_labeled_images(
        self,
        files_by_label: Mapping[str, Sequence[str | Path]],
        *,
        config: dict[str, Any],
        max_upload_bytes: int,
    ) -> dict[str, Any]:
        manifest = self.ensure_initialized()
        active_database, _, active_version = self.active_paths()
        active_record = self._version_record(manifest, active_version)
        known_hashes = set(active_record.get("source_hashes", []))
        uploaded_hashes: set[str] = set()
        pending_database = self.source_workspace / ROI_DATABASE_FILENAME
        pending_rows = (
            load_image_annotations(pending_database)
            if pending_database.is_file()
            else []
        )
        prepared: list[tuple[Path, str, str, str, str, np.ndarray]] = []
        roi_settings = config.get("thermal_roi", {})
        roi_token = hashlib.sha256(
            json.dumps(roi_settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]

        for label, files in files_by_label.items():
            if label not in LABELS:
                raise ValueError(f"unsupported focus-quality label: {label}")
            for file_value in files:
                path = Path(file_value).expanduser().resolve()
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(f"unsupported image extension: {path.suffix}")
                if not path.is_file():
                    raise FileNotFoundError(f"uploaded image does not exist: {path}")
                size = path.stat().st_size
                if size > max_upload_bytes:
                    raise ValueError(
                        f"{path.name} exceeds {max_upload_bytes // (1024 * 1024)} MiB"
                    )
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest in known_hashes or digest in uploaded_hashes:
                    raise ValueError(f"duplicate image in the active dataset: {path.name}")
                uploaded_hashes.add(digest)
                image_data: ImageData = decode_image_bytes(
                    data,
                    max_dimension=int(config.get("image", {}).get("max_dimension", 12000)),
                    float_range=config.get("image", {}).get("float_range"),
                )
                panel, _ = extract_thermal_roi(image_data.image, config)
                source_name = f"{digest}{path.suffix.lower()}"
                image_name = f"{digest}_{roi_token}.png"
                prepared.append((path, label, digest, source_name, image_name, panel))

        if not prepared and not pending_rows:
            raise ValueError("select at least one labeled image to create a dataset version")

        maximum_version = max(
            int(manifest["latest_version"]),
            *(int(record["version"]) for record in manifest["versions"]),
        )
        new_version = maximum_version + 1
        final_dir = self._version_dir(self.versions_dir, new_version)
        staging_dir = self.versions_dir / f".{final_dir.name}-{uuid.uuid4().hex}.tmp"
        staging_dir.mkdir(parents=True)
        staged_assets = self.root / f".import-{uuid.uuid4().hex}"
        (staged_assets / "images").mkdir(parents=True)
        (staged_assets / "sources").mkdir(parents=True)
        try:
            database = staging_dir / ROI_DATABASE_FILENAME
            shutil.copy2(active_database, database)
            asset_records: list[tuple[Path, Path]] = []
            new_hashes = set(known_hashes)
            active_rows = load_image_annotations(database)
            active_by_source = {
                str(Path(row["source_path"]).resolve()).casefold(): row
                for row in active_rows
                if row["source_path"]
            }
            for row in pending_rows:
                relative = Path(row["image_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe pending annotation image path: {relative}")
                pending_image = (self.source_workspace / relative).resolve()
                if not pending_image.is_relative_to(self.source_workspace):
                    raise ValueError(f"pending image is outside data directory: {pending_image}")
                if not pending_image.is_file():
                    raise FileNotFoundError(f"pending annotation image is missing: {pending_image}")
                source_path = Path(row["source_path"]).resolve()
                source_key = str(source_path).casefold()
                existing = active_by_source.get(source_key)
                if existing is not None:
                    if existing["label"] != row["label"]:
                        with closing(sqlite3.connect(database)) as connection, connection:
                            connection.execute(
                                "UPDATE image_annotations SET label = ? WHERE image_path = ?",
                                (row["label"], existing["image_path"]),
                            )
                    continue
                source_hash = (
                    self._sha256_file(source_path) if source_path.is_file() else None
                )
                if source_hash and source_hash in new_hashes:
                    continue
                crop_hash = self._sha256_file(pending_image)
                image_name = f"{crop_hash}_{pending_image.name}"
                asset_relative = (Path("annotations") / image_name).as_posix()
                staged_image = staged_assets / "images" / image_name
                shutil.copy2(pending_image, staged_image)
                image_destination = self.assets_dir / asset_relative
                asset_records.append((staged_image, image_destination))
                save_image_annotation(
                    database,
                    {
                        "image_path": asset_relative,
                        "label": row["label"],
                        "source_path": str(source_path),
                    },
                )
                if source_hash:
                    new_hashes.add(source_hash)
            for path, label, digest, source_name, image_name, panel in prepared:
                staged_source = staged_assets / "sources" / source_name
                staged_source.write_bytes(path.read_bytes())
                staged_image = staged_assets / "images" / image_name
                write_png(staged_image, panel)
                source_destination = self.assets_dir / "sources" / source_name
                image_destination = self.assets_dir / "images" / image_name
                asset_records.extend(
                    [(staged_source, source_destination), (staged_image, image_destination)]
                )
                save_image_annotation(
                    database,
                    {
                        "image_path": (Path("images") / image_name).as_posix(),
                        "label": label,
                        "source_path": str(source_destination),
                    },
                )
                new_hashes.add(digest)

            with closing(sqlite3.connect(database)) as connection, connection:
                sample_count = int(
                    connection.execute("SELECT COUNT(*) FROM image_annotations").fetchone()[0]
                )
            for staged, destination in asset_records:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    os.replace(staged, destination)

            os.replace(staging_dir, final_dir)
            records = manifest["versions"]
            assert isinstance(records, list)
            records.append(
                {
                    "version": new_version,
                    "parent": active_version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "sample_count": sample_count,
                    "source_hashes": sorted(new_hashes),
                }
            )
            manifest["latest_version"] = new_version
            manifest["active_version"] = new_version
            self._write_manifest(manifest)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if staged_assets.exists():
                shutil.rmtree(staged_assets)
        return self.status()
