"""Pure-Python service layer backing the Gradio web UI."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from annotation_storage import (
    LABELS,
    ROI_DATABASE_FILENAME,
    load_image_annotations,
    load_image_rois,
    save_image_roi,
    save_panel_label,
    write_png,
)
from classifier import (
    ARCHITECTURES,
    load_classifier,
)
from decision import STATUS_LABELS_ZH
from image_io import (
    IMAGE_SUFFIXES,
    ImageData,
    collect_image_paths,
    decode_image_bytes,
    load_image_for,
)
from inference import INFERENCE_BATCH_SIZE, InferenceEngine, load_config
from thermal_roi import (
    ThermalBox,
    detect_black_border_roi,
    draw_thermal_roi,
    extract_thermal_roi,
)
from visualization import render_annotated_image

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("thermal_focus_quality.webui")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
PANEL_CACHE_SIZE = 9
THUMB_MAX_WIDTH = 720
THUMB_MAX_HEIGHT = 960
MAX_ANNOTATION_SESSIONS = 64
ANNOTATION_CLAIM_TTL_SECONDS = 30 * 60
LOG_TAIL_CHARS = 60000


def _utf8_env() -> dict[str, str]:
    """Return a subprocess env that forces Python child processes to use UTF-8 I/O.

    Windows Chinese locale defaults to GBK (cp936), which mismatches our
    ``encoding="utf-8"`` log-file and causes mojibake in the WebUI log panel.
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class TrainingJobs:
    """Manage a single local training subprocess and expose its recent log."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise RuntimeError("a training job is already running")
            job_id = uuid.uuid4().hex
            output = Path(options["output"]).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            log_path = output / "webui-training.log"
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "train_classifier.py"),
                "--database",
                str(Path(options["database"]).expanduser().resolve()),
                "--image-root",
                str(Path(options["image_root"]).expanduser().resolve()),
                "--output",
                str(output),
                "--epochs",
                str(options["epochs"]),
                "--architecture",
                str(options["architecture"]),
                "--batch-size",
                str(options["batch_size"]),
                "--input-size",
                str(options["input_size"]),
                "--learning-rate",
                str(options["learning_rate"]),
                "--seed",
                str(options["seed"]),
                "--device",
                str(options["device"]),
            ]
            if options.get("resume"):
                command.extend(["--resume", str(Path(options["resume"]).expanduser().resolve())])
            if options.get("pretrained"):
                command.append("--pretrained")
            log_file = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_utf8_env(),
                )
            except OSError:
                log_file.close()
                raise
            self._job = {
                "job_id": job_id,
                "status": "running",
                "return_code": None,
                "log_path": log_path,
                "log_offset": 0,
                "log_tail": "",
                "process": process,
                "log_file": log_file,
                "output": str(output),
            }
            threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
            return {"job_id": job_id, "status": "running", "output": str(output)}

    def _watch(self, job_id: str) -> None:
        with self._lock:
            job = self._job
            if not job or job["job_id"] != job_id:
                return
            process = job["process"]
            log_file = job["log_file"]
        return_code = process.wait()
        log_file.close()
        with self._lock:
            if self._job and self._job["job_id"] == job_id:
                self._job["return_code"] = return_code
                self._job["status"] = "completed" if return_code == 0 else "failed"

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._job
            if not job or job["job_id"] != job_id:
                return None
            log_path = job["log_path"]
            offset = job["log_offset"]
            tail = job["log_tail"]
            status = {
                "job_id": job["job_id"],
                "status": job["status"],
                "return_code": job["return_code"],
                "output": job["output"],
            }
        try:
            with log_path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                if size < offset:
                    offset = 0
                stream.seek(offset)
                chunk = stream.read()
        except OSError as exc:
            LOGGER.warning("could not read training log %s: %s", log_path, exc)
            status["log"] = f"训练日志读取失败：{exc}"
            return status
        text = (tail + chunk.decode("utf-8", "replace"))[-LOG_TAIL_CHARS:]
        with self._lock:
            if self._job and self._job["job_id"] == job_id:
                self._job["log_offset"] = offset + len(chunk)
                self._job["log_tail"] = text
        status["log"] = text
        return status


def _device_name(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_workspace_name(value: Any) -> str:
    name = str(value or "").strip()
    if name and (
        Path(name).name != name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("workspace name must be a single folder name")
    return name


def _validate_training_options(payload: dict[str, Any]) -> dict[str, Any]:
    workspace_name = _validate_workspace_name(payload.get("workspace_name", ""))
    if workspace_name:
        workspace_root = Path(
            str(payload.get("workspace_root") or ROOT / "data")
        ).expanduser().resolve()
        workspace_dir = _find_workspace_dir(workspace_name, workspace_root)
        if workspace_dir is None:
            raise ValueError(f"workspace does not exist: {workspace_name}")
        database = workspace_dir / ROI_DATABASE_FILENAME
        image_root = _workspace_output_dir(workspace_root, workspace_dir)
        output = image_root / "runs"
    else:
        database = Path(str(payload.get("database", ""))).expanduser()
        image_root = Path(str(payload.get("image_root", database.parent))).expanduser()
        output = Path(str(payload.get("output", ""))).expanduser()
    if not database.is_file():
        raise ValueError(f"annotation database does not exist: {database}")
    if not workspace_name and not str(payload.get("output", "")).strip():
        raise ValueError("output directory is required")
    pretrained = payload.get("pretrained", True)
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be a boolean")
    try:
        options: dict[str, Any] = {
            "database": str(database.resolve()),
            "image_root": str(image_root.resolve()),
            "output": str(output.resolve()),
            "workspace_name": workspace_name or None,
            "resume": str(payload.get("resume", "")).strip(),
            "epochs": int(payload.get("epochs", 30)),
            "architecture": str(payload.get("architecture", "mobilenet_v3_large")),
            "batch_size": int(payload.get("batch_size", 16)),
            "input_size": int(payload.get("input_size", 224)),
            "learning_rate": float(payload.get("learning_rate", 1e-4)),
            "seed": int(payload.get("seed", 42)),
            "device": str(payload.get("device", "auto")),
            "pretrained": pretrained,
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid training parameter: {exc}") from exc
    if min(options["epochs"], options["batch_size"], options["input_size"]) <= 0:
        raise ValueError("epochs, batch size, and input size must be positive")
    if options["architecture"] not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if not math.isfinite(options["learning_rate"]) or options["learning_rate"] <= 0:
        raise ValueError("learning rate must be finite and positive")
    if options["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if options["resume"]:
        resume = Path(options["resume"]).expanduser()
        if not resume.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume}")
        options["resume"] = str(resume.resolve())
    return options


def _collect_annotation_images(input_dir: str | Path) -> list[Path]:
    return collect_image_paths(input_dir)


def _workspace_image_dir(workspace: Path) -> Path:
    """优先使用 images_source/ 子目录，避免把标注产物 images/ 当作原图。"""
    source = workspace / "images_source"
    return source if source.is_dir() else workspace


def _source_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def _direct_image_count(directory: Path) -> int:
    """统计文件夹直接存放的图片数（不递归子文件夹）。"""
    try:
        return sum(
            1
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    except OSError:
        return 0


def _workspace_image_count(directory: Path) -> int:
    """递归统计工作区图片，避免为计数构造和排序完整路径列表。"""
    target = directory.expanduser().resolve()
    if not target.is_dir():
        raise NotADirectoryError(f"image folder does not exist: {target}")
    return sum(
        1
        for item in target.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def _is_annotation_output(directory: Path) -> bool:
    """标注输出目录中的裁剪图不应被再次发现为原图工作区。"""
    return (
        (directory / "images").is_dir()
        and not (directory / "images_source").is_dir()
        and not _direct_image_count(directory)
    )


def _workspace_candidates(child: Path) -> list[Path]:
    if child.is_symlink() or not child.is_dir() or _is_annotation_output(child):
        return []
    if (child / "images_source").is_dir() or _direct_image_count(child):
        return [child]
    return [
        grandchild
        for grandchild in sorted(child.iterdir(), key=lambda path: path.name.casefold())
        if grandchild.is_dir()
        and not grandchild.is_symlink()
        and not _is_annotation_output(grandchild)
    ]


def _find_workspace_dir(
    workspace_name: str, root_dir: str | Path | None = None
) -> Path | None:
    """Locate a workspace directly under its root or one grouping folder below."""
    data_root = Path(root_dir).expanduser().resolve() if root_dir else ROOT / "data"
    direct = data_root / workspace_name
    if direct.is_dir():
        return direct
    if not data_root.is_dir():
        return None
    for entry in sorted(data_root.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        candidate = entry / workspace_name
        if candidate.is_dir():
            return candidate
    return None


def _workspace_output_dir(root: Path, directory: Path) -> Path:
    """工作区位于 data/ 下时产物写入工作区文件夹本身，否则统一输出到 data/<名称>/。"""
    data_root = (ROOT / "data").resolve()
    resolved = directory.resolve()
    if resolved.is_relative_to(data_root):
        return resolved
    return data_root / directory.name


def _workspace_record(root: Path, directory: Path) -> dict[str, Any] | None:
    image_dir = _workspace_image_dir(directory)
    image_count = _workspace_image_count(image_dir)
    if not image_count:
        return None
    name = directory.name
    output_dir = _workspace_output_dir(root, directory)
    return {
        "name": name,
        "root_dir": str(root),
        "image_dir": str(image_dir.resolve()),
        "image_count": image_count,
        "output_dir": str(output_dir),
        "data_dir": str(output_dir),
        "annotation_database": str(directory / ROI_DATABASE_FILENAME),
        "runs_dir": str(output_dir / "runs"),
    }


def _list_workspaces(root_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"workspace root does not exist: {root}")
    children = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        candidate_groups = pool.map(
            _workspace_candidates, children
        )
        candidates = [
            candidate for group in candidate_groups for candidate in group
        ]
        records = pool.map(
            lambda directory: _workspace_record(root, directory), candidates
        )
        return sorted(
            (record for record in records if record is not None),
            key=lambda record: record["name"].casefold(),
            reverse=True,
        )


def _read_training_metadata(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "training_config.json"
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def list_trained_models(runs_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Scan runs/ and workspace folders for best.pt checkpoints, newest first."""
    if runs_root:
        roots = [Path(runs_root).expanduser().resolve()]
    else:
        # 默认同时扫描旧版 runs/ 与 data/ 下各工作区文件夹内的 runs/。
        roots = [ROOT / "runs", ROOT / "data"]
    entries: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for best in root.rglob("best.pt"):
            resolved = str(best.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                stat_result = best.stat()
            except OSError:
                continue
            metadata = _read_training_metadata(best.parent)
            if best.parent.name == "runs" and best.parent != root:
                workspace = best.parent.parent.name
            else:
                workspace = best.parent.name
            entries.append(
                (
                    stat_result.st_mtime,
                    {
                        "path": str(best),
                        "workspace": workspace,
                        "modified": datetime.fromtimestamp(
                            stat_result.st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                        "architecture": str(metadata.get("architecture", "未知")),
                        "input_size": int(metadata.get("input_size", 0)) or None,
                    },
                )
            )
    entries.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in entries]


class InspectionDownloadJobs:
    """Run download_inspection_pictures.py as background subprocesses with streaming logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        json_input: str,
        output_dir: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "download_inspection_pictures.py"),
            json_input,
        ]
        if output_dir:
            command.extend(["-o", output_dir])
        if dry_run:
            command.append("--dry-run")

        # 子进程输出经管道读入内存，不在磁盘写日志文件
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_env(),
            bufsize=1,
        )
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "return_code": None,
                "log_tail": "",
                "process": process,
                "dry_run": dry_run,
                "json_input": json_input,
                "output_dir": output_dir,
            }
        threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
        return {
            "job_id": job_id,
            "status": "running",
            "dry_run": dry_run,
        }

    def _watch(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            process = job["process"]
        if process.stdout is not None:
            for chunk in process.stdout:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None:
                        break
                    job["log_tail"] = (job["log_tail"] + chunk)[-LOG_TAIL_CHARS:]
        return_code = process.wait()
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["return_code"] = return_code
                job["status"] = "completed" if return_code == 0 else "failed"

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "return_code": job["return_code"],
                "dry_run": job.get("dry_run"),
                "log": job["log_tail"],
            }


class WebUIService:
    """Hold the WebUI state and expose the operations behind the former HTTP API."""

    def __init__(self) -> None:
        self.jobs = TrainingJobs()
        self.download_jobs = InspectionDownloadJobs()
        self.config_path = Path(
            os.environ.get("THERMAL_FOCUS_CONFIG", str(ROOT / "configs" / "default.yaml"))
        ).expanduser()
        self._model_cache: dict[tuple[str, str], tuple[InferenceEngine, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()
        self._annotation_sessions: dict[str, dict[str, Any]] = {}
        self._annotation_lock = threading.RLock()
        self._annotation_claims: dict[tuple[str, str], tuple[str, float]] = {}

    def list_workspaces(self, root_dir: str | Path) -> dict[str, Any]:
        root_text = str(root_dir).strip()
        if not root_text:
            raise ValueError("workspace root directory is required")
        workspaces = _list_workspaces(root_text)
        return {
            "root_dir": str(Path(root_text).expanduser().resolve()),
            "workspaces": workspaces,
        }

    def list_models(self, runs_root: str | Path | None = None) -> dict[str, Any]:
        models = list_trained_models(runs_root)
        return {
            "runs_dir": str(Path(runs_root).expanduser().resolve())
            if runs_root
            else str(ROOT / "runs"),
            "models": models,
        }

    def get_annotation_session(self, session_id: str) -> dict[str, Any]:
        with self._annotation_lock:
            session = self._annotation_sessions.get(session_id)
        if session is None:
            raise ValueError("annotation session not found")
        return session

    def _annotation_labels(self, session: dict[str, Any]) -> dict[str, str]:
        labels = {
            _source_key(Path(row["source_path"])): row["label"]
            for row in load_image_annotations(session["roi_database_path"])
            if row["source_path"]
        }
        session["labels"] = labels
        return labels

    def _annotation_claim_key(
        self, session: dict[str, Any], index: int
    ) -> tuple[str, str]:
        return str(session["output_dir"].resolve()).casefold(), _source_key(
            session["paths"][index]
        )

    def _release_annotation_claim(self, session_id: str, session: dict[str, Any]) -> None:
        claimed_index = session.get("claimed_index")
        if claimed_index is not None:
            claim_key = self._annotation_claim_key(session, claimed_index)
            claim = self._annotation_claims.get(claim_key)
            if claim is not None and claim[0] == session_id:
                del self._annotation_claims[claim_key]
        session["claimed_index"] = None

    def _active_annotation_claim(self, claim_key: tuple[str, str]) -> str | None:
        claim = self._annotation_claims.get(claim_key)
        if claim is None:
            return None
        session_id, expires_at = claim
        if expires_at <= time.monotonic():
            del self._annotation_claims[claim_key]
            return None
        return session_id

    def _claim_annotation_index(
        self,
        session_id: str,
        session: dict[str, Any],
        index: int,
    ) -> bool:
        labels = session["labels"]
        claim_key = self._annotation_claim_key(session, index)
        owner = self._active_annotation_claim(claim_key)
        if _source_key(session["paths"][index]) in labels or (
            owner is not None and owner != session_id
        ):
            return False
        self._annotation_claims[claim_key] = (
            session_id,
            time.monotonic() + ANNOTATION_CLAIM_TTL_SECONDS,
        )
        session["claimed_index"] = index
        return True

    def _claim_next_annotation(
        self, session_id: str, session: dict[str, Any], start_index: int
    ) -> int | None:
        self._annotation_labels(session)
        total = len(session["paths"])
        for offset in range(total):
            index = (start_index + offset) % total
            if self._claim_annotation_index(session_id, session, index):
                return index
        return None

    def annotation_image_size(self, session_id: str) -> tuple[int, int]:
        session = self.get_annotation_session(session_id)
        height, width = self._image_at(session, session["index"]).shape[:2]
        return width, height

    def _image_at(self, session: dict[str, Any], index: int) -> np.ndarray:
        """Decode the original image; only the current image is kept in memory."""
        cached = session["full_image"]
        if index == session["index"] and cached is not None and cached[0] == index:
            return cached[1]
        image = load_image_for(session["paths"][index], session["config"]).image
        if index == session["index"]:
            session["full_image"] = (index, image)
        return image

    def _image_path_key(self, session: dict[str, Any], index: int) -> str:
        return session["paths"][index].relative_to(session["image_dir"]).as_posix().casefold()

    def _roi_for_image(
        self, session: dict[str, Any], index: int
    ) -> ThermalBox:
        image_rois: dict[int, ThermalBox] = session["image_rois"]
        roi = image_rois.get(index)
        if roi is not None:
            image = self._image_at(session, index)
            height, width = image.shape[:2]
            roi_width = min(roi[2], width)
            roi_height = min(roi[3], height)
            fitted_roi = (
                min(roi[0], width - roi_width),
                min(roi[1], height - roi_height),
                roi_width,
                roi_height,
            )
            image_rois[index] = fitted_roi
            return fitted_roi
        image = self._image_at(session, index)
        detected_roi = detect_black_border_roi(image)
        if detected_roi is not None:
            roi = detected_roi
        else:
            config = dict(session["config"])
            settings = dict(config["thermal_roi"])
            settings["mode"] = "centered"
            config["thermal_roi"] = settings
            _, roi = extract_thermal_roi(image, config)
        image_rois[index] = roi
        return roi

    def _image_config(
        self, session: dict[str, Any], index: int
    ) -> dict[str, Any]:
        x, y, width, height = self._roi_for_image(session, index)
        config = dict(session["config"])
        config["thermal_roi"] = {
            **session["config"]["thermal_roi"],
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "mode": "fixed",
        }
        return config

    def _panel_at(
        self, session: dict[str, Any], index: int
    ) -> tuple[np.ndarray, ThermalBox]:
        """Return the cropped thermal panel, keeping a bounded LRU of decoded panels."""
        cached = session["panels"].get(index)
        if cached is not None:
            return cached
        panel, box = extract_thermal_roi(
            self._image_at(session, index), self._image_config(session, index)
        )
        session["panels"][index] = (panel, box)
        while len(session["panels"]) > PANEL_CACHE_SIZE:
            session["panels"].pop(next(iter(session["panels"])))
        return panel, box

    def _clear_image_cache(self, session: dict[str, Any]) -> None:
        session["panels"].clear()
        session["full_image"] = None

    def render_annotation(self, session: dict[str, Any], *, original: bool) -> dict[str, Any]:
        current = session["index"]
        source = session["paths"][current]
        with self._annotation_lock:
            labels = self._annotation_labels(session)
            claim_key = self._annotation_claim_key(session, current)
            claim_owner = self._active_annotation_claim(claim_key)
            if claim_owner == session.get("session_id"):
                self._annotation_claims[claim_key] = (
                    claim_owner,
                    time.monotonic() + ANNOTATION_CLAIM_TTL_SECONDS,
                )
            claimed_by_other = claim_owner is not None and claim_owner != session["session_id"]
        panel, box = self._panel_at(session, current)
        image = self._image_at(session, current)
        displayed = (
            draw_thermal_roi(image.copy(), box)
            if original
            else panel
        )
        preview_errors: list[str] = []

        def _load_thumb(index: int) -> np.ndarray | None:
            path = session["paths"][index]
            try:
                panel, box = self._panel_at(session, index)
            except (OSError, ValueError):
                preview_errors.append(path.name)
                return None
            thumb = (
                draw_thermal_roi(self._image_at(session, index).copy(), box)
                if original
                else panel
            )
            thumb_h, thumb_w = thumb.shape[:2]
            scale = min(
                THUMB_MAX_WIDTH / thumb_w, THUMB_MAX_HEIGHT / thumb_h, 1.0
            )
            if scale < 1.0:
                thumb = cv2.resize(
                    thumb,
                    (max(1, round(thumb_w * scale)), max(1, round(thumb_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

        # Neighbour thumbnails double as the gallery preview (click a thumbnail
        # to enlarge it), so they are kept large enough to judge focus quality.
        gallery: list[np.ndarray] = []
        gallery_indices: list[int] = []
        total = len(session["paths"])
        window = 8
        start_idx = max(0, min(current - 3, total - window))
        for idx in range(start_idx, min(start_idx + window, total)):
            rgb = _load_thumb(idx)
            if rgb is None:
                continue
            gallery.append(rgb)
            gallery_indices.append(idx)

        return {
            "index": current,
            "total": len(session["paths"]),
            "filename": source.name,
            "label": labels.get(_source_key(source)),
            "claimed_by_other": claimed_by_other,
            "labeled_count": len(labels),
            "preview": cv2.cvtColor(displayed, cv2.COLOR_BGR2RGB),
            "gallery": gallery,
            "gallery_indices": gallery_indices,
            "preview_errors": preview_errors,
            "roi": {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3]),
            },
            "image_size": {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            },
        }

    def start_annotation(
        self,
        workspace_root: str | Path,
        workspace_name: str,
    ) -> dict[str, Any]:
        workspace_root = str(workspace_root).strip()
        workspace_name = _validate_workspace_name(workspace_name)
        if not workspace_root or not workspace_name:
            raise ValueError("select a workspace")
        config = load_config(self.config_path)
        root_path = Path(workspace_root).expanduser().resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(f"workspace root does not exist: {root_path}")
        workspace_candidate = root_path / workspace_name
        if not workspace_candidate.is_dir():
            # 分部归组结构：在根目录的一级子文件夹中查找同名工作区。
            matches = [
                entry
                for entry in root_path.iterdir()
                if entry.is_dir()
                and not entry.is_symlink()
                and (entry / workspace_name).is_dir()
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"workspace name is ambiguous under root: {workspace_name}"
                )
            if not matches:
                raise ValueError(f"workspace does not exist: {workspace_name}")
            workspace_candidate = matches[0] / workspace_name
        if workspace_candidate.is_symlink():
            raise ValueError("workspace folder cannot be a symbolic link")
        workspace_dir = workspace_candidate.resolve()
        if (
            workspace_dir.parent != root_path
            and workspace_dir.parent.parent != root_path
        ) or not workspace_dir.is_dir():
            raise ValueError(f"workspace does not exist: {workspace_name}")
        image_dir = _workspace_image_dir(workspace_dir)
        source_paths = _collect_annotation_images(image_dir)
        if not source_paths:
            raise ValueError(f"workspace contains no images: {workspace_name}")
        data_root = (ROOT / "data").resolve()
        if workspace_dir.resolve().is_relative_to(data_root):
            # data/ 下的工作区：标注产物直接写入工作区文件夹。
            output_path = workspace_dir
        else:
            output_path = ROOT / "data" / workspace_name
        output_path.mkdir(parents=True, exist_ok=True)
        roi_database_path = workspace_dir / ROI_DATABASE_FILENAME
        stored_rois = load_image_rois(roi_database_path)
        resolved_image_dir = image_dir.resolve()
        stored_rois_by_index: dict[int, ThermalBox] = {}
        for index, path in enumerate(source_paths):
            key = path.relative_to(resolved_image_dir).as_posix().casefold()
            if key in stored_rois:
                stored_rois_by_index[index] = stored_rois[key]
        session_id = uuid.uuid4().hex
        labels: dict[str, str] = {}
        session = {
            "paths": source_paths,
            "session_id": session_id,
            "index": 0,
            "output_dir": output_path,
            "workspace_name": workspace_name,
            "workspace_dir": workspace_dir,
            "image_dir": resolved_image_dir,
            "roi_database_path": roi_database_path,
            "config": config,
            "labels": labels,
            "claimed_index": None,
            "image_rois": stored_rois_by_index,
            "panels": {},
            "full_image": None,
        }
        with self._annotation_lock:
            while len(self._annotation_sessions) >= MAX_ANNOTATION_SESSIONS:
                expired_id = next(iter(self._annotation_sessions))
                expired = self._annotation_sessions.pop(expired_id)
                self._release_annotation_claim(expired_id, expired)
            self._annotation_sessions[session_id] = session
            next_index = self._claim_next_annotation(session_id, session, 0)
            if next_index is not None:
                session["index"] = next_index
            else:
                next_unlabeled = next(
                    (
                        index
                        for index, path in enumerate(source_paths)
                        if _source_key(path) not in session["labels"]
                    ),
                    None,
                )
                if next_unlabeled is not None:
                    session["index"] = next_unlabeled
        return {
            "session_id": session_id,
            "total": len(source_paths),
            "index": session["index"],
            "workspace_name": workspace_name,
            "input_dir": str(image_dir),
            "output_dir": str(output_path),
            "roi_database_path": str(roi_database_path),
        }

    def navigate(self, session_id: str, index: int) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        try:
            index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("image index must be an integer") from exc
        if not 0 <= index < len(session["paths"]):
            raise ValueError("image index is out of range")
        with self._annotation_lock:
            self._release_annotation_claim(session_id, session)
            session["index"] = index
            self._annotation_labels(session)
            self._claim_annotation_index(session_id, session, index)
        return self.render_annotation(session, original=False)

    def claim_next_annotation(self, session_id: str) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        with self._annotation_lock:
            current = session["index"]
            self._release_annotation_claim(session_id, session)
            next_index = self._claim_next_annotation(
                session_id, session, current + 1
            )
            if next_index is not None:
                session["index"] = next_index
        return {
            "claimed": next_index is not None,
            **self.render_annotation(session, original=False),
        }

    def save_label(self, session_id: str, label: str, advance: bool = True) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not isinstance(advance, bool):
            raise ValueError("advance must be a boolean")
        with self._annotation_lock:
            index = session["index"]
            source = session["paths"][index]
            source_key = _source_key(source)
            labels = self._annotation_labels(session)
            claim_key = self._annotation_claim_key(session, index)
            claim_owner = self._active_annotation_claim(claim_key)

            if source_key in labels:
                self._release_annotation_claim(session_id, session)
                next_index = self._claim_next_annotation(
                    session_id, session, index + 1
                )
                if next_index is not None:
                    session["index"] = next_index
                saved = None
                skipped = True
            else:
                if claim_owner not in (None, session_id):
                    raise ValueError(
                        "该图片已由其他网页领取，请标注其他未分配图片"
                    )
                if claim_owner is None and not self._claim_annotation_index(
                    session_id, session, index
                ):
                        raise ValueError("该图片已标注或已分配给其他网页")
                panel, _ = self._panel_at(session, index)
                roi = self._roi_for_image(session, index)
                save_image_roi(
                    session["roi_database_path"],
                    self._image_path_key(session, index),
                    roi,
                )
                saved = save_panel_label(
                    session["output_dir"],
                    session["roi_database_path"],
                    source,
                    label,
                    panel,
                )
                session["labels"][source_key] = label
                self._release_annotation_claim(session_id, session)
                skipped = None
                if advance:
                    next_index = self._claim_next_annotation(
                        session_id, session, index + 1
                    )
                    if next_index is not None:
                        session["index"] = next_index

        return {
            "saved": saved,
            "skipped": skipped,
            **self.render_annotation(session, original=False),
        }

    def _recrop_batch(
        self,
        session: dict[str, Any],
        rows: list[dict[str, str]],
        config: dict[str, Any],
    ) -> list[tuple[Path, np.ndarray]]:
        """Re-crop every labeled panel with the new ROI, decoding images in parallel."""

        def _recrop(row: dict[str, str]) -> tuple[Path, np.ndarray]:
            source_path = Path(row.get("source_path", ""))
            image_path = Path(row.get("image_path", ""))
            if not source_path.is_file():
                raise ValueError(
                    f"cannot update ROI: labeled source image is missing: {source_path}"
                )
            if image_path.is_absolute() or ".." in image_path.parts:
                raise ValueError(f"cannot update ROI: unsafe label image path: {image_path}")
            image = load_image_for(source_path, config).image
            panel, _ = extract_thermal_roi(image, config)
            return session["output_dir"] / image_path, panel

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            return list(pool.map(_recrop, rows))

    def update_roi(
        self,
        session_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        save: bool = False,
    ) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        try:
            roi = {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ROI coordinates: {exc}") from exc
        if roi["x"] < 0 or roi["y"] < 0 or roi["width"] <= 0 or roi["height"] <= 0:
            raise ValueError("ROI coordinates must be non-negative with positive size")
        index = session["index"]
        candidate = dict(session["config"])
        candidate["thermal_roi"] = {
            **session["config"]["thermal_roi"],
            **roi,
            # Explicit coordinates take precedence over centered mode.
            "mode": "fixed",
        }
        extract_thermal_roi(self._image_at(session, index), candidate)
        roi_box = (roi["x"], roi["y"], roi["width"], roi["height"])
        session["image_rois"][index] = roi_box
        self._clear_image_cache(session)
        if save:
            source_key = _source_key(session["paths"][index])
            current_row = next(
                (
                    row
                    for row in load_image_annotations(session["roi_database_path"])
                    if row["source_path"]
                    and _source_key(Path(row["source_path"])) == source_key
                ),
                None,
            )
            updated = self._recrop_batch(session, [current_row], candidate) if current_row else []
            for destination, panel in updated:
                write_png(destination, panel)
            save_image_roi(
                session["roi_database_path"],
                self._image_path_key(session, index),
                roi_box,
            )
            return {
                "updated_labels": len(updated),
                **self.render_annotation(session, original=False),
            }
        return self.render_annotation(session, original=False)

    def predict(
        self,
        file_paths: list[str | Path],
        model: str,
        device: str = "auto",
    ) -> dict[str, Any]:
        if not file_paths:
            raise ValueError("select at least one image")
        resolved_device = _device_name(device)
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"classifier checkpoint does not exist: {model_path}")
        key = (str(model_path), resolved_device)
        with self._cache_lock:
            cached = self._model_cache.get(key)
            if cached is None:
                config = load_config(self.config_path)
                classifier, metadata = load_classifier(model_path, device=resolved_device)
                engine = InferenceEngine(
                    config,
                    classifier=classifier,
                    classifier_metadata=metadata,
                    device=resolved_device,
                )
                cached = (engine, metadata)
                self._model_cache.clear()
                self._model_cache[key] = cached
        engine, _ = cached
        results: list[dict[str, Any] | None] = [None] * len(file_paths)
        for offset in range(0, len(file_paths), INFERENCE_BATCH_SIZE):
            loaded: list[tuple[int, Path, ImageData]] = []
            batch_paths = file_paths[offset : offset + INFERENCE_BATCH_SIZE]
            for index, file_path in enumerate(batch_paths, start=offset):
                path = Path(file_path)
                filename = path.name
                try:
                    if path.stat().st_size > MAX_UPLOAD_BYTES:
                        results[index] = {
                            "filename": filename,
                            "error": f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                        }
                        continue
                    data = path.read_bytes()
                    settings = engine.config.get("image", {})
                    image_data = decode_image_bytes(
                        data,
                        max_dimension=int(settings.get("max_dimension", 12000)),
                        float_range=settings.get("float_range"),
                    )
                    loaded.append((index, path, image_data))
                except (OSError, ValueError, RuntimeError) as exc:
                    results[index] = {"filename": filename, "error": str(exc)}

            predictions = engine.predict_batch(
                [(image_data, path.name) for _, path, image_data in loaded]
            )
            for (index, path, image_data), outcome in zip(loaded, predictions):
                filename = path.name
                if isinstance(outcome, Exception):
                    results[index] = {"filename": filename, "error": str(outcome)}
                    continue
                prediction, thermal_image, thermal_box = outcome
                try:
                    annotated = render_annotated_image(
                        image_data.image,
                        thermal_box,
                        label=str(
                            prediction.get("label_zh", STATUS_LABELS_ZH["uncertain"])
                        ),
                        status=str(prediction["status"]),
                    )
                    results[index] = {
                        "filename": filename,
                        "prediction": prediction,
                        "annotated_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "thermal_image": cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB),
                    }
                except (OSError, ValueError, RuntimeError) as exc:
                    results[index] = {"filename": filename, "error": str(exc)}
        if any(result is None for result in results):
            raise RuntimeError("batch prediction did not produce a result for every image")
        return {"results": [result for result in results if result is not None]}

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        options = _validate_training_options(payload)
        return self.jobs.start(options)

    def training_status(self, job_id: str) -> dict[str, Any]:
        status = self.jobs.status(job_id)
        if status is None:
            raise ValueError("training job not found")
        return status

    def start_download(
        self,
        json_input: str,
        output_dir: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self.download_jobs.start(json_input, output_dir=output_dir, dry_run=dry_run)

    def download_status(self, job_id: str) -> dict[str, Any]:
        status = self.download_jobs.status(job_id)
        if status is None:
            raise ValueError("download job not found")
        return status
