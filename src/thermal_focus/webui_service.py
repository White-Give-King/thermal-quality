"""Pure-Python service layer backing the Gradio web UI."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .annotation_storage import (
    LABELS,
    save_panel_label,
    save_roi_coordinates,
    write_png,
)
from .classifier import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE,
    load_classifier,
)
from .decision import STATUS_LABELS_ZH
from .image_io import collect_image_paths, decode_image_bytes, load_image_for
from .inference import InferenceEngine, load_config
from .thermal_roi import ThermalBox, draw_thermal_roi, extract_thermal_roi
from .visualization import render_annotated_image

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("thermal_focus.webui")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
PANEL_CACHE_SIZE = 9
THUMB_MAX_WIDTH = 360
THUMB_MAX_HEIGHT = 480
MAX_ANNOTATION_SESSIONS = 4
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
                "--labels",
                str(Path(options["labels"]).expanduser().resolve()),
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
        labels = ROOT / "data" / workspace_name / "labels.csv"
        output = ROOT / "runs" / workspace_name
    else:
        labels = Path(str(payload.get("labels", ""))).expanduser()
        output = Path(str(payload.get("output", ""))).expanduser()
    if not labels.is_file():
        raise ValueError(f"labels CSV does not exist: {labels}")
    if not workspace_name and not str(payload.get("output", "")).strip():
        raise ValueError("output directory is required")
    pretrained = payload.get("pretrained", False)
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be a boolean")
    try:
        options: dict[str, Any] = {
            "labels": str(labels.resolve()),
            "output": str(output.resolve()),
            "workspace_name": workspace_name or None,
            "resume": str(payload.get("resume", "")).strip(),
            "epochs": int(payload.get("epochs", 20)),
            "architecture": str(payload.get("architecture", DEFAULT_ARCHITECTURE)),
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


def _list_workspaces(root_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"workspace root does not exist: {root}")
    workspaces: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if child.is_symlink() or not child.is_dir():
            continue
        image_dir = _workspace_image_dir(child)
        images = _collect_annotation_images(image_dir)
        if not images:
            continue
        name = child.name
        workspaces.append(
            {
                "name": name,
                "root_dir": str(root),
                "image_dir": str(image_dir.resolve()),
                "image_count": len(images),
                "data_dir": str(ROOT / "data" / name),
                "labels_csv": str(ROOT / "data" / name / "labels.csv"),
                "runs_dir": str(ROOT / "runs" / name),
            }
        )
    return workspaces


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
    """Scan the runs directory for best.pt checkpoints, newest first."""
    root = Path(runs_root).expanduser().resolve() if runs_root else ROOT / "runs"
    if not root.is_dir():
        return []
    entries: list[tuple[float, dict[str, Any]]] = []
    for best in root.rglob("best.pt"):
        try:
            stat_result = best.stat()
        except OSError:
            continue
        metadata = _read_training_metadata(best.parent)
        entries.append(
            (
                stat_result.st_mtime,
                {
                    "path": str(best),
                    "workspace": best.parent.name if best.parent != root else "未分组",
                    "modified": datetime.fromtimestamp(stat_result.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
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
        self._annotation_lock = threading.Lock()

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

    def _image_at(self, session: dict[str, Any], index: int) -> np.ndarray:
        """Decode the original image; only the current image is kept in memory."""
        cached = session["full_image"]
        if index == session["index"] and cached is not None and cached[0] == index:
            return cached[1]
        image = load_image_for(session["paths"][index], session["config"]).image
        if index == session["index"]:
            session["full_image"] = (index, image)
        return image

    def _panel_at(
        self, session: dict[str, Any], index: int
    ) -> tuple[np.ndarray, ThermalBox]:
        """Return the cropped thermal panel, keeping a bounded LRU of decoded panels."""
        cached = session["panels"].get(index)
        if cached is not None:
            return cached
        panel, box = extract_thermal_roi(
            self._image_at(session, index), session["config"]
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
        panel, box = self._panel_at(session, current)
        displayed = (
            draw_thermal_roi(self._image_at(session, current).copy(), box)
            if original
            else panel
        )
        labels = session["labels"]
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
            "labeled_count": len(labels),
            "preview": cv2.cvtColor(displayed, cv2.COLOR_BGR2RGB),
            "gallery": gallery,
            "gallery_indices": gallery_indices,
            "preview_errors": preview_errors,
            "roi": {
                key: int(session["config"]["thermal_roi"][key])
                for key in ("x", "y", "width", "height")
            },
        }

    def start_annotation(self, workspace_root: str | Path, workspace_name: str) -> dict[str, Any]:
        workspace_root = str(workspace_root).strip()
        workspace_name = _validate_workspace_name(workspace_name)
        if not workspace_root or not workspace_name:
            raise ValueError("select a workspace")
        config = load_config(self.config_path)
        root_path = Path(workspace_root).expanduser().resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(f"workspace root does not exist: {root_path}")
        workspace_candidate = root_path / workspace_name
        if workspace_candidate.is_symlink():
            raise ValueError("workspace folder cannot be a symbolic link")
        workspace_dir = workspace_candidate.resolve()
        if workspace_dir.parent != root_path or not workspace_dir.is_dir():
            raise ValueError(f"workspace does not exist: {workspace_name}")
        image_dir = _workspace_image_dir(workspace_dir)
        source_paths = _collect_annotation_images(image_dir)
        if not source_paths:
            raise ValueError(f"workspace contains no images: {workspace_name}")
        output_path = ROOT / "data" / workspace_name
        output_path.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        labels: dict[str, str] = {}
        csv_path = output_path / "labels.csv"
        if csv_path.is_file():
            with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    if row.get("source_path") and row.get("label"):
                        labels[_source_key(Path(row["source_path"]))] = row["label"]
        session = {
            "paths": source_paths,
            "index": 0,
            "output_dir": output_path,
            "workspace_name": workspace_name,
            "config": config,
            "labels": labels,
            "panels": {},
            "full_image": None,
        }
        with self._annotation_lock:
            while len(self._annotation_sessions) >= MAX_ANNOTATION_SESSIONS:
                self._annotation_sessions.pop(next(iter(self._annotation_sessions)))
            self._annotation_sessions[session_id] = session
        return {
            "session_id": session_id,
            "total": len(source_paths),
            "workspace_name": workspace_name,
            "input_dir": str(image_dir),
            "output_dir": str(output_path),
        }

    def navigate(self, session_id: str, index: int) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        try:
            index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("image index must be an integer") from exc
        if not 0 <= index < len(session["paths"]):
            raise ValueError("image index is out of range")
        session["index"] = index
        return self.render_annotation(session, original=False)

    def save_label(self, session_id: str, label: str, advance: bool = True) -> dict[str, Any]:
        session = self.get_annotation_session(session_id)
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not isinstance(advance, bool):
            raise ValueError("advance must be a boolean")
        index = session["index"]
        source = session["paths"][index]
        panel, _ = self._panel_at(session, index)
        row = save_panel_label(session["output_dir"], source, label, panel)
        session["labels"][_source_key(source)] = label
        if advance and index + 1 < len(session["paths"]):
            session["index"] = index + 1
        return {"saved": row, **self.render_annotation(session, original=False)}

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
        candidate = dict(session["config"])
        candidate["thermal_roi"] = {
            **session["config"]["thermal_roi"],
            **roi,
        }
        extract_thermal_roi(self._image_at(session, session["index"]), candidate)
        if save:
            rows: list[dict[str, str]] = []
            csv_path = session["output_dir"] / "labels.csv"
            if csv_path.is_file():
                with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
            updated = self._recrop_batch(session, rows, candidate) if rows else []
            save_roi_coordinates(self.config_path, tuple(roi[key] for key in ("x", "y", "width", "height")))
            for destination, panel in updated:
                write_png(destination, panel)
            session["config"] = candidate
            self._clear_image_cache(session)
            return {
                "updated_labels": len(updated),
                **self.render_annotation(session, original=False),
            }
        session["config"] = candidate
        self._clear_image_cache(session)
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
        results: list[dict[str, Any]] = []
        for file_path in file_paths:
            path = Path(file_path)
            filename = path.name
            try:
                if path.stat().st_size > MAX_UPLOAD_BYTES:
                    results.append(
                        {
                            "filename": filename,
                            "error": f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                        }
                    )
                    continue
                data = path.read_bytes()
                settings = engine.config.get("image", {})
                image_data = decode_image_bytes(
                    data,
                    max_dimension=int(settings.get("max_dimension", 12000)),
                    float_range=settings.get("float_range"),
                )
                prediction, thermal_image, thermal_box = engine.predict_array(
                    image_data,
                    image_name=filename,
                )
                annotated = render_annotated_image(
                    image_data.image,
                    thermal_box,
                    label=str(
                        prediction.get("label_zh", STATUS_LABELS_ZH["uncertain"])
                    ),
                    status=str(prediction["status"]),
                )
                results.append(
                    {
                        "filename": filename,
                        "prediction": prediction,
                        "annotated_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "thermal_image": cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB),
                    }
                )
            except (OSError, ValueError, RuntimeError) as exc:
                results.append({"filename": filename, "error": str(exc)})
        return {"results": results}

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
