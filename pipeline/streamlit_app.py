from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "pipeline" / "scripts"
CHECKPOINTS = ROOT / "pipeline" / "checkpoints"
OCR_REFERENCES = ROOT / "datasets" / "prepared" / "12_ocr_soft_resized_v1" / "characters"
RUNS_ROOT = ROOT / "results" / "app_runs"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned[:80] or "estampage"


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_run(upload_name: str, data: bytes) -> dict[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_ROOT / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(upload_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    input_path = input_dir / f"{safe_stem(upload_name)}{suffix}"
    input_path.write_bytes(data)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "upload_name": upload_name,
        "upload_sha256": file_digest(data),
        "input_path": str(input_path),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_dir() -> Path | None:
    metadata = st.session_state.get("run")
    return Path(metadata["input_path"]).parents[1] if metadata else None


def restoration_dir() -> Path | None:
    metadata = st.session_state.get("run")
    if not metadata:
        return None
    stem = safe_stem(Path(metadata["input_path"]).name)
    return run_dir() / "restoration" / stem


def artifact(path: Path | None, name: str) -> Path | None:
    candidate = path / name if path else None
    return candidate if candidate and candidate.exists() else None


def run_command(stage: str, command: list[str]) -> bool:
    current_run = run_dir()
    if current_run is None:
        st.error("Create a run from an uploaded image first.")
        return False

    logs_dir = current_run / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage}.log"
    display = st.empty()
    started = time.monotonic()
    lines: list[str] = ["COMMAND", subprocess.list2cmdline(command), ""]

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())
            display.code("\n".join(lines[-18:]), language="text")
        code = process.wait()
    except Exception as exc:
        lines.append(f"ERROR: {exc}")
        code = -1

    elapsed = time.monotonic() - started
    lines.extend(["", f"exit_code={code}", f"elapsed_seconds={elapsed:.2f}"])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st.session_state.setdefault("stage_logs", {})[stage] = str(log_path)
    if code != 0:
        st.session_state.setdefault("stage_status", {})[stage] = "failed"
        st.error(f"{stage.title()} failed. Review the log below.")
        display.code("\n".join(lines[-30:]), language="text")
        return False

    st.session_state.setdefault("stage_status", {})[stage] = "complete"
    display.empty()
    st.success(f"{stage.title()} completed in {elapsed:.1f} seconds.")
    return True


def csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def suggest_segmentation_params(restored: Path) -> dict[str, int | float | bool]:
    mask_path = restored / "final_keep_mask.png"
    if not mask_path.exists():
        mask_path = restored / "keep_mask.png"
    if not mask_path.exists():
        return {"min_area": 24, "min_height": 28, "max_height": 150, "max_width": 180, "line_step": 105, "threshold": 0.25, "use_presence_filter": True}

    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    binary = (mask >= 96).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height_floor = max(8, int(round(mask.shape[0] * 0.02)))
    glyph_heights = [int(h) for _, _, _, h, area in stats[1:] if area >= 24 and h >= height_floor]
    if not glyph_heights:
        return {"min_area": 24, "min_height": 28, "max_height": 150, "max_width": 180, "line_step": 105, "threshold": 0.25, "use_presence_filter": True}

    median_height = float(np.median(glyph_heights))
    projection = binary.sum(axis=1).astype(np.float32).reshape(-1, 1)
    projection = cv2.GaussianBlur(
        projection,
        (1, 0),
        sigmaX=0,
        sigmaY=max(8.0, median_height * 0.6),
    ).ravel()
    radius = max(20, int(round(median_height * 1.5)))
    peaks: list[int] = []
    for y in np.argsort(projection)[::-1]:
        if projection[y] < projection.max() * 0.18:
            break
        if all(abs(int(y) - peak) > radius for peak in peaks):
            peaks.append(int(y))
    peaks.sort()
    dense_page = len(stats) > 500
    if dense_page:
        line_step = int(round(median_height * 1.3125))
    else:
        line_step = int(round(float(np.median(np.diff(peaks))))) if len(peaks) >= 2 else int(round(median_height * 1.4))

    scale = median_height / 80.0
    return {
        "min_area": max(12, int(round(24 * scale * scale))),
        "min_height": max(8, int(round(median_height * 0.35))),
        "max_height": max(150, int(round(median_height * 1.65))),
        "max_width": max(180, int(round(median_height * 1.5))),
        "line_step": max(30, line_step),
        "threshold": 0.25,
        "use_presence_filter": dense_page,
    }


def image_panel(path: Path | None, caption: str) -> None:
    if path is not None and path.exists():
        st.image(str(path), caption=caption, width="stretch")


def download(path: Path | None, label: str, mime: str) -> None:
    if path is not None and path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, width="stretch")


def stage_badge(label: str, ready: bool, complete: bool) -> None:
    state = "Complete" if complete else "Ready" if ready else "Waiting"
    css = "complete" if complete else "ready" if ready else "waiting"
    st.markdown(f'<div class="stage-badge {css}"><strong>{label}</strong><span>{state}</span></div>', unsafe_allow_html=True)


st.set_page_config(page_title="RAHAS Workbench", page_icon="R", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: 0 !important;}
      .stage-badge {border: 1px solid #d6d9de; border-radius: 6px; padding: .7rem .8rem; display:flex; justify-content:space-between; align-items:center; background:#fff;}
      .stage-badge span {font-size:.78rem; font-weight:700; text-transform:uppercase;}
      .stage-badge.complete {border-left:4px solid #138a52;}
      .stage-badge.complete span {color:#0d7544;}
      .stage-badge.ready {border-left:4px solid #ca7a08;}
      .stage-badge.ready span {color:#9b5900;}
      .stage-badge.waiting {border-left:4px solid #8a9099; color:#686e77;}
      [data-testid="stMetric"] {border:1px solid #dfe2e6; border-radius:6px; padding:.65rem .8rem; background:#fff;}
      [data-testid="stFileUploader"] {border-radius:6px;}
      .stButton button, .stDownloadButton button {border-radius:6px; min-height:2.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAHAS Estampage Workbench")
st.caption("Restoration · Character segmentation · Brahmi OCR")

uploaded = st.file_uploader("Estampage image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
uploaded_data = uploaded.getvalue() if uploaded else None
uploaded_hash = file_digest(uploaded_data) if uploaded_data else None
active_hash = st.session_state.get("run", {}).get("upload_sha256")

if uploaded_data and uploaded_hash != active_hash:
    with Image.open(uploaded) as preview_image:
        preview = preview_image.convert("RGB")
    left, right = st.columns([2, 1])
    with left:
        st.image(preview, caption=uploaded.name, width="stretch")
    with right:
        st.metric("Width", f"{preview.width:,} px")
        st.metric("Height", f"{preview.height:,} px")
        st.metric("File size", f"{len(uploaded_data) / 1_048_576:.2f} MB")
        if st.button("Create run", type="primary", width="stretch"):
            st.session_state["run"] = new_run(uploaded.name, uploaded_data)
            st.session_state["stage_status"] = {}
            st.session_state["stage_logs"] = {}
            st.rerun()

metadata = st.session_state.get("run")
if metadata:
    current_run = run_dir()
    restored_dir = restoration_dir()
    restored_image = artifact(restored_dir, "restored_dark_ocr.png")
    final_seg_dir = current_run / "segmentation" / "final"
    final_manifest = artifact(final_seg_dir, "prompt_manifest.csv")
    final_ocr_dir = current_run / "ocr" / "final"
    ocr_manifest = artifact(final_ocr_dir, "ocr_manifest.csv")

    statuses = st.session_state.setdefault("stage_status", {})
    status_cols = st.columns(3)
    with status_cols[0]:
        stage_badge("1 · Restoration", True, restored_image is not None)
    with status_cols[1]:
        stage_badge("2 · Segmentation", restored_image is not None, final_manifest is not None)
    with status_cols[2]:
        stage_badge("3 · OCR", final_manifest is not None, ocr_manifest is not None)

    with st.sidebar:
        st.subheader("Active run")
        st.code(metadata["run_id"], language="text")
        st.text(metadata["upload_name"])
        st.caption(str(current_run.relative_to(ROOT)))
        if st.button("Clear active run", width="stretch"):
            for key in ("run", "stage_status", "stage_logs"):
                st.session_state.pop(key, None)
            st.rerun()

    restoration_tab, segmentation_tab, ocr_tab = st.tabs(["Restoration", "Segmentation", "OCR"])

    with restoration_tab:
        st.subheader("Adaptive restoration")
        source_path = Path(metadata["input_path"])
        rest_cols = st.columns([1, 1, 1])
        with rest_cols[0]:
            method = st.segmented_control("Route", ["auto", "sparse_stroke", "letter_recall"], default="auto")
        with rest_cols[1]:
            profile_size = st.number_input("Profile max side", min_value=512, max_value=4096, value=1024, step=128)
        with rest_cols[2]:
            dense_size = st.number_input("Dense work max side", min_value=1024, max_value=4096, value=2304, step=128)

        if st.button("Run restoration", type="primary", width="stretch"):
            command = [
                sys.executable,
                str(SCRIPTS / "adaptive_restore_estampage_v1.py"),
                "--input", str(source_path),
                "--output", str(current_run / "restoration"),
                "--preview", str(current_run / "restoration_previews"),
                "--force_method", method or "auto",
                "--profile_max_side", str(profile_size),
                "--dense_work_max_side", str(dense_size),
            ]
            if run_command("restoration", command):
                st.rerun()

        restored_image = artifact(restored_dir, "restored_dark_ocr.png")
        if restored_image:
            image_cols = st.columns(2)
            with image_cols[0]:
                image_panel(source_path, "Uploaded estampage")
            with image_cols[1]:
                image_panel(restored_image, "Restored OCR image")
            action_cols = st.columns(3)
            with action_cols[0]:
                download(restored_image, "Download restored image", "image/png")
            with action_cols[1]:
                download(artifact(restored_dir, "soft_alpha.png"), "Download soft alpha", "image/png")
            with action_cols[2]:
                download(artifact(restored_dir, "selector_profile.txt"), "Download profile", "text/plain")

    with segmentation_tab:
        st.subheader("Character segmentation")
        if not restored_image:
            st.info("Complete restoration before segmentation.")
        else:
            suggested = suggest_segmentation_params(restored_dir)
            with st.expander("Segmentation controls"):
                control_cols = st.columns(3)
                with control_cols[0]:
                    min_area = st.number_input("Minimum component area", 4, 2000, int(suggested["min_area"]))
                with control_cols[1]:
                    min_height = st.number_input("Minimum character height", 8, 1000, int(suggested["min_height"]))
                with control_cols[2]:
                    max_height = st.number_input("Maximum character height", 40, 1500, int(suggested["max_height"]))
                control_cols = st.columns(3)
                with control_cols[0]:
                    max_width = st.number_input("Maximum character width", 40, 1500, int(suggested["max_width"]))
                with control_cols[1]:
                    line_step = st.number_input("Line step", 20, 1200, int(suggested["line_step"]))
                with control_cols[2]:
                    presence_threshold = st.slider("Presence threshold", 0.05, 0.95, float(suggested["threshold"]), 0.01)
                use_presence_filter = st.checkbox("Apply learned presence filter", value=bool(suggested["use_presence_filter"]))
                attach_fragments = st.checkbox("Attach nearby detached fragments to candidate boxes", value=False)

            if st.button("Run segmentation", type="primary", width="stretch"):
                candidates = current_run / "segmentation" / "candidates"
                filtered = current_run / "segmentation" / "filtered"
                final = current_run / "segmentation" / "final"
                commands: list[tuple[str, list[str]]] = [
                    (
                        "segmentation_candidates",
                        [sys.executable, str(SCRIPTS / "generate_episam_component_prompts_v1.py"),
                         "--input-dir", str(restored_dir), "--original-image", str(restored_image),
                         "--output", str(candidates), "--min-area", str(min_area),
                         "--min-height", str(min_height), "--max-height", str(max_height),
                         "--max-width", str(max_width), "--line-step", str(line_step)]
                        + (["--attach-fragments"] if attach_fragments else []),
                    ),
                ]
                if use_presence_filter:
                    commands.append((
                        "segmentation_presence",
                        [sys.executable, str(SCRIPTS / "filter_rahas_detector_prompts_v1.py"),
                         "--image", str(restored_image), "--manifest", str(candidates / "prompt_manifest.csv"),
                         "--checkpoint", str(CHECKPOINTS / "rahas_character_presence_v1" / "best.pt"),
                         "--output", str(filtered), "--threshold", str(presence_threshold)],
                    ))
                grouping_manifest = filtered / "prompt_manifest.csv" if use_presence_filter else candidates / "prompt_manifest.csv"
                commands.append((
                        "segmentation_grouping",
                        [sys.executable, str(SCRIPTS / "group_multicomponent_graphemes_v1.py"),
                         "--input-dir", str(restored_dir), "--manifest", str(grouping_manifest),
                         "--image", str(restored_image), "--output", str(final), "--line-step", str(line_step)],
                    ))
                success = True
                for stage, command in commands:
                    if not run_command(stage, command):
                        success = False
                        break
                if success:
                    st.session_state["stage_status"]["segmentation"] = "complete"
                    st.rerun()

            final_manifest = artifact(final_seg_dir, "prompt_manifest.csv")
            segment_rows = csv_rows(final_manifest)
            if final_manifest:
                metric_cols = st.columns(3)
                metric_cols[0].metric("Characters", f"{len(segment_rows):,}")
                metric_cols[1].metric("Connected components", sum(r.get("grouping_reason") == "connected_component" for r in segment_rows))
                metric_cols[2].metric("Three-dot groups", sum(r.get("grouping_reason") == "i_three_dot" for r in segment_rows))
                image_panel(artifact(final_seg_dir, "grouped_overlay.png"), "Final character boxes")
                download(final_manifest, "Download segmentation manifest", "text/csv")
                with st.expander("Manifest preview"):
                    st.dataframe(segment_rows[:200], width="stretch", hide_index=True)

    with ocr_tab:
        st.subheader("Brahmi OCR")
        if not final_manifest:
            st.info("Complete segmentation before OCR.")
        else:
            ocr_cols = st.columns(2)
            with ocr_cols[0]:
                workers = st.number_input("OCR workers", 0, 8, 2)
            with ocr_cols[1]:
                memory_mode = st.segmented_control("Recognition memory", ["exemplar", "centroid"], default="exemplar")

            if st.button("Run OCR", type="primary", width="stretch"):
                command = [
                    sys.executable,
                    str(SCRIPTS / "infer_rahas_spatial_proto_manifest_v1.py"),
                    "--image", str(restored_image),
                    "--manifest", str(final_manifest),
                    "--checkpoint", str(CHECKPOINTS / "rahas_spatial_proto_v1" / "best.pt"),
                    "--characters", str(OCR_REFERENCES),
                    "--output", str(final_ocr_dir),
                    "--memory-mode", memory_mode or "exemplar",
                    "--prototype-per-class", "0",
                    "--workers", str(workers),
                ]
                if run_command("ocr", command):
                    st.rerun()

            ocr_manifest = artifact(final_ocr_dir, "ocr_manifest.csv")
            ocr_rows = csv_rows(ocr_manifest)
            if ocr_manifest:
                accepted = sum(r.get("ocr_status") == "accepted" for r in ocr_rows)
                geometry = sum(r.get("ocr_status") == "geometry_confirmed" for r in ocr_rows)
                unknown = sum(r.get("ocr_status", "").startswith("unknown") for r in ocr_rows)
                metric_cols = st.columns(4)
                metric_cols[0].metric("Total", len(ocr_rows))
                metric_cols[1].metric("Accepted", accepted)
                metric_cols[2].metric("Geometry confirmed", geometry)
                metric_cols[3].metric("Unknown", unknown)
                image_panel(artifact(final_ocr_dir, "ocr_overlay.png"), "OCR result")
                text_path = artifact(final_ocr_dir, "ocr_output.txt")
                if text_path:
                    st.text_area("Recognized text", text_path.read_text(encoding="utf-8"), height=180)
                download(text_path, "Download OCR text", "text/plain")
                download(ocr_manifest, "Download OCR manifest", "text/csv")
                with st.expander("OCR manifest preview"):
                    st.dataframe(ocr_rows[:300], width="stretch", hide_index=True)

    logs = st.session_state.get("stage_logs", {})
    if logs:
        with st.expander("Run logs"):
            for stage, path_text in logs.items():
                path = Path(path_text)
                st.markdown(f"**{stage.replace('_', ' ').title()}**")
                st.code(path.read_text(encoding="utf-8", errors="replace")[-6000:], language="text")
else:
    st.info("Upload an estampage image and create a run.")
