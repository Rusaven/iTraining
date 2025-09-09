# Import libraries
import os
import sys
import time
import json
import shutil
import zipfile
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Directories & constants
# -----------------------------
WORKDIR    = Path.cwd()
DATA_DIR   = WORKDIR  / "datasets"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = WORKDIR  / "models"
EXPORT_DIR = WORKDIR  / "exports"
RUNS_DIR   = WORKDIR  / "runs"
TMP_DIR    = WORKDIR  / "tmp"

for d in [DATA_DIR, IMAGES_DIR, LABELS_DIR, MODELS_DIR, EXPORT_DIR, RUNS_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilities: file I/O, labels, plotting, exports
# -----------------------------
def save_uploaded_file(uploaded_file, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def extract_zip_to(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    return dest_dir

def list_images(folder: Path) -> List[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])

def ensure_data_yaml():
    p = WORKDIR / "data.yaml"
    if not p.exists():
        p.write_text("train: datasets/images\nval: datasets/images\nnc: 1\nnames: ['object']")
    return p

# YOLO label helpers
def load_yolo_label(txt_path: Path) -> List[Dict]:
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append({"class": cls, "xc": xc, "yc": yc, "w": w, "h": h})
    return boxes

def yolo_to_pixel(box: Dict, img_w: int, img_h: int):
    xc, yc, bw, bh = box['xc'], box['yc'], box['w'], box['h']
    wbox, hbox = bw * img_w, bh * img_h
    left = xc * img_w - wbox / 2
    top = yc * img_h - hbox / 2
    return left, top, wbox, hbox

def save_labels_yolo(img_path: Path, labels: List[Dict]) -> Path:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LABELS_DIR / (img_path.stem + ".txt")
    img = Image.open(img_path)
    W, H = img.size
    with open(out_file, "w") as f:
        for l in labels:
            # expected keys: left, top, width, height, class
            x0 = l['left']; y0 = l['top']; wbox = l['width']; hbox = l['height']
            x1 = x0 + wbox; y1 = y0 + hbox
            xc = (x0 + x1) / 2 / W
            yc = (y0 + y1) / 2 / H
            bw = wbox / W
            bh = hbox / H
            f.write(f"{int(l['class'])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return out_file

def draw_overlay_boxes(pil_img: Image.Image, boxes_pixel: List[Dict], classes: Optional[List[str]] = None, color=(255,0,0)):
    img = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(overlay)
    for b in boxes_pixel:
        left, top, wbox, hbox = b['left'], b['top'], b['width'], b['height']
        rect = [left, top, left + wbox, top + hbox]
        draw.rectangle(rect, outline=color+(255,), width=2)
        if 'class' in b and classes:
            try:
                clsname = classes[int(b['class'])]
            except Exception:
                clsname = str(b.get('class',''))
            draw.text((left, max(0, top - 12)), clsname, fill=color+(255,))
    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")

def export_dataset_zip(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in IMAGES_DIR.glob("*"):
            z.write(p, arcname=f"images/{p.name}")
        for p in LABELS_DIR.glob("*"):
            z.write(p, arcname=f"labels/{p.name}")
        dy = WORKDIR / "data.yaml"
        if dy.exists():
            z.write(dy, arcname="data.yaml")
    return out_path

def plot_results_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    fig, axs = plt.subplots(2, 1, figsize=(8,6))
    x = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
    # Loss related
    loss_cols = [c for c in df.columns if 'loss' in c]
    if loss_cols:
        for c in loss_cols:
            axs[0].plot(x, df[c], label=c)
    axs[0].set_title("Loss")
    axs[0].legend()
    # Metric columns: map50, map, precision, recall
    metric_cols = [c for c in df.columns if c in ('map50','map','precision','recall') or 'map' in c]
    for c in metric_cols:
        axs[1].plot(x, df[c], label=c)
    axs[1].set_title("Metrics")
    axs[1].legend()
    plt.tight_layout()
    return fig

def simple_pdf_report(summary: dict, per_class_df: Optional[pd.DataFrame]=None, out_pdf: Optional[Path]=None) -> Path:
    if out_pdf is None:
        out_pdf = EXPORT_DIR / f"eval_report_{int(time.time())}.pdf"
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis('off')
    y = 0.95
    ax.text(0.01, y, "Evaluation Report", fontsize=18, weight='bold')
    y -= 0.04
    ax.text(0.01, y, f"Timestamp: {time.ctime()}", fontsize=9)
    y -= 0.04
    for k, v in summary.items():
        ax.text(0.01, y, f"{k}: {v}", fontsize=10)
        y -= 0.03
    if per_class_df is not None:
        y -= 0.02
        ax.text(0.01, y, "Per-class metrics (first rows):", fontsize=11, weight='bold')
        y -= 0.02
        tbl = per_class_df.head(12)
        table = ax.table(cellText=tbl.values, colLabels=tbl.columns, loc='lower left', bbox=[0.01, 0.01, 0.98, y-0.01])
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_pdf

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="iTraining", layout="wide")
st.title("iTraining - Label, Train, and Test")
st.markdown("Fully functionable website for labeling, training, and testing your computer vision datasets.")

tabs = st.tabs(["Labeling", "Training", "Testing", "Optional - Detection"])

# -----------------------------
# TAB 1: LABELING
# -----------------------------
with tabs[0]:
    col_Management, col_Canvas, col_Utilities = st.columns([1,4,1])

    with col_Management:
        st.subheader("Upload")
        zip_upload = st.file_uploader("Upload images .ZIP", type=["zip"], key="label_zip_upload")
        if zip_upload:
            tmp_zip = TMP_DIR / f"images_upload_{int(time.time())}.zip"
            save_uploaded_file(zip_upload, tmp_zip)
            extract_zip_to(tmp_zip, IMAGES_DIR)
            st.success(f"Extracted images to {IMAGES_DIR}")

        multi_up = st.file_uploader("Or upload images", type=["jpg","png","jpeg"], accept_multiple_files=True, key="label_multi_upload")
        if multi_up:
            for f in multi_up:
                save_uploaded_file(f, IMAGES_DIR / f.name)
            st.success("Saved uploaded images to datasets/images")

        st.subheader("Feature")
        if st.button("Export dataset", key="label_export_btn"):
            outp = EXPORT_DIR / f"dataset_export_{int(time.time())}.zip"
            export_dataset_zip(outp)
            st.success(f"Export ready: {outp}")
            st.download_button("Download exported dataset", data=open(outp, "rb"), file_name=outp.name, key="label_export_download")

        if st.button("Clear labels folder", key="label_clear_btn"):
            for f in LABELS_DIR.glob("*"):
                f.unlink()
            st.success("Cleared labels folder")

        if st.button("Check data.yaml", key="label_ensure_yaml"):
            ensure_data_yaml()
            st.success("data.yaml ensured in working dir")

    with col_Canvas:
        st.subheader("Canvas")
        images = list_images(IMAGES_DIR)
        if not images:
            st.warning("No images in datasets/images. Upload first!")
        else:
            if "label_idx" not in st.session_state:
                st.session_state["label_idx"] = 0
            idx = st.session_state["label_idx"]
            total = len(images)
            st.write(f"Image {idx+1}/{total} — {images[idx].name}")

            # thumbnails (first 24)
            thumbs_cols = st.columns(6)
            for i,p in enumerate(images[:24]):
                try:
                    thumb = Image.open(p)
                    thumb.thumbnail((80,80))
                    if thumbs_cols[i % 6].button(p.name, key=f"label_thumb_{i}"):
                        st.session_state["label_idx"] = i
                        idx = i
                except Exception:
                    continue

            nav_l, nav_c, nav_r = st.columns([1,8,1])
            with nav_l:
                if st.button("⬅ Prev", key="label_prev"):
                    if st.session_state["label_idx"] > 0:
                        st.session_state["label_idx"] -= 1
            with nav_r:
                if st.button("Next ➡", key="label_next"):
                    if st.session_state["label_idx"] < total - 1:
                        st.session_state["label_idx"] += 1

            selected_path = images[st.session_state["label_idx"]]
            pil_img = Image.open(selected_path)
            st.image(pil_img, use_column_width=True)

            st.markdown("**Classes (one per line)**")
            classes_text = st.text_area("Classes", value="\n".join(["person","car","dog"]), height=120, key="label_classes_text")
            classes = [c.strip() for c in classes_text.splitlines() if c.strip()]
            if not classes:
                st.error("Define at least one class (one per line).")

            # load existing labels and overlay
            existing = load_yolo_label(LABELS_DIR / (selected_path.stem + ".txt"))
            overlay_boxes = []
            for b in existing:
                left, top, wbox, hbox = yolo_to_pixel(b, pil_img.width, pil_img.height)
                overlay_boxes.append({"left": left, "top": top, "width": wbox, "height": hbox, "class": b['class']})
            pil_overlay = draw_overlay_boxes(pil_img, overlay_boxes, classes) if overlay_boxes else pil_img

            if st_canvas is None:
                st.error("Install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
            else:
                max_w, max_h = 1200, 800
                disp_w = min(pil_overlay.width, max_w)
                disp_h = min(pil_overlay.height, max_h)
                canvas_res = st_canvas(
                    background_image=pil_overlay,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="rect",
                    key=f"label_canvas_{selected_path.name}"
                )

                if canvas_res is not None and canvas_res.json_data is not None:
                    objs = canvas_res.json_data.get("objects", [])
                    st.write(f"{len(objs)} object(s) drawn")
                    new_labels = []
                    for i_obj, obj in enumerate(objs):
                        left = obj.get("left",0); top = obj.get("top",0)
                        width_b = obj.get("width",0); height_b = obj.get("height",0)
                        sel_cls = st.selectbox(f"Class for object #{i_obj+1}", options=classes, index=0, key=f"label_objclass_{selected_path.name}_{i_obj}")
                        new_labels.append({"left": left, "top": top, "width": width_b, "height": height_b, "class": classes.index(sel_cls)})
                    if st.button("Save labels (overwrite)", key=f"label_save_{selected_path.name}"):
                        saved = save_labels_yolo(selected_path, new_labels)
                        st.success(f"Saved labels to {saved}")
                else:
                    st.info("Draw boxes on the canvas and click Save. Existing labels are shown as overlay (non-editable on canvas).")

            # progress
            labeled_count = sum(1 for p in images if (LABELS_DIR / (p.stem + ".txt")).exists())
            st.progress(int(100 * labeled_count / total))

    with col_Utilities:
        st.subheader("Utilities")
        if st.button("Open images path", key="label_util_open_images"):
            st.write(IMAGES_DIR)
        if st.button("Open labels path", key="label_util_open_labels"):
            st.write(LABELS_DIR)
        if st.button("Label stats", key="label_util_stats"):
            imgs = list_images(IMAGES_DIR)
            total = len(imgs)
            labeled = sum(1 for p in imgs if (LABELS_DIR / (p.stem + ".txt")).exists())
            st.write(f"Total images: {total}")
            st.write(f"Labeled: {labeled}")
            st.write(f"Unlabeled: {total - labeled}")

# -----------------------------
# TAB 2: TRAINING
# -----------------------------
with tabs[1]:
    col_Setup, col_Control = st.columns([2,1])

    with col_Setup:
        st.subheader("Upload")
        uploaded_yaml = st.file_uploader("Upload data.yaml (train/val/names)", type=['yaml','yml'], key="train_yaml_upload")
        if uploaded_yaml:
            save_uploaded_file(uploaded_yaml, WORKDIR / "data.yaml")
            st.success("Saved data.yaml to working directory.")
        elif (WORKDIR / "data.yaml").exists():
            st.info(f"Found existing data.yaml: {WORKDIR/'data.yaml'}")
        else:
            st.warning("No data.yaml found. Upload or place data.yaml in working directory.")

        st.markdown("**Images folder preview**")
        images = list_images(IMAGES_DIR)
        st.write(f"Images in {IMAGES_DIR}: {len(images)}")
        if st.button("Refresh image list", key="train_refresh_images"):
            images = list_images(IMAGES_DIR)
            st.experimental_rerun()

        st.subheader("Model selection")
        yolo_version = st.selectbox("YOLO Version", ["YOLOv5","YOLOv8","YOLOv11"], index=0, key="train_yolo_version")
        model_size = st.selectbox("Model size", ["Nano","Small","Medium","Large","Extra Large"], index=1, key="train_model_size")
        default_weights = f"yolov{yolo_version[-1]}{model_size}.pt" if yolo_version != "YOLOv5" else f"yolov5{model_size}.pt"
        custom_weights = st.file_uploader("Upload starting weights (.pt)", type=['pt'], key="train_custom_weights")
        if custom_weights:
            save_uploaded_file(custom_weights, MODELS_DIR / custom_weights.name)
            st.success("Uploaded starting weights saved.")
        
        st.subheader("Parameter")
        batch = st.selectbox("Batch size", [4,6,8,12,16,32], index=3, key="train_batch")
        epochs = st.selectbox("Epochs", list(range(100,1001,100)), index=2, key="train_epochs")
        imgsz = st.slider("Image size", 128, 1024, 640, step=32, key="train_imgsz")
        has_cuda = False
        try:
            import torch as _torch
            has_cuda = _torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if has_cuda:
            device = st.radio("Device", ["CUDA (GPU)","CPU"], index=0, key="train_device")
        else:
            device = st.radio("Device", ["CPU"], index=0, key="train_device")
            st.warning("CUDA not available — training on CPU may be slow.")

        st.subheader("Augmentation")
        aug_flip = st.checkbox("Horizontal flip", value=True, key="train_aug_flip")
        aug_mosaic = st.checkbox("Mosaic augmentation", value=False, key="train_aug_mosaic")
        aug_hsv = st.checkbox("HSV augmentation", value=True, key="train_aug_hsv")
        resume_checkpoint = st.file_uploader("Resume checkpoint (.pt)", type=['pt'], key="train_resume_checkpoint")

    with col_Control:
        st.subheader("Control panel")
        start_training = st.button("Start Training", key="train_start_btn")
        stop_training = st.button("Stop Training", key="train_stop_btn")
        st.write("After training you can export model formats (placeholders for conversion).")
        export_formats = st.multiselect("Export formats", ["pt","onnx","tflite"], default=["pt"], key="train_export_formats")
        st.checkbox("Save results.csv if present", value=True, key="train_save_results_csv")
        st.markdown("Recent run folders:")
        for runp in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
            st.write(runp.name)

    # Training execution
    if start_training:
        if not (WORKDIR / "data.yaml").exists():
            st.error("data.yaml is required in working directory (or upload).")
        else:
            run_id = int(time.time())
            run_dir = RUNS_DIR / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            st.info(f"Created run dir: {run_dir}")

            config = {
                "version": yolo_version,
                "model_size": model_size,
                "batch": batch,
                "epochs": epochs,
                "imgsz": imgsz,
                "device": device,
                "augment": {"flip": aug_flip, "mosaic": aug_mosaic, "hsv": aug_hsv}
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            # YOLOv5 via repo script (non-blocking subprocess)
            if yolo_version == "YOLOv5":
                yolov5_repo = WORKDIR / "yolov5"
                if not yolov5_repo.exists():
                    st.error("yolov5 repo not found — clone https://github.com/ultralytics/yolov5 into working dir to use script.")
                else:
                    weights_arg = str(MODELS_DIR / custom_weights.name) if custom_weights else default_weights
                    cmd = [
                        sys.executable, str(yolov5_repo / "train.py"),
                        "--img", str(imgsz), "--batch", str(batch), "--epochs", str(epochs),
                        "--data", str(WORKDIR / "data.yaml"),
                        "--weights", weights_arg,
                        "--project", str(run_dir), "--name", "exp"
                    ]
                    if resume_checkpoint:
                        rp = save_uploaded_file(resume_checkpoint, MODELS_DIR / resume_checkpoint.name)
                        cmd += ["--resume", str(rp)]
                    st.write("Running:", " ".join(cmd))
                    try:
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        log_box = st.empty()
                        progress = st.progress(0)
                        i = 0
                        while True:
                            line = proc.stdout.readline()
                            if line == "" and proc.poll() is not None:
                                break
                            if line:
                                i += 1
                                if i % 5 == 0:
                                    progress.progress(min(0.95, i/1000))
                                log_box.text(line)
                        proc.wait()
                        st.success("YOLOv5 training finished. Check run folder for artifacts.")
                    except Exception as e:
                        st.error(f"YOLOv5 training failed: {e}")
            else:
                # ultralytics API training (blocking)
                if YOLO is None:
                    st.error("ultralytics not installed — pip install ultralytics to train YOLOv8/YOLOv11")
                else:
                    weights_to_use = str(MODELS_DIR / custom_weights.name) if custom_weights else default_weights
                    st.write("Using weights:", weights_to_use)
                    try:
                        model = YOLO(weights_to_use)
                        st.info("Starting ultralytics training (blocking call). This may take long.")
                        res = model.train(data=str(WORKDIR / "data.yaml"), epochs=epochs, batch=batch, imgsz=imgsz, device=device)
                        st.success("Ultralytics training finished.")
                        try:
                            ckpt = getattr(model, "ckpt_path", None)
                            if ckpt and Path(ckpt).exists():
                                dest = MODELS_DIR / Path(ckpt).name
                                shutil.copy(ckpt, dest)
                                st.info(f"Copied checkpoint to {dest}")
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Ultralytics training error: {e}")

            # After training, try to find results.csv for plotting
            candidates = list(run_dir.rglob("results.csv")) + list(run_dir.rglob("results*.csv"))
            if candidates:
                csvp = candidates[0]
                st.write("Found results CSV:", csvp)
                fig = plot_results_csv(csvp)
                if fig:
                    st.pyplot(fig)
            else:
                # search in common runs
                common = list(RUNS_DIR.glob("**/results.csv"))
                if common:
                    st.write("Found results.csv in other runs:", common[0])
                    fig = plot_results_csv(common[0])
                    if fig:
                        st.pyplot(fig)
                else:
                    st.info("No results CSV found in run directory.")

    if stop_training:
        st.warning("Stop requested — if training started subprocesses please kill processes manually on your system.")

# -----------------------------
# TAB 3: TESTING & EVALUATION
# -----------------------------
with tabs[2]:
    col_Test, col_Export = st.columns([2,1])

    with col_Test:
        st.subheader("Setup")
        model_upload = st.file_uploader("Upload model (.pt) for evaluation", type=['pt'], key="test_model_upload")
        model_existing = st.selectbox("Or choose existing model", [p.name for p in MODELS_DIR.glob("*.pt")] if any(MODELS_DIR.glob("*.pt")) else ["--none--"], key="test_model_select")
        yaml_upload = st.file_uploader("Upload data.yaml for test", type=['yaml','yml'], key="test_yaml_upload")
        test_images_zip = st.file_uploader("Upload test images ZIP", type=['zip'], key="test_images_zip")
        conf_th = st.slider("Confidence threshold", 0.0, 1.0, 0.25, key="test_conf")
        iou_th = st.slider("IoU threshold", 0.0, 1.0, 0.45, key="test_iou")
        run_eval = st.button("Run evaluation", key="test_run_eval")

        st.subheader("Quick inference")
        quick_img = st.file_uploader("Upload image for quick inference", type=['jpg','png','jpeg'], key="test_quick_img")
        run_quick = st.button("Run quick inference", key="test_run_quick")

    with col_Export:
        st.subheader("Export")
        export_csv_btn = st.button("Export last eval CSV", key="test_export_csv")
        export_pdf_btn = st.button("Export last eval PDF", key="test_export_pdf")
        st.markdown("Saved reports:")
        for p in sorted(EXPORT_DIR.glob("*"), key=lambda q: q.stat().st_mtime, reverse=True)[:8]:
            st.write(p.name)

    # prepare selected model path
    selected_model_path = None
    if model_upload:
        selected_model_path = MODELS_DIR / model_upload.name
        save_uploaded_file(model_upload, selected_model_path)
    elif model_existing and model_existing != "--none--":
        selected_model_path = MODELS_DIR / model_existing

    # extract test images if uploaded
    if test_images_zip:
        tmpz = TMP_DIR / f"test_images_{int(time.time())}.zip"
        save_uploaded_file(test_images_zip, tmpz)
        extract_zip_to(tmpz, WORKDIR / "test_images")
        st.success("Extracted test images to ./test_images")

    # quick inference
    if run_quick:
        if quick_img is None:
            st.error("Upload an image first")
        elif selected_model_path is None:
            st.error("Select or upload a model first")
        elif YOLO is None:
            st.error("ultralytics not installed")
        else:
            tmp = TMP_DIR / f"tmp_infer_{int(time.time())}.jpg"
            save_uploaded_file(quick_img, tmp)
            try:
                model = YOLO(str(selected_model_path))
                preds = model.predict(source=str(tmp), conf=conf_th, iou=iou_th)
                try:
                    rendered = preds[0].plot()
                    st.image(Image.fromarray(rendered), caption="Quick inference")
                except Exception:
                    st.write(preds[0].boxes)
            except Exception as e:
                st.error(f"Inference failed: {e}")

    # run evaluation
    if run_eval:
        if selected_model_path is None:
            st.error("Select or upload a model first")
        else:
            yaml_path = None
            if yaml_upload:
                yaml_path = WORKDIR / f"test_data_{int(time.time())}.yaml"
                save_uploaded_file(yaml_upload, yaml_path)
            elif (WORKDIR / "data.yaml").exists():
                yaml_path = WORKDIR / "data.yaml"
            else:
                st.error("No data.yaml available for evaluation. Upload one or place data.yaml in working dir.")
                yaml_path = None

            if yaml_path and YOLO is not None:
                try:
                    model = YOLO(str(selected_model_path))
                    st.info("Running evaluation — this may take time based on dataset size.")
                    metrics = model.val(data=str(yaml_path), conf=conf_th, iou=iou_th)
                    st.success("Evaluation completed.")
                    # summary extraction
                    summary = {}
                    box = getattr(metrics, "box", None)
                    if box is not None:
                        summary["mAP50"] = getattr(box, "map50", None)
                        summary["mAP50-95"] = getattr(box, "map", None)
                        summary["precision"] = getattr(box, "mp", None)
                        summary["recall"] = getattr(box, "mr", None)
                    st.write("Summary:", summary)

                    # per-class metrics
                    per_class_df = None
                    if hasattr(metrics, "box_results") and metrics.box_results is not None:
                        try:
                            per_class_df = pd.DataFrame(metrics.box_results)
                        except Exception:
                            per_class_df = None
                    elif hasattr(metrics, "results") and metrics.results is not None:
                        try:
                            per_class_df = pd.DataFrame(metrics.results)
                        except Exception:
                            per_class_df = None

                    if per_class_df is not None:
                        st.subheader("Per-class metrics")
                        st.dataframe(per_class_df)
                        csvp = EXPORT_DIR / f"eval_{int(time.time())}.csv"
                        per_class_df.to_csv(csvp, index=False)
                        st.success(f"Saved per-class CSV to {csvp}")

                    # confusion matrix
                    try:
                        if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
                            cm_fig = metrics.confusion_matrix.plot(normalize=True, show=False)
                            st.pyplot(cm_fig)
                    except Exception as e:
                        st.warning(f"Confusion matrix plotting not available: {e}")

                    # PR curves
                    try:
                        if hasattr(metrics, "plot_pr") and callable(metrics.plot_pr):
                            pr_fig = metrics.plot_pr()
                            st.pyplot(pr_fig)
                    except Exception:
                        pass

                    # create PDF report
                    try:
                        pdf_out = simple_pdf_report(summary, per_class_df, out_pdf=EXPORT_DIR / f"eval_{int(time.time())}.pdf")
                        st.success(f"Saved PDF report to {pdf_out}")
                    except Exception as e:
                        st.warning(f"Could not create PDF report: {e}")

                except Exception as e:
                    st.error(f"Evaluation error: {e}")
            else:
                st.error("ultralytics not installed or yaml missing")

    # export CSV/PDF buttons
    if export_csv_btn:
        csvs = sorted(EXPORT_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csvs:
            st.download_button("Download latest CSV", data=open(csvs[0], "rb"), file_name=csvs[0].name, key="test_download_csv")
        else:
            st.warning("No CSV found in exports")

    if export_pdf_btn:
        pdfs = sorted(EXPORT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pdfs:
            st.download_button("Download latest PDF", data=open(pdfs[0], "rb"), file_name=pdfs[0].name, key="test_download_pdf")
        else:
            st.warning("No PDF found in exports")

# -----------------------------
# TAB 4: DETECTION / INFERENCE
# -----------------------------
with tabs[3]:
    col_Settings, col_Info = st.columns([2,1])

    with col_Settings:
        st.subheader("Model")
        det_model_upload = st.file_uploader("Upload trained model (.pt)", type=['pt'], key="detect_model_upload")
        det_model_choice = st.selectbox("Or choose existing model", [p.name for p in MODELS_DIR.glob("*.pt")] if any(MODELS_DIR.glob("*.pt")) else ["--none--"], key="detect_model_choice")

        st.subheader("Settings")
        det_device = st.radio("Device", ["CUDA","CPU"] if (torch is not None and torch.cuda.is_available()) else ["CPU"], index=0, key="detect_device_radio")
        det_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, step=0.01, key="detect_conf_slider")
        det_iou = st.slider("IoU threshold", 0.0, 1.0, 0.45, step=0.01, key="detect_iou_slider")
        det_imgsz = st.slider("Inference image size", 224, 1280, 640, step=32, key="detect_imgsz_slider")
        det_input_type = st.radio("Input type", ["Image","Video","Webcam"], key="detect_input_radio")
        # file uploader per input mode
        if det_input_type in ("Image","Video"):
            det_input_file = st.file_uploader(f"Upload {det_input_type}", type=['jpg','png','jpeg','mp4','avi','mov'], key="detect_input_file")
        else:
            det_input_file = None
        start_detect = st.button("Start Detection", key="detect_start_btn")

    with col_Info:
        st.subheader("Model info")
        det_model_path = None
        if det_model_upload:
            det_model_path = MODELS_DIR / det_model_upload.name
            save_uploaded_file(det_model_upload, det_model_path)
        elif det_model_choice and det_model_choice != "--none--":
            det_model_path = MODELS_DIR / det_model_choice

        if det_model_path:
            st.write("Model file:", det_model_path)
            if YOLO is None:
                st.warning("ultralytics not installed — limited inspection")
            else:
                try:
                    det_model = YOLO(str(det_model_path))
                    names = getattr(det_model, "names", None)
                    st.write("Number of classes:", len(names) if names else "Unknown")
                    st.write("Classes sample:", list(names.items())[:10] if names else "Unknown")
                    if torch is not None and hasattr(det_model, "model"):
                        try:
                            total_params = sum(p.numel() for p in det_model.model.parameters())
                            st.write("Total parameters:", f"{total_params:,}")
                        except Exception:
                            pass
                except Exception as e:
                    st.warning(f"Could not load model for inspection: {e}")
        else:
            st.info("Select or upload a model to view info")

    if det_input_type == "Webcam":
        if cv2 is None:
            st.error("opencv-python not installed — webcam requires opencv")
        elif det_model_path is None:
            st.error("Select or upload a model first")
        elif YOLO is None:
            st.error("ultralytics not installed")
        else:
            # --- pakai tombol, bukan checkbox ---
            if "webcam_active" not in st.session_state:
                st.session_state.webcam_active = False

            col1, col2 = st.columns(2)
            if col1.button("▶️ Start Webcam", key="detect_webcam_start"):
                st.session_state.webcam_active = True
            if col2.button("⏹️ Stop Webcam", key="detect_webcam_stop"):
                st.session_state.webcam_active = False

            if st.session_state.webcam_active:
                try:
                    model = YOLO(str(det_model_path))
                    cap = cv2.VideoCapture(0)
                    stframe = st.empty()
                    fps_box = st.empty()
                    frame_count = 0
                    t0 = time.time()

                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("No frame from webcam. Exiting stream.")
                            break

                        preds = model.predict(
                            frame, conf=det_conf, iou=det_iou,
                            imgsz=det_imgsz, device=det_device, verbose=False
                        )
                        try:
                            rendered = preds[0].plot()
                            stframe.image(rendered, channels="BGR")
                        except Exception:
                            stframe.image(frame, channels="BGR")

                        frame_count += 1
                        fps = frame_count / (time.time() - t0 + 1e-6)
                        fps_box.metric("FPS", f"{fps:.2f}")

                        time.sleep(0.01)

                    cap.release()
                    st.info("Webcam stream stopped.")
                except Exception as e:
                    st.error(f"Webcam detection error: {e}")