# test.py

# =========================
# Imports
# =========================
import os
import time
import shutil
import zipfile
import tempfile
from pathlib import Path
import io

# Scientific & ML
import cv2
import numpy as np
import torch
import psutil
import pandas as pd
from PIL import Image

# Streamlit & YOLO
import streamlit as st
from ultralytics import YOLO

# ReportLab untuk export PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors

# FPDF (opsional, jika dipakai)
from fpdf import FPDF


# =========================
# Paths & Initial Setup
# =========================
WORKDIR = Path("workdir")

# Subdirectories
DATASETS_DIR = WORKDIR / "datasets"
YAML_DIR     = WORKDIR / "yaml"
CLASSES_DIR  = WORKDIR / "classes"
MODELS_DIR   = WORKDIR / "models"
RUNS_DIR     = WORKDIR / "runs"

# Pastikan semua folder tersedia
for directory in [DATASETS_DIR, YAML_DIR, CLASSES_DIR, MODELS_DIR, RUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Streamlit session state
st.session_state.setdefault("uploader_index", 0)


# =========================
# Helper Functions
# =========================
def list_images(folder: Path, exts=(".jpg", ".jpeg", ".png")) -> list[Path]:
    """List all image files in a folder (recursively)."""
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def save_uploaded_bytes(uploaded_file, save_path: Path) -> Path:
    """Save uploaded file bytes into a path."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path

def _move_extracted_to_root(extract_dir: Path) -> None:
    """
    Flatten extracted ZIP if wrapped inside a single root folder.
    Example:
    extract_dir/
        some_folder/  -> contents moved up into extract_dir
    """
    items = list(extract_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        wrapper = items[0]
        for child in wrapper.iterdir():
            dest = extract_dir / child.name
            if dest.exists():
                if child.is_dir():
                    for sub in child.iterdir():
                        target = dest / sub.name
                        if target.exists() and target.is_file():
                            try:
                                target.unlink()
                            except Exception:
                                pass
                        shutil.move(str(sub), str(dest))
                else:
                    shutil.move(str(child), str(dest))
            else:
                shutil.move(str(child), str(dest))
        shutil.rmtree(wrapper, ignore_errors=True)

def read_classes_file() -> list[str]:
    """Read first classes.txt in CLASSES_DIR, return list of class names."""
    candidates = list(CLASSES_DIR.glob("*.txt"))
    if not candidates:
        return []
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


# =========================
# Dataset Upload Tab
# =========================
def dataset_upload_tab():
    col_setup, col_info = st.columns([2, 1])

    with col_setup:
        # -----------------
        # Dataset upload
        # -----------------
        st.subheader("Dataset")
        st.write(
            "Unggah dataset dalam format `.ZIP` yang berisi folder `train`, `val`, dan `test`, "
            "pastikan masing-masing folder memiliki subfolder `images` dan `labels` di dalamnya."
            )

        uploader_key = f"dataset_zip_{st.session_state.uploader_index}"
        dataset_zip = st.file_uploader(
            "Unggah file `.ZIP` dataset (`train`/`val`/`test`)",
            type=["zip"],
            key=uploader_key,
        )

        # Fungsi upload dataset
        if dataset_zip is not None:
            try:
                if DATASETS_DIR.exists():
                    shutil.rmtree(DATASETS_DIR, ignore_errors=True)
                DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.error(f"Gagal membersihkan folder `datasets` lama: {e}")
                return

            zip_path = WORKDIR / "uploaded_dataset.zip"
            try:
                save_uploaded_bytes(dataset_zip, zip_path)
            except Exception as e:
                st.error(f"Gagal menyimpan `.ZIP`: {e}")
                return

            tmp_extract_dir = WORKDIR / f"_tmp_extract_{int(time.time())}"
            tmp_extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Extract ZIP
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.namelist():
                        if "__MACOSX" in member:
                            continue
                        zf.extract(member, tmp_extract_dir)

                # Flatten structure
                _move_extracted_to_root(tmp_extract_dir)

                # Move extracted content to datasets
                for item in tmp_extract_dir.iterdir():
                    dest = DATASETS_DIR / item.name
                    if dest.exists():
                        if item.is_dir():
                            for sub in item.iterdir():
                                target = dest / sub.name
                                if target.exists() and target.is_file():
                                    try:
                                        target.unlink()
                                    except Exception:
                                        pass
                                shutil.move(str(sub), str(dest))
                        else:
                            shutil.move(str(item), str(dest))
                    else:
                        shutil.move(str(item), str(dest))

                # Cleanup
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                try:
                    zip_path.unlink()
                except Exception:
                    pass

                st.success(f"Dataset berhasil diekstrak ke `{DATASETS_DIR}`")
            except Exception as e:
                st.error(f"Gagal mengekstrak `.ZIP`: {e}")
                shutil.rmtree(DATASETS_DIR, ignore_errors=True)

        st.markdown("---")

        # -----------------
        # Classes.txt
        # -----------------
        st.subheader("Classes.txt")
        uploaded_classes = st.file_uploader(
            "Unggah file `classes.txt` yang berisi `class` label dataset.",
            type=["txt"],
            key=f"classes_upload_{st.session_state.uploader_index}",
        )

        # Fungsi upload classes.txt atau manual
        if uploaded_classes is not None:
            try:
                classes_file = CLASSES_DIR / "classes.txt"
                if classes_file.exists():
                    classes_file.unlink()
                save_uploaded_bytes(uploaded_classes, classes_file)
                st.success("`Classes.txt` berhasil diunggah ke folder `workdir/classes`.")

                names = read_classes_file()
                if names:
                    st.info(f"Detected {len(names)} `classes`: {names}")
            except Exception as e:
                st.error(f"Gagal upload `classes.txt`: {e}")
        else:
            st.write("Atau buat `class` secara manual pada kolom di bawah ini:")

            if "num_classes" not in st.session_state:
                st.session_state.num_classes = 1

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("➕ Add Column"):
                    st.session_state.num_classes += 1
            with col_b:
                if st.button("➖ Subtract Column") and st.session_state.num_classes > 1:
                    st.session_state.num_classes -= 1

            class_inputs = []
            for i in range(st.session_state.num_classes):
                v = st.text_input(f"Class name {i+1}", key=f"class_name_{i}")
                class_inputs.append(v.strip())

            if st.button("Generate `classes.txt`"):
                names = [n for n in class_inputs if n]
                if not names:
                    st.error("Isi minimal satu nama kelas.")
                else:
                    CLASSES_DIR.mkdir(parents=True, exist_ok=True)
                    with open(CLASSES_DIR / "classes.txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(names))
                    st.success(f"`Classes.txt` dibuat ({len(names)} kelas).")
                    st.code("\n".join(names), language="text")

        st.markdown("---")

        # -----------------
        # Data.YAML
        # -----------------
        st.subheader("Data.YAML")
        yaml_path = YAML_DIR / "data.yaml"
        current_names = read_classes_file()

        # Fungsi generate data.YAML
        if current_names:
            st.write("Detected `classes`:", current_names)
        else:
            st.write("No `classes.txt` detected yet.")

        if yaml_path.exists():
            st.info(f"Found existing `data.yaml`: {yaml_path}")
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception:
                pass
        else:
            st.code(
                "train: <path ke train/images>\n"
                "val: <path ke val/images>\n"
                "test: <path ke test/images>\n\n"
                "nc: <jumlah kelas>\n"
                "names: [<daftar kelas>]",
                language="yaml",
            )

        # Fungsi tombol generate data.YAML
        if st.button("Generate `data.YAML`"):
            names = read_classes_file()
            if not DATASETS_DIR.exists() or not any(DATASETS_DIR.iterdir()):
                st.error("Dataset belum ada di `workdir/datasets`. Upload `.ZIP` terlebih dahulu.")
            elif not names:
                st.error("`Classes.txt` belum ada. Upload atau buat `classes.txt` terlebih dahulu.")
            else:
                train_dir = str((DATASETS_DIR / "train" / "images").resolve())
                val_dir = str((DATASETS_DIR / "val" / "images").resolve())
                test_dir = str((DATASETS_DIR / "test" / "images").resolve())

                yaml_content = (
                    f"train: {train_dir}\n"
                    f"val: {val_dir}\n"
                    f"test: {test_dir}\n\n"
                    f"nc: {len(names)}\n"
                    f"names: {names}\n"
                )

                YAML_DIR.mkdir(parents=True, exist_ok=True)
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(yaml_content)

                st.success(f"`Data.yaml` berhasil dibuat di `{yaml_path}`")
                st.code(yaml_content, language="yaml")

    with col_info:
        # -----------------
        # Dataset info
        # -----------------
        # Fungsi preview informasi dataset
        st.subheader("Dataset Info")
        if DATASETS_DIR.exists() and any(DATASETS_DIR.iterdir()):
            rows = []
            for split in ["Train", "Val", "Test"]:
                img_dir = DATASETS_DIR / split / "images"
                lbl_dir = DATASETS_DIR / split / "labels"

                imgs = list_images(img_dir)
                lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

                rows.append(("Split", split))
                rows.append(("`Images`", len(imgs)))
                rows.append(("`Labels`", len(lbls)))

            df_info = pd.DataFrame(rows, columns=["Keterangan", "Jumlah"])
            st.table(df_info)
        else:
            st.info("Belum ada `dataset` terunggah ke `workdir/datasets`.")

        # Fungsi tombol delete
        if st.button("Delete `workdir`"):
            try:
                if WORKDIR.exists():
                    shutil.rmtree(WORKDIR, ignore_errors=True)
                for d in [DATASETS_DIR, YAML_DIR, CLASSES_DIR, MODELS_DIR, RUNS_DIR]:
                    d.mkdir(parents=True, exist_ok=True)
                st.session_state.uploader_index += 1
                st.success("`workdir` berhasil dihapus.")
                st.rerun()
            except Exception as e:
                st.error(f"Gagal reset `workdir`: {e}")

# =========================
# Training Tab
# =========================
def training_tab():
    
    # -----------------
    # Pre-processing
    # -----------------
    st.subheader("Pre-processing")
    apply_aug    = st.checkbox("Augmentation")
    apply_resize = st.checkbox("Resize")
    apply_norm   = st.checkbox("Normalize")

    # Fungsi augmentation
    if apply_aug:
        st.markdown("---")
        st.markdown("**Augmentation parameters**")
        fliplr = st.slider("`Flip Probability`", 0.0, 1.0, 0.5, 0.05)
        rotate = st.slider("`Rotation (deg)`", 0, 90, 0, 1)
        brightness = st.slider("`Brightness factor`", 0.0, 2.0, 1.0, 0.05)
    else:
        fliplr, rotate, brightness = 0.0, 0, 0.0

    # Fungsi resize
    if apply_resize:
        st.markdown("---")
        st.markdown("**Resize parameter**")
        imgsz = st.slider("`Image Size (px)`", 32, 2048, 640, step=32)
    else:
        imgsz = 640

    # Fungsi normalize
    if apply_norm:
        st.markdown("---")
        st.markdown("**Normalization parameter**")
        norm_min = st.slider("`Normalize Min`", 0.0, 1.0, 0.0, 0.01)
        norm_max = st.slider("`Normalize Max`", 0.0, 1.0, 1.0, 0.01)
    else:
        norm_min, norm_max = 0.0, 1.0

    # Fungsi tombol apply
    if st.button("Apply `pre-processing`"):
        chosen = []
        if apply_aug: chosen.append("Augmentation")
        if apply_resize: chosen.append("Resize")
        if apply_norm: chosen.append("Normalize")
        st.success(f"Parameters applied: `{', '.join(chosen)}`")

    # -----------------
    # Hyperparameter tuning
    # -----------------
    st.markdown("---")
    st.subheader("Hyperparameter Tuning")
    
    # Index parameter
    lr = st.slider("`Learning Rate`", 1e-5, 1e-1, 1e-3, 1e-5)
    momentum = st.slider("`Momentum`", 0.0, 1.0, 0.937, 0.001)
    weight_decay = st.slider("`Weight Decay`", 0.0, 0.01, 0.0005, 0.0001)
    warmup_epochs = st.slider("`Warmup Epochs`", 0, 10, 3, 1)
    dropout = st.slider("`Dropout`", 0.0, 0.9, 0.0, 0.05)

    # Fungsi tombol apply
    if st.button("Apply `hyperparameter`"):
        st.success(
            f"Hyperparameter applied: `lr={lr}, momentum={momentum}, "
            f"wd={weight_decay}, warmup={warmup_epochs}, dropout={dropout}`"
        )

    # -----------------
    # Training
    # -----------------
    st.markdown("---")
    st.subheader("Training")
    
    # Fungsi preview data.YAML
    yaml_path = YAML_DIR / "data.yaml"
    if yaml_path.exists():
        st.code(open(yaml_path, "r", encoding="utf-8").read(), language="yaml")
    else:
        st.warning("`Data.YAML` belum tersedia. Buat di `Tab Dataset` terlebih dahulu.")

    # Index parameter otomatis
    if torch.cuda.is_available():
        default_device = "CUDA"
        default_batch = 32
        default_epochs = 100
        default_imgsz = 640
    else:
        default_device = "CPU"
        default_batch = 8
        default_epochs = 50
        default_imgsz = 320

    # Index parameter model YOLO
    yolo_version = st.radio("`YOLO version`", ["YOLOv5", "YOLOv8", "YOLO11"], index=1)

    # Index parameter size YOLO
    model_size_label = st.selectbox(
        "`YOLO model size`",
        ["Nano", "Small", "Medium", "Large", "Extra Large"],
        index=1
    )
    size_map = {"Nano": "n", "Small": "s", "Medium": "m", "Large": "l", "Extra Large": "x"}
    model_size = size_map[model_size_label]

    # Index parameter device
    device = st.radio("`Device`", ["CPU", "CUDA"], index=0 if default_device == "CPU" else 1)

    # Index parameter batch, epoch, dan imgsz
    batch  = st.selectbox("`Batch size`", [4, 8, 16, 32, 64, 128], index=[4, 8, 16, 32, 64, 128].index(default_batch))
    epochs = st.selectbox("`Epochs`", [10, 50, 100, 200, 300, 500, 1000, 1500, 2000], index=[10, 50, 100, 200, 300, 500, 1000, 1500, 2000].index(default_epochs))
    imgsz  = st.slider   ("`Image size`", 32, 2048, default_imgsz, step=32)

    # Fungsi set nama model
    model_name = f"{yolo_version.lower()}{model_size}.pt"

    # Fungsi tombol start
    if st.button("Start `training`"):
        if YOLO is None:
            st.error("`Ultralytics YOLO` belum terinstall.")
        elif not yaml_path.exists():
            st.error("`Data.YAML` tidak ditemukan.")
        else:
            try:
                st.info(f"Loading `model {model_name}` ...")
                model = YOLO(model_name)

                st.info("Memulai `training` ...")
                results = model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    device=0 if device == "CUDA" else "CPU",
                    project=RUNS_DIR,
                    name="train",  # otomatis: train, train2, dst
                )
                st.success("`Training` selesai!")

                # Ambil folder hasil training dari results
                save_dir = Path(results.save_dir)   # contoh: runs/train2
                best_model = save_dir / "weights/best.pt"

                if best_model.exists():
                    exp_name = save_dir.name  # contoh: train, train2

                    # --- Simpan hanya ke workdir/models ---
                    workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
                    try:
                        workdir_models.mkdir(parents=True, exist_ok=True)
                        target_model = workdir_models / f"{yolo_version.lower()}{model_size}_{exp_name}_best.pt"
                        shutil.copy2(best_model, target_model)
                        st.success(f"`Best.pt` model disalin ke `{target_model}`")
                    except Exception as e:
                        st.error(f"Gagal copy ke `workdir/models`: {e}")
                else:
                    st.error("`best.pt` tidak ditemukan di `folder weights`.")

            except Exception as e:
                st.error(f"`Training` error: {e}")

    # -----------------
    # Deployment
    # -----------------
    st.markdown("---")
    st.subheader("Deployment")

    # Index model export
    export_options = ["ONNX", "TorchScript"]
    selected_export = st.multiselect("Pilih format `export`", export_options)

    # Fungsi ambil model dari workdir/models
    workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
    models_available = list(workdir_models.glob("*.pt"))

    if models_available:
        # Fungsi otomatis pilih model terbaru
        models_available = sorted(models_available, key=os.path.getmtime, reverse=True)
        selected_model = st.selectbox("Pilih Model", models_available, index=0)
    else:
        st.warning("Belum ada model di `workdir/models`.")
        selected_model = None

    # Fungsi tombol export
    if st.button("Export `Model`"):
        if selected_model is None:
            st.error("Tidak ada `model` dipilih.")
        elif YOLO is None:
            st.error("`Ultralytics YOLO` belum terinstall.")
        else:
            try:
                st.info("`Export` sedang berjalan ...") 
                model = YOLO(str(selected_model))
                if "ONNX" in selected_export:
                    model.export(format="onnx")
                if "TorchScript" in selected_export:
                    model.export(format="torchscript")
                st.success("`Export` selesai!")
            except Exception as e:
                st.error(f"`Export` error: {e}")

# =========================
# Test Tab
# =========================
def test_tab():
    
    # Fungsi cek YOLO
    @st.cache_resource
    def load_model(path):
        try:
            return YOLO(str(path))
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return None
    
    #======================
    # SET-UP
    #======================
    col_test, col_testinfo = st.columns([2,1])

    with col_test:
        # -------------------------------
        # Check model
        # -------------------------------
        st.subheader("Check model")
        uploaded_model = st.file_uploader("Upload model `(.pt)`", type=["pt"])
        st.write("Atau gunakan hasil `model train` terakhir:")

        # Fungsi upload
        if uploaded_model:
            # Validasi ukuran file
            if uploaded_model.size > 500_000_000:  # 500MB limit
                st.warning("File terlalu besar, mungkin akan lambat saat load.")
            
            temp_model_path = Path(tempfile.gettempdir()) / uploaded_model.name
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            latest_model_path = temp_model_path
            st.success(f"Model upload digunakan: `{uploaded_model.name}`")
        else:
            workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
            models_available = sorted(workdir_models.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not models_available:
                st.warning("Belum ada model di `workdir/models` dan tidak ada `model upload`.")
                st.stop()
            latest_model_path = models_available[0]
            st.info(f"`Model train` terakhir digunakan: `{latest_model_path.name}`")

        # Load model dengan caching
        model = load_model(latest_model_path)
        if model is None:
            st.stop()

    with col_testinfo:
        # -------------------------------
        # Check model info
        # -------------------------------
        st.write("`Class` yang ada di `model`:")
        classes = list(model.names.values()) if isinstance(model.names, dict) else model.names
        st.table(classes)

    st.markdown("---")

    #======================
    # TEST 1
    #======================

    # Fungsi cek YOLO
    @st.cache_resource
    def load_model(path):
        try:
            return YOLO(str(path))
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return None
    
    # Fungsi state
    if "eval_results" not in st.session_state:
        st.session_state["eval_results"] = []

    if "eval_preview" not in st.session_state:
        st.session_state["eval_preview"] = None

    col_eval, col_evalinfo = st.columns([2,1])

    with col_eval:
        # ===============================
        # 1.1 Evaluation
        # ===============================
        st.subheader("Evaluation")
        uploaded_test = st.file_uploader(
            "Upload dataset config `data.yaml`", 
            type=["yaml"], key="eval_data"
        )
        st.write("Atau gunakan `data.yaml` terakhir yang ada di `workdir/yaml/`.")

        # Fungsi upload / gunakan yaml terakhir
        if uploaded_test:
            temp_yaml = Path(tempfile.mkdtemp()) / uploaded_test.name
            with open(temp_yaml, "wb") as f:
                f.write(uploaded_test.getbuffer())
            eval_data = str(temp_yaml)
            st.session_state["eval_preview"] = eval_data  # simpan ke session state
            st.success("Dataset `config` digunakan.")
        else:
            yaml_dir = Path("/Users/aditya/Downloads/riset/workdir/yaml")
            yaml_files = sorted(yaml_dir.glob("*.yaml"), key=lambda x: x.stat().st_mtime, reverse=True)
            if yaml_files:
                eval_data = str(yaml_files[0])
                st.session_state["eval_preview"] = eval_data
                st.info(f"Menggunakan `config dataset` terakhir: `{yaml_files[0].name}`")
            else:
                eval_data = None
                st.session_state["eval_preview"] = None
                st.warning("Tidak ada `data.yaml` ditemukan di `workdir/yaml/`.")

        run_eval = st.button("Start `evaluation`")

    with col_evalinfo:
        # ===============================
        # 1.2 Evaluation info
        # ===============================
        st.subheader("Evaluation Info")
        if run_eval and eval_data:
            with st.info("`Evaluation` sedang berjalan ..."):
                try:
                    model = YOLO("yolov8n.pt")  # ganti model sesuai kebutuhan
                    metrics = model.val(data=eval_data, split="test")
                    f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
                    eval_result = {
                        "`Timestamp`": str(pd.Timestamp.now()),
                        "`Test Type`": "Evaluation",
                        "`mAP50`": metrics.box.map50,
                        "`mAP50-95`": metrics.box.map,
                        "`Precision`": metrics.box.mp,
                        "`Recall`": metrics.box.mr,
                        "`F1`": f1,
                    }

                    st.session_state["eval_results"].append(eval_result)
                    st.success("`Evaluation` selesai!")

                except Exception as e:
                    st.error(f"Gagal menjalankan evaluation: {e}")

        # Fungsi tampilkan hasil terakhir
        if st.session_state.get("eval_results"):
            st.markdown("Hasil Evaluasi:")
            st.table(st.session_state["eval_results"][-1])
    
    st.markdown("---")

    #======================
    # TEST 2
    #======================

    # Fungsi cek YOLO
    @st.cache_resource
    def load_yolo_model(path):
        try:
            return YOLO(str(path))
        except Exception as e:
            st.error(f"Gagal load model benchmark: {e}")
            return None
    
    # Fungsi state
    if "bench_image" not in st.session_state:
        st.session_state["bench_image"] = None  # untuk gambar benchmark

    if "bench_results" not in st.session_state:
        st.session_state["bench_results"] = []  # untuk tabel benchmark

    col_bench, col_benchinfo = st.columns([2,1])

    with col_bench:
        # ===============================
        # 2.1 Benchmark
        # ===============================
        st.subheader("Benchmark")
        uploaded_model = st.file_uploader("Upload model `.pt` untuk benchmark", type=["pt"], key="bench_model")
        if uploaded_model:
            temp_model_path = Path(tempfile.gettempdir()) / uploaded_model.name
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            bench_model_path = temp_model_path
            st.success(f"Model upload digunakan: `{uploaded_model.name}`")
        else:
            workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
            models_available = sorted(workdir_models.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not models_available:
                st.warning("Belum ada model di `workdir/models` dan tidak ada model upload.")
                st.stop()
            bench_model_path = models_available[0]
            st.info(f"Model train terakhir digunakan: `{bench_model_path.name}`")

        bench_model = load_yolo_model(bench_model_path)
        if bench_model is None:
            st.stop()

        uploaded_input = st.file_uploader("Upload sample `image/video` untuk `benchmark`", type=["jpg","png","mp4"], key="bench_input")

        conf = st.slider("`Confidence Threshold`", 0.0, 1.0, 0.25, 0.05)
        iou = st.slider("`IoU Threshold`", 0.0, 1.0, 0.45, 0.05)
        max_det = st.slider("`Max Detections`", min_value=1, max_value=1000, value=100)

        run_bench = st.button("Jalankan `benchmark`")

    with col_benchinfo:
        # ===============================
        # 2.2 Benchmark info
        # ===============================
        st.subheader("Benchmark Info")
        if run_bench:
            if uploaded_input is None:
                st.warning("Silakan upload file gambar/video sebelum menjalankan benchmark!")
            else:
                with st.spinner("Benchmark sedang berjalan ..."):
                    temp_path = Path(tempfile.mkdtemp()) / uploaded_input.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_input.getbuffer())
                    bench_input = str(temp_path)
                    is_image = uploaded_input.type.startswith("image")

                    # Run inference 5x untuk rata-rata latency
                    t0 = time.time()
                    results = None
                    for _ in range(5):
                        results = bench_model.predict(
                            bench_input,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            verbose=False
                        )
                    t1 = time.time()
                    latency = (t1 - t0) / 5
                    fps = 1 / latency

                    # Resource usage
                    cpu_usage = psutil.cpu_percent(interval=1)
                    process = psutil.Process(os.getpid())
                    ram_usage = process.memory_info().rss / (1024*1024)

                    # Append hasil tabel ke session state
                    bench_result = {
                        "`Timestamp`": str(pd.Timestamp.now()),
                        "`Test Type`": "Benchmark",
                        "`Latency (s)`": latency,
                        "`FPS`": fps,
                        "`Model Size (MB)`": bench_model_path.stat().st_size/(1024*1024),
                        "`CPU Usage (%)`": cpu_usage,
                        "`RAM Usage (MB)`": ram_usage,
                        "`GPU Available`": torch.cuda.is_available(),
                    }
                    st.session_state["bench_results"].append(bench_result)

                    # Simpan hasil gambar ke session state
                    if is_image and results is not None:
                        st.session_state["bench_image"] = results[0].plot()

                    st.success("`Benchmark` selesai!")

        # Fungsi tampilkan gambar
        if st.session_state["bench_image"] is not None:
            st.image(st.session_state["bench_image"], caption="Hasil Benchmark (prediksi)", use_container_width=True)

        # Fungsi tampilkan tabel
        if st.session_state["bench_results"]:
            st.markdown("Hasil Benchmark:")
            st.table(st.session_state["bench_results"][-1])

    st.markdown("---")

    #======================
    # TEST 3
    #======================

    # Fungsi state
    if "func_results" not in st.session_state:
        st.session_state["func_results"] = []

    if "func_image" not in st.session_state:
        st.session_state["func_image"] = None

    if "func_info_current" not in st.session_state:
        st.session_state["func_info_current"] = {}

    col_func, col_funcinfo = st.columns([2,1])

    with col_func:
        # ===============================
        # 3.1 Functional test
        # ===============================
        st.subheader("Functional Test")
        uploaded_model = st.file_uploader("Upload model `.pt`", type=["pt"], key="func_model")
        if uploaded_model:
            temp_model_path = Path(tempfile.gettempdir()) / uploaded_model.name
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = temp_model_path
        else:
            workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
            models_available = sorted(workdir_models.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not models_available:
                st.warning("Belum ada model di `workdir/models`.")
                st.stop()
            model_path = models_available[0]

        model = YOLO(str(model_path))

        # Reset hasil jika input mode berubah
        selected_mode = st.radio("Pilih input", ["`Image`","`Video`","`Webcam`"], key="func_input_mode")
        # Reset hasil hanya untuk info current & image preview
        if selected_mode != st.session_state.get("last_func_mode"):
            st.session_state["func_image"] = None
            st.session_state["func_info_current"] = {}
            st.session_state["last_func_mode"] = selected_mode

        # Fungsi tombol reset manual
        if st.button("Reset `functional history`"):
            st.session_state["func_results"] = []
            st.success("`Functional test history` berhasil direset.")

        # Index parameter
        test_mode = selected_mode
        conf_func = st.slider("`Confidence Threshold`",0.0,1.0,0.25,0.05,key="func_conf")
        iou_func = st.slider("`IoU Threshold`",0.0,1.0,0.45,0.05,key="func_iou")
        max_det_func = st.slider("`Max Detections`",1,1000,100,key="func_max_det")

        test_image, test_video = None, None
        run_test = False
        start_webcam = False
        stop_webcam = False

        # Fungsi image, video, dan webcam
        if test_mode=="`Image`":
            test_image = st.file_uploader("Upload gambar", type=["jpg","png"], key="func_input_image")
            run_test = st.button("Start `test`", key="func_start")
        elif test_mode=="`Video`":
            test_video = st.file_uploader("Upload video", type=["mp4","avi"], key="func_input_video")
            run_test = st.button("Start `test`", key="func_start")
        elif test_mode=="`Webcam`":
            cam_ports=[0,1,2,3,4]
            webcam_port=st.selectbox("Pilih camera port", cam_ports, key="func_cam_port")
            if "webcam_running" not in st.session_state:
                st.session_state["webcam_running"]=False
            start_webcam=st.button("Start `webcam`", key="func_start_webcam")
            stop_webcam=st.button("Stop `webcam`", key="func_stop_webcam")

    with col_funcinfo:
        # ===============================
        # 3.2 Functional test info
        # ===============================
        st.subheader("Functional Test Info")

        # Fungsi image dan video
        if test_mode in ["`Image`","`Video`"] and run_test:
            if test_mode == "`Image`" and test_image is not None:
                file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                results = model.predict(img, conf=conf_func, iou=iou_func, max_det=max_det_func, verbose=False)
                img_plot = results[0].plot()
                img_rgb = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)

                # tampilkan hasil deteksi
                st.image(img_rgb, channels="RGB", caption="Prediction Result")

                # info table
                info_dict = {
                    "`Test Type`": "Functional Test",
                    "`Input Type`": "Image",
                    "`Filename`": test_image.name,
                    "`Width`": img.shape[1],
                    "`Height`": img.shape[0],
                    "`Num Detections`": len(results[0].boxes),
                    "`Detected Classes`": [model.names[int(box.cls)] for box in results[0].boxes],
                    "`Confidence Threshold`": conf_func,
                    "`IoU Threshold`": iou_func,
                    "`Max Detections`": max_det_func
                }
                df_vert = pd.DataFrame(list(info_dict.items()), columns=["Parameter", "Value"])
                st.table(df_vert)

                # simpan ke session & export
                st.session_state["func_image"] = img_rgb
                st.session_state["func_results"].append(info_dict)
                st.success("Functional Test selesai!")

            elif test_mode == "`Video`" and test_video is not None:
                st.video(test_video)
                info_dict = {
                    "`Test Type`": "Functional Test",
                    "`Input Type`": "Video",
                    "`Filename`": test_video.name,
                    "`Note`": "Frame-by-frame analysis belum diimplementasikan",
                    "`Confidence Threshold`": conf_func,
                    "`IoU Threshold`": iou_func,
                    "`Max Detections`": max_det_func
                }
                df_vert = pd.DataFrame(list(info_dict.items()), columns=["Parameter", "Value"])
                st.table(df_vert)

                st.session_state["func_results"].append(info_dict)
                st.success("`Functional Test` selesai!")

        # Fungsi webcam
        if test_mode == "`Webcam`":
            webcam_placeholder = st.empty()
            table_placeholder = st.empty()
            last_save_time = time.time()

            if start_webcam:
                st.session_state["webcam_running"] = True
                cap = cv2.VideoCapture(webcam_port)

                while st.session_state["webcam_running"]:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    start_time = time.time()
                    results = model.predict(frame, conf=conf_func, iou=iou_func, max_det=max_det_func, verbose=False)
                    latency = time.time() - start_time
                    fps = 1 / latency if latency > 0 else 0

                    img_plot = results[0].plot()
                    img_rgb = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)

                    # Simpan frame terakhir
                    st.session_state["func_image"] = img_rgb

                    # Update info current
                    st.session_state["func_info_current"] = {
                        "`Test Type`": "Functional Test",
                        "`Input Type`": "Webcam",
                        "`Resolution`": f"{frame.shape[1]}x{frame.shape[0]}",
                        "`Num Detections`": len(results[0].boxes),
                        "`Detected Classes`": [model.names[int(box.cls)] for box in results[0].boxes],
                        "`Latency (s)`": round(latency,3),
                        "`FPS`": round(fps,2),
                        "`Camera Port`": webcam_port,
                        "`Confidence Threshold`": conf_func,
                        "`IoU Threshold`": iou_func,
                        "`Max Detections`": max_det_func
                    }

                    # Tampilkan webcam + tabel realtime
                    webcam_placeholder.image(img_rgb, channels="RGB", caption="Webcam Prediction")
                    df_vert = pd.DataFrame(list(st.session_state["func_info_current"].items()), columns=["Parameter","Value"])
                    table_placeholder.table(df_vert)

                    # Simpan ke export tiap 1 detik
                    if time.time() - last_save_time >= 1:
                        st.session_state["func_results"].append(st.session_state["func_info_current"].copy())
                        last_save_time = time.time()

                    time.sleep(0.03)

                cap.release()

            if stop_webcam:
                st.session_state["webcam_running"] = False
                if st.session_state.get("func_info_current"):
                    # Simpan sekali lagi saat stop
                    st.session_state["func_results"].append(st.session_state["func_info_current"].copy())

                    # Tampilkan gambar terakhir + tabel terakhir
                    st.image(st.session_state["func_image"], channels="RGB", caption="Final Webcam Frame")
                    df_vert = pd.DataFrame(list(st.session_state["func_info_current"].items()), columns=["Parameter","Value"])
                    st.table(df_vert)

                st.success("Webcam test dihentikan & hasil terakhir disimpan.")

    st.markdown("---")

    #======================
    # TEST 4
    #======================

    # Fungsi state
    if "robust_results" not in st.session_state:
        st.session_state["robust_results"] = []

    if "robust_images" not in st.session_state:
        st.session_state["robust_images"] = {}
    
    col_robust, col_robustinfo = st.columns([2,1])

    with col_robust:
        # ===============================
        # 4.1 Robustness test
        # ===============================
        st.subheader("Robustness Test")
        uploaded_model = st.file_uploader("Upload model `.pt`", type=["pt"], key="robust_model")
        st.write("Atau gunakan hasil `model train` terakhir:")

        if uploaded_model:
            temp_model_path = Path(tempfile.gettempdir()) / uploaded_model.name
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = temp_model_path
            st.success(f"Model upload digunakan: `{uploaded_model.name}`")
        else:
            workdir_models = Path("/Users/aditya/Downloads/riset/workdir/models")
            models_available = sorted(workdir_models.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not models_available:
                st.warning("Belum ada model di `workdir/models`.")
                st.stop()
            model_path = models_available[0]
            st.info(f"Model terakhir digunakan: `{model_path.name}`")

        model = YOLO(str(model_path))

        input_type = st.radio("Pilih input", ["`Image`", "`Video`"], key="robust_input_mode")
        uploaded_input = None
        if input_type == "`Image`":
            uploaded_input = st.file_uploader("Upload gambar", type=["jpg","png"], key="robust_input")
        elif input_type == "`Video`":
            uploaded_input = st.file_uploader("Upload video", type=["mp4","avi"], key="robust_input_video")

        perturbations = st.multiselect(
            "Pilih jenis gangguan",
            ["Gaussian Noise", "Blur", "Rotate", "Brightness"]
        )
        start_test = st.button("Start Robustness Test")

    with col_robustinfo:
        # ===============================
        # 4.2 Robustness test info
        # ===============================
        st.subheader("Robustness Info")
        if start_test:
            if uploaded_input is None:
                st.warning("Silakan upload file sebelum menjalankan Robustness Test.")
            elif not perturbations:
                st.warning("Silakan pilih minimal satu jenis gangguan.")
            else:
                if input_type != "`Image`":
                    st.warning("Robustness Test untuk Video belum diimplementasikan.")
                    st.stop()

                img_bytes_original = uploaded_input.getvalue()
                tabs = st.tabs(perturbations)

                for i, perturbation in enumerate(perturbations):
                    with tabs[i]:
                        # Decode ulang setiap tab
                        img_bytes = np.frombuffer(img_bytes_original, dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

                        # Apply perturbation
                        if perturbation == "Gaussian Noise":
                            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                            img = cv2.add(img, noise)
                        elif perturbation == "Blur":
                            img = cv2.GaussianBlur(img, (7,7), 0)
                        elif perturbation == "Rotate":
                            M = cv2.getRotationMatrix2D((img.shape[1]//2,img.shape[0]//2), 15, 1)
                            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                        elif perturbation == "Brightness":
                            img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)

                        # Run prediction
                        t0 = time.time()
                        results = model.predict(img, verbose=False)
                        t1 = time.time()
                        latency = t1 - t0
                        fps = 1 / latency

                        img_plot = results[0].plot()
                        st.session_state["robust_images"][perturbation] = img_plot

                        info_dict = {
                            "`Test Type`": f"Robustness - {perturbation}",
                            "`Input Type`": "Image",
                            "`Filename`": uploaded_input.name,
                            "`Perturbation`": perturbation,
                            "`Width`": img.shape[1],
                            "`Height`": img.shape[0],
                            "`Num Detections`": len(results[0].boxes),
                            "`Detected Classes`": [model.names[int(box.cls)] for box in results[0].boxes] if len(results[0].boxes) > 0 else [],
                            "`Latency (s)`": round(latency, 3),
                            "`FPS`": round(fps, 2)
                        }
                        st.session_state["robust_results"].append(info_dict)

                        # Gambar di atas
                        st.image(img_plot, caption=f"{perturbation} Prediction", use_container_width=True)

                        # Tabel vertikal di bawah
                        df_vert = pd.DataFrame(list(info_dict.items()), columns=["Parameter", "Value"])
                        st.table(df_vert)

                st.success("`Robustness Test` selesai!")
    
    st.markdown("---")

    #======================
    # EXPORT
    #======================

    # Fungsi export csv
    def export_csv(data, filename):
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode("utf-8"), filename

    # Fungsi export pdf
    def export_pdf(data, filename, title):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph(title, styles["Heading1"]))
        elements.append(Spacer(1, 12))

        # Convert dict/list to DataFrame
        df = pd.DataFrame(data)

        # Konversi semua cell jadi Paragraph supaya bisa dirender
        def make_paragraph(x):
            return Paragraph(str(x), styles["Normal"])

        table_data = [[make_paragraph(col) for col in df.columns]] + [
            [make_paragraph(cell) for cell in row] for row in df.values.tolist()
        ]

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ]))
        elements.append(table)

        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf, filename

    # Fungsi tampilan hasil
    test_sections = [
        ("Evaluation", "eval_results"),
        ("Benchmark", "bench_results"),
        ("Functional", "func_results"),
        ("Robustness", "robust_results"),
    ]

    for label, key in test_sections:
        if st.session_state.get(key):
            st.markdown(f"### {label} Results")
            df = pd.DataFrame(st.session_state[key])
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                csv_data, filename = export_csv(st.session_state[key], f"{label.lower()}_results.csv")
                st.download_button(
                    label=f"Export {label} CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )

            with col2:
                pdf_data, filename = export_pdf(st.session_state[key], f"{label.lower()}_results.pdf", f"{label} Results")
                st.download_button(
                    label=f"Export {label} PDF",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf"
                )
    
# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="iTraining", layout="wide")
    st.title("iTraining")

    tabs = st.tabs(["Dataset", "Training", "Benchmark"])
    with tabs[0]:
        dataset_upload_tab()
    with tabs[1]:
        training_tab()
    with tabs[2]:
        test_tab()

if __name__ == "__main__":
    main()