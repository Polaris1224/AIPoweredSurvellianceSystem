# 🛰️ AI‑Powered Surveillance — Streamlit App

**Real‑time person detection, tracking, and short‑term behaviour anomaly classification** using **YOLO (Ultralytics)**, **DeepSORT**, and a lightweight **MobileNetV2 + LSTM** temporal model.

---

## 🚀 Quick overview

* Detect people in video frames with YOLO and track IDs with DeepSORT.
* Extract per‑track crops and run a short‑term temporal classifier (MobileNetV2 features → LSTM) to flag anomalous behaviour.
* Streamlit UI for easy upload, preview, live labeling, alert log, and optional red‑frame flash on anomalies.
* Designed for **memory‑safe training** with `tf.data` sliding windows and runs on **CPU** (GPU optional).

---

## 📁 Project structure

```
ai-surveillance/
├─ app.py                    # Streamlit UI + inference loop
├─ behavior_model.py         # Model architecture (MobileNetV2 + LSTM) wrapper
├─ detection.py              # YOLO + DeepSORT glue code
├─ train_dataset.py          # Training pipeline / tf.data sliding windows
├─ utils.py                  # I/O, video helpers, crop & tracking utilities
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ data/
   ├─ training_videos/       # input videos (you add)
   ├─ training_vol/          # paired .mat files (you add)
   └─ models/
      ├─ behavior_model.weights.h5
      └─ behavior_model_config.json
```

---

## ✨ Features

* ✅ Person detection with Ultralytics YOLO
* ✅ Multi‑object tracking with DeepSORT (per‑ID track history)
* ✅ Temporal behaviour classifier (per‑track short clips) using MobileNetV2 → LSTM
* ✅ Streamlit UI: upload, preview, per‑ID labels, alert log
* ✅ Optional red‑frame flash when anomaly detected
* ✅ Memory‑safe dataset creation via `tf.data` sliding windows
* ✅ Cross‑platform: Windows / macOS / Linux (Python 3.10)
* ✅ Docker (CPU) friendly; optional GPU path for acceleration

---

## 🛠️ Requirements

* Python **3.10.x** (64‑bit)
* Packages: see `requirements.txt`
* Optional: Docker Desktop (to run containerized CPU image)

> If you plan to use GPU later, install TensorFlow with GPU support and the appropriate CUDA/cuDNN matching TF 2.15.

---

## 🧩 Installation (local, without Docker)

1. Create & activate a virtual environment

**Windows (PowerShell)**

```powershell
py -3.10 -m venv env
.\env\Scripts\Activate
```

**macOS / Linux**

```bash
python3.10 -m venv env
source env/bin/activate
```

2. Update pip and install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "ultralytics>=8.0,<9"
```

3. Prepare data folders (if not present)

```bash
mkdir -p data/training_videos data/training_vol data/models
```

---

## ⚙️ Training the behaviour model (optional)

If you already have `data/models/behavior_model*.{h5,json}` you can skip training.

```bash
python train_dataset.py
```

This creates:

* `data/models/behavior_model.weights.h5`
* `data/models/behavior_model_config.json`

**Notes**

* Training uses `tf.data` sliding windows (memory‑safe). Adjust the sequence length (T) inside `train_dataset.py` consistently with `app.py`.
* Default sequence length used by the codebase: **16 frames** (changeable).

---

## ▶️ Run the Streamlit app

```bash
streamlit run app.py
```

In the browser UI:

* Set **Sequence length (frames)** to the trained value (default: `16`).
* Adjust **YOLO confidence** (try `0.35–0.50`) if detections are missing or noisy.
* Upload a video and click **Run**.
* Toggle **Red flash on anomaly** and set **Flash duration (frames)** to taste.
* Enable **FPS limit** on low‑power CPUs for smoother UI.

---

## 🐳 Docker (CPU‑only)

**Build** (project root where `Dockerfile` & `requirements.txt` live):

```bash
docker build -t ai-surveillance:cpu .
```

**Run** (mount `data/` into container):

**Windows PowerShell** (use ABSOLUTE path):

```powershell
docker run --rm -p 8501:8501 ^
  -v "C:\full\path\to\project\data:/app/data" ^
  ai-surveillance:cpu
```

**macOS / Linux**

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" ai-surveillance:cpu
```

**Optional: run training inside the container**

```bash
# Windows
docker run --rm -v "C:\full\path\to\project\data:/app/data" ai-surveillance:cpu python train_dataset.py

# macOS/Linux
docker run --rm -v "$(pwd)/data:/app/data" ai-surveillance:cpu python train_dataset.py
```

---

## 💡 Usage tips & configurable knobs

* **Sequence length (T)**: must match the length used during training (default `16`). You may change it in both `train_dataset.py` and `app.py` — keep them synchronized.
* **YOLO confidence**: lower for weak detections, higher to reduce false positives.
* **Flash duration**: number of frames to show red border on anomaly.
* **FPS limit**: enable to reduce CPU and UI update rate on low‑end machines.

---

## 🐞 Troubleshooting

**`requirements.txt` not found**

* Run commands from the project root (where the file exists).

**`ModuleNotFoundError: tensorflow`**

```bash
pip install tensorflow==2.15.0
```

**Weights load error / shape mismatch**

* Ensure `behavior_model.py` architecture matches the model used at training time.
* If Keras vs tf.keras mismatch occurs, uninstall standalone Keras packages:

```bash
pip uninstall -y keras keras-nightly keras-tuner
pip install tensorflow==2.15.0
```

**No/weak detections**

* Lower YOLO confidence in the sidebar or use clearer footage.

**Video can't open**

* Re‑encode to H.264 MP4 (e.g., `ffmpeg -i input.mov -c:v libx264 -preset veryfast output.mp4`).

**Docker volume mount issues on Windows**

* Use absolute path and ensure drive sharing is enabled in Docker Desktop.

---

## 🧑‍💻 Development (PyCharm / IDE)

* Open the project root (contains `app.py`, `requirements.txt`).
* Create Python 3.10 virtualenv as project interpreter.
* Install dependencies and Ultralytics:

```bash
pip install -r requirements.txt
pip install "ultralytics>=8.0,<9"
```

**Run/Debug configurations**

* Training: Script = `train_dataset.py`; Working dir = project root
* App: Module = `streamlit`; Parameters = `run app.py`; Working dir = project root

---

## 🔬 Implementation stack

Streamlit, OpenCV, NumPy, Pandas, SciPy, scikit‑learn, TensorFlow 2.15 (tf.keras), Ultralytics YOLO, deep‑sort‑realtime.

---

## ✍️ Recommended small improvements (future)

* Add a lightweight web socket for lower latency UI streaming.
* Provide a small sample video in `data/training_videos/sample.mp4` for quick demo.
* Add unit tests for `detection.py` and `utils.py` (mock YOLO output).
* Add a small model quantized variant for very low‑power CPU inference.

---

## 🙋 Feedback & help

If you want I can:

* 🎨 Reformat the README as a shorter one‑page cheat sheet
* 🧪 Add example `ffmpeg` commands for preprocessing
* 🧩 Produce a minimal `docker-compose.yml` and a small sample video

Just tell me which and I’ll add it.

---

*Made with ❤️ — change the sequence length anywhere in the codebase but keep train & runtime in sync.*
