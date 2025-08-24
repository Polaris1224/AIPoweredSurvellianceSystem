AI‑Powered Surveillance System
Streamlit app for real‑time person detection, tracking, and short‑term behavior anomaly classification using YOLO, DeepSORT, and a MobileNetV2+LSTM sequence model.

Features
Person detection (Ultralytics YOLO) and multi‑object tracking (DeepSORT)

Temporal classification on per‑track crops (MobileNetV2 features + LSTM)

Streamlit UI: upload/preview, per‑ID labels, alert log, optional red‑frame flash on anomaly

Memory‑safe training with tf.data sliding windows

CPU‑only and Docker (CPU) support; optional GPU path available later

Project structure
text
.
├─ app.py
├─ behavior_model.py
├─ detection.py
├─ train_dataset.py
├─ utils.py
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ data/
   ├─ training_videos/         # input videos (you add)
   ├─ training_vol/            # paired .mat files (you add)
   └─ models/                  # saved artifacts after training
      ├─ behavior_model.weights.h5
      └─ behavior_model_config.json
Requirements
Python 3.10.x (64‑bit)

Windows/macOS/Linux

Optional: Docker Desktop (for containerized run)

Installation (local, without Docker)
Create and activate a virtual environment

Windows PowerShell

text
py -3.10 -m venv env
.\env\Scripts\Activate
macOS/Linux

text
python3.10 -m venv env
source env/bin/activate
Install dependencies

text
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "ultralytics>=8.0,<9"
Prepare data folders

text
mkdir -p data/training_videos
mkdir -p data/training_vol
mkdir -p data/models
Train the behavior model (optional if artifacts already exist)

text
python train_dataset.py
This creates:

data/models/behavior_model.weights.h5

data/models/behavior_model_config.json

Run the app

text
streamlit run app.py
In the browser:

Set Sequence length (frames) to the trained value (default 16)

Adjust YOLO confidence (try 0.35–0.50)

Upload a video and click Run

Docker (CPU‑only)
Ensure files are at project root (app.py, requirements.txt, Dockerfile). Keep large data in data/.

Build the image

text
docker build -t ai-surveillance:cpu .
Run the app container with the host data mounted

Windows PowerShell (absolute path recommended)

text
docker run --rm -p 8501:8501 ^
  -v "C:\full\path\to\project\data:/app/data" ^
  ai-surveillance:cpu
macOS/Linux

text
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" ai-surveillance:cpu
Optional: run training inside the container

Windows

text
docker run --rm -v "C:\full\path\to\project\data:/app/data" ai-surveillance:cpu python train_dataset.py
macOS/Linux

text
docker run --rm -v "$(pwd)/data:/app/data" ai-surveillance:cpu python train_dataset.py
Usage tips
Sequence length (T): must match the value used for training (default 16)

Red flash on anomaly: enable in sidebar; adjust “Flash duration (frames)”

FPS limit: enable in sidebar for smoother UI on slower CPUs

Troubleshooting
requirements.txt not found

Run commands from the project root where requirements.txt is located.

ModuleNotFoundError: tensorflow

Install explicitly:

text
pip install tensorflow==2.15.0
Weight load error or shape mismatch (e.g., “Conv1 expected 1 variables, got 0”)

Ensure only tf.keras is used (no standalone keras package):

text
pip uninstall -y keras keras-nightly keras-tuner
pip install tensorflow==2.15.0
Make sure behavior_model.py matches the architecture used during training.

Retrain in this environment:

text
python train_dataset.py
No detections or weak detections

Lower YOLO confidence in the sidebar or use clearer footage.

Video cannot open

Re‑encode to H.264 MP4 and retry.

Docker volume mount issues on Windows

Use an absolute path and ensure drive sharing is enabled in Docker Desktop.

Development (PyCharm)
Open the project root (where app.py and requirements.txt live)

Create a Python 3.10 virtualenv interpreter

Install:

text
pip install -r requirements.txt
pip install "ultralytics>=8.0,<9"
Run/Debug configurations:

Training: Script = train_dataset.py; Working dir = project root

App: Module = streamlit; Parameters = run app.py; Working dir = project root

ech stack
Streamlit, OpenCV, NumPy, Pandas, SciPy, scikit‑learn

TensorFlow 2.15 (tf.keras)

Ultralytics YOLO, deep‑sort‑realtime

Docker (CPU image provided)
