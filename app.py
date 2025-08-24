import os
import cv2
import time
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

from detection import DetectorTracker
from behavior_model import build_behavior_model
from utils import preprocess_frames

# ---------------- UI ----------------
st.set_page_config(page_title="AI-Powered Surveillance", layout="wide")
st.title("🛡️ AI-Powered Surveillance System")

with st.sidebar:
    st.header("Settings")
    BAN_FRAMES = st.number_input("Sequence length (frames)", 8, 64, 16, 1)
    conf_thresh = st.slider("YOLO confidence", 0.1, 0.9, 0.4, 0.05)
    run_fps_limit = st.checkbox("Limit FPS (preview only)", value=False)
    max_fps = st.slider("Max FPS", 1, 30, 12) if run_fps_limit else None

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
run_button = st.button("Run")

# ---------------- Load tracker + model ----------------
@st.cache_resource(show_spinner=False)
def _get_tracker(conf):
    return DetectorTracker(conf_thresh=conf)

def _load_class_names_and_num_classes(default_seq_len):
    """Read model config if available; fallback to 2-class (normal/anomaly)."""
    cfg_path = os.path.join("data", "models", "behavior_model_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        num_classes = int(cfg.get("num_classes", 2))
        class_names = cfg.get("class_names", ["normal", "anomaly"])
        seq_len = int(cfg.get("seq_len", default_seq_len))
    else:
        num_classes = 2
        class_names = ["normal", "anomaly"]
        seq_len = default_seq_len
    return class_names, num_classes, seq_len

@st.cache_resource(show_spinner=False)
def _get_model(BAN_FRAMES):
    class_names, num_classes, seq_len = _load_class_names_and_num_classes(BAN_FRAMES)
    model = build_behavior_model(input_shape=(BAN_FRAMES, 64, 64, 3), num_classes=num_classes)

    # Updated filename for Keras 3 save_weights rule
    weights_path = os.path.join("data", "models", "behavior_model.weights.h5")

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
        except Exception as e:
            st.warning(f"Model weights shape mismatch with current settings: {e}")
    return model, class_names

if uploaded_file and run_button:
    detector_tracker = _get_tracker(conf_thresh)
    model, CLASS_NAMES = _get_model(BAN_FRAMES)

    # Save upload to a temp file
    tpath = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}").name
    with open(tpath, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(tpath)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    alert_log = []
    track_histories = {}     # tid -> list of RGB crops (64x64)
    last_pred_label = {}     # tid -> last predicted label idx

    stframe = st.empty()
    alert_placeholder = st.empty()
    log_placeholder = st.empty()

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_count += 1
            tracks = detector_tracker.detect_and_track(frame_bgr)
            vis = frame_bgr.copy()

            # Update histories with crops
            for tid, (x, y, w, h) in tracks:
                x = max(0, int(x)); y = max(0, int(y))
                w = max(1, int(w)); h = max(1, int(h))
                x2 = min(vis.shape[1], x + w)
                y2 = min(vis.shape[0], y + h)

                crop_bgr = vis[y:y2, x:x2]
                if crop_bgr.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(cv2.resize(crop_bgr, (64, 64)), cv2.COLOR_BGR2RGB)

                if tid not in track_histories:
                    track_histories[tid] = []
                track_histories[tid].append(crop_rgb)
                if len(track_histories[tid]) > BAN_FRAMES:
                    track_histories[tid] = track_histories[tid][-BAN_FRAMES:]

            # Periodically run behavior classification
            if frame_count % BAN_FRAMES == 0:
                anomalies = []
                for tid, crops in list(track_histories.items()):
                    if len(crops) < BAN_FRAMES:
                        continue
                    X = preprocess_frames(crops, size=(64, 64), rgb_already=True, add_batch=True)  # (1,T,64,64,3)
                    pred = model.predict(X, verbose=0)  # (1,num_classes)
                    label_idx = int(np.argmax(pred, axis=1)[0])
                    last_pred_label[tid] = label_idx
                    if label_idx != 0:  # non-normal class
                        anomalies.append((tid, label_idx))

                if anomalies:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    details = [f"ID {tid} → {CLASS_NAMES[label]}" for tid, label in anomalies]
                    alert_text = f"⚠️ Anomalies detected at {ts}: " + "; ".join(details)
                    alert_log.append({"timestamp": ts, "details": alert_text})
                    alert_placeholder.error(alert_text)
                    log_placeholder.dataframe(pd.DataFrame(alert_log), use_container_width=True)

            # Draw tracks + last predicted class
            for tid, (x, y, w, h) in tracks:
                x = int(x); y = int(y); w = int(w); h = int(h)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_idx = last_pred_label.get(tid, 0)
                label_name = CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else "unknown"
                text = f'ID {tid} | {label_name}'
                cv2.putText(vis, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 255, 20), 2)

            # Show frame
            stframe.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB")

            # Optional FPS limit for smoother UI
            if run_fps_limit and max_fps:
                now = time.time()
                elapsed = now - last_time
                min_interval = 1.0 / max_fps
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_time = time.time()

    finally:
        cap.release()
        if alert_log:
            log_placeholder.dataframe(pd.DataFrame(alert_log), use_container_width=True)
