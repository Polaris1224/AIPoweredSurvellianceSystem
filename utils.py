import os
import cv2
import numpy as np
import scipy.io

# ---------------- Basic frame preprocessing ----------------
def preprocess_frames(frames, size=(64, 64), rgb_already=False, add_batch=False):
    """
    frames: list/array of frames (H,W,3)
    - If rgb_already=False, assumes BGR and converts to RGB
    Returns (T,H,W,C) in [0,1], or (1,T,H,W,C) if add_batch=True
    """
    processed = []
    for f in frames:
        if f is None or (hasattr(f, "size") and f.size == 0):
            continue
        fr = cv2.resize(f, size)
        if not rgb_already:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = fr.astype("float32") / 255.0
        processed.append(fr)
    arr = np.array(processed, dtype=np.float32)
    if add_batch:
        arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- Dataset helpers ----------------
def _sorted_files(folder, exts):
    return sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts])

def pair_videos_and_mats(video_dir, vol_dir):
    """
    Pairs files by sorted order. Assumes matching counts. Adjust if your naming differs.
    Returns list of (video_path, mat_path).
    """
    vids = _sorted_files(video_dir, {".mp4", ".avi", ".mov"})
    mats = _sorted_files(vol_dir, {".mat"})
    if len(vids) != len(mats):
        raise RuntimeError(f"Video/Mat count mismatch: {len(vids)} vs {len(mats)}")
    return [(os.path.join(video_dir, v), os.path.join(vol_dir, m)) for v, m in zip(vids, mats)]

def read_video_frames_rgb(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames  # list of RGB frames

def load_mask_volume(mat_path):
    """
    Loads 'vol'/'Vol'/'mask' from .mat and returns per-frame boolean flags (T,).
    Handles arrays and MATLAB cell arrays robustly.
    """
    data = scipy.io.loadmat(mat_path, simplify_cells=True)

    vol = None
    for k in ("vol", "Vol", "mask"):
        if k in data and data[k] is not None:
            vol = data[k]
            break
    if vol is None:
        raise KeyError(f"No ['vol'|'Vol'|'mask'] in {mat_path}")

    # Normalize to ndarray
    if isinstance(vol, (list, tuple)):
        vol = np.array(vol, dtype=object)
    else:
        vol = np.array(vol)

    # If it's a sequence of 2D masks (cell array), stack to (T,H,W)
    if vol.dtype == object:
        flat = [np.asarray(x) for x in np.ravel(vol)]
        if len(flat) == 0:
            raise ValueError(f"Empty mask volume in {mat_path}")
        vol = np.stack(flat, axis=0)  # (T,H,W)

    # Binarize
    vol = (vol > 0)

    # Reduce to per-frame flags
    if vol.ndim == 3:
        # Try both interpretations: (T,H,W) and (H,W,T)
        flags_axis0 = vol.any(axis=(1, 2))  # assumes axis 0 is time
        flags_axis2 = vol.any(axis=(0, 1))  # assumes axis 2 is time
        frame_flags = flags_axis2 if flags_axis2.sum() >= flags_axis0.sum() else flags_axis0
    elif vol.ndim == 2:
        # Ambiguous 2D; choose axis with smaller length as time
        time_axis = 0 if vol.shape[0] < vol.shape[1] else 1
        frame_flags = vol.any(axis=1 - time_axis)
    elif vol.ndim == 1:
        frame_flags = vol.astype(bool)
    else:
        raise ValueError(f"Unsupported mask volume shape {vol.shape} in {mat_path}")

    return np.asarray(frame_flags, dtype=bool).ravel()

def make_sequences_from_frames(frames_rgb, frame_flags, seq_len=16, stride=4, resize=(64, 64)):
    """
    Creates (X, y) where:
      X: list of np.array with shape (seq_len, 64, 64, 3), values in [0,1]
      y: list of int labels (0=normal, 1=anomaly) — 2 classes
    A sequence is labeled anomalous if ANY frame in the window is anomalous.
    """
    T = min(len(frames_rgb), len(frame_flags))
    X, Y = [], []
    if T < seq_len:
        return X, Y

    for start in range(0, T - seq_len + 1, stride):
        end = start + seq_len
        clip = frames_rgb[start:end]
        flags = frame_flags[start:end]
        label = 1 if np.any(flags) else 0

        clip_arr = preprocess_frames(clip, size=resize, rgb_already=True, add_batch=False)  # (seq_len,H,W,C)
        if clip_arr.shape[0] == seq_len:
            X.append(clip_arr)
            Y.append(label)
    return X, Y
