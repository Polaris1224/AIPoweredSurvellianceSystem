"""
Streaming training to avoid RAM issues and with Keras 3-compliant weights filename.
"""

import os
import random
import tensorflow as tf

from behavior_model import build_behavior_model, save_model_config
from utils import pair_videos_and_mats, read_video_frames_rgb, load_mask_volume, make_sequences_from_frames

# ---------------- Robust paths ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
TRAIN_VIDEO_DIR = os.path.join(DATA_DIR, "training_videos")
TRAIN_VOL_DIR   = os.path.join(DATA_DIR, "training_vol")

# ---------------- Config ----------------
SEQ_LEN = 16
STRIDE  = 4          # increase (e.g., 8–12) to produce fewer clips if training is slow
RESIZE  = (64, 64)
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 20
VAL_RATIO = 0.2

# Keras 3 requires .weights.h5 suffix
WEIGHTS_PATH = os.path.join(DATA_DIR, "models", "behavior_model.weights.h5")

def _ensure_dirs():
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
    for d in [TRAIN_VIDEO_DIR, TRAIN_VOL_DIR]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing folder: {d}. Create it and add files.")

def split_pairs(video_dir, vol_dir, val_ratio=0.2, seed=42):
    pairs = pair_videos_and_mats(video_dir, vol_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_train = max(1, int(len(pairs) * (1 - val_ratio)))
    return pairs[:n_train], pairs[n_train:]

def dataset_from_pairs(pairs):
    """
    Yields one (seq, one_hot_label) at a time to keep memory low.
    seq shape: (SEQ_LEN, RESIZE[0], RESIZE[1], 3), dtype float32 in [0,1].
    """
    def gen():
        for vpath, mpath in pairs:
            frames = read_video_frames_rgb(vpath)        # list of RGB frames
            flags  = load_mask_volume(mpath)             # (T,) bool
            X, Y = make_sequences_from_frames(
                frames, flags, seq_len=SEQ_LEN, stride=STRIDE, resize=RESIZE
            )
            for x, y in zip(X, Y):
                yield x.astype("float32"), tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)

    spec_x = tf.TensorSpec(shape=(SEQ_LEN, RESIZE[0], RESIZE[1], 3), dtype=tf.float32)
    spec_y = tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
    return tf.data.Dataset.from_generator(gen, output_signature=(spec_x, spec_y))

def main():
    _ensure_dirs()

    # Split by videos so validation clips come from different videos
    train_pairs, val_pairs = split_pairs(TRAIN_VIDEO_DIR, TRAIN_VOL_DIR, val_ratio=VAL_RATIO, seed=42)

    train_ds = dataset_from_pairs(train_pairs).shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = dataset_from_pairs(val_pairs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_behavior_model(input_shape=(SEQ_LEN, RESIZE[0], RESIZE[1], 3), num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            WEIGHTS_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Ensure final snapshot saved too
    model.save_weights(WEIGHTS_PATH)
    save_model_config(num_classes=NUM_CLASSES, seq_len=SEQ_LEN, class_names=["normal", "anomaly"])

    print(f"Saved weights to: {WEIGHTS_PATH}")
    print("Saved config to: data/models/behavior_model_config.json")

if __name__ == "__main__":
    main()
