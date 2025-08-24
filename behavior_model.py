import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_behavior_model(input_shape=(16, 64, 64, 3), num_classes=2):
    """
    TimeDistributed MobileNetV2 + LSTM for sequence classification.
    Default num_classes=2 (normal, anomaly). If you trained 3-class, pass num_classes=3.
    """
    inputs = layers.Input(shape=input_shape)

    # Preprocess for MobileNetV2 (expects RGB 0..255 into preprocess_input)
    def _preproc(x):
        return tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

    x = layers.TimeDistributed(layers.Lambda(_preproc), name="mobilenet_preproc")(inputs)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[1], input_shape[2], input_shape[3]),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    x = layers.TimeDistributed(base, name="mobilenet")(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="gap")(x)   # (T, F)
    x = layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, name="lstm")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model_config(num_classes=2, seq_len=16, class_names=None, path="data/models/behavior_model_config.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if class_names is None:
        class_names = ["normal", "anomaly"] if num_classes == 2 else [str(i) for i in range(num_classes)]
    cfg = {"num_classes": int(num_classes), "seq_len": int(seq_len), "class_names": list(class_names)}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
