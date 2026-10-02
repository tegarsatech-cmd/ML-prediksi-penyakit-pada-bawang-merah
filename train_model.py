"""
Script Pelatihan Model Deep Learning Penyakit Bawang Merah 4 Kategori:
1. Busuk Daun (Hawar Daun)
2. Moler (Layu Fusarium)
3. Sehat (Daun Sehat)
4. Trotol (Bercak Ungu / Alternaria porri)

Arsitektur: MobileNetV2 dengan ModelCheckpoint Otomatis & Deep Fine-Tuning
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Pastikan output stream aman dari encoding error di Windows cp1252
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

DATASET_DIR = "d:/pendeteksi bawang merah citra  digital/dataset_temp"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_PATH = "d:/pendeteksi bawang merah citra  digital/model_bawang_final.keras"
CLASS_SAVE_PATH = "d:/pendeteksi bawang merah citra  digital/class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"TensorFlow Version: {tf.__version__}")
print("Memuat dataset citra...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
print(f"Kelas terdeteksi ({len(class_names)}): {class_names}")

# Simpan class_names sekarang
with open(CLASS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
print(f"Label kelas tersimpan ke: {CLASS_SAVE_PATH}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    
    # 1. Augmentasi data
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.12)(x)
    x = layers.RandomZoom(0.12)(x)
    
    # 2. Normalisasi bawaan model [0, 255] -> [-1, 1]
    x = layers.Rescaling(1./127.5, offset=-1.0, name="rescaling_mobilenet")(x)
    
    # 3. Backbone MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    
    # 4. Top Classification Head (Canonical CAM Architecture)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(x)
    
    model = models.Model(inputs, outputs, name="AgroScan_MobileNetV2_4Class")
    return model, base_model

model, base_model = build_model(len(class_names))
print("\n" + "="*50)
print("Ringkasan Model:")
model.summary()

# Checkpoint callback untuk menyimpan otomatis bobot terbaik ke file final
ckpt_callback = callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ==============================================================================
# TAHAP 1: WARM-UP CLASSIFIER HEAD (Backbone Frozen)
# ==============================================================================
print("\n" + "="*50)
print("TAHAP 1: Melatih Classifier Head (Backbone Frozen, 2 Epochs)...")
print("="*50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[ckpt_callback],
    verbose=1
)

# ==============================================================================
# TAHAP 2: DEEP FINE-TUNING (Unfreeze Top Layers MobileNetV2)
# ==============================================================================
print("\n" + "="*50)
print("TAHAP 2: Membuka Top Layers MobileNetV2 untuk Deep Fine-Tuning (4 Epochs)...")
print("="*50)

base_model.trainable = True
for layer in base_model.layers[:90]:
    layer.trainable = False

print(f"Total layer backbone: {len(base_model.layers)}, Trainable: {len([l for l in base_model.layers if l.trainable])}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb_list = [
    ckpt_callback,
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4,
    callbacks=cb_list,
    verbose=1
)

# ==============================================================================
# EVALUASI PADA DATA TEST
# ==============================================================================
print("\n" + "="*50)
print("EVALUASI AKURASI PADA DATA UJI (TEST SET):")
print("="*50)

# Muat model terbaik yang disimpan checkpoint
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"HASIL AKHIR: Test Accuracy = {test_acc*100:.2f}% | Test Loss = {test_loss:.4f}")

# Simpan ulang untuk memastikan
best_model.save(MODEL_SAVE_PATH)
print(f"Model terbaik berhasil disimpan ke: {MODEL_SAVE_PATH}")
print("Pelatihan selesai 100%!")
