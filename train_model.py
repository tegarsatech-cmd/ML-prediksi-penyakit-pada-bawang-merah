"""
Script Pelatihan Model Deep Learning Penyakit Bawang Merah 4 Kategori:
1. Busuk Daun (Hawar Daun)
2. Moler (Layu Fusarium)
3. Sehat (Daun Sehat)
4. Trotol (Bercak Ungu / Alternaria porri)

Dataset: dataset_clean (Zero Data Leakage - Stratified Group Split by Base Image ID)
Arsitektur: MobileNetV2 dengan Heavy Field Augmentation & Two-Phase Fine-Tuning
"""

import os
import sys
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Pastikan output stream aman dari encoding error di Windows cp1252
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_clean")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model_bawang_final.keras")
CLASS_SAVE_PATH = os.path.join(BASE_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

print(f"TensorFlow Version: {tf.__version__}")
print(f"Memuat dataset bersih dari: {DATASET_DIR}")

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

# Simpan class_names
with open(CLASS_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)
print(f"Label kelas tersimpan ke: {CLASS_SAVE_PATH}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    
    # 1. Augmentasi data luar ruangan yang kokoh (simulasi sinar matahari, rotasi, zoom)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom((-0.15, 0.15))(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness(0.2)(x)
    
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
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(x)
    
    model = models.Model(inputs, outputs, name="AgroScan_MobileNetV2_Clean")
    return model, base_model

model, base_model = build_model(len(class_names))
print("\n" + "="*50)
print("Ringkasan Model:")
model.summary()

ckpt_callback = callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ==============================================================================
# TAHAP 1: WARM-UP CLASSIFIER HEAD (Backbone Frozen, 3 Epochs)
# ==============================================================================
print("\n" + "="*50)
print("TAHAP 1: Melatih Classifier Head (Backbone Frozen, 3 Epochs)...")
print("="*50)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=[ckpt_callback],
    verbose=1
)

# ==============================================================================
# TAHAP 2: DEEP FINE-TUNING (Unfreeze Top Layers 100-154 MobileNetV2, 6 Epochs)
# ==============================================================================
print("\n" + "="*50)
print("TAHAP 2: Membuka Top Layers MobileNetV2 untuk Deep Fine-Tuning (6 Epochs)...")
print("="*50)

base_model.trainable = True
for layer in base_model.layers[:100]:
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
    epochs=6,
    callbacks=cb_list,
    verbose=1
)

# ==============================================================================
# EVALUASI PADA DATA TEST MURNI (ZERO LEAKAGE TEST SET)
# ==============================================================================
print("\n" + "="*50)
print("EVALUASI AKURASI PADA DATA UJI BERSIH (TEST SET):")
print("="*50)

best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
print(f"HASIL AKHIR UJI (TEST SET): Akurasi = {test_acc*100:.2f}% | Loss = {test_loss:.4f}")

# Simpan ulang final
best_model.save(MODEL_SAVE_PATH)
print(f"Model terbaik berhasil disimpan ke: {MODEL_SAVE_PATH}")
print("Pelatihan selesai 100%!")
