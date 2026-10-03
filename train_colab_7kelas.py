# ==============================================================================
# SCRIPT LENGKAP PELATIHAN MODEL 7 KELAS (GOOGLE COLAB)
# Arsitektur: EfficientNet-B0 PyTorch -> TorchScript (.pt) & meta.json
# Fitur: Class Balancing (Daun Sehat & Penyakit 100% Seimbang, Tanpa Bias)
# ==============================================================================

# 1. Install Library Pendukung (Otomatis jika belum terpasang di Colab/Python)
import sys
import subprocess

for pkg in ["kagglehub", "torchvision", "torch", "Pillow", "tqdm", "scikit-learn"]:
    try:
        mod_name = "sklearn" if pkg == "scikit-learn" else ("PIL" if pkg == "Pillow" else pkg)
        __import__(mod_name)
    except ImportError:
        print(f"Menginstal paket {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

import os
import shutil
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import classification_report
import kagglehub

# ------------------------------------------------------------------------------
# 2. Inisialisasi Device & Bersihkan Cache GPU
# ------------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device yang Digunakan:", device)
if torch.cuda.is_available():
    print("   Nama GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------------------------
# 3. Download Dataset Publik Kaggle Bawang Merah (7 Kategori)
# ------------------------------------------------------------------------------
print("\n[Langkah 1/6] Mengunduh dataset publik dari Kaggle...")
raw_data_dir = kagglehub.dataset_download("tejasbargujepatil/onion-diseases")
print("✅ Dataset mentah berhasil diunduh ke:", raw_data_dir)

# ------------------------------------------------------------------------------
# 4. Kurasi & Penyeimbangan Dataset (Mencegah Bias ke Hawar Daun)
# ------------------------------------------------------------------------------
print("\n[Langkah 2/6] Menyusun dan menyeimbangkan data 7 kelas...")
BASE_DATASET = Path("/content/dataset_7kelas_balanced")
if BASE_DATASET.exists():
    shutil.rmtree(BASE_DATASET)

# 7 Kelas Target Resmi
CLASSES = [
    "downy_mildew",       # 1. Embun Bulu
    "healthy",            # 2. Daun Sehat & Normal (Dijamin Seimbang!)
    "iris_yellow_virus",  # 3. Virus Iris Kuning (IYSV)
    "leaf_blight",        # 4. Hawar Daun (Hanya helai daun, foto umbi dibuang)
    "moler",              # 5. Layu Moler (Fusarium)
    "purple_blotch",      # 6. Bercak Ungu / Trotol
    "rust"                # 7. Karat Daun
]

# Mapping Folder: Ambil HANYA foto daun murni.
# CATATAN: "Bulb_blight-D" sengaja DIBUANG karena isinya foto umbi di tanah, bukan daun.
FOLDER_MAPPING = {
    "Downy mildew": "downy_mildew",
    "Healthy leaves": "healthy",
    "onion1": "healthy",
    "Iris yellow virus_augment": "iris_yellow_virus",
    "Virosis-D": "iris_yellow_virus",
    "stemphylium Leaf Blight": "leaf_blight",
    "Botrytis Leaf Blight": "leaf_blight",
    "Xanthomonas Leaf Blight": "leaf_blight",
    "Fusarium-D": "moler",
    "Purple blotch": "purple_blotch",
    "Alternaria_D": "purple_blotch",
    "Rust": "rust"
}

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (BASE_DATASET / split / cls).mkdir(parents=True, exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
all_images = {cls: [] for cls in CLASSES}

for root, _, files in os.walk(raw_data_dir):
    folder_name = os.path.basename(root)
    if folder_name in FOLDER_MAPPING:
        t_cls = FOLDER_MAPPING[folder_name]
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                all_images[t_cls].append(os.path.join(root, f))

# CLASS BALANCING: Batasi sampel maksimal per kelas agar proporsi adil
random.seed(42)
MAX_PER_CLASS = 450

for cls, paths in all_images.items():
    random.shuffle(paths)
    sel = paths[:MAX_PER_CLASS]
    n_train = int(len(sel) * 0.8)
    n_val = int(len(sel) * 0.1)
    
    for p in sel[:n_train]:
        shutil.copy(p, BASE_DATASET / "train" / cls / Path(p).name)
    for p in sel[n_train:n_train + n_val]:
        shutil.copy(p, BASE_DATASET / "val" / cls / Path(p).name)
    for p in sel[n_train + n_val:]:
        shutil.copy(p, BASE_DATASET / "test" / cls / Path(p).name)

# ------------------------------------------------------------------------------
# 5. Data Loaders & Augmentasi Citra
# ------------------------------------------------------------------------------
print("\n[Langkah 3/6] Mempersiapkan Data Loader & Augmentasi...")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_ds = ImageFolder(str(BASE_DATASET / "train"), transform=train_tf)
val_ds = ImageFolder(str(BASE_DATASET / "val"), transform=eval_tf)
test_ds = ImageFolder(str(BASE_DATASET / "test"), transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

# Hitung bobot per kelas (Class Weights) agar penalti loss adil
class_counts = [0] * len(CLASSES)
for _, lbl in train_ds.samples:
    class_counts[lbl] += 1
weights = [len(train_ds) / (len(CLASSES) * c) if c > 0 else 1.0 for c in class_counts]
weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

print(f"📊 Jumlah Sampel Latih per Kelas: {dict(zip(train_ds.classes, class_counts))}")

# ------------------------------------------------------------------------------
# 6. Membangun Model EfficientNet-B0 & Training Seimbang
# ------------------------------------------------------------------------------
print("\n[Langkah 4/6] Memuat EfficientNet-B0 dan Memulai Pelatihan...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_f = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_f, len(CLASSES))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=0.03)

# Fase A: Warmup Classifier Head (3 Epoch)
for param in model.features.parameters():
    param.requires_grad = False
opt_head = optim.AdamW(model.classifier.parameters(), lr=1e-3)

print("\n--- FASE A: WARMUP CLASSIFIER (3 EPOCH) ---")
for ep in range(1, 4):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt_head.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt_head.step()
        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    print(f"Warmup Epoch {ep}/3: Loss={running_loss/total:.4f}, Acc={correct/total*100:.2f}%")

# Fase B: Fine-Tuning Semua Layer (9 Epoch)
for param in model.features.parameters():
    param.requires_grad = True
opt_full = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=9, eta_min=1e-6)

print("\n--- FASE B: FULL FINE-TUNING (9 EPOCH) ---")
best_acc = 0.0
best_w = None

for ep in range(1, 10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {ep}/9"):
        x, y = x.to(device), y.to(device)
        opt_full.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt_full.step()
        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    # Validasi
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            v_correct += pred.eq(y).sum().item()
            v_total += y.size(0)
    
    v_acc = v_correct / v_total
    scheduler.step()
    print(f"Epoch {ep:02d}: Train Acc={correct/total*100:.1f}%, Val Acc={v_acc*100:.1f}%")
    if v_acc > best_acc:
        best_acc = v_acc
        best_w = model.state_dict().copy()

# Pasang bobot terbaik
model.load_state_dict(best_w)

# ------------------------------------------------------------------------------
# 7. Evaluasi Akurasi Per Kelas di Test Set (Memastikan Daun Sehat Akurat)
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("[Langkah 5/6] LAPORAN AKURASI PER KELAS (TEST DATASET):")
print("="*60)
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        _, p = model(x).max(1)
        y_pred.extend(p.cpu().numpy())
        y_true.extend(y.numpy())

print(classification_report(y_true, y_pred, target_names=train_ds.classes, digits=2))

# ------------------------------------------------------------------------------
# 8. Export TorchScript Model (.pt) & meta.json
# ------------------------------------------------------------------------------
print("\n[Langkah 6/6] Mengekspor TorchScript & meta.json...")
model.eval().to("cpu")
dummy = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, dummy)
ts_path = "/content/bawang_efficientnet_b0_ts.pt"
traced.save(ts_path)
print("✅ Model TorchScript berhasil disimpan:", ts_path)

c_keys = train_ds.classes
label_map = {
    "downy_mildew": "Downy mildew",
    "healthy": "Sehat",
    "iris_yellow_virus": "Iris Yellow Spot Virus (IYSV)",
    "leaf_blight": "Hawar Daun (Stemphylium / Colletotrichum)",
    "moler": "Moler",
    "purple_blotch": "Bercak Ungu / Trotol (Alternaria porri)",
    "rust": "Rust"
}
c_labels = [label_map[k] for k in c_keys]

meta = {
    "arch": "efficientnet_b0",
    "class_keys": c_keys,
    "class_labels_id": c_labels,
    "img_size": 224,
    "mean": MEAN,
    "std": STD,
    "temperature": 0.5,        # Temperature adil, tidak bias ekstrem!
    "conf_threshold": 0.65,
    "data_config": "balanced_v2"
}

meta_path = "/content/meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("✅ File meta.json berhasil disimpan:", meta_path)

# 9. Kemas ke ZIP & Download Otomatis
import zipfile

zip_path = "/content/model_bawang_7kelas_balanced.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    if os.path.exists(ts_path):
        zipf.write(ts_path, arcname=os.path.basename(ts_path))
    if os.path.exists(meta_path):
        zipf.write(meta_path, arcname=os.path.basename(meta_path))
print("✅ File ZIP berhasil dibuat:", zip_path)

try:
    from google.colab import files
    files.download(zip_path)
    print("\n🎉 SELESAI! File 'model_bawang_7kelas_balanced.zip' otomatis diunduh ke laptop Anda!")
except ImportError:
    print(f"\n🎉 Model dan meta telah selesai dikemas di: {zip_path}")
