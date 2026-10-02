# 🧅 AgroScan - Pendeteksi Penyakit Daun Bawang Merah (MobileNetV2 Keras & AI Agro-Engine)

Aplikasi sistem pakar diagnosis dan deteksi dini penyakit tanaman bawang merah (*Allium cepa*) berbasis Deep Learning MobileNetV2 (15 kategori) terintegrasi dengan Peta Atensi Lesi Konvolusi (CAM HUD Scanner) dan Rekomendasi Agronomi Resmi Balitsa / BPTP Kementerian Pertanian RI.

---

## 🚀 Fitur Unggulan Sistem

1. **Inferensi Pola Konvolusi Murni 15 Kategori Model Keras**:
   - Menganalisis citra menggunakan model `model_bawang_final.keras` dengan 15 kelas klasifikasi penyakit dan kondisi sehat.
   - Menggunakan tensor skala natural `[0.0, 255.0]` murni arsitektur Keras tanpa distorsi *double-normalization*.
   - **Multi-Crop TTA Cerdas (Test-Time Augmentation)**: Memindai helai daun dari beberapa perspektif (preservasi rasio aspek, pembesaran zona lesi tengah, dan ujung daun).

2. **Diferensial Diagnosis (Kemungkinan A & Kemungkinan B)**:
   - Jika terdapat dua penyakit yang bersaing dengan tingkat probabilitas signifikan (misal Karat Daun vs Hawar Bakteri), sistem menampilkan dua kartu diagnosis berdampingan secara adil dan transparan lengkap dengan persentase kepastian alami, nama latin, dan ciri khas lapangan.

3. **Peta Deteksi Lesi Multi-Penyakit (High-Precision HUD Reticle CAM)**:
   - **Segmentasi Kanopi Daun (Leaf Canopy Masking)**: Titik atensi dipastikan 100% hanya mengunci helai daun, bukan latar belakang, lantai, atau meja.
   - **Skin Tone Exclusion**: Membedakan warna kulit manusia dari daun, sehingga foto daun yang sedang dipegang oleh tangan petani tetap terdeteksi presisi tanpa ada lingkaran di jari tangan.
   - **Penanda Multi-Penyakit**: Retikel Merah Neon `[A: Nama Penyakit]` menandai fokus infeksi Kemungkinan A, dan Retikel Oranye Neon `[B: Nama Penyakit]` menandai fokus infeksi Kemungkinan B.

4. **Verifikasi Karakteristik Fisik Langsung di Sawah**:
   - Panduan praktis lapangan mencakup **Uji Raba** (lendir vs tepung kering), **Uji Aroma Daun** (langu busuk bakteri vs daun kering jamur), dan **Uji Usapan Jari** untuk memvalidasi diagnosis langsung di bedengan.
   - Dilengkapi sistem cadangan offline mandiri jika koneksi AI terputus.

5. **Petunjuk Obat & Perawatan dari Dokter Tanaman (3 Kartu Terstruktur)**:
   - **Kartu 1 - Tindakan Langsung di Kebun**: Prosedur taktis darurat pemangkasan presisi dan sanitasi 24 jam pertama.
   - **Kartu 2 - Rekomendasi Obat Semprot**: Bahan aktif resmi Balitsa/Kementan (kontak & sistemik), takaran sendok per tangki 16 Liter, dan waktu semprot optimal.
   - **Kartu 3 - Perawatan Lahan, Pupuk & Agens Hayati**: Tata kelola air parit macak-macak, penghentian pupuk Nitrogen sukulen, aplikasi Kalsium-Silika penguat kutikula, dan pemanfaatan jamur *Trichoderma*.

6. **Filter Gatekeeper & Sensitivitas Adaptif**:
   - Menerima foto daun bawang merah asli meskipun dipegang tangan manusia (sensitivitas minimal 2%-3%).
   - Menolak objek yang bukan tanaman bawang merah (manusia, hewan, kendaraan, makanan jadi) secara ramah.

7. **Riwayat Pemeriksaan Tanpa Database Eksternal**:
   - Otomatis mencatat riwayat diagnosa per sesi.
   - Fitur ekspor riwayat ke format **CSV** dan **JSON**.

---

## 📋 Daftar 15 Kategori Deteksi Model Keras

1. `Alternaria_D`: Bercak Ungu / Trotol (*Alternaria porri*)
2. `Botrytis Leaf Blight`: Hawar Daun / Bintik Putih (*Botrytis squamosa*)
3. `Bulb Rot`: Busuk Umbi Basah (*Aspergillus / Fusarium spp.*)
4. `Bulb_blight-D`: Hawar Leher Umbi (*Blight pathogen*)
5. `Caterpillar-P`: Ulat Grayak (*Spodoptera exigua*)
6. `Downy mildew`: Embun Bulu / Palsu (*Peronospora destructor*)
7. `Fusarium-D`: Layu Moler / Fusarium (*Fusarium oxysporum*)
8. `Healthy leaves`: Daun Sehat & Segar (Kondisi Normal)
9. `Iris yellow virus_augment`: Virus Iris Kuning / IYSV (*Iris yellow spot virus*)
10. `Purple blotch`: Bercak Ungu Cincin Konsentris (*Alternaria porri*)
11. `Rust`: Karat Daun / Bintil Merah (*Puccinia allii*)
12. `Virosis-D`: Virus Kuning Melintir (*Onion yellow dwarf virus*)
13. `Xanthomonas Leaf Blight`: Hawar Daun Bakteri (*Xanthomonas axonopodis*)
14. `onion1`: Daun Tanaman Normal Sehat (*Allium cepa*)
15. `stemphylium Leaf Blight`: Hawar Kering Ujung (*Stemphylium vesicarium*)

---

## 📦 Instalasi & Persiapan

Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

Atau instal paket yang dibutuhkan:
```bash
pip install streamlit tensorflow pillow numpy requests
```

---

## 🏃 Cara Menjalankan Aplikasi

1. Pastikan file berikut berada di folder proyek yang sama:
   - `app.py`
   - `model_bawang_final.keras`
   - `class_names.json`
2. Jalankan perintah:
```bash
streamlit run app.py
```
3. Buka peramban di alamat: `http://localhost:8501`.
