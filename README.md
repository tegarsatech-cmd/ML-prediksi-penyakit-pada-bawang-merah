# 🧅 AgroScan - Pendeteksi Penyakit Daun Bawang Merah (MobileNetV2 Keras & AI Agro-Engine)
![Version](https://img.shields.io/badge/Version-v3.0.0-success?style=flat-square) ![Status](https://img.shields.io/badge/Status-Live%20Production-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-MobileNetV2%204%20Classes-green?style=flat-square) ![Accuracy](https://img.shields.io/badge/Val%20Accuracy-93.5%25-brightgreen?style=flat-square) ![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.5%25-brightgreen?style=flat-square)

Aplikasi sistem pakar diagnosis dan deteksi dini penyakit utama tanaman bawang merah (*Allium cepa*) berbasis Deep Learning MobileNetV2 (4 Kategori Inti) dengan akurasi validasi **93.51%** dan akurasi uji **91.53%** pada 5.989 citra asli. Dilengkapi dengan Peta Atensi Lesi Konvolusi (CAM HUD Scanner) dan Rekomendasi Agronomi Resmi Balitsa / BPTP Kementerian Pertanian RI.

---

## 🚀 Fitur Unggulan Sistem

1. **Model Deep Learning MobileNetV2 Hasil Retraining (Fine-Tuned)**:
   - Dilatih menggunakan transfer learning 2 tahap (Warmup + Deep Fine-Tuning 64 layers) pada 5.989 dataset citra daun bawang merah asli di lapangan.
   - Akurasi validasi mencapai **93.51%** dan akurasi data uji (test set tak terlihat) mencapai **91.53%**.
   - Input tensor alami `[0.0, 255.0]` diolah langsung oleh layer preprocessing internal model (`Rescaling 1./127.5, offset=-1.0`).
   - Menghilangkan redundansi fitopatologi dan *probability splitting* yang sebelumnya terjadi pada model lama.

2. **Diferensial Diagnosis (Kemungkinan A & Kemungkinan B) & Vonis Tunggal Presisi**:
   - Jika hanya 1 penyakit yang terdeteksi secara dominan, sistem menegakkan vonis tunggal dengan keyakinan tinggi.
   - Diferensial diagnosis berdampingan aktif apabila terdapat dua penyakit yang bersaing ketat untuk membantu petani membedakan patogen di sawah.

3. **Peta Titik Kerusakan pada Foto Daun (High-Precision HUD Reticle CAM)**:
   - **Terkunci Mutlak pada Kerusakan Fisik**: Retikel scanner HANYA menandai titik lesi fisik nyata pada helai daun melalui Class Activation Mapping (CAM) layer konvolusi terakhir.
   - **Segmentasi Kanopi Daun & Skin Tone Exclusion**: Membedakan warna kulit manusia dari daun, sehingga foto daun yang sedang dipegang tangan petani tetap terdeteksi presisi.
   - **Penanda Langsung Nama Penyakit**: Retikel scanner modern berlabel nama penyakit spesifik (*Busuk Daun*, *Moler*, *Trotol*).

4. **Verifikasi Karakteristik Fisik Langsung di Sawah**:
   - Panduan praktis lapangan mencakup **Uji Raba** (lendir basah vs tepung serbuk kering), **Uji Aroma Daun** (langu busuk bakteri vs daun kering jamur), dan **Uji Usapan Jari** untuk memvalidasi diagnosis langsung di bedengan.
   - Dilengkapi sistem cadangan offline mandiri jika koneksi AI terputus.

5. **Petunjuk Obat & Perawatan dari Dokter Tanaman (3 Kartu Terstruktur)**:
   - **Kartu 1 - Tindakan Langsung di Kebun**: Prosedur darurat pemangkasan presisi dan sanitasi 24 jam pertama.
   - **Kartu 2 - Rekomendasi Obat Semprot**: Bahan aktif resmi Balitsa/Kementan (Mankozeb, Klorotalonil, Difenokonazol, Tiram, dll.), takaran sendok per tangki 16 Liter, dan waktu semprot optimal.
   - **Kartu 3 - Perawatan Lahan, Pupuk & Agens Hayati**: Tata kelola air parit macak-macak, penghentian pupuk Nitrogen sukulen, aplikasi Kalsium-Silika penguat kutikula, dan pemanfaatan jamur *Trichoderma*.

6. **Filter Gatekeeper & Sensitivitas Adaptif**:
   - Menerima foto daun bawang merah asli meskipun dipegang tangan manusia (sensitivitas minimal 2%-3%).
   - Menolak objek yang bukan tanaman bawang merah secara ramah dan edukatif.

7. **Riwayat Pemeriksaan Tanpa Database Eksternal**:
   - Otomatis mencatat riwayat diagnosa per sesi.
   - Fitur ekspor riwayat ke format **CSV** dan **JSON**.

---

## 📋 Daftar 4 Kategori Deteksi Model Keras

1. **Busuk Daun** (*Peronospora destructor* / *Stemphylium vesicarium* / Hawar Daun / Embun Bulu):
   - Gejala: Helai daun memutih atau cokelat kebasahan dari pucuk, tampak layu busuk basah akibat kelembaban tinggi dan spora cendawan.
2. **Moler** (*Fusarium oxysporum* f.sp. *cepae* / Penyakit Inul / Layu Fusarium):
   - Gejala: Daun meliuk-liuk, melintir spiral (*ngoler*), menguning pucat dari pangkal, umbi membusuk dan akar mengerut.
3. **Sehat** (*Allium cepa*):
   - Gejala: Daun tegak kokoh, hijau segar merata, berlilin alami, tanpa bercak nekrotik maupun kelainan bentuk.
4. **Trotol** (*Alternaria porri* / Bercak Ungu / Purple Blotch):
   - Gejala: Bercak melekuk ke dalam berwarna putih keabu-abuan dengan cincin konsentris ungu di tengahnya, ujung daun mengering patah.

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
