# 🧅 AgroScan - Pendeteksi Penyakit Daun Bawang Merah (EfficientNet-B0 TorchScript & AI Agro-Engine)
![Version](https://img.shields.io/badge/Version-v3.2.0-success?style=flat-square) ![Status](https://img.shields.io/badge/Status-Live%20Production-blue?style=flat-square) ![Model](https://img.shields.io/badge/Model-EfficientNet--B0%20TorchScript%207%20Classes-green?style=flat-square)

Aplikasi sistem pakar diagnosis dan deteksi dini penyakit utama tanaman bawang merah (*Allium cepa*) berbasis Deep Learning **EfficientNet-B0 (TorchScript)** dengan 7 kategori kelas (6 penyakit utama + 1 sehat). Dilengkapi dengan Peta Deteksi Lesi HUD Scanner dan Rekomendasi Agronomi Resmi Balitsa / BPTP Kementerian Pertanian RI.

---

## 🚀 Fitur Unggulan Sistem

1. **Model Deep Learning EfficientNet-B0 (TorchScript - Balanced v2)**:
   - Dilatih menggunakan transfer learning pada dataset citra daun bawang merah asli di lapangan dengan sampling seimbang (balanced class weighting).
   - Model disimpan dalam format TorchScript (`.pt`) — dimuat langsung dengan `torch.jit.load()` tanpa perlu definisi arsitektur.
   - Pipeline prediksi menggunakan **Test-Time Augmentation (TTA)** horizontal flip untuk meningkatkan akurasi.
   - Normalisasi input: ImageNet mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]`.
   - Temperature-scaled softmax (`T=0.5`) untuk kalibrasi probabilitas yang proporsional dan tidak bias.
   - Confidence threshold `0.65` — jika keyakinan di bawah ambang batas, sistem menampilkan "Perlu Verifikasi Lapangan" alih-alih menebak.

2. **Diferensial Diagnosis (Kemungkinan A & Kemungkinan B) & Vonis Tunggal Presisi**:
   - Jika hanya 1 penyakit yang terdeteksi secara dominan, sistem menegakkan vonis tunggal dengan keyakinan tinggi.
   - Diferensial diagnosis berdampingan aktif apabila terdapat dua penyakit yang bersaing ketat untuk membantu petani membedakan patogen di sawah.

3. **Peta Titik Kerusakan pada Foto Daun (HUD Lesion Scanner Presisi Tinggi)**:
   - **Terkunci Mutlak pada Kerusakan Fisik**: Retikel scanner HANYA menandai titik lesi fisik nyata pada helai daun (anti-melengser, anti-acak).
   - **Akurasi Tinggi untuk Seluruh 7 Kategori**: Seluruh kelas patogen (Bercak Ungu, Hawar Daun, Karat Daun, Layu Moler, IYSV, hingga Daun Sehat) ditandai dengan fokus tunggal setara ketepatan Embun Bulu.
   - **Segmentasi Kanopi Daun & Skin Tone Exclusion**: Membedakan warna kulit manusia dari daun, sehingga foto daun yang sedang dipegang tangan petani tetap terdeteksi presisi.
   - **Penanda Langsung Nama Penyakit**: Retikel Merah untuk penyakit primer, Oranye untuk diferensial kedua, dan Hijau Zamrud untuk validasi daun sehat.

4. **Buku Panduan Penggunaan Web di Sidebar**:
   - Panduan interaktif ringkas, padat, dan jelas mengenai cara pengambilan foto yang benar (jarak 10-20 cm, fokus helai daun).
   - Penjelasan teknis pengaturan ambang keyakinan (Confidence Threshold) dan sensitivitas kanopi daun.
   - Tata cara membaca retikel HUD scanner dan manajemen penyimpanan riwayat per butir.

5. **Petunjuk Obat & Perawatan dari Dokter Tanaman (3 Kartu Terstruktur)**:
   - **Kartu 1 - Tindakan Langsung di Kebun**: Prosedur darurat pemangkasan presisi dan sanitasi 24 jam pertama.
   - **Kartu 2 - Rekomendasi Obat Semprot**: Bahan aktif resmi Balitsa/Kementan (Mankozeb, Klorotalonil, Difenokonazol, Tiram, dll.), takaran sendok per tangki 16 Liter, dan waktu semprot optimal.
   - **Kartu 3 - Perawatan Lahan, Pupuk & Agens Hayati**: Tata kelola air parit macak-macak, penghentian pupuk Nitrogen sukulen, aplikasi Kalsium-Silika penguat kutikula, dan pemanfaatan jamur *Trichoderma*.

6. **Filter Gatekeeper & Sensitivitas Adaptif**:
   - Menerima foto daun bawang merah asli meskipun dipegang tangan manusia (sensitivitas minimal 2%-3%).
   - Menolak objek yang bukan tanaman bawang merah secara ramah dan edukatif.
   - Batas ukuran upload: maksimal **5 MB** (format: JPG, JPEG, PNG, WEBP).

7. **Riwayat Pemeriksaan & Penyimpanan Mandiri**:
   - Pengguna bebas memilih kapan menyimpan diagnosa dengan tombol `💾 Simpan Hasil ke Riwayat`.
   - Riwayat dapat dipantau dan dihapus per butir (`🗑️`) langsung di sidebar.

---

## 📋 Daftar 7 Kategori Deteksi Model

1. **Downy mildew (Embun Bulu / Peronospora destructor)**:
   - Gejala: Daun layu pucat dengan lapisan beludru halus keunguan/kelabu saat udara lembap.
2. **Sehat (*Allium cepa*)**:
   - Gejala: Daun tegak kokoh, hijau segar merata, berlilin alami, tanpa bercak nekrotik maupun kelainan bentuk.
3. **Iris Yellow Spot Virus (IYSV)** — Virus Iris Kuning:
   - Gejala: Bercak klorotik berbentuk ketupat kuning jerami di tengah helai daun, terkadang memiliki pulau hijau (green islands).
4. **Hawar Daun (Stemphylium / Colletotrichum)**:
   - Gejala: Ujung daun menguning kecokelatan kering merambat ke bawah, atau bercak melekuk kebasahan pada helai daun.
5. **Moler / Layu Fusarium (*Fusarium oxysporum*)**:
   - Gejala: Daun meliuk-liuk memutar abnormal (spiral/inul), tanaman layu cepat dan pangkal batang melunak.
6. **Bercak Ungu / Trotol (*Alternaria porri*)**:
   - Gejala: Bercak melekuk ke dalam berbentuk cincin konsentris bertepung keunguan/gelap di tengah helai daun.
7. **Rust (Karat Daun / *Puccinia allii*)**:
   - Gejala: Bintil-bintil kecil menonjol (pustula) warna oranye terang hingga merah bata kecokelatan mirip karat besi.

---

## 📦 Instalasi & Persiapan

Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

Atau instal paket yang dibutuhkan secara manual:
```bash
pip install streamlit torch pillow numpy pandas requests groq
```

---

## 🏃 Cara Menjalankan Aplikasi

1. Pastikan file berikut berada di folder proyek yang sama:
   - `app.py` — Aplikasi Streamlit utama
   - `bawang_efficientnet_b0_ts.pt` — Model EfficientNet-B0 TorchScript
   - `meta.json` — Konfigurasi kelas, ukuran gambar, mean/std, temperature, dan threshold
2. Jalankan perintah:
```bash
streamlit run app.py
```
3. Buka peramban di alamat: `http://localhost:8501`.

> ⚠️ **Catatan**: Hasil prediksi hanya alat bantu, bukan diagnosis final.
