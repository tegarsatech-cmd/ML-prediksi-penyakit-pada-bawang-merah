# 🧅 AgroScan - Pendeteksi Penyakit Daun Bawang Merah (Streamlit & Keras DL)

Aplikasi web interaktif modern untuk klasifikasi dan deteksi dini penyakit daun tanaman bawang merah berbasis Deep Learning Keras dan Streamlit.

## 🚀 Fitur Unggulan

1. **Dual Input Fleksibel**:
   - Unggah berkas gambar (JPG, JPEG, PNG, WEBP).
   - Ambil foto secara instan menggunakan kamera perangkat (`st.camera_input`).

2. **Filter Out-of-Distribution (OOD)**:
   - Confidence threshold filter (default 55%). Jika citra buram, bukan daun bawang, atau probabilitas model rendah, aplikasi menampilkan peringatan ramah dan menolak vonis salah.

3. **Visualisasi Interaktif & Edukasi Petani**:
   - Status visual: Badge Hijau untuk daun sehat (`Healthy leaves`), Merah/Oranye untuk penyakit atau hama.
   - Distribusi probabilitas Top-3 dengan progress bar interaktif.
   - Rekomendasi penanganan lengkap: Gejala klinis, pencegahan budidaya, pengendalian kimiawi/hayati, serta saran cepat.

4. **Penyimpanan Riwayat Sisi Browser (Tanpa Database Eksternal)**:
   - Disimpan di memori browser pengguna (`session_state` & LocalStorage support).
   - Tombol unduh riwayat ke format **CSV** dan **JSON**.
   - Tombol kosongkan semua riwayat & fitur impor riwayat JSON.

5. **Ensiklopedia Penyakit Bawang Merah**:
   - Database lengkap: Bercak Ungu (*Alternaria porri*), Embun Bulu (*Peronospora destructor*), Hawar Daun (*Stemphylium vesicarium*), Karat Daun (*Puccinia allii*), Layu Fusarium / Moler (*Fusarium oxysporum*), Ulat Grayak (*Spodoptera exigua*), dan Daun Sehat.

6. **Optimasi & Resiliensi**:
   - Model Keras dan file JSON label dimuat efisien menggunakan `@st.cache_resource`.
   - Error handling ramah: Dilengkapi mode pratinjau/simulasi jika file `model_bawang_final.keras` belum diletakkan di folder.

---

## 📦 Instalasi & Persiapan

Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

Atau instal pustaka secara individual:
```bash
pip install streamlit tensorflow pillow numpy pandas
```

---

## 🏃 Cara Menjalankan Aplikasi

1. Pastikan file berikut berada di folder yang sama:
   - `app.py`
   - `model_bawang_final.keras` (file model Keras Anda)
   - `class_names.json` (daftar label kelas)
2. Jalankan perintah:
```bash
streamlit run app.py
```
3. Buka peramban di alamat `http://localhost:8501`.
