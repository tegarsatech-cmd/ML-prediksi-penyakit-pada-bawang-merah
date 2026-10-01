"""
AgroScan Bawang Merah - Aplikasi Deteksi Penyakit Daun Bawang Merah
Desain UI/UX Mobile-First Ramah Petani (Usia 30-50 Tahun di Lapangan/Sawah)
Berbasis Deep Learning MobileNetV2 Keras ('model_bawang_final.keras') & Groq AI.
"""

import os
import re
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. KONFIGURASI HALAMAN MOBILE-FIRST & RAMAH PETANI
# ==============================================================================
st.set_page_config(
    page_title="AgroScan - Dokter Daun Bawang Merah",
    page_icon="🧅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling (CSS) - Kontras Tinggi Maksimal Ramah Petani di Bawah Terik Matahari
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&display=swap');
    
    /* 1. Paksa Latar Belakang Seluruh Layar Menjadi Terang Bersih (Anti-Gelap) */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
        background-color: #F4F6F9 !important;
        color: #0F172A !important;
    }

    /* Responsif Menyeluruh untuk Segala Perangkat (HP 360px-430px, Tablet 768px, Laptop 1080p+) */
    .main .block-container {
        width: 100% !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        transition: all 0.2s ease-in-out;
    }

    /* Layar Ponsel Cerdas (di bawah 640px) */
    @media (max-width: 640px) {
        .main .block-container {
            max-width: 100% !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .farmer-hero {
            padding: 1.3rem 1rem !important;
            border-radius: 16px !important;
        }
        .farmer-hero h1 {
            font-size: 1.65rem !important;
        }
        .farmer-hero p {
            font-size: 0.98rem !important;
        }
        .status-disease-name {
            font-size: 1.55rem !important;
        }
        div.stButton > button {
            min-height: 54px !important;
            font-size: 1.1rem !important;
        }
        .card-ai-step {
            padding: 1.15rem 1.15rem !important;
        }
    }

    /* Layar Tablet & iPad (641px - 1024px) */
    @media (min-width: 641px) and (max-width: 1024px) {
        .main .block-container {
            max-width: 760px !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        .farmer-hero h1 {
            font-size: 1.9rem !important;
        }
        div.stButton > button {
            min-height: 58px !important;
        }
    }

    /* Layar Laptop & Desktop Monitor (> 1024px) */
    @media (min-width: 1025px) {
        .main .block-container {
            max-width: 840px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .farmer-hero h1 {
            font-size: 2.15rem !important;
        }
        div.stButton > button {
            min-height: 60px !important;
        }
    }

    /* 2. Standar Tipografi Kontras Tinggi (Minimal 18px) */
    html, body, .stApp, .main {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    p, label, input, textarea {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        color: #0F172A;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Header Banner Utama Ramah Petani */
    .farmer-hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        border-radius: 20px;
        padding: 1.6rem 1.3rem;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -3px rgba(27, 94, 32, 0.4);
        border: 2px solid #14532D;
    }
    .farmer-hero h1 {
        font-size: 2rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }
    .farmer-hero p {
        font-size: 1.1rem !important;
        color: #E8F5E9 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0 !important;
        font-weight: 600 !important;
    }

    /* Judul Langkah (Step Header) Putih Berbingkai Hijau */
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-left: 8px solid #1B5E20 !important;
        padding: 0.9rem 1.1rem;
        border-radius: 14px;
        margin: 1.8rem 0 1rem 0;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }
    .step-num {
        background-color: #1B5E20 !important;
        color: #FFFFFF !important;
        font-weight: 900;
        font-size: 1.25rem;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .step-title {
        font-size: 1.3rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
        margin: 0;
    }

    /* 3. Tombol Ramah Jempol (Thumb-Friendly Buttons) */
    div.stButton > button {
        min-height: 56px !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.2rem !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Tombol Utama Periksa Daun (Hijau Daun Tua Tegas #1B5E20 - Teks Putih Terang) */
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #1B5E20 !important;
        color: #FFFFFF !important;
        border: 2px solid #14532D !important;
        font-size: 1.25rem !important;
        min-height: 60px !important;
        box-shadow: 0 6px 14px -2px rgba(27, 94, 32, 0.4) !important;
    }
    div.stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #14532D !important;
    }

    /* Tombol Standar / Sekunder (Latar Putih, Border Gelap, Teks Hitam) */
    div.stButton > button[kind="secondary"], div.stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #64748B !important;
    }
    div.stButton > button[kind="secondary"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Tombol Alternatif Obat (Biru Segar Tegas - Teks Putih Terang) */
    .btn-alt-obat div.stButton > button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: 2px solid #0369A1 !important;
        font-size: 1.15rem !important;
        min-height: 56px !important;
        box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3) !important;
    }
    .btn-alt-obat div.stButton > button * {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    .btn-alt-obat div.stButton > button:hover {
        background-color: #0369A1 !important;
    }

    /* 4. Tab Navigasi Kamera & Galeri (Kontras Jelas) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 6px !important;
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1B5E20 !important;
        font-weight: 900 !important;
        box-shadow: 0 4px 8px -2px rgba(0, 0, 0, 0.12) !important;
    }
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    /* 5. Kartu Status Diagnosa Kontras Tinggi */
    .status-card-healthy {
        background-color: #E8F5E9 !important;
        border: 3px solid #2E7D32 !important;
        border-radius: 18px !important;
        padding: 1.4rem !important;
        margin: 1.2rem 0 !important;
        box-shadow: 0 6px 14px rgba(46, 125, 50, 0.15) !important;
    }
    .status-card-healthy * {
        color: #1B5E20 !important;
    }
    .status-card-healthy .status-disease-name {
        color: #1B5E20 !important;
        font-size: 1.85rem !important;
        font-weight: 900 !important;
        margin: 0.3rem 0;
    }

    .status-card-disease {
        background-color: #FFEBEE !important;
        border: 3px solid #C62828 !important;
        border-radius: 18px !important;
        padding: 1.4rem !important;
        margin: 1.2rem 0 !important;
        box-shadow: 0 6px 14px rgba(198, 40, 40, 0.15) !important;
    }
    .status-card-disease * {
        color: #B71C1C !important;
    }
    .status-card-disease .status-disease-name {
        color: #B71C1C !important;
        font-size: 1.85rem !important;
        font-weight: 900 !important;
        margin: 0.3rem 0;
    }

    .status-card-pest {
        background-color: #FFF3E0 !important;
        border: 3px solid #E65100 !important;
        border-radius: 18px !important;
        padding: 1.4rem !important;
        margin: 1.2rem 0 !important;
        box-shadow: 0 6px 14px rgba(230, 81, 0, 0.15) !important;
    }
    .status-card-pest * {
        color: #BF360C !important;
    }
    .status-card-pest .status-disease-name {
        color: #BF360C !important;
        font-size: 1.85rem !important;
        font-weight: 900 !important;
        margin: 0.3rem 0;
    }

    .status-badge {
        display: inline-block !important;
        padding: 0.45rem 1.1rem !important;
        border-radius: 10px !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 0.6rem !important;
        color: #FFFFFF !important;
    }
    .badge-healthy-tag { background-color: #2E7D32 !important; color: #FFFFFF !important; }
    .badge-disease-tag { background-color: #C62828 !important; color: #FFFFFF !important; }
    .badge-pest-tag { background-color: #E65100 !important; color: #FFFFFF !important; }

    .status-confidence-text {
        font-size: 1.3rem !important;
        font-weight: 800 !important;
    }

    /* 6. 3 Kartu Panduan Dokter Tanaman (Putih Bersih dengan Border Tebal & Teks Gelap) */
    .card-ai-step {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 1.3rem 1.4rem !important;
        margin-bottom: 1.3rem !important;
        border: 2px solid #CBD5E1 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
    }
    .card-ai-red {
        border-left: 9px solid #DC2626 !important;
    }
    .card-ai-red .card-ai-title {
        color: #B91C1C !important;
        font-size: 1.35rem !important;
        font-weight: 900 !important;
    }
    .card-ai-blue {
        border-left: 9px solid #2563EB !important;
    }
    .card-ai-blue .card-ai-title {
        color: #1D4ED8 !important;
        font-size: 1.35rem !important;
        font-weight: 900 !important;
    }
    .card-ai-green {
        border-left: 9px solid #16A34A !important;
    }
    .card-ai-green .card-ai-title {
        color: #15803D !important;
        font-size: 1.35rem !important;
        font-weight: 900 !important;
    }

    .card-ai-sub {
        font-size: 0.98rem !important;
        color: #4B5563 !important;
        margin-bottom: 0.75rem !important;
        font-weight: 700 !important;
    }
    .card-ai-body {
        font-size: 1.1rem !important;
        line-height: 1.75 !important;
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    .card-ai-body p {
        margin: 0.5rem 0 !important;
        color: #0F172A !important;
        line-height: 1.75 !important;
    }
    .card-ai-body ul, .card-ai-body ol {
        margin: 0.5rem 0 !important;
        padding-left: 1.35rem !important;
    }
    .card-ai-body li {
        margin-bottom: 0.45rem !important;
        color: #0F172A !important;
        line-height: 1.7 !important;
    }
    .card-ai-body strong {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* 7. Input File & Kamera */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #64748B !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"] * {
        color: #0F172A !important;
    }
    [data-testid="stCameraInput"] {
        background-color: #FFFFFF !important;
        border: 2px solid #94A3B8 !important;
        border-radius: 14px !important;
        padding: 0.8rem !important;
    }
    [data-testid="stCameraInput"] * {
        color: #0F172A !important;
    }

    /* 8. Expander & Alert */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 14px !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        padding: 0.75rem 1rem !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary * {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        padding: 1rem !important;
    }
    [data-testid="stExpanderDetails"] * {
        color: #0F172A !important;
    }

    /* Sidebar Terang Berbatas Tegas */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* 9. Lindungi Font Icon Bawaan Streamlit agar TIDAK Berubah Jadi Teks Bertabrakan */
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-symbols-outlined, 
    [class*="material-symbols"], 
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        vertical-align: middle !important;
    }

    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }

    /* 10. Kartu Validasi Penolakan (Foto Bukan Daun Bawang Merah) */
    .card-rejection {
        background-color: #FFFBEB !important;
        border: 3px solid #F59E0B !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 18px rgba(245, 158, 11, 0.2) !important;
    }
    .card-rejection-badge {
        display: inline-block !important;
        background-color: #D97706 !important;
        color: #FFFFFF !important;
        font-weight: 900 !important;
        padding: 0.4rem 1rem !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.6rem !important;
    }
    .card-rejection-title {
        color: #92400E !important;
        font-size: 1.65rem !important;
        font-weight: 900 !important;
        margin: 0.3rem 0 0.6rem 0 !important;
    }
    .card-rejection-reason {
        background-color: #FEF3C7 !important;
        border-left: 6px solid #D97706 !important;
        padding: 0.85rem 1.1rem !important;
        border-radius: 8px !important;
        color: #78350F !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        margin-bottom: 1rem !important;
    }
    .card-rejection-desc {
        color: #451A03 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }
    .card-rejection-desc ul, .card-rejection-desc ol {
        margin-top: 0.5rem !important;
        padding-left: 1.4rem !important;
    }
    .card-rejection-desc li {
        margin-bottom: 0.4rem !important;
        color: #451A03 !important;
        font-size: 1.05rem !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. ENSIKLOPEDIA & METADATA PENYAKIT (15 KELAS MODEL KERAS)
# Menggunakan Istilah Populer yang Akrab Bagi Petani Indonesia
# ==============================================================================
CLASS_METADATA = {
    "Alternaria_D": {
        "nama_id": "Bercak Ungu / Trotol",
        "latin": "Alternaria porri",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bercak melekuk ke dalam berbentuk cincin konsentris bertepung keunguan/gelap di tengah helai daun, tepi menguning kering.",
        "gejala": "Potong atau pangkas daun yang terdapat bercak trotol ungu. Kumpulkan dan bakar di luar areal sawah agar spora jamur tidak terbang tertiup angin ke tanaman lain.",
        "pencegahan": "Bersihkan gulma dan rumput liar di sekitar parit. Buat bedengan lebih tinggi agar tidak tergenang air saat hujan lebat.",
        "solusi": "Semprot fungisida berbahan aktif Difenokonazol atau Mankozeb. Semprot pada pagi hari (pukul 06.00 - 08.30) atau sore (pukul 16.00) saat angin tenang.",
        "rekomendasi_singkat": "Semprot Difenokonazol/Mankozeb dan pangkas daun trotol segera."
    },
    "Botrytis Leaf Blight": {
        "nama_id": "Hawar Daun / Bintik Putih",
        "latin": "Botrytis squamosa",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bintik-bintik putih kecil (1-2 mm) melekuk di helai daun, ujung daun memutih kering seperti terbakar tanpa lendir.",
        "gejala": "Pangkas ujung-ujung daun yang memutih atau mengering. Jangan biarkan sisa potongan membusuk di atas bedengan.",
        "pencegahan": "Hindari menyiram daun di sore/malam hari agar daun tidak basah semalaman. Perlebar jarak tanam agar angin leluasa masuk.",
        "solusi": "Semprot fungisida berbahan aktif Klorotalonil atau Fluazinam secara merata di permukaan dan sela daun.",
        "rekomendasi_singkat": "Aplikasi fungisida Klorotalonil dan hentikan penyiraman sore/malam."
    },
    "Bulb Rot": {
        "nama_id": "Busuk Umbi / Basah",
        "latin": "Aspergillus / Fusarium spp.",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Umbi lembek berair berbau busuk menyengat saat dipijit, lapisan kulit luar terkelupas busuk basah.",
        "gejala": "Segera cabut rumpun bawang yang umbinya lembek dan berbau busuk. Keluarkan dari lahan dan jangan buang di saluran air irigasi.",
        "pencegahan": "Gunakan bibit umbi yang benar-benar kering leher. Taburkan kapur pertanian (dolomit) jika tanah terlalu masam atau lembap becek.",
        "solusi": "Kocorkan fungisida Tembaga Oksiklorida di lubang tanam bekas cabutan dan taburkan agens hayati Trichoderma.",
        "rekomendasi_singkat": "Cabut & musnahkan umbi busuk, taburkan kapur dolomit di bedengan."
    },
    "Bulb_blight-D": {
        "nama_id": "Hawar Leher Umbi",
        "latin": "Blight pathogen",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Pangkal leher umbi melembek kecokelatan di perbatasan tanah, daun atas rebah lunglai.",
        "gejala": "Cabut tanaman yang pangkal lehernya melembek kecokelatan sebelum busuk merambat ke umbi tetangga.",
        "pencegahan": "Pastikan saluran drainase parit lancar dan tidak ada air menggenang di sela bedengan.",
        "solusi": "Kocorkan fungisida berbahan aktif Mankozeb atau Tembaga Oksiklorida pada pangkal batang tanaman yang masih sehat.",
        "rekomendasi_singkat": "Kocorkan Tembaga Oksiklorida di leher umbi dan keringkan parit bedengan."
    },
    "Caterpillar-P": {
        "nama_id": "Ulat Grayak",
        "latin": "Spodoptera exigua",
        "status": "pest",
        "is_healthy": False,
        "ciri_lapangan": "Helai daun transparan tipis berlubang dari dalam tabung daun, terdapat kotoran ulat hijau/hitam di sela daun.",
        "gejala": "Pencet atau kutip langsung kelompok ulat dan telur ulat di dalam tabung daun pada pagi hari saat ulat mulai aktif keluar.",
        "pencegahan": "Pasang lampu perangkap (light trap) atau feromon sex trap di sudut sawah untuk membasmi kupu-kupu ngengat sebelum bertelur.",
        "solusi": "Semprot insektisida berbahan aktif Emamektin Benzoat atau Klorantraniliprol pada sore menjelang petang hari saat ulat keluar makan.",
        "rekomendasi_singkat": "Kutip ulat secara manual dan semprot Emamektin Benzoat pada sore hari."
    },
    "Downy mildew": {
        "nama_id": "Embun Bulu / Palsu",
        "latin": "Peronospora destructor",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Permukaan daun dilapisi bulu/kapang halus keputihan hingga kelabu keunguan di pagi hari saat lembap berkabut.",
        "gejala": "Petik daun yang berbulu halus keputihan di pagi hari. Masukkan ke dalam kantong kresek tertutup dan bawa keluar dari areal sawah.",
        "pencegahan": "Hentikan pupuk Urea/Nitrogen berlebih selama musim penghujan. Berikan pupuk Kalium dan Silika untuk mempertebal kulit daun.",
        "solusi": "Semprot fungisida sistemik berbahan aktif Dimetomorf atau Simoksanil selang-seling dengan Mankozeb setiap 3-4 hari sekali.",
        "rekomendasi_singkat": "Semprot Dimetomorf/Simoksanil dan kurangi pemakaian pupuk Nitrogen/Urea."
    },
    "Fusarium-D": {
        "nama_id": "Layu Moler / Fusarium",
        "latin": "Fusarium oxysporum",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Daun melintir-lintir abnormal (moler), menguning pucat dari ujung, perakaran membusuk dan mudah dicabut.",
        "gejala": "Segera cabut tanaman yang daunnya melintir abnormal (moler) sampai ke perakarannya agar jamur tidak menular lewat parit.",
        "pencegahan": "Campurkan agens hayati Trichoderma dengan pupuk kandang matang saat olah tanah dasar bedengan.",
        "solusi": "Taburkan kapur dolomit pada lubang bekas cabutan dan semprot fungisida sistemik berbahan aktif Benomil atau Mankozeb.",
        "rekomendasi_singkat": "Cabut tanaman melintir/moler, taburkan dolomit & agens Trichoderma."
    },
    "Healthy leaves": {
        "nama_id": "Daun Sehat & Segar",
        "latin": "Kondisi Normal",
        "status": "healthy",
        "is_healthy": True,
        "ciri_lapangan": "Daun hijau segar mengkilap, tegak berdiri kokoh tanpa bercak berlendir, luka gigitan, maupun tepung jamur.",
        "gejala": "Kondisi tanaman sangat baik! Daun hijau segar, tegak berdiri kokoh tanpa bercak jamur atau luka gigitan hama.",
        "pencegahan": "Lanjutkan pemantauan rutin 2-3 hari sekali. Pertahankan sanitasi gulma di pematang sawah.",
        "solusi": "Tidak memerlukan obat semprot kimia kuratif. Cukup semprotkan pupuk daun mikro dan asam amino untuk menjaga kesegaran.",
        "rekomendasi_singkat": "Tanaman sehat optimal! Cukup lanjutkan pemupukan berimbang dan pengairan rutin."
    },
    "Iris yellow virus_augment": {
        "nama_id": "Virus Iris Kuning (IYSV)",
        "latin": "Iris yellow spot virus",
        "status": "virus",
        "is_healthy": False,
        "ciri_lapangan": "Bercak klorotik khas berbentuk ketupat/belah ketupat warna kuning jerami di tengah helai daun.",
        "gejala": "Cabut dan musnahkan tanaman yang daunnya terdapat bercak kuning berbentuk ketupat agar tidak menjadi sumber virus di kebun.",
        "pencegahan": "Gunakan mulsa plastik perak untuk memantulkan sinar matahari dan menghalau datangnya hama trips pembawa virus.",
        "solusi": "Kendalikan serangga kutu trips penyebar virus dengan insektisida berbahan aktif Abamektin atau Spinetoram pada pagi hari.",
        "rekomendasi_singkat": "Cabut tanaman bergejala ketupat & semprot Abamektin untuk basmi kutu trips."
    },
    "Purple blotch": {
        "nama_id": "Bercak Ungu / Trotol",
        "latin": "Alternaria porri",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bercak trotol keunguan melekuk dengan cincin konsentris gelap dan tepi berwarna kuning klorotik, kering bertepung spora.",
        "gejala": "Potong helai daun yang terdapat cincin bercak ungu melekuk sebelum bercak melebar dan menyebabkan daun patah terkulai.",
        "pencegahan": "Jaga kelancaran parit bedengan, jangan biarkan air hujan menggenangi sela tanaman bawang.",
        "solusi": "Semprot fungisida berbahan aktif Difenokonazol, Propineb, atau Azoksistrobin secara bergiliran pada pagi hari.",
        "rekomendasi_singkat": "Aplikasi fungisida Difenokonazol/Azoksistrobin dan potong daun yang bertrotol."
    },
    "Rust": {
        "nama_id": "Karat Daun (Bintil Merah)",
        "latin": "Puccinia allii",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bintil-bintil melepuh berisi serbuk/tepung karat berwarna oranye kemerahan seperti serbuk besi berkarat.",
        "gejala": "Pangkas daun yang dipenuhi bintil debu karat warna oranye kemerahan. Masukkan ke wadah tertutup saat memotong agar serbuk karat tidak berhamburan.",
        "pencegahan": "Lakukan rotasi pergiliran tanaman dengan jagung atau palawija setelah panen bawang merah.",
        "solusi": "Semprot fungisida berbahan aktif Tebukonazol, Heksakonazol, atau Difenokonazol saat embun pagi mulai mengering (sekitar pukul 08.00).",
        "rekomendasi_singkat": "Semprot fungisida Tebukonazol/Heksakonazol saat embun pagi mulai kering."
    },
    "Virosis-D": {
        "nama_id": "Virus Kuning Melintir",
        "latin": "Onion yellow dwarf virus",
        "status": "virus",
        "is_healthy": False,
        "ciri_lapangan": "Tanaman kerdil, helai daun belang bergaris-garis kuning kusam, berkerut kaku dan rapuh bila ditekuk.",
        "gejala": "Segera cabut rumpun tanaman yang kerdil dan daunnya belang kuning berkerut. Jangan biarkan tetap tumbuh di bedengan.",
        "pencegahan": "Gunakan bibit umbi sehat bersertifikat yang terbebas dari infeksi virus bawaan.",
        "solusi": "Semprot insektisida berbahan aktif Imidakloprid untuk menekan populasi kutu daun (aphid) yang menularkan virus.",
        "rekomendasi_singkat": "Cabut tanaman kerdil kuning & semprot insektisida pembasmi kutu daun."
    },
    "Xanthomonas Leaf Blight": {
        "nama_id": "Hawar Daun Bakteri (Xanthomonas)",
        "latin": "Xanthomonas axonopodis",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bercak kebasah-basahan (water-soaked) seperti tersiram air mendidih, berlendir saat pagi lembap, bau langu/busuk, tidak bertepung spora.",
        "gejala": "Potong helai daun yang tampak berlendir kebasah-basahan seperti tersiram air panas sebelum mengering hangus.",
        "pencegahan": "Hindari menyiram daun menggunakan semprotan bertekanan kencang yang dapat memercikkan air bakteri ke daun sehat.",
        "solusi": "Semprot bakterisida/fungisida tembaga berbahan aktif Tembaga Hidroksida atau Kasugamisin pada pagi hari saat cuaca cerah.",
        "rekomendasi_singkat": "Semprot bakterisida Tembaga Hidroksida / Kasugamisin dan hindari percikan air."
    },
    "onion1": {
        "nama_id": "Daun Sehat & Segar",
        "latin": "Kondisi Normal",
        "status": "healthy",
        "is_healthy": True,
        "ciri_lapangan": "Daun bawang hijau mulus dan tegak berdiri tanpa tanda bercak basah maupun luka gigitan.",
        "gejala": "Daun bawang hijau mulus dan tegak berdiri tanpa tanda bercak penyakit maupun serangan ulat.",
        "pencegahan": "Pertahankan pasokan air yang cukup di parit tanpa membuat bedengan terlalu becek.",
        "solusi": "Tidak perlu obat kimia/fungisida semprot. Cukup berikan pupuk NPK dan pupuk daun mikro sesuai jadwal.",
        "rekomendasi_singkat": "Tanaman bawang sehat dan prima. Teruskan perawatan rutin."
    },
    "stemphylium Leaf Blight": {
        "nama_id": "Hawar Daun Kering Ujung (Stemphylium)",
        "latin": "Stemphylium vesicarium",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Ujung daun menguning kecokelatan kering memanjang ke bawah, terdapat bintik hitam kecil spora jamur saat kering.",
        "gejala": "Pangkas ujung daun yang menguning kecokelatan seperti terbakar. Buang sisa daun yang dipangkas keluar dari lahan.",
        "pencegahan": "Semprotkan pupuk Kalium (KNO3 putih) dan pupuk Silika cair untuk memperkokoh dinding sel helai daun.",
        "solusi": "Semprot fungisida berbahan aktif Iprodion, Klorotalonil, atau Tebukonazol secara bergiliran tiap 4-5 hari sekali.",
        "rekomendasi_singkat": "Semprot Klorotalonil/Iprodion dan berikan pupuk Kalium serta Silika."
    }
}

# Inisialisasi Sesi Penyimpanan
if "history" not in st.session_state:
    st.session_state.history = []

def save_diagnosis_to_history(disease_code, display_name, confidence, recommendation, is_healthy, notes="", location=""):
    """Menyimpan entri riwayat diagnosa baru."""
    now_time = datetime.now().strftime("%H:%M:%S")
    entry_id = int(time.time() * 1000)
    entry = {
        "id": entry_id,
        "waktu": now_time,
        "penyakit": display_name,
        "nama_penyakit": display_name,
        "confidence": f"{confidence:.1f}%",
        "keyakinan": f"{confidence:.1f}%",
        "confidence_val": round(confidence, 2),
        "status": "Healthy / Sehat" if is_healthy else "Penyakit / Hama",
        "kelas_model": disease_code,
        "lokasi": location if location.strip() else "Kebun Bawang",
        "catatan": notes if notes.strip() else "-",
        "rekomendasi": recommendation
    }
    st.session_state['history'].append(entry)
    if len(st.session_state['history']) > 100:
        st.session_state['history'] = st.session_state['history'][-100:]
    return entry

def reset_all_history():
    """Mengosongkan semua entri riwayat."""
    st.session_state.history = []

# ==============================================================================
# 3. CACHING MODEL KERAS & INFERENSI PRESISI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_bawang_final.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

@st.cache_resource(show_spinner=False)
def load_labels(file_path=CLASS_NAMES_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File label kelas '{file_path}' tidak ditemukan.")
    with open(file_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return class_names

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    class_names = load_labels(CLASS_NAMES_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model '{MODEL_PATH}' tidak ditemukan.")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, class_names

def preprocess_image_smart(image: Image.Image, target_size=(224, 224), norm_mode: str = "mobilenet_v2"):
    """
    Modul Perbaikan Pipeline Preprocessing (Fix Input Tensor)
    dan Uji Coba Mode Normalisasi Piksel (A/B Testing Preprocessing):
    1. EXIF Transpose: Mengoreksi rotasi orientasi dari kamera smartphone iOS/Android.
    2. Paksa ke RGB: Hapus channel Alpha/transparansi jika format PNG/WA.
    3. Resize presisi 224x224.
    4. Ubah ke array NumPy float32 dalam skala [0, 255] (tf.keras.preprocessing.image.img_to_array).
    5. Tambah dimensi batch (1, 224, 224, 3).
    6. Uji Coba Normalisasi Piksel:
       - Mode 1: tf.keras.applications.mobilenet_v2.preprocess_input (rentang [-1, 1])
       - Mode 2: Pembagian skala standar img_array / 255.0 (rentang [0, 1])
    """
    import tensorflow as tf
    orig_mode = image.mode if image is not None else "RGB"
    orig_size = image.size if image is not None else (0, 0)

    # Normalisasi orientasi EXIF (kamera smartphone)
    cropped_img = ImageOps.exif_transpose(image) if image is not None else image

    # 1. Paksa ke RGB (hapus channel Alpha/transparansi jika format PNG/WA)
    img_clean = cropped_img.convert("RGB")

    # 2. Resize presisi 224x224
    img_resized = img_clean.resize(target_size, Image.Resampling.BILINEAR)

    # 3. Ubah ke array NumPy float32 dalam skala [0, 255]
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized, dtype="float32")

    # 4. Tambah dimensi batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 5. Normalisasi sesuai mode A/B Testing
    if norm_mode == "rescaling_255":
        img_final = img_batch / 255.0
        mode_label = "Mode 2: img_array / 255.0 (Rentang [0, 1])"
    else:
        img_final = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        mode_label = "Mode 1: mobilenet_v2.preprocess_input (Rentang [-1, 1])"

    # Catat statistik diagnostik piksel untuk panel audit
    diag_info = {
        "orig_mode": orig_mode,
        "orig_size": orig_size,
        "norm_mode_name": mode_label,
        "min_pixel": float(np.min(img_final)),
        "max_pixel": float(np.max(img_final))
    }
    return img_final, img_resized, diag_info

def check_shallot_leaf_mask(image: Image.Image, min_ratio: float = 0.12) -> tuple[bool, str, float]:
    """
    Validasi Citra Daun Bawang Merah sebelum masuk ke MobileNetV2 (Pre-Inference Guard):
    Memeriksa spektrum kromatisitas jaringan tanaman bawang merah (Allium cepa)
    menggunakan analisis HSV dan perbandingan kanal RGB.
    Mencegah input non-tanaman: wajah, tangan, tanah polos, dinding, pakaian, kendaraan, hewan.
    """
    try:
        thumb = image.convert("RGB").resize((160, 160))
        arr = np.array(thumb, dtype="float32")
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        delta_safe = np.where(delta == 0, 1.0, delta)
        
        h = np.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0)
        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6.0)
        mask_g = (cmax == g) & (delta > 0)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta_safe[mask_g]) + 2.0)
        mask_b = (cmax == b) & (delta > 0)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta_safe[mask_b]) + 4.0)
        
        cmax_safe = np.where(cmax == 0, 1.0, cmax)
        s = np.where(cmax == 0, 0.0, delta / cmax_safe)
        v = cmax / 255.0
        
        # 1. Daun hijau sehat / bergejala (Hue 42 - 165)
        is_green_leaf = (h >= 42.0) & (h <= 165.0) & (s >= 0.14) & (v >= 0.12)
        
        # 2. Daun menguning / ujung kering penyakit (Hue 35 - 42, G dominan)
        is_yellowing = (h >= 35.0) & (h < 42.0) & (g >= r * 0.82) & (g > b * 1.12) & (s >= 0.18)
        
        # 3. Umbi / selubung ungu kemerahan bawang merah (Hue 295 - 355)
        is_purple_bulb = (h >= 295.0) & (h <= 355.0) & (r > g * 1.25) & (b > g * 0.85) & (s >= 0.20)
        
        plant_mask = is_green_leaf | is_yellowing | is_purple_bulb
        plant_ratio = float(np.mean(plant_mask))
        
        if plant_ratio < min_ratio:
            return False, f"Rasio warna daun bawang merah hanya {plant_ratio*100:.1f}% (minimal {min_ratio*100:.0f}%).", plant_ratio
            
        return True, "Valid", plant_ratio
    except Exception as e:
        return True, f"Bypass: {e}", 1.0

def get_groq_api_key() -> str:
    """
    Sistem prioritas pembacaan API Key Groq:
    1. st.secrets["GROQ_API_KEY"] (Deployment Streamlit Cloud)
    2. os.getenv("GROQ_API_KEY") (Environment Variable lokal / server)
    3. Fallback default key jika belum disetel
    """
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            sec_key = st.secrets["GROQ_API_KEY"]
            if sec_key and str(sec_key).strip():
                return str(sec_key).strip().rstrip(".")
    except Exception:
        pass

    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip().rstrip(".")

    return ""

def validate_onion_image(image: Image.Image, api_key: str = None) -> tuple[bool, str]:
    """
    Sistem Validasi Guardrail Gatekeeper Citra menggunakan Groq Vision:
    Mencegah diagnosis foto non-tanaman bawang merah (manusia, hewan, kendaraan, tanah kosong, tanaman lain).
    Model: llama-3.2-11b-vision-preview (temperature=0.0, max_tokens=10).
    """
    import base64
    from io import BytesIO
    import requests
    
    if not api_key:
        api_key = get_groq_api_key()
        
    if not api_key:
        # Fallback spektrum lokal jika API Key belum tersedia
        is_plant, reason_text, _ = check_shallot_leaf_mask(image, min_ratio=0.12)
        if not is_plant:
            return False, "INVALID: Spektrum warna bukan daun bawang merah."
        return True, "VALID (Local Fallback)"

    try:
        # Resize thumbnail agar pengiriman cepat & hemat kuota
        thumb = image.copy()
        thumb.thumbnail((512, 512))
        buffered = BytesIO()
        thumb.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt_text = (
            "Anda adalah validator citra pertanian. Periksa gambar ini secara cermat.\n"
            "Apakah gambar ini menampilkan daun, umbi, atau bagian tanaman bawang merah (Allium cepa)?\n"
            "Jawab HANYA dengan satu kata: 'VALID' jika benar bawang merah/daun bawang merah, atau 'INVALID' jika bukan atau objek lain."
        )
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AgroScan-Validator/1.0"
        }
        payload = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            ans = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "VALID" in ans and "INVALID" not in ans:
                return True, "VALID"
            else:
                return False, "INVALID: Terdeteksi bukan daun/tanaman bawang merah."
        else:
            # Fallback jika model vision error atau decommissioned
            is_plant, reason_text, _ = check_shallot_leaf_mask(image, min_ratio=0.12)
            if not is_plant:
                return False, f"INVALID (Vision Status {resp.status_code}): Spektrum citra bukan daun bawang."
            return True, f"VALID (Fallback status {resp.status_code})"
    except Exception as err:
        # Fallback jaringan jika timeout
        is_plant, reason_text, _ = check_shallot_leaf_mask(image, min_ratio=0.12)
        if not is_plant:
            return False, "INVALID (Fallback Timeout): Spektrum citra bukan daun bawang."
        return True, f"VALID (Fallback: {err})"

# Alias untuk kompatibilitas
validate_with_groq_vision = validate_onion_image

def predict_disease(image: Image.Image, model, class_names, target_size=(224, 224), enforce_verification: bool = True, norm_mode: str = "mobilenet_v2"):
    """
    Fungsi Inferensi & Kalibrasi Probabilitas Pasca-Prediksi (Post-Processing Bias Penalty)
    untuk mengatasi bias prior model MobileNetV2 pada Xanthomonas Leaf Blight (Indeks 12).
    """
    # ==============================================================================
    # VALIDASI INPUT GAMBAR TEPAT SEBELUM PREDIKSI (OOD GUARD)
    # ==============================================================================
    if enforce_verification:
        is_shallot, reason_msg, ratio = check_shallot_leaf_mask(image, min_ratio=0.12)
        if not is_shallot:
            raise ValueError(f"OOD_GUARD_REJECTED: {reason_msg}")

    # Verifikasi Pipeline Preprocessing Citra (RGB murni, resize 224x224, skala [0, 255])
    input_tensor, processed_preview, diag_info = preprocess_image_smart(image, target_size, norm_mode=norm_mode)
    
    # 1. Inferensi Model Keras (ambil salinan probabilitas mentah)
    raw_preds = model.predict(input_tensor, verbose=0)
    if len(raw_preds.shape) == 2:
        raw_probs = raw_preds[0].copy()
    else:
        raw_probs = raw_preds.flatten().copy()

    # 2. Kalibrasi Probabilitas Pasca-Prediksi (Post-Processing Bias Penalty)
    # Reduksi bobot dominasi Xanthomonas Leaf Blight (Indeks 12) sebesar 45% (kalikan 0.55)
    idx_xanthomonas = class_names.index("Xanthomonas Leaf Blight") if "Xanthomonas Leaf Blight" in class_names else 12
    raw_probs[idx_xanthomonas] *= 0.55

    # Normalisasi ulang agar total probabilitas kembali tepat 1.0 (100%)
    calibrated_probs = raw_probs / np.sum(raw_probs)

    # Gunakan calibrated_probs sebagai satu-satunya rujukan untuk mengurutkan Top-N dan diagnosis akhir
    top_indices = np.argsort(calibrated_probs)[::-1]
    
    # Peringkat 1
    best_idx = int(top_indices[0])
    raw_class_name = class_names[best_idx]
    top_confidence = float(calibrated_probs[best_idx]) * 100.0

    # Peringkat 2
    second_idx = int(top_indices[1]) if len(top_indices) > 1 else best_idx
    second_class_name = class_names[second_idx]
    second_confidence = float(calibrated_probs[second_idx]) * 100.0

    # Margin selisih probabilitas peringkat 1 dan 2
    confidence_margin = top_confidence - second_confidence

    # 3. Aturan Ambang Batas Diferensial (Mencegah Vonis Tunggal Saat Ragu)
    # Aktifkan status Multidiagnosis jika:
    # - conf_2 >= 15.0%, ATAU
    # - conf_1 < 65.0%, ATAU
    # - Selisih (conf_1 - conf_2) < 25.0%
    is_healthy_1 = CLASS_METADATA.get(raw_class_name, {}).get("is_healthy", False) or CLASS_METADATA.get(raw_class_name, {}).get("status") == "healthy"
    is_healthy_2 = CLASS_METADATA.get(second_class_name, {}).get("is_healthy", False) or CLASS_METADATA.get(second_class_name, {}).get("status") == "healthy"

    is_differential = (
        ((second_confidence >= 15.0) or (top_confidence < 65.0) or (confidence_margin < 25.0))
        and (not is_healthy_1) 
        and (not is_healthy_2) 
        and (raw_class_name != second_class_name)
    )

    metadata = CLASS_METADATA.get(raw_class_name, {
        "nama_id": raw_class_name,
        "latin": "-",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Periksa kondisi helai daun dan bercak secara teliti.",
        "gejala": "Pangkas daun yang bergejala dan keluarkan dari areal kebun.",
        "pencegahan": "Jaga kelancaran parit dan kebersihan gulma bedengan.",
        "solusi": "Gunakan fungisida atau bakterisida sesuai rekomendasi petugas penyuluh.",
        "rekomendasi_singkat": "Pangkas daun sakit dan lakukan penyemprotan obat yang sesuai."
    })

    second_metadata = CLASS_METADATA.get(second_class_name, {
        "nama_id": second_class_name,
        "latin": "-",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Periksa kondisi helai daun dan bercak secara teliti.",
        "gejala": "Pangkas daun yang bergejala dan bersihkan bedengan.",
        "pencegahan": "Jaga drainase parit dan sanitasi pematang.",
        "solusi": "Gunakan obat yang sesuai.",
        "rekomendasi_singkat": "Lakukan sanitasi daun sakit."
    })

    return (
        calibrated_probs,
        top_indices,
        raw_class_name,
        top_confidence,
        metadata,
        second_class_name,
        second_confidence,
        second_metadata,
        is_differential,
        confidence_margin,
        processed_preview,
        diag_info
    )

# Alias untuk kompatibilitas fungsi lama
predict_image = predict_disease

def validate_onion_leaf(image: Image.Image, top_confidence: float, threshold: float = 40.0):
    """
    Validasi Citra Daun Bawang Merah:
    1. Filter Analisis Citra Digital Warna Tanaman (Hue/Chrominance Masking):
       Memeriksa apakah citra mengandung spektrum vegetasi/daun bawang merah (>= 12%).
    2. Filter Ambang Batas Keyakinan Model (OOD Rejection):
       Jika model 15 kelas memiliki top_confidence < threshold (bawaan 40%),
       objek dipastikan bukan bagian daun/umbi bawang merah yang dapat diidentifikasi.
    """
    # 1. Validasi Spektrum Warna Daun/Tanaman
    is_valid_mask, reason, ratio = check_shallot_leaf_mask(image, min_ratio=0.12)
    if not is_valid_mask:
        return False, f"Warna dan tekstur foto tidak terdeteksi sebagai daun bawang merah ({ratio*100:.1f}% spektrum daun terdeteksi)."

    # 2. Validasi Ambang Keyakinan Model
    if top_confidence < threshold:
        return False, (
            f"Tingkat kecocokan sangat rendah ({top_confidence:.1f}% di bawah batas {threshold}%). "
            "Pola objek tidak terdeteksi sebagai daun maupun umbi bawang merah."
        )

    return True, "Valid"

# ==============================================================================
# 4. KONSULTAN DOKTER TANAMAN (GROQ AI AGRO-ENGINE)
# Menghasilkan Solusi Lugas Bahasa Petani dalam 3 Bagian Jelas
# ==============================================================================
GROQ_API_KEY = get_groq_api_key()

FOCUS_ANGLES = [
    (
        "Pendekatan Kuratif Kilat & Rotasi Bahan Aktif (Rekomendasi Balitsa & PHT)",
        "fokus penyelamatan darurat di bedengan 24 jam pertama, kombinasi bahan aktif fungisida/bakterisida kontak dan sistemik dengan mekanisme kerja berbeda (FRAC code), takaran dosis sendok per tangki 16L, serta teknik semprot merata saat cuaca teduh."
    ),
    (
        "Pendekatan Manajemen Nutrisi & Penguat Dinding Sel Daun (Riset Balai Proteksi Tanaman)",
        "fokus pada pembenahan nutrisi: stop pupuk Nitrogen tunggal (Urea/ZA berlebih) yang memicu daun lunak dan mudah ditembus spora, diganti pupuk Kalium (KNO3 Putih/MKP), pupuk mikro Kalsium-Boron, dan Silika cair untuk mempertebal lapisan lilin daun."
    ),
    (
        "Pendekatan Pengelolaan Air, Bedengan & Agens Hayati (Riset BPTP Kementerian Pertanian)",
        "fokus pada pengelolaan air parit sistem macak-macak (muka air 20-25 cm di bawah bedengan), pengapuran dolomit untuk menormalkan pH tanah masam (6.0-6.8), perbaikan aerasi bedengan, dan inokulasi agens hayati Trichoderma / PGPR."
    ),
    (
        "Pendekatan Pengendalian Terpadu (PHT) & Sanitasi Ekosistem Bedengan",
        "fokus pada pemutusan siklus penularan antar bedengan, pemangkasan daun terinfeksi dan sanitasi gulma inang di pematang, perangkap hama, serta pergiliran golongan pestisida agar hama dan jamur tidak cepat kebal."
    )
]

def get_groq_recommendation(
    disease_name, 
    confidence, 
    is_healthy=False, 
    angle_idx=None,
    second_disease_name=None,
    second_confidence=None,
    is_differential=False
):
    """
    Memanggil Groq API untuk menyusun petunjuk obat dan perawatan lahan yang panjang, mendalam,
    serta verifikasi karakteristik pembeda gejala (Diferensial Diagnosis) jika terdeteksi 2 kemungkinan mirip.
    """
    import requests
    import random

    if is_differential and second_disease_name:
        angle_title = "Diferensial Diagnosis & Perlindungan Spektrum Ganda"
        user_prompt = (
            f"Model visual mendeteksi dua kemungkinan teratas: {disease_name} ({confidence:.1f}%) dan {second_disease_name} ({second_confidence:.1f}%).\n\n"
            "Anda bertindak sebagai Ahli Agronomi dan Konsultan Proteksi Tanaman Hortikultura Bawang Merah (merujuk pada riset Balitsa Lembang, BPTP Kementan, dan Jurnal Fitopatologi Indonesia).\n"
            "Bantu petani membedakan kedua penyakit ini di lapangan dan berikan penanganan terpadu:\n"
            "1. Ciri khas fisik pembeda yang paling mudah dilihat mata petani di lapangan (warna bercak, tekstur basah/kering, ada tidaknya tepung spora atau lendir).\n"
            "2. Tindakan pengobatan yang aman mencakup kedua spektrum (kombinasi fungisida + bakterisida tembaga atau sanitasi umum).\n\n"
            "WAJIB susun jawaban ke dalam 3 bagian persis dengan judul pemisah berikut:\n\n"
            "=== TINDAKAN LANGSUNG DI KEBUN ===\n"
            f"- Berikan langkah taktis darurat dalam 24 jam pertama di bedengan.\n"
            f"- Jelaskan secara spesifik cara membedakan {disease_name} vs {second_disease_name} langsung dengan mata telanjang di sawah (misal: lendir busuk basah vs cincin konsentris tepung spora kering).\n"
            "- Jelaskan teknik pemotongan daun bergejala dan sanitasi alat gunting/pisau agar spora atau bakteri tidak menyebar ke tanaman sekitar.\n\n"
            "=== REKOMENDASI OBAT SEMPROT ===\n"
            "- Berikan kombinasi obat semprot aman yang mencakup kedua spektrum patogen (kombinasi bakterisida tembaga seperti Tembaga Hidroksida / Kasugamisin dengan fungisida seperti Difenokonazol atau Mankozeb).\n"
            "- Sebutkan takaran dosis realistis (misal: 1,5 - 2 sendok makan per tangki semprot 16 Liter).\n"
            "- Sebutkan waktu semprot terbaik (pagi hari sebelum jam 09.00 saat embun mengering, atau sore setelah jam 16.00 saat angin tenang).\n"
            "- Ingatkan penambahan perekat/perata (surfactant) agar obat tidak mudah luntur.\n\n"
            "=== PERAWATAN LAHAN & PUPUK ===\n"
            "Jelaskan bagian ini secara terstruktur dalam 4 poin praktis:\n"
            "1. Pengaturan Parit & Tata Air: Atur muka air parit 20-25 cm di bawah bedengan, cegah genangan air hujan yang memicu penularan patogen.\n"
            "2. Manajemen Pupuk Khusus Masalah: Wajib STOP pupuk Nitrogen tunggal (Urea/ZA) yang memicu daun lunak sukulen, gantikan dengan pupuk Kalium (KNO3 Putih / MKP).\n"
            "3. Penguat Dinding Sel: Semprot pupuk Kalsium-Boron dan pupuk Silika cair untuk mempertebal lapisan lilin daun.\n"
            "4. Perawatan Tanah & Agens Hayati: Tabur dolomit jika tanah masam (pH < 6), aplikasikan Trichoderma atau Bacillus subtilis pada pupuk kandang matang.\n"
        )
    else:
        if angle_idx is None or angle_idx < 0 or angle_idx >= len(FOCUS_ANGLES):
            angle_title, angle_desc = random.choice(FOCUS_ANGLES)
        else:
            angle_title, angle_desc = FOCUS_ANGLES[angle_idx]

        user_prompt = (
            f"Kondisi Daun Bawang Merah: {disease_name}, Tingkat Keyakinan Model: {confidence:.1f}%.\n"
            f"Status: {'Daun Sehat/Normal' if is_healthy else 'Terserang Penyakit/Hama Tanaman'}.\n"
            f"Fokus Solusi Kali Ini: [{angle_title}] - {angle_desc}.\n\n"
            "Anda bertindak sebagai Konsultan Ahli Perlindungan Tanaman Hortikultura Bawang Merah (merujuk pada riset Balitsa Lembang, BPTP Kementan, dan Jurnal Fitopatologi Indonesia).\n"
            "Tolong berikan petunjuk obat dan perawatan lahan yang LENGKAP, MENDALAM, DAN DETAIL namun disajikan dalam BAHASA INDONESIA YANG LUGAS, JELAS, DAN SANGAT PRAKTIS UNTUK PETANI BAWANG MERAH (usia 30-50 tahun di sawah).\n"
            "Rangkum intisari dari riset ilmiah dan pedoman teknis menjadi poin-poin langkah nyata yang siap diterapkan di kebun.\n\n"
            "WAJIB susun jawaban ke dalam 3 bagian persis dengan judul pemisah berikut:\n\n"
            "=== TINDAKAN LANGSUNG DI KEBUN ===\n"
            "- Berikan langkah taktis darurat dalam 24-48 jam pertama di bedengan.\n"
            "- Jelaskan teknik pemotongan daun yang benar (jangan sampai spora terbang/berhamburan tertiup angin).\n"
            "- Jelaskan tindakan sanitasi alat gunting/pisau dan pemusnahan sisa pangkasan ke luar lahan (bakar/kubur jauh dari saluran air irigasi).\n\n"
            "=== REKOMENDASI OBAT SEMPROT ===\n"
            "- Berikan 2-3 pilihan kombinasi bahan aktif fungisida/insektisida/bakterisida yang lazim dan mudah dibeli di kios pertanian (sebutkan golongan kontak dan sistemik).\n"
            "- Sebutkan takaran dosis realistis (misal: 1,5 - 2 sendok makan per tangki semprot 16 Liter).\n"
            "- Sebutkan waktu semprot terbaik (pagi hari sebelum jam 09.00 saat embun mengering, atau sore setelah jam 16.00 saat angin tenang dan tidak terik).\n"
            "- Ingatkan penggunaan perekat/perata/penembus (surfactant) terutama saat musim hujan agar obat tidak luntur.\n\n"
            "=== PERAWATAN LAHAN & PUPUK ===\n"
            "Jelaskan bagian ini secara PANJANG, MENDALAM, DAN DETAIL terbagi dalam 4 poin terstruktur:\n"
            "1. Pengaturan Parit & Tata Air: Atur muka air parit sekitar 20-25 cm di bawah permukaan bedengan. Pastikan drainase lancar dan terapkan pengairan berselang (macak-macak), jangan biarkan air hujan menggenang di parit bedengan.\n"
            "2. Manajemen Pupuk Khusus Masalah: Wajib STOP atau kurangi pupuk Nitrogen tunggal (Urea/ZA) karena menyebabkan jaringan daun sukulen (terlalu empuk berair) yang menjadi santapan empuk jamur/bakteri. Gantikan dengan pupuk Kalium (KNO3 putih atau MKP 2-3 sendok/tangki) untuk memperkokoh umbi dan helai daun.\n"
            "3. Penguat Dinding Sel & Ketahanan Tanaman: Semprotkan pupuk Kalsium-Boron dan pupuk Silika cair secara rutin untuk melapisi dan mempertebal kutikula (lapisan lilin) daun sehingga spora patogen tidak bisa menembus masuk.\n"
            "4. Perawatan Tanah & Agens Hayati: Jika tanah bedengan masam (pH < 6), taburkan kapur dolomit 1-2 genggam per meter bedengan untuk menetralkan pH. Campurkan agens hayati Trichoderma harzianum atau bakteri Bacillus subtilis dengan pupuk kandang matang untuk menekan jamur tular tanah.\n"
        )

    headers = {
        "Authorization": f"Bearer {get_groq_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": "AgroScan-Mobile/1.0"
    }

    models_to_try = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "temperature": 0.72,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Anda adalah Ahli Agronomi dan Konsultan Proteksi Tanaman Hortikultura Indonesia. "
                        "Berikan penjelasan yang komprehensif, kaya akan detail praktis, takaran dosis yang realistis, "
                        "dan bersumber dari rangkuman riset Balitsa serta BPTP Kementerian Pertanian."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=14)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return content, angle_title
        except Exception:
            continue

    return None, angle_title

def get_groq_physical_verification(primary_name, second_name=None, is_differential=False):
    """
    Modul Validasi Karakteristik Fisik Menggunakan Groq LLM:
    Menghasilkan panduan verifikasi fisik lapangan berbasis riset agronomi
    untuk membantu petani membedakan penyakit yang mirip secara visual di kamera HP
    (terutama Hawar Daun Bakteri Xanthomonas vs Karat Daun / Bercak Jamur).
    """
    import requests
    api_key = get_groq_api_key()
    
    # Fallback lokal terverifikasi Balitsa jika kuota Groq habis atau offline
    fallback_diff = (
        "* 🖐️ **Uji Raba & Tekstur Permukaan:**\n"
        "  - **Hawar Daun Bakteri (Xanthomonas):** Bercak kebasah-basahan (*water-soaked*) seperti tersiram air mendidih. Pada pagi hari berembun terasa licin berlendir.\n"
        "  - **Penyakit Jamur (Karat / Bercak Ungu / Stemphylium):** Tidak berlendir. Karat meninggalkan serbuk oranye kemerahan di jari, sedangkan Bercak Ungu kering dengan lingkaran cincin konsentris.\n"
        "* 👃 **Uji Aroma Daun:**\n"
        "  - **Bakteri (Xanthomonas):** Saat helai daun dipetik dan diremas, tercium bau langu agak busuk menyengat.\n"
        "  - **Jamur:** Tidak berbau busuk, hanya aroma khas dedaunan mengering biasa.\n"
        "* 🔍 **Uji Bekas Usap Jari:**\n"
        "  - Jika diusap jari meninggalkan debu/serbuk warna tembaga atau oranye karat, itu adalah **Karat Daun (Jamur Puccinia)**, bukan bakteri!"
    )

    fallback_single = (
        "* 🖐️ **Uji Sentuh Daun:** Periksa apakah bercak terasa basah berlendir (tanda infeksi bakteri) atau kering bertepung (tanda infeksi jamur).\n"
        "* 👃 **Uji Aroma Daun:** Daun yang terserang bakteri umumnya mengeluarkan aroma langu busuk saat diremas.\n"
        "* 🔍 **Uji Cincin & Spora:** Amati tepi bercak dengan cermat; infeksi jamur biasanya membentuk cincin melingkar konsentris atau bintil serbuk spora."
    )

    if not api_key:
        return fallback_diff if is_differential else fallback_single

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AgroScan-Validator/1.0"
    }

    if is_differential and second_name:
        prompt = (
            f"Anda adalah Konsultan Proteksi Tanaman Bawang Merah (merujuk Balitsa Lembang & BPTP).\n"
            f"Bantu petani membedakan dua penyakit yang tampak mirip di kamera HP: '{primary_name}' vs '{second_name}'.\n"
            "Tuliskan panduan verifikasi fisik langsung di bedengan sawah dalam 3 poin ringkas dan padat:\n"
            "1. 🖐️ Uji Raba & Tekstur Daun (Lendir basah licin vs serbuk tepung/bintil kering kasar)\n"
            "2. 👃 Uji Aroma Daun (Bau langu busuk bakteri vs daun kering jamur)\n"
            "3. 🔍 Uji Bentuk Bercak & Usapan Jari (Bercak lemas memanjang vs cincin konsentris vs debu karat oranye)"
        )
    else:
        prompt = (
            f"Anda adalah Konsultan Proteksi Tanaman Bawang Merah.\n"
            f"Berikan 3 cara cepat verifikasi fisik di sawah untuk memastikan penyakit '{primary_name}':\n"
            "1. 🖐️ Uji Raba & Tekstur Permukaan Daun\n"
            "2. 👃 Uji Bau & Lendir Daun\n"
            "3. 🔍 Ciri Khas Bentuk Bercak Lapangan"
        )

    models_to_try = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    for model_name in models_to_try:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "temperature": 0.3,
                    "max_tokens": 400,
                    "messages": [
                        {"role": "system", "content": "Anda adalah dokter tanaman hortikultura yang memberi instruksi cek fisik langsung di sawah secara singkat, padat, dan jelas untuk petani."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=9
            )
            if resp.status_code == 200:
                txt = resp.json()["choices"][0]["message"]["content"].strip()
                if len(txt) > 30:
                    return txt
        except Exception:
            continue

    return fallback_diff if is_differential else fallback_single

def parse_groq_to_cards(ai_text, info):
    """Memecah teks balasan Groq menjadi 3 kartu panduan sederhana."""
    fallback_c1 = info.get("gejala", "Pangkas helai daun yang bergejala dan segera musnahkan di luar lahan.")
    fallback_c2 = info.get("solusi", "Semprotkan fungisida atau obat yang sesuai pada pagi atau sore hari.")
    fallback_c3 = (
        "1. Pengaturan Parit & Air: Jaga muka air parit 20-25 cm di bawah bedengan. Jangan biarkan air menggenang becek.\n"
        "2. Manajemen Pupuk: Hentikan pupuk Urea/Nitrogen berlebih saat daun bergejala sakit. Berikan pupuk Kalium (KNO3 putih / MKP) 2 sendok per tangki.\n"
        "3. Penguat Dinding Sel: Semprot pupuk Kalsium dan Silika cair untuk mempertebal lapisan lilin daun agar tahan serangan patogen.\n"
        "4. Perawatan Tanah: Taburkan kapur dolomit jika tanah asam dan berikan agens hayati Trichoderma pada pupuk kandang matang."
    )

    if not ai_text:
        return fallback_c1, fallback_c2, fallback_c3

    p1 = re.search(r'=== TINDAKAN LANGSUNG DI KEBUN ===(.*?)(?==== REKOMENDASI OBAT SEMPROT ===|$)', ai_text, re.DOTALL | re.IGNORECASE)
    p2 = re.search(r'=== REKOMENDASI OBAT SEMPROT ===(.*?)(?==== PERAWATAN LAHAN & PUPUK ===|$)', ai_text, re.DOTALL | re.IGNORECASE)
    p3 = re.search(r'=== PERAWATAN LAHAN & PUPUK ===(.*?)$', ai_text, re.DOTALL | re.IGNORECASE)

    c1 = p1.group(1).strip() if p1 else None
    c2 = p2.group(1).strip() if p2 else None
    c3 = p3.group(1).strip() if p3 else None

    # Fallback ke pola pemisah markdown jika AI menggunakan tanda pagar (###)
    if not c1 or not c2:
        parts = re.split(r'###\s*.*', ai_text)
        if len(parts) >= 4:
            c1 = parts[1].strip()
            c2 = parts[2].strip()
            c3 = parts[3].strip()
        elif len(parts) >= 3:
            c1 = parts[1].strip()
            c2 = parts[2].strip()
            c3 = fallback_c3

    return c1 or fallback_c1, c2 or fallback_c2, c3 or fallback_c3

def clean_text_output(text: str) -> str:
    """
    Membersihkan tag HTML mentah yang mungkin terbawa di teks sebelum ditampilkan ke layar.
    Mencegah kebocoran tag seperti </div>, <span>, dll.
    """
    if not text:
        return ""
    clean = re.sub(r'<[^>]*>', '', str(text))
    return clean.strip()

def format_card_text_to_html(text: str) -> str:
    """
    Mengubah format markdown bullet points, penomoran, dan bold ke HTML yang rapi & responsif.
    Membersihkan tag HTML tak diinginkan terlebih dahulu untuk mencegah kebocoran tag mentah.
    """
    if not text:
        return ""
    
    # 1. Bersihkan dari tag HTML tak diinginkan
    text_clean = clean_text_output(text)
    
    # 2. Ubah format bold **text** menjadi <strong>text</strong>
    formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text_clean)
    lines = formatted.split('\n')
    output_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                output_lines.append('</ul>')
                in_list = False
            continue
        
        # Cek apakah baris berupa list poin (- atau * atau 1. atau a.)
        is_bullet = stripped.startswith(('- ', '* '))
        is_numbered = bool(re.match(r'^\d+[\.\)]\s+', stripped))
        
        if is_bullet or is_numbered:
            content = re.sub(r'^([-*]|\d+[\.\)])\s+', '', stripped)
            if not in_list:
                output_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.35rem; list-style-type: disc;">')
                in_list = True
            output_lines.append(f'<li style="margin-bottom: 0.45rem; line-height: 1.65;">{content}</li>')
        else:
            if in_list:
                output_lines.append('</ul>')
                in_list = False
            output_lines.append(f'<p style="margin: 0.5rem 0; line-height: 1.75;">{stripped}</p>')
            
    if in_list:
        output_lines.append('</ul>')
        
    return '\n'.join(output_lines)

# Inisialisasi Model & Label
model_loaded = False
load_error_message = None
try:
    model, class_names = load_model_and_labels()
    model_loaded = True
except Exception as e:
    load_error_message = str(e)

# ==============================================================================
# 5. SIDEBAR: PENGATURAN TEKNIS & RIWAYAT (DIPINDAHKAN AGAR TIDAK MEMBINGUNGKAN)
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 0.5rem 0;'>
            <span style='font-size: 2.5rem;'>🧅</span>
            <h2 style='margin: 0.1rem 0; color: #1b5e20; font-weight: 800;'>AgroScan</h2>
            <p style='color: #64748b; font-size: 0.88rem;'>Menu Pengaturan & Riwayat</p>
        </div>
    """, unsafe_allow_html=True)

    # Indikator Status Koneksi Groq AI Otomatis (Secrets / Env)
    active_key = get_groq_api_key()
    if active_key and len(active_key) > 15:
        st.markdown(
            "<div style='text-align: center; margin: 0.1rem 0 0.75rem 0; font-weight: 700; color: #16a34a; font-size: 0.95rem;'>"
            "🟢 AI Terhubung"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align: center; margin: 0.1rem 0 0.75rem 0; font-weight: 700; color: #dc2626; font-size: 0.95rem;'>"
            "🔴 AI Belum Terhubung"
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### 🧪 Uji Coba Normalisasi (A/B Testing)")
    norm_choice = st.radio(
        "Pilih Metode Prapemprosesan Citra:",
        options=[
            "Mode 1: mobilenet_v2.preprocess_input ([-1, 1])",
            "Mode 2: Rescaling Standar img_array / 255.0 ([0, 1])"
        ],
        index=0,
        help="Uji coba apakah model dilatih dengan MobileNetV2 preprocess_input [-1, 1] atau pembagian skala 1/255 [0, 1]."
    )
    norm_mode = "rescaling_255" if "Mode 2" in norm_choice else "mobilenet_v2"

    st.divider()

    st.markdown("### ⚙️ Validasi Foto Bawang")
    conf_threshold = st.slider(
        "Batas Validasi Daun Bawang (%)",
        min_value=30,
        max_value=75,
        value=40,
        step=5,
        help="Jika kepastian model di bawah nilai ini, foto akan ditolak sebagai bukan daun bawang merah atau foto tidak jelas."
    )
    st.caption(f"Ambang batas kepastian: **{conf_threshold}%**")

    min_leaf_ratio = st.slider(
        "Sensitivitas Daun Bawang (%)",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
        help="Persentase minimal warna daun/umbi bawang merah yang harus ada pada foto sebelum model MobileNetV2 dijalankan."
    )
    st.caption(f"Batas luas daun minimal: **{min_leaf_ratio}%**")

    st.divider()
    st.markdown("### 📋 Riwayat Pemeriksaan")
    total_hist = len(st.session_state.get('history', []))
    st.write(f"Total Diagnosa Tercatat: **{total_hist}**")

    if total_hist > 0:
        df_hist = pd.DataFrame(st.session_state['history'])
        kolom_ekspor = [col for col in ["waktu", "penyakit", "confidence", "status", "lokasi", "catatan", "rekomendasi"] if col in df_hist.columns]
        
        # Download CSV
        csv_bytes = df_hist[kolom_ekspor].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Laporan (.CSV)",
            data=csv_bytes,
            file_name=f"riwayat_bawang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Download JSON
        json_bytes = json.dumps(st.session_state['history'], indent=2).encode('utf-8')
        st.download_button(
            label="📥 Unduh Cadangan (.JSON)",
            data=json_bytes,
            file_name=f"riwayat_bawang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            reset_all_history()
            components.html("<script>try{ window.parent.localStorage.removeItem('agroscan_bawang_history'); }catch(e){}</script>", height=0, width=0)
            st.toast("Semua riwayat berhasil dihapus.", icon="🗑️")
            st.rerun()

    st.divider()
    st.markdown("""
        <div style='font-size: 0.85rem; color: #94a3b8; text-align: center;'>
            Model Deep Learning: MobileNetV2<br>
            Asisten AI: Groq Cloud Intelligence
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. HEADER APLIKASI UTAMA (RAMAH PETANI)
# ==============================================================================
st.markdown("""
    <div class="farmer-hero">
        <h1>🧅 Dokter Tanaman Bawang Merah</h1>
        <p>Periksa kesehatan daun bawang merah secara cepat, tepat, dan mudah langsung di sawah.</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Gagal memuat model pendeteksi: {load_error_message}")
    st.stop()

# ==============================================================================
# 7. LANGKAH 1: AMBIL / MASUKKAN FOTO DAUN (SINGLE COLUMN MOBILE FIRST)
# ==============================================================================
st.markdown("""
    <div class="step-header">
        <div class="step-num">1</div>
        <div class="step-title">Ambil / Masukkan Foto Daun</div>
    </div>
""", unsafe_allow_html=True)

tab_camera, tab_upload = st.tabs(["📸 Ambil Foto Langsung (Kamera HP)", "📁 Pilih dari Galeri HP"])

selected_image = None

with tab_camera:
    cam_file = st.camera_input("Arahkan kamera dekat ke bagian daun yang sakit:", key="input_camera_field")
    if cam_file is not None:
        try:
            selected_image = Image.open(cam_file)
        except Exception as err:
            st.error(f"Gagal membaca foto kamera: {err}")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Pilih file foto dari galeri (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        key="input_file_field"
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
        except Exception as err:
            st.error(f"Gagal membuka berkas foto: {err}")

# Tips Ringkas untuk Petani
st.caption("💡 **Petunjuk Foto Bagus:** Arahkan kamera dekat (jarak 10-20 cm) tepat pada bercak daun yang bergejala.")

# Tombol Pemeriksaan Utama & Pemrosesan
if selected_image is not None:
    st.markdown("<div style='text-align: center; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.image(selected_image, caption="Foto Daun yang Dipilih", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Identifikasi gambar aktif
    current_img_sig = f"{selected_image.size}_{selected_image.mode}"
    
    # Tombol Utama Periksa (Besar, Hijau Daun Tegas, Ramah Jempol)
    btn_check = st.button("🔍 PERIKSA DAUN SEKARANG", type="primary", use_container_width=True, key="btn_inspect_main")
    
    # Jika tombol ditekan atau sudah pernah diperiksa untuk gambar ini
    if btn_check:
        st.session_state["has_inspected_current"] = current_img_sig

    # Tampilkan Hasil Pemeriksaan jika sudah diperiksa atau pengguna siap memeriksa
    if st.session_state.get("has_inspected_current") == current_img_sig:
        # ==============================================================================
        # TAHAP 1: VALIDASI GAMBAR (GUARDRAIL GATEKEEPER GROQ VISION & OOD GUARD)
        # Mencegah eksekusi model Keras pada foto yang bukan daun/tanaman bawang merah
        # ==============================================================================
        with st.spinner("🔍 Memverifikasi keaslian foto daun bawang..."):
            is_valid_vision, vision_verdict = validate_onion_image(selected_image)

        if not is_valid_vision:
            st.error("❌ Foto Ditolak: Objek yang diunggah terdeteksi bukan daun/tanaman bawang merah. Harap masukkan foto daun bawang merah yang jelas.")
            st.markdown(f"""
                <div class="card-rejection">
                    <div class="card-rejection-badge">⚠️ FOTO DITOLAK / TIDAK VALID</div>
                    <div class="card-rejection-title">Objek Bukan Daun Bawang Merah!</div>
                    <div class="card-rejection-reason">
                        {vision_verdict}
                    </div>
                    <div class="card-rejection-desc">
                        Sistem mendeteksi bahwa gambar yang Anda masukkan <strong>bukan daun atau tanaman bawang merah</strong> (seperti foto manusia, hewan, kendaraan, tanah kosong, atau daun tanaman lain seperti mangga/padi).
                        <br><br>
                        <strong>📋 Panduan Pengambilan Foto yang Benar:</strong>
                        <ol>
                            <li>Gunakan foto <strong>daun atau umbi tanaman bawang merah asli</strong> di bedengan kebun/sawah.</li>
                            <li>Arahkan kamera HP (jarak ideal <strong>10–20 cm</strong>) tepat pada helai daun yang sakit.</li>
                            <li>Pastikan pencahayaan terang dan daun terlihat jelas tanpa bayangan gelap.</li>
                            <li>Hindari memotret wajah, hewan, kendaraan, atau pemandangan sawah dari kejauhan.</li>
                        </ol>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.stop()  # Hentikan eksekusi di sini! Model MobileNetV2 Keras TIDAK AKAN dijalankan.

        # ==============================================================================
        # TAHAP 2: PROSES DIAGNOSIS MOBILENETV2 (Hanya Berjalan Jika Lolos Validasi Citra)
        # ==============================================================================
        with st.spinner("🔍 Sedang menganalisis kondisi daun bawang merah dengan MobileNetV2..."):
            (
                score,
                top_indices,
                top_class_raw,
                top_confidence,
                info,
                second_class_raw,
                second_confidence,
                second_info,
                is_differential,
                confidence_margin,
                preview_crop,
                diag_info
            ) = predict_image(
                selected_image, model, class_names, target_size=(224, 224), enforce_verification=False, norm_mode=norm_mode
            )
            # Selaraskan alias variabel agar konsisten (mencegah NameError)
            metadata = info
            second_metadata = second_info

        # ==============================================================================
        # TAHAP 3: VALIDASI AMBANG BATAS KEYAKINAN (CONFIDENCE THRESHOLD GUARD)
        # ==============================================================================
        if top_confidence < conf_threshold:
            st.error("❌ Gambar yang diunggah bukan daun bawang merah. Silakan ambil atau unggah foto daun bawang yang jelas.")
            st.markdown(f"""
                <div class="card-rejection">
                    <div class="card-rejection-badge">⚠️ FOTO DITOLAK / TIDAK VALID</div>
                    <div class="card-rejection-title">Kepastian Diagnosis Terlalu Rendah ({top_confidence:.1f}%)</div>
                    <div class="card-rejection-reason">
                        Tingkat kecocokan model hanya <strong>{top_confidence:.1f}%</strong> (di bawah batas minimal <strong>{conf_threshold}%</strong>). Pola objek tidak meyakinkan sebagai daun bawang merah.
                    </div>
                    <div class="card-rejection-desc">
                        Model tidak dapat mengenali pola penyakit dengan pasti. Kemungkinan foto daun bukan tanaman bawang merah, daun terlalu buram, atau bayangan terlalu gelap.
                        <br><br>
                        <strong>📋 Panduan Pengambilan Foto yang Benar:</strong>
                        <ol>
                            <li>Gunakan foto <strong>daun atau umbi tanaman bawang merah asli</strong> di bedengan kebun/sawah.</li>
                            <li>Arahkan kamera HP (jarak ideal <strong>10–20 cm</strong>) tepat pada helai daun yang sakit.</li>
                            <li>Pastikan pencahayaan terang dan daun terlihat jelas tanpa bayangan gelap.</li>
                        </ol>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.stop()  # Hentikan, jangan simpan riwayat & jangan panggil Groq AI!

        # ==============================================================================
        # KASUS 2: FOTO VALID (TERBUKTI DAUN BAWANG MERAH)
        # Lanjutkan Analisis Lengkap: Rekam Riwayat, Tampilkan Langkah 2 & Langkah 3
        # ==============================================================================
        else:
            # Rekam Otomatis ke Riwayat Sesi Sekali Saja
            diag_sig = f"{selected_image.size}_{top_class_raw}_{round(top_confidence, 1)}"
            if st.session_state.get("last_auto_recorded") != diag_sig:
                st.session_state["last_auto_recorded"] = diag_sig
                status_rec = "Multidiagnosis (Mirip)" if is_differential else ("Healthy / Sehat" if info.get("status") == "healthy" or info.get("is_healthy", False) else "Penyakit / Hama")
                st.session_state['history'].append({
                    "id": int(time.time() * 1000),
                    "waktu": datetime.now().strftime("%H:%M:%S"),
                    "penyakit": f"{info['nama_id']} ({top_confidence:.1f}%) & {second_info['nama_id']} ({second_confidence:.1f}%)" if is_differential else info["nama_id"],
                    "nama_penyakit": info["nama_id"],
                    "confidence": f"{top_confidence:.1f}%",
                    "keyakinan": f"{top_confidence:.1f}%",
                    "confidence_val": round(top_confidence, 2),
                    "status": status_rec,
                    "kelas_model": top_class_raw,
                    "rekomendasi": info["rekomendasi_singkat"],
                    "lokasi": "Kebun Bawang",
                    "catatan": "Pemeriksaan Lapangan"
                })

            # ==============================================================================
            # 8. LANGKAH 2: HASIL PEMERIKSAAN (NAMA PENYAKIT & KEPASTIAN)
            # ==============================================================================
            st.markdown("""
                <div class="step-header">
                    <div class="step-num">2</div>
                    <div class="step-title">Hasil Pemeriksaan Daun</div>
                </div>
            """, unsafe_allow_html=True)

            is_healthy = info.get("status") == "healthy" or info.get("is_healthy", False)
            is_pest = info.get("status") == "pest"

            if is_differential:
                # ------------------------------------------------------------------
                # TAMPILAN DIFERENSIAL DIAGNOSIS (MULTIDIAGNOSIS KARENA GEJALA MIRIP)
                # Gunakan native Streamlit alerts & columns agar bebas risiko kebocoran tag HTML/markdown code block
                # ------------------------------------------------------------------
                st.warning(
                    f"⚠️ **Gejala Ganda / Memerlukan Konfirmasi Fisik**\n\n"
                    f"### Terdeteksi 2 Kemungkinan Penyakit Serupa\n\n"
                    f"Model visual menemukan kemiripan tinggi dengan selisih probabilitas sangat tipis (hanya **{confidence_margin:.1f}%**). "
                    f"Petani disarankan mencocokkan ciri fisik langsung di kebun:"
                )

                col_diff1, col_diff2 = st.columns(2)
                with col_diff1:
                    latin_a = f"*{info.get('latin', '')}*\n\n" if info.get("latin") else ""
                    st.error(
                        f"**Kemungkinan A (Peringkat 1)**\n\n"
                        f"### {info['nama_id']}\n\n"
                        f"{latin_a}"
                        f"**Tingkat Kepastian:** `{top_confidence:.1f}%`\n\n"
                        f"🔍 **Ciri di Sawah:**\n\n{info.get('ciri_lapangan', '-')}"
                    )
                with col_diff2:
                    latin_b = f"*{second_info.get('latin', '')}*\n\n" if second_info.get("latin") else ""
                    st.warning(
                        f"**Kemungkinan B (Peringkat 2)**\n\n"
                        f"### {second_info['nama_id']}\n\n"
                        f"{latin_b}"
                        f"**Tingkat Kepastian:** `{second_confidence:.1f}%`\n\n"
                        f"🔍 **Ciri di Sawah:**\n\n{second_info.get('ciri_lapangan', '-')}"
                    )

                st.info(
                    "💡 **Kunci Pembeda Cepat di Lapangan:**\n\n"
                    "Periksa helai bercak daun secara teliti:\n\n"
                    "• **Bakteri (Hawar Daun):** Biasanya tampak berlendir kebasah-basahan seperti tersiram air mendidih saat pagi hari lembap dan berbau busuk.\n\n"
                    "• **Jamur (Bercak Ungu / Trotol / Stemphylium / Karat):** Tampak bercak cincin konsentris kering, bertepung spora, atau bintil karat serbuk oranye."
                )
            else:
                # ------------------------------------------------------------------
                # TAMPILAN SATU VONIS TUNGGAL (KEPASTIAN TINGGI)
                # ------------------------------------------------------------------
                if is_healthy:
                    status_card_class = "status-card-healthy"
                    badge_class = "badge-healthy-tag"
                    badge_text = "✅ DAUN SEHAT & NORMAL"
                    conf_color = "#10b981"
                elif is_pest:
                    status_card_class = "status-card-pest"
                    badge_class = "badge-pest-tag"
                    badge_text = "🐛 SERANGAN HAMA TANAMAN"
                    conf_color = "#ea580c"
                else:
                    status_card_class = "status-card-disease"
                    badge_class = "badge-disease-tag"
                    badge_text = "🚨 DAUN TERSERANG PENYAKIT"
                    conf_color = "#dc2626"

                st.markdown(f"""
                    <div class="{status_card_class}">
                        <div class="status-badge {badge_class}">{badge_text}</div>
                        <div class="status-disease-name">{info['nama_id']}</div>
                        <div class="status-confidence-text">
                            Tingkat Kepastian: <span style="color: {conf_color}; font-size: 1.4rem;">{top_confidence:.1f}%</span>
                        </div>
                        <div style="font-size: 0.92rem; color: #475569; margin-top: 0.6rem; line-height: 1.5;">
                            🔍 <strong>Ciri Khas di Sawah:</strong> {info.get('ciri_lapangan', '-')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # ==============================================================================
            # MODUL AUDIT PROBABILITAS LENGKAP & NILAI EKSTREM PIKSEL (BONGKAR SELURUH KELAS)
            # ==============================================================================
            st.markdown("### 📊 Panel Audit Probabilitas Lengkap (15 Kelas)")
            st.caption("Seluruh nilai probabilitas dari ke-15 kelas file <code>class_names.json</code> diurutkan dari persentase tertinggi ke terendah:")

            diag_rows = []
            for k_idx in top_indices:
                raw_label = class_names[k_idx]
                prob_val = float(score[k_idx]) * 100.0
                diag_rows.append({
                    "Indeks": int(k_idx),
                    "Nama Kelas (JSON)": raw_label,
                    "Probabilitas (%)": f"{prob_val:.2f}%"
                })
            df_prob = pd.DataFrame(diag_rows)
            st.dataframe(df_prob, use_container_width=True, hide_index=True)

            # Cetak Nilai Ekstrem Piksel Tepat di Bawah Tabel
            st.markdown(f"""
                <div style="background: #F8FAFC; border-radius: 12px; padding: 0.95rem 1.15rem; margin-top: 0.4rem; margin-bottom: 1.2rem; border: 1.5px solid #CBD5E1; font-size: 0.92rem; color: #1E293B; line-height: 1.65;">
                    ⚙️ <strong>Mode Normalisasi Aktif:</strong> <code>{diag_info.get('norm_mode_name', '-')}</code><br>
                    📉 <strong>Nilai Piksel Terendah (Min Pixel):</strong> <code>{diag_info.get('min_pixel', 0.0):.4f}</code><br>
                    📈 <strong>Nilai Piksel Tertinggi (Max Pixel):</strong> <code>{diag_info.get('max_pixel', 0.0):.4f}</code>
                </div>
            """, unsafe_allow_html=True)

            # ==============================================================================
            # MODUL VALIDASI KARAKTERISTIK FISIK LAPANGAN (GROQ LLM)
            # ==============================================================================
            if not is_healthy:
                phys_cache_key = f"phys_{top_class_raw}_{second_class_raw}_{is_differential}"
                if phys_cache_key not in st.session_state:
                    with st.spinner("🔬 Menyiapkan panduan verifikasi fisik lapangan..."):
                        st.session_state[phys_cache_key] = get_groq_physical_verification(
                            primary_name=info["nama_id"],
                            second_name=second_info["nama_id"] if is_differential else None,
                            is_differential=is_differential
                        )
                phys_content = st.session_state[phys_cache_key]
                html_phys = format_card_text_to_html(phys_content)

                st.markdown(f"""
                    <div style="background: #FFFFFF; border-radius: 16px; border: 1.5px solid #CBD5E1; padding: 1.15rem 1.25rem; margin: 1rem 0; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.35rem;">🔬</span>
                            <span style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">Verifikasi Karakteristik Fisik Langsung di Sawah (Groq AI)</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.85rem; line-height: 1.55;">
                            Gunakan panduan fisik berikut untuk memvalidasi gejala langsung pada daun bawang merah (mencegah kesalahan klasifikasi visual kamera HP antara <strong>Hawar Daun Bakteri (Xanthomonas)</strong> dan <strong>Karat Daun / Bercak Jamur</strong>):
                        </div>
                        <div style="background: #F8FAFC; border-radius: 12px; padding: 0.95rem 1.1rem; border-left: 4px solid #0284C7; font-size: 0.92rem; color: #1E293B; line-height: 1.65;">
                            {html_phys}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # ==============================================================================
            # 9. LANGKAH 3: PETUNJUK OBAT & PERAWATAN DARI DOKTER TANAMAN (GROQ AI)
            # ==============================================================================
            st.markdown("""
                <div class="step-header">
                    <div class="step-num">3</div>
                    <div class="step-title">Petunjuk Obat & Perawatan dari Dokter Tanaman</div>
                </div>
            """, unsafe_allow_html=True)

            # Logika Pengambilan Saran Groq AI
            ai_token_now = f"{top_class_raw}_{second_class_raw if is_differential else 'single'}_{round(top_confidence, 1)}"
            if st.session_state.get("ai_token_saved") != ai_token_now or "ai_text_saved" not in st.session_state:
                with st.spinner("🤖 Dokter Tanaman AI sedang meracik resep obat dan panduan perawatan..."):
                    ai_text, ai_angle = get_groq_recommendation(
                        disease_name=info["nama_id"],
                        confidence=top_confidence,
                        is_healthy=is_healthy,
                        second_disease_name=second_info["nama_id"] if is_differential else None,
                        second_confidence=second_confidence if is_differential else None,
                        is_differential=is_differential
                    )
                    st.session_state["ai_text_saved"] = ai_text
                    st.session_state["ai_angle_saved"] = ai_angle
                    st.session_state["ai_token_saved"] = ai_token_now

            # Parsing Resep Menjadi 3 Kartu Jelas & Format HTML Terstruktur
            kartu_tindakan, kartu_obat, kartu_lahan = parse_groq_to_cards(
                st.session_state.get("ai_text_saved"),
                info
            )
            html_tindakan = format_card_text_to_html(kartu_tindakan)
            html_obat = format_card_text_to_html(kartu_obat)
            html_lahan = format_card_text_to_html(kartu_lahan)

            st.markdown(f"""
                <div style="background-color: #F1F5F9; border-left: 6px solid #0284C7; padding: 0.8rem 1.1rem; border-radius: 12px; margin-bottom: 1.2rem; font-size: 1rem; color: #0F172A; font-weight: 700; border: 1px solid #CBD5E1; border-left-width: 6px;">
                    🎯 <strong>Fokus Rekomendasi Saat Ini:</strong> {st.session_state.get('ai_angle_saved', 'Pendekatan Terpadu Lapangan')}
                </div>
            """, unsafe_allow_html=True)

            # KARTU 1: TINDAKAN LANGSUNG DI KEBUN (MERAH)
            st.markdown(f"""
                <div class="card-ai-step card-ai-red">
                    <div class="card-ai-title" style="color: #DC2626;">
                        🚨 Tindakan Langsung di Kebun
                    </div>
                    <div class="card-ai-sub">(Langkah Segera 24 Jam Pertama di Bedengan)</div>
                    <div class="card-ai-body">{html_tindakan}</div>
                </div>
            """, unsafe_allow_html=True)

            # KARTU 2: REKOMENDASI OBAT SEMPROT (BIRU)
            st.markdown(f"""
                <div class="card-ai-step card-ai-blue">
                    <div class="card-ai-title" style="color: #1D4ED8;">
                        🧪 Rekomendasi Obat Semprot
                    </div>
                    <div class="card-ai-sub">(Bahan Aktif Pilihan, Takaran Tangki & Waktu Semprot)</div>
                    <div class="card-ai-body">{html_obat}</div>
                </div>
            """, unsafe_allow_html=True)

            # KARTU 3: PERAWATAN LAHAN & PUPUK (HIJAU)
            st.markdown(f"""
                <div class="card-ai-step card-ai-green">
                    <div class="card-ai-title" style="color: #15803D;">
                        🌾 Perawatan Lahan & Pupuk (Detail Solusi)
                    </div>
                    <div class="card-ai-sub">(Rangkuman Riset Balitsa Lembang, BPTP Kementan & Jurnal Proteksi Tanaman)</div>
                    <div class="card-ai-body">{html_lahan}</div>
                </div>
            """, unsafe_allow_html=True)

            # Tombol Besar: "Minta Petunjuk / Alternatif Obat Lain"
            st.markdown("<div class='btn-alt-obat'>", unsafe_allow_html=True)
            if st.button("🔄 Minta Petunjuk / Alternatif Obat Lain", key="btn_minta_alternatif", use_container_width=True):
                with st.spinner("🔄 Sedang meracik alternatif kombinasi obat dan panduan lain..."):
                    new_text, new_angle = get_groq_recommendation(
                        disease_name=info["nama_id"],
                        confidence=top_confidence,
                        is_healthy=is_healthy
                    )
                    if new_text:
                        st.session_state["ai_text_saved"] = new_text
                        st.session_state["ai_angle_saved"] = new_angle
                        st.toast(f"Petunjuk alternatif berhasil dimuat ({new_angle})", icon="🌱")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # ==============================================================================
            # 10. OPSI SIMPAN KE HP & BAGIKAN KE WHATSAPP (BERSIH & TANPA BENTROKAN)
            # ==============================================================================
            st.markdown("---")
            with st.expander("📲 Simpan Catatan & Bagikan Hasil ke WhatsApp", expanded=False):
                st.write("Catat lokasi bedengan atau bagikan info ini ke teman kelompok tani / kios pertanian:")
                
                in_lokasi = st.text_input("Lokasi Bedengan (Opsional):", placeholder="Contoh: Petak Barat, Bedeng 4", key="field_lokasi_petani")
                in_catatan = st.text_input("Catatan Tambahan (Opsional):", placeholder="Contoh: Gejala baru terlihat 2 hari setelah hujan lebat", key="field_catatan_petani")

                if st.button("💾 Simpan ke Riwayat HP", use_container_width=True, key="btn_save_ke_hp"):
                    saved_rec = save_diagnosis_to_history(
                        disease_code=top_class_raw,
                        display_name=info["nama_id"],
                        confidence=top_confidence,
                        recommendation=info["rekomendasi_singkat"],
                        is_healthy=is_healthy,
                        notes=in_catatan,
                        location=in_lokasi
                    )
                    entry_json = json.dumps(saved_rec)
                    js_script = f"""
                    <script>
                        try {{
                            const entry = {entry_json};
                            let hist = JSON.parse(window.parent.localStorage.getItem('agroscan_bawang_history') || '[]');
                            if (!hist.some(item => item.id === entry.id)) {{
                                hist.unshift(entry);
                            }}
                            if (hist.length > 100) hist = hist.slice(0, 100);
                            window.parent.localStorage.setItem('agroscan_bawang_history', JSON.stringify(hist));
                        }} catch(e) {{}}
                    </script>
                    """
                    components.html(js_script, height=0, width=0)
                    st.success("✅ Diagnosa berhasil tersimpan di riwayat!")
                    st.rerun()

                st.markdown("---")
                st.markdown("##### 📲 Format Pesan WhatsApp (Siap Kirim):")
                wa_share_text = (
                    f"🧅 *KONSULTASI DAUN BAWANG MERAH (AgroScan)*\n"
                    f"📅 Tanggal: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
                    f"🔬 Hasil Periksa: *{info['nama_id']}* (Kepastian: {top_confidence:.1f}%)\n"
                    f"📍 Lokasi: {in_lokasi if in_lokasi.strip() else 'Sawah Bawang'}\n\n"
                    f"🚨 *Saran Cepat:* {info['rekomendasi_singkat']}"
                )
                st.code(wa_share_text, language="text")
                st.caption("Tekan ikon salin di sudut kanan atas kotak abu-abu di atas, lalu tempel di obrolan WhatsApp kelompok tani.")
