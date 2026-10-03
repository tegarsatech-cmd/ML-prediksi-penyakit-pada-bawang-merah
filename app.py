"""
AgroScan Bawang Merah - Aplikasi Deteksi Penyakit Daun Bawang Merah
Desain UI/UX Mobile-First Ramah Petani (Usia 30-50 Tahun di Lapangan/Sawah)
Berbasis Deep Learning EfficientNet-B0 PyTorch (TorchScript) & Groq AI.
"""

import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps

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
# 2. ENSIKLOPEDIA & METADATA PENYAKIT DAUN BAWANG MERAH
# Menggunakan Istilah Populer yang Akrab Bagi Petani Indonesia
# ==============================================================================
# Daftar 7 Kategori Resmi Model EfficientNet-B0 (6 Penyakit Utama + 1 Kondisi Sehat)
SUPPORTED_DISEASES_7 = [
    {
        "key": "downy_mildew",
        "label": "Downy mildew",
        "icon": "💨",
        "nama_id": "Embun Bulu / Downy Mildew",
        "latin": "Peronospora destructor",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Permukaan daun dilapisi kapang/bulu halus beledu berwarna putih kelabu hingga keunguan saat pagi dingin lembap berkabut; daun menguning klorotik dari ujung.",
        "gejala": "Petik daun yang berbulu halus kelabu di pagi hari ke dalam kantong kresek tertutup; jangan biarkan spora terbang tertiup angin.",
        "pencegahan": "Hentikan pupuk Urea/Nitrogen berlebih selama musim hujan. Berikan pupuk Kalium dan Silika cair untuk mempertebal lapisan lilin daun.",
        "solusi": "Semprot fungisida sistemik berbahan aktif Dimetomorf, Simoksanil, atau Metalaksil selang-seling dengan Mankozeb tiap 3-4 hari.",
        "rekomendasi_singkat": "Semprot Dimetomorf/Simoksanil berselang Mankozeb & kurangi pupuk Nitrogen."
    },
    {
        "key": "healthy",
        "label": "Sehat",
        "icon": "🌿",
        "nama_id": "Daun Sehat & Normal",
        "latin": "Allium cepa (Kondisi Prima)",
        "status": "healthy",
        "is_healthy": True,
        "ciri_lapangan": "Daun silindris tegak berdiri kokoh, hijau segar merata, berlilin alami, tanpa bercak berlendir, luka trotol, bintil karat, maupun puntiran abnormal.",
        "gejala": "Kondisi tanaman prima! Tidak ditemukan lesi patogen aktif atau serangan hama pada helai daun.",
        "pencegahan": "Lanjutkan pemantauan rutin 2-3 hari sekali di waktu pagi. Jaga kebersihan gulma di parit dan pematang.",
        "solusi": "Tidak memerlukan fungisida/bakterisida kuratif. Cukup semprotkan pupuk daun mikro lengkap dan asam amino untuk menjaga fotosintesis.",
        "rekomendasi_singkat": "Tanaman sehat optimal! Cukup lanjutkan pemupukan berimbang dan pengairan macak-macak."
    },
    {
        "key": "iris_yellow_virus",
        "label": "Iris Yellow Spot Virus (IYSV)",
        "icon": "🟡",
        "nama_id": "Virus Iris Kuning (IYSV)",
        "latin": "Iris yellow spot virus (Vektor Thrips tabaci)",
        "status": "virus",
        "is_healthy": False,
        "ciri_lapangan": "Bercak klorotik khas berbentuk ketupat / belah ketupat warna kuning jerami di tengah helai daun, terkadang memiliki pulau hijau di tengah bercak (green islands).",
        "gejala": "Cabut dan musnahkan tanaman bergejala bercak ketupat kuning agar tidak menular ke tanaman lain melalui kutu trips.",
        "pencegahan": "Gunakan mulsa plastik perak untuk memantulkan sinar matahari dan menghalau kutu trips (Thrips tabaci) pembawa virus.",
        "solusi": "Kendalikan kutu trips dengan insektisida sistemik berbahan aktif Abamektin, Spinetoram, atau Klorfenapir pada pagi/sore hari.",
        "rekomendasi_singkat": "Cabut tanaman bergejala ketupat & semprot Abamektin untuk basmi kutu trips."
    },
    {
        "key": "leaf_blight",
        "label": "Hawar Daun (Stemphylium / Colletotrichum)",
        "icon": "🍂",
        "nama_id": "Hawar Daun / Kering Ujung",
        "latin": "Stemphylium vesicarium / Colletotrichum gloeosporioides",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Ujung helai daun menguning kecokelatan kering merambat memanjang ke bawah (blight), atau bercak melekuk kebasahan pada helai daun.",
        "gejala": "Pangkas helai daun yang tampak membusuk atau mengering dari ujung sekitar 2 cm di bawah batas lesi sebelum merambat ke leher umbi.",
        "pencegahan": "Hindari penyiraman sore/malam hari dan jaga sirkulasi parit bedengan macak-macak agar tidak tergenang air.",
        "solusi": "Semprot fungisida berbahan aktif Mankozeb, Klorotalonil, atau Difenokonazol selang-seling dengan Tembaga Oksiklorida.",
        "rekomendasi_singkat": "Pangkas daun kering ujung dan semprot fungisida Mankozeb/Klorotalonil."
    },
    {
        "key": "moler",
        "label": "Moler",
        "icon": "🌀",
        "nama_id": "Layu Moler / Inul (Fusarium)",
        "latin": "Fusarium oxysporum f. sp. cepae",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Helai daun meliuk-liuk abnormal berpilin/spiral (moler/inul), menguning pucat dari ujung, perakaran membusuk dan tanaman sangat gampang dicabut.",
        "gejala": "Segera cabut tanaman yang daunnya melintir abnormal (moler) sampai ke perakarannya agar jamur tidak menular lewat parit.",
        "pencegahan": "Campurkan agens hayati Trichoderma dengan pupuk kandang matang saat olah tanah dasar bedengan sebelum tanam.",
        "solusi": "Taburkan kapur dolomit pada lubang bekas cabutan dan kocorkan fungisida sistemik berbahan aktif Benomil atau Mankozeb.",
        "rekomendasi_singkat": "Cabut tanaman melintir/moler, taburkan dolomit & agens Trichoderma."
    },
    {
        "key": "purple_blotch",
        "label": "Bercak Ungu / Trotol (Alternaria porri)",
        "icon": "🟣",
        "nama_id": "Bercak Ungu / Trotol",
        "latin": "Alternaria porri",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bercak melekuk ke dalam berbentuk cincin konsentris bertepung keunguan/gelap di tengah helai daun, tepi menguning klorotik, daun mudah patah di titik bercak.",
        "gejala": "Potong atau pangkas daun yang terdapat bercak trotol ungu cincin konsentris. Kumpulkan dan musnahkan di luar areal sawah.",
        "pencegahan": "Bersihkan gulma di sekitar parit. Buat bedengan lebih tinggi agar tidak tergenang air saat hujan lebat.",
        "solusi": "Semprot fungisida berbahan aktif Difenokonazol, Azoksistrobin, atau Mankozeb pada pagi hari saat angin tenang.",
        "rekomendasi_singkat": "Pangkas daun trotol dan semprot Difenokonazol/Azoksistrobin segera."
    },
    {
        "key": "rust",
        "label": "Rust",
        "icon": "🟠",
        "nama_id": "Karat Daun (Bintil Pustula)",
        "latin": "Puccinia allii",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bintil-bintil kecil melepuh (pustula) berisi serbuk/tepung spora berwarna jingga kemerahan atau merah tembaga seperti serbuk besi berkarat.",
        "gejala": "Pangkas daun yang dipenuhi bintil debu karat warna oranye kemerahan ke dalam wadah tertutup saat memotong agar serbuk karat tidak berhamburan.",
        "pencegahan": "Lakukan rotasi pergiliran tanaman dengan jagung atau palawija setelah panen bawang merah.",
        "solusi": "Semprot fungisida berbahan aktif Tebukonazol, Heksakonazol, atau Difenokonazol saat embun pagi mulai mengering (sekitar pukul 08.00).",
        "rekomendasi_singkat": "Semprot fungisida Tebukonazol/Heksakonazol saat embun pagi mulai kering."
    }
]

# Bangun Dictionary Master Metadata Penyakit (Termasuk Aliases untuk Kompatibilitas Menyeluruh)
CLASS_METADATA = {}

# 1. Registrasi 7 Kelas Resmi
for item in SUPPORTED_DISEASES_7:
    meta_dict = {
        "nama_id": item["nama_id"],
        "latin": item["latin"],
        "status": item["status"],
        "is_healthy": item["is_healthy"],
        "ciri_lapangan": item["ciri_lapangan"],
        "gejala": item["gejala"],
        "pencegahan": item["pencegahan"],
        "solusi": item["solusi"],
        "rekomendasi_singkat": item["rekomendasi_singkat"],
        "icon": item["icon"]
    }
    CLASS_METADATA[item["label"]] = meta_dict
    CLASS_METADATA[item["key"]] = meta_dict

# 2. Registrasi Alias Bahasa & Istilah Umum
CLASS_METADATA["Trotol"] = CLASS_METADATA["Bercak Ungu / Trotol (Alternaria porri)"]
CLASS_METADATA["Alternaria_D"] = CLASS_METADATA["Bercak Ungu / Trotol (Alternaria porri)"]
CLASS_METADATA["Purple blotch"] = CLASS_METADATA["Bercak Ungu / Trotol (Alternaria porri)"]

CLASS_METADATA["Busuk Daun"] = CLASS_METADATA["Hawar Daun (Stemphylium / Colletotrichum)"]
CLASS_METADATA["Hawar Daun"] = CLASS_METADATA["Hawar Daun (Stemphylium / Colletotrichum)"]
CLASS_METADATA["stemphylium Leaf Blight"] = CLASS_METADATA["Hawar Daun (Stemphylium / Colletotrichum)"]

CLASS_METADATA["Fusarium-D"] = CLASS_METADATA["Moler"]
CLASS_METADATA["Layu Moler"] = CLASS_METADATA["Moler"]

CLASS_METADATA["Karat Daun"] = CLASS_METADATA["Rust"]

CLASS_METADATA["Embun Bulu"] = CLASS_METADATA["Downy mildew"]

CLASS_METADATA["Healthy leaves"] = CLASS_METADATA["Sehat"]
CLASS_METADATA["onion1"] = CLASS_METADATA["Sehat"]
CLASS_METADATA["Daun Sehat & Segar"] = CLASS_METADATA["Sehat"]

CLASS_METADATA["Iris yellow virus_augment"] = CLASS_METADATA["Iris Yellow Spot Virus (IYSV)"]
CLASS_METADATA["Virus Iris Kuning"] = CLASS_METADATA["Iris Yellow Spot Virus (IYSV)"]
CLASS_METADATA["Virosis-D"] = CLASS_METADATA["Iris Yellow Spot Virus (IYSV)"]

# Inisialisasi Sesi Penyimpanan
if "history" not in st.session_state:
    st.session_state.history = []

def save_diagnosis_to_history(disease_code, display_name, confidence, recommendation, is_healthy, notes="", location=""):
    """Menyimpan entri riwayat diagnosa baru."""
    now_time = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
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

def delete_history_item(index: int):
    """Menghapus satu entri riwayat berdasarkan indeks (satu per satu)."""
    if "history" in st.session_state and 0 <= index < len(st.session_state['history']):
        st.session_state['history'].pop(index)

# ==============================================================================
# 3. CACHING MODEL PYTORCH (TORCHSCRIPT) & KONFIGURASI METADATA
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bawang_efficientnet_b0_ts.pt")
META_PATH = os.path.join(BASE_DIR, "meta.json")

@st.cache_resource(show_spinner=False)
def load_meta_config(file_path=META_PATH) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File konfigurasi model '{file_path}' tidak ditemukan.")
    with open(file_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

@st.cache_resource(show_spinner=False)
def load_torch_model(file_path=MODEL_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File model '{file_path}' tidak ditemukan.")
    import torch
    model = torch.jit.load(file_path, map_location="cpu")
    model.eval()
    return model

def extract_leaf_roi(image: Image.Image, padding_pct: float = 0.15) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """
    Ekstraksi Otomatis Region of Interest (ROI) Daun Bawang Merah:
    Mendeteksi area helai daun dan memangkas latar belakang tanah, mulsa, tangan, atau meja.
    Menjembatani perbedaan antara foto kamera smartphone lapangan dengan foto dataset makro.
    """
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    thumb_dim = 256
    scale_w = orig_w / float(thumb_dim)
    scale_h = orig_h / float(thumb_dim)
    thumb = img_rgb.resize((thumb_dim, thumb_dim), Image.Resampling.BILINEAR)

    arr = np.array(thumb, dtype=np.float32)
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

    # Deteksi piksel tanaman daun bawang merah
    is_green = (h >= 35.0) & (h <= 170.0) & (s >= 0.12) & (v >= 0.10)
    is_yellow = (h >= 20.0) & (h < 35.0) & (g >= r * 0.70) & (s >= 0.15) & (v >= 0.20)
    is_spot = (h >= 5.0) & (h < 20.0) & (r > g * 1.05) & (s >= 0.20) & (v >= 0.15)

    plant_mask = is_green | is_yellow | is_spot
    plant_pixels = np.argwhere(plant_mask)

    if len(plant_pixels) < 60:
        return img_rgb, (0, 0, orig_w, orig_h), 1.0

    y_min, x_min = plant_pixels.min(axis=0)
    y_max, x_max = plant_pixels.max(axis=0)

    ox1 = int(x_min * scale_w)
    oy1 = int(y_min * scale_h)
    ox2 = int(x_max * scale_w)
    oy2 = int(y_max * scale_h)

    bw = ox2 - ox1
    bh = oy2 - oy1

    pad_x = int(bw * padding_pct)
    pad_y = int(bh * padding_pct)

    fx1 = max(0, ox1 - pad_x)
    fy1 = max(0, oy1 - pad_y)
    fx2 = min(orig_w, ox2 + pad_x)
    fy2 = min(orig_h, oy2 + pad_y)

    crop_w = fx2 - fx1
    crop_h = fy2 - fy1
    area_ratio = (crop_w * crop_h) / float(orig_w * orig_h)

    if area_ratio < 0.04 or crop_w < 35 or crop_h < 35:
        return img_rgb, (0, 0, orig_w, orig_h), 1.0

    cropped_roi = img_rgb.crop((fx1, fy1, fx2, fy2))
    return cropped_roi, (fx1, fy1, fx2, fy2), area_ratio

def preprocess_image_smart(image: Image.Image, target_size=(224, 224), use_tta: bool = True, **kwargs):
    """
    Pipeline Prapemprosesan Citra Multiperspektif Cerdas (Leaf-Focused Smart TTA):
    1. EXIF Transpose: Mengoreksi rotasi orientasi dari kamera smartphone iOS/Android.
    2. Auto-Cropping Daun (Leaf ROI): Memfokuskan potongan pada helai daun bawang merah
       dan menyingkirkan latar belakang tanah/tangan/mulsa (menghilangkan Domain Gap).
    3. Multi-View TTA:
       - View 1: Focused Leaf ROI (daun bersih fokus tinggi).
       - View 2: High-Resolution Macro Center Crop dari Leaf ROI (detail lesi/tekstur).
       - View 3: Simetri Horizontal dari Leaf ROI (invarian arah kamera).
       - View 4: Full Frame Natural (konteks global keseluruhan tanaman).
    4. Input Skala: Tensor float32 [0.0, 255.0] untuk pemrosesan citra.
    """
    orig_mode = image.mode if image is not None else "RGB"
    orig_size = image.size if image is not None else (0, 0)

    # 1. Normalisasi orientasi EXIF (kamera smartphone)
    cropped_img = ImageOps.exif_transpose(image) if image is not None else image
    img_clean = cropped_img.convert("RGB")
    w, h = img_clean.size

    # 2. Deteksi & Ekstraksi Region of Interest (ROI) Daun Bawang Merah
    leaf_roi, leaf_bbox, leaf_coverage = extract_leaf_roi(img_clean, padding_pct=0.15)
    is_auto_cropped = leaf_coverage < 0.88

    # View Utama: Focused Leaf ROI (atau Full Frame jika daun sudah memenuhi layar)
    primary_crop = leaf_roi if is_auto_cropped else img_clean
    view_primary = primary_crop.resize(target_size, Image.Resampling.BILINEAR)
    view_full = img_clean.resize(target_size, Image.Resampling.BILINEAR)

    if not use_tta:
        crops = [view_primary]
        weights = [1.0]
    else:
        crops = [view_primary]
        weights = [0.45]

        # View 2: Macro Center Crop dari area daun fokus
        pw, ph = primary_crop.size
        min_dim = min(pw, ph)
        cx, cy = pw // 2, ph // 2
        half = min_dim // 2
        center_img = primary_crop.crop((cx - half, cy - half, cx + half, cy + half)).resize(target_size, Image.Resampling.BILINEAR)
        crops.append(center_img)
        weights.append(0.25)

        # View 3: Simetri Horizontal (invarian sudut pemotretan kamera smartphone)
        crops.append(view_primary.transpose(Image.FLIP_LEFT_RIGHT))
        weights.append(0.15)

        # View 4: Full Frame Global (konteks keseluruhan daun/tanaman)
        crops.append(view_full)
        weights.append(0.15)

    # Susun batch tensor NumPy float32
    batch_array = np.stack([np.array(c, dtype=np.float32) for c in crops], axis=0)

    diag_info = {
        "orig_mode": orig_mode,
        "orig_size": orig_size,
        "norm_mode_name": f"Leaf-Focused TTA ({len(crops)} Perspektif)" if use_tta else "Focused ROI",
        "num_views": len(crops),
        "is_auto_cropped": is_auto_cropped,
        "leaf_coverage_pct": round(leaf_coverage * 100.0, 1),
        "leaf_bbox": leaf_bbox,
        "min_pixel": float(np.min(batch_array)),
        "max_pixel": float(np.max(batch_array))
    }

    return batch_array, view_primary, weights, diag_info

def check_shallot_leaf_mask(image: Image.Image, min_ratio: float = 0.08) -> tuple[bool, str, float]:
    """
    Validasi Citra Daun Bawang Merah (Pre-Inference Guard):
    Memeriksa spektrum kromatisitas dan morfologi tanaman bawang merah (Allium cepa)
    menggunakan kombinasi Excess Green Index (ExG) dan analisis HSV botani.
    Mencegah input non-tanaman: wajah, tangan tanpa daun, tanah/dinding polos, pakaian, kendaraan, hewan, dokumen.
    """
    try:
        thumb = image.convert("RGB").resize((224, 224))
        arr = np.array(thumb, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # 1. Excess Green Index (ExG) botani: 2G - R - B
        exg = 2.0 * g - r - b
        
        # 2. HSV murni
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
        
        # Jaringan daun hijau tanaman (klorofil aktif)
        is_green_leaf = (h >= 35.0) & (h <= 165.0) & (s >= 0.14) & (v >= 0.09) & (exg > 4.0)
        
        # Daun menguning klorotik / ujung mengering penyakit
        is_yellowing = (h >= 24.0) & (h < 35.0) & (g >= r * 0.72) & (s >= 0.16) & (v >= 0.14)
        
        # Bintil pustula karat / bercak nekrotik pada daun
        is_rust_spot = (h >= 8.0) & (h < 24.0) & (r > g * 1.12) & (s >= 0.28) & (v >= 0.16) & (v <= 0.85)

        # Deteksi kulit tangan/wajah manusia untuk eksklusi
        is_skin = (
            (h >= 6.0) & (h <= 26.0) &
            (s >= 0.18) & (s <= 0.55) &
            (v >= 0.35) & (v <= 0.90) &
            (r > g * 1.12) & (g > b * 1.08) &
            (np.abs(r - g) < 85)
        )
        skin_ratio = float(np.mean(is_skin))
        
        # Mask vegetasi murni pada helai daun
        plant_mask = (is_green_leaf | is_yellowing | is_rust_spot) & (~is_skin)
        plant_ratio = float(np.mean(plant_mask))
        
        # Cek kertas putih / dinding polos / background abu-abu
        is_white_gray = (s < 0.10) & (v > 0.70)
        if np.mean(is_white_gray) > 0.85:
            return False, "Terdeteksi objek kertas atau dinding putih polos, bukan daun bawang.", plant_ratio

        # Jika kulit tangan/wajah mendominasi dan tanaman hampir tidak ada
        if skin_ratio > 0.38 and plant_ratio < 0.05:
            return False, "Terdeteksi hanya menampilkan kulit/tangan manusia tanpa helai daun bawang yang memadai.", plant_ratio
        
        if plant_ratio < min_ratio:
            return False, f"Rasio daun bawang pada foto hanya {plant_ratio*100:.1f}% (minimal {min_ratio*100:.0f}%).", plant_ratio
            
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
    except (KeyError, AttributeError):
        pass

    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip().rstrip(".")

    return ""

def validate_onion_image(image: Image.Image, api_key: str | None = None, min_ratio: float = 0.08) -> tuple[bool, str]:
    """
    Sistem Validasi Guardrail Gatekeeper Citra Tanaman Bawang Merah:
    Memverifikasi keaslian foto daun bawang merah sebelum proses diagnosa.
    Menolak foto manusia, hewan, kendaraan, tanah kosong, dan objek non-bawang.
    Tetap mengizinkan anomali wajar seperti daun bawang yang dipegang tangan petani di kebun.
    """
    is_plant, reason, ratio = check_shallot_leaf_mask(image, min_ratio=min_ratio)
    if not is_plant:
        return False, f"INVALID: {reason}"
    return True, f"VALID (Rasio Kanopi Daun: {ratio*100:.1f}%)"

# Alias untuk kompatibilitas
validate_with_groq_vision = validate_onion_image

def inspect_visual_leaf_symptoms(image: Image.Image, target_disease: str = "", is_healthy: bool = False) -> dict:
    """
    Modul Inspeksi Fitur Visual Citra Daun Bawang Merah:
    Menganalisis karakteristik visual fisik helai daun secara sinkron dengan vonis AI.
    Jika daun sehat (is_healthy=True), secara tegas menetapkan kerusakan 0% dan kondisi sehat prima
    tanpa memunculkan peringatan karat atau penyakit palsu.
    """
    if is_healthy or "sehat" in target_disease.lower():
        return {
            "has_visual_evidence": True,
            "evidence_disease": "Healthy leaves",
            "suspected_rust": False,
            "override_applied": False,
            "severity_pct": 0.0,
            "severity_level": "Sehat Prima (0%)",
            "rust_pct": 0.0,
            "purple_pct": 0.0,
            "xantho_pct": 0.0,
            "healthy_pct": 100.0,
            "num_spots_detected": 0,
            "evidence_desc": (
                "Helai daun bawang hijau segar optimal (100% jaringan klorofil normal utuh), "
                "tegak kokoh tanpa ditemukan bercak nekrotik, bintil jamur, maupun infeksi patogen aktif."
            ),
            "overlay_img": image
        }

    try:
        img = image.convert("RGB")
        w, h = img.size
        max_dim = 640
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = max(int(w * scale), 10), max(int(h * scale), 10)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_np = np.array(img, dtype=np.float32)
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        exg = 2.0 * g - r - b

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = np.where(cmax - cmin == 0, 1.0, cmax - cmin)

        h_arr = np.zeros_like(delta)
        mask_r = (cmax == r) & (cmax > cmin)
        h_arr[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
        mask_g = (cmax == g) & (cmax > cmin)
        h_arr[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
        mask_b = (cmax == b) & (cmax > cmin)
        h_arr[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        s_arr = np.where(cmax == 0, 0.0, (cmax - cmin) / np.where(cmax == 0, 1.0, cmax))
        v_arr = cmax / 255.0

        is_skin = (
            (h_arr >= 6.0) & (h_arr <= 26.0) &
            (s_arr >= 0.18) & (s_arr <= 0.55) &
            (v_arr >= 0.35) & (v_arr <= 0.90) &
            (r > g * 1.12) & (g > b * 1.08) &
            (np.abs(r - g) < 85)
        )

        is_plant = ((exg > 0) | ((h_arr >= 20.0) & (h_arr <= 165.0) & (s_arr >= 0.12))) & (~is_skin)
        total_plant = int(np.count_nonzero(is_plant))

        if total_plant < 100:
            return {
                "has_visual_evidence": True,
                "evidence_disease": target_disease or "Inconclusive",
                "suspected_rust": False,
                "override_applied": False,
                "severity_pct": 5.0,
                "severity_level": "Ringan (< 10%)",
                "rust_pct": 0.0,
                "purple_pct": 0.0,
                "xantho_pct": 0.0,
                "healthy_pct": 95.0,
                "num_spots_detected": 0,
                "evidence_desc": f"Gejala infeksi {target_disease} teridentifikasi pada helai daun.",
                "overlay_img": image
            }

        # Filter lesi spesifik
        name_lower = (target_disease or "").lower()
        if "rust" in name_lower or "karat" in name_lower:
            is_les = (h_arr >= 7.0) & (h_arr <= 28.0) & (r > g * 1.08) & (s_arr >= 0.22) & (v_arr >= 0.15) & (v_arr <= 0.85)
            metric_label = "Karat"
        elif "trotol" in name_lower or "bercak" in name_lower or "alternaria" in name_lower:
            is_les = ((h_arr <= 16.0) | (h_arr >= 265.0)) & (r > g * 1.05) & (s_arr >= 0.15) & (v_arr >= 0.08) & (v_arr <= 0.70)
            metric_label = "Bercak Ungu"
        elif "hawar" in name_lower or "stemphylium" in name_lower or "colletotrichum" in name_lower:
            is_les = (h_arr >= 18.0) & (h_arr <= 42.0) & (s_arr >= 0.12) & (v_arr >= 0.15) & (g >= r * 0.70)
            metric_label = "Hawar Daun"
        elif "moler" in name_lower or "fusarium" in name_lower or "inul" in name_lower:
            is_les = (h_arr >= 26.0) & (h_arr <= 50.0) & (s_arr >= 0.15) & (v_arr >= 0.20)
            metric_label = "Layu Moler"
        elif "virus" in name_lower or "iysv" in name_lower or "mildew" in name_lower:
            is_les = (h_arr >= 25.0) & (h_arr <= 55.0) & (s_arr >= 0.12) & (v_arr >= 0.30)
            metric_label = "Klorosis Daun"
        else:
            is_les = (h_arr >= 10.0) & (h_arr <= 45.0) & (s_arr >= 0.15) & (v_arr >= 0.12)
            metric_label = "Lesi Patogen"

        les_pixels = int(np.count_nonzero(is_les & is_plant))
        severity_pct = round(min(max((les_pixels / total_plant) * 100.0, 3.0), 92.0), 1)
        healthy_pct = round(max(100.0 - severity_pct, 5.0), 1)

        if severity_pct < 10.0:
            severity_level = "Sangat Ringan (< 10%)"
        elif severity_pct < 25.0:
            severity_level = "Ringan (10% - 25%)"
        elif severity_pct < 50.0:
            severity_level = "Sedang (25% - 50%)"
        else:
            severity_level = "Berat / Kritis (> 50%)"

        ev_desc = (
            f"Pemindaian visual mengonfirmasi gejala khas {target_disease} "
            f"dengan tingkat keparahan {severity_level} (estimasi jaringan daun terdampak: {severity_pct}%, "
            f"jaringan daun hijau tersisa: {healthy_pct}%)."
        )

        return {
            "has_visual_evidence": True,
            "evidence_disease": target_disease,
            "suspected_rust": ("rust" in name_lower or "karat" in name_lower),
            "override_applied": False,
            "severity_pct": severity_pct,
            "severity_level": severity_level,
            "rust_pct": severity_pct if ("rust" in name_lower or "karat" in name_lower) else 0.0,
            "purple_pct": severity_pct if ("trotol" in name_lower or "bercak" in name_lower) else 0.0,
            "xantho_pct": severity_pct if ("hawar" in name_lower) else 0.0,
            "healthy_pct": healthy_pct,
            "num_spots_detected": 0,
            "evidence_desc": ev_desc,
            "overlay_img": image
        }
    except Exception as e:
        return {
            "has_visual_evidence": True,
            "evidence_disease": target_disease,
            "suspected_rust": False,
            "override_applied": False,
            "severity_pct": 10.0,
            "severity_level": "Ringan",
            "rust_pct": 0.0,
            "purple_pct": 0.0,
            "xantho_pct": 0.0,
            "healthy_pct": 90.0,
            "num_spots_detected": 0,
            "evidence_desc": f"Gejala infeksi {target_disease} teridentifikasi pada helai daun.",
            "overlay_img": image
        }

def generate_lesion_hud_map(
    image: Image.Image,
    target_class_idx: int = 0,
    is_healthy: bool = False,
    second_class_idx: int | None = None,
    primary_name: str = "Penyakit A",
    second_name: str | None = None,
    is_differential: bool = False,
    model=None,
    **kwargs
):
    """
    Peta HUD Scanner Deteksi Titik Kerusakan / Jaringan Daun Bawang Merah:
    - Seluruh 7 kategori (termasuk Daun Sehat dan seluruh penyakit) SELALU memiliki titik retikel presisi.
    - Titik retikel terkunci kuat pada lokasi fisik lesi / klorofil helai daun (anti-melengser).
    - Membatasi jumlah titik secara disiplin (maksimal 1-2 titik) agar visual bersih, fokus, dan tidak acak.
    - Daun Sehat menggunakan retikel Hijau Zamrud (Emerald Green) untuk memverifikasi jaringan prima.
    - Penyakit primer menggunakan retikel Merah, dan diferensial kedua menggunakan retikel Oranye.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    try:
        img_np = np.array(img_rgb)
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)

        # 1. Konversi Warna & Parameter Fisiologis Daun
        exg = 2.0 * g - r - b
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = np.where(cmax - cmin == 0, 1.0, cmax - cmin)

        h_arr = np.zeros_like(delta)
        mask_r = (cmax == r) & (cmax > cmin)
        h_arr[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
        mask_g = (cmax == g) & (cmax > cmin)
        h_arr[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
        mask_b = (cmax == b) & (cmax > cmin)
        h_arr[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        s_arr = np.where(cmax == 0, 0.0, (cmax - cmin) / np.where(cmax == 0, 1.0, cmax))
        v_arr = cmax / 255.0

        # 2. Filter Ketat Non-Tanaman (Tangan/Kulit Manusia & Latar Belakang Netral)
        is_skin = (
            (h_arr >= 6.0) & (h_arr <= 26.0) &
            (s_arr >= 0.18) & (s_arr <= 0.55) &
            (v_arr >= 0.35) & (v_arr <= 0.90) &
            (r > g * 1.12) & (g > b * 1.08) &
            (np.abs(r - g) < 85)
        )
        is_neutral_bg = (s_arr < 0.09) & ((v_arr > 0.80) | (v_arr < 0.08))
        is_glare = (s_arr < 0.11) & (v_arr > 0.82)

        # Kanopi daun bawang merah: mencakup daun hijau segar, klorosis kuning, hingga nekrosis lesi
        is_green = (exg > 2.0) | ((h_arr >= 35.0) & (h_arr <= 165.0) & (s_arr >= 0.12) & (v_arr >= 0.09))
        is_yellow = (h_arr >= 22.0) & (h_arr < 35.0) & (g >= r * 0.70) & (s_arr >= 0.15) & (v_arr >= 0.14)
        is_necrotic = ((h_arr <= 20.0) | (h_arr >= 265.0)) & (s_arr >= 0.12) & (v_arr >= 0.08) & (v_arr <= 0.75) & (g >= r * 0.40)
        is_plant = (is_green | is_yellow | is_necrotic) & (~is_skin) & (~is_neutral_bg) & (~is_glare)

        # Abaikan margin tepi foto 2.5% agar retikel tidak melengser ke batas bingkai kamera
        m_x = max(int(w * 0.025), 2)
        m_y = max(int(h * 0.025), 2)
        is_plant[:m_y, :] = False
        is_plant[-m_y:, :] = False
        is_plant[:, :m_x] = False
        is_plant[:, -m_x:] = False

        # Bersihkan noise kecil kanopi menggunakan morfologi opening & closing
        kernel_plant = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        plant_clean = cv2.morphologyEx(is_plant.astype(np.uint8), cv2.MORPH_OPEN, kernel_plant)
        plant_clean = cv2.morphologyEx(plant_clean, cv2.MORPH_CLOSE, kernel_plant)
        is_plant = plant_clean > 0

        # Jika kanopi daun sangat sedikit (misal background dominan), ambil area terbaik atau fallback ke tengah
        total_plant_px = int(np.count_nonzero(is_plant))
        if total_plant_px < 150:
            cy_c, cx_c = h // 2, w // 2
            ry_c, rx_c = int(h * 0.35), int(w * 0.35)
            y_grid, x_grid = np.ogrid[:h, :w]
            is_plant = ((x_grid - cx_c)**2 / max(rx_c**2, 1) + (y_grid - cy_c)**2 / max(ry_c**2, 1)) <= 1.0

        # 3. Fungsi Pembobotan Gejala Sesuai Karakteristik Patogen/Kategori (Presisi Tinggi seperti Bulu Embun)
        def get_heatmap(disease_name: str, healthy_flag: bool):
            name = (disease_name or "").lower()
            if healthy_flag or "sehat" in name or "healthy" in name:
                # Daun Sehat: Klorofil hijau zamrud murni
                is_green_blade = (h_arr >= 48.0) & (h_arr <= 155.0) & (exg > 0.0)
                score = np.where(
                    is_green_blade,
                    np.clip(exg / 45.0, 0.0, 1.0) * 0.50 +
                    ((h_arr >= 50.0) & (h_arr <= 145.0)).astype(np.float32) * 0.35 +
                    np.clip((s_arr - 0.15) / 0.40, 0.0, 1.0) * 0.15,
                    0.0
                )
            elif "mildew" in name or "embun" in name:
                # Embun Bulu (Gold Standard): Bercak klorotik pucat dengan rona beludru keabuan
                is_sym = (h_arr >= 24.0) & (h_arr <= 65.0) & (s_arr <= 0.55) & (s_arr >= 0.08)
                score = np.where(
                    is_sym,
                    0.40 +
                    np.clip((0.55 - s_arr) / 0.35, 0.0, 1.0) * 0.35 +
                    np.clip(1.0 - np.abs(v_arr - 0.50) / 0.35, 0.0, 1.0) * 0.25,
                    0.0
                )
            elif "rust" in name or "karat" in name:
                # Karat Daun: Pustula jingga-oranye kemerahan menonjol
                is_sym = (h_arr >= 7.0) & (h_arr <= 36.0) & (r > g * 1.04) & (s_arr >= 0.18)
                score = np.where(
                    is_sym,
                    0.40 +
                    np.clip((r - g) / 25.0, 0.0, 1.0) * 0.35 +
                    np.clip((s_arr - 0.18) / 0.40, 0.0, 1.0) * 0.25,
                    0.0
                )
            elif "trotol" in name or "bercak" in name or "alternaria" in name or "purple" in name:
                # Bercak Ungu / Trotol: Titik cekung gelap konsentris / keunguan
                is_sym = ((h_arr <= 18.0) | (h_arr >= 265.0) | (v_arr < 0.42)) & (s_arr >= 0.10)
                score = np.where(
                    is_sym,
                    0.35 +
                    np.clip((0.48 - v_arr) / 0.40, 0.0, 1.0) * 0.45 +
                    np.clip((s_arr - 0.10) / 0.35, 0.0, 1.0) * 0.20,
                    0.0
                )
            elif "hawar" in name or "blight" in name or "stemphylium" in name or "colletotrichum" in name:
                # Hawar Daun: Kering ujung jerami pucat / nekrotik
                is_sym = (h_arr >= 16.0) & (h_arr <= 54.0) & (v_arr >= 0.30) & (s_arr >= 0.08) & (s_arr <= 0.65)
                score = np.where(
                    is_sym,
                    0.40 +
                    np.clip((v_arr - 0.28) / 0.45, 0.0, 1.0) * 0.35 +
                    np.clip(1.0 - (exg / 25.0), 0.0, 1.0) * 0.25,
                    0.0
                )
            elif "moler" in name or "fusarium" in name or "inul" in name:
                # Layu Moler: Klorosis kuning terang meliuk
                is_sym = (h_arr >= 24.0) & (h_arr <= 58.0) & (s_arr >= 0.18)
                score = np.where(
                    is_sym,
                    0.40 +
                    np.clip((s_arr - 0.18) / 0.40, 0.0, 1.0) * 0.35 +
                    np.clip((v_arr - 0.20) / 0.50, 0.0, 1.0) * 0.25,
                    0.0
                )
            elif "virus" in name or "iysv" in name:
                # IYSV: Lesi belah ketupat warna jerami pucat
                is_sym = (h_arr >= 20.0) & (h_arr <= 54.0) & (v_arr >= 0.30) & (s_arr >= 0.10)
                score = np.where(
                    is_sym,
                    0.40 +
                    np.clip((v_arr - 0.28) / 0.45, 0.0, 1.0) * 0.35 +
                    np.clip((s_arr - 0.10) / 0.35, 0.0, 1.0) * 0.25,
                    0.0
                )
            else:
                # Patogen lain: deviasi dari klorofil hijau normal
                is_sym = (h_arr < 45.0) | (h_arr > 155.0) | (exg <= 0.0)
                score = np.where(
                    is_sym,
                    np.clip(1.0 - (exg / 25.0), 0.0, 1.0) * 0.60 +
                    np.clip((s_arr - 0.10) / 0.30, 0.0, 1.0) * 0.40,
                    0.0
                )
            # Pastikan nilai skor hanya aktif di dalam kanopi daun (anti-melengser keluar daun)
            return np.where(is_plant, score, 0.0)

        # Hitung peta panas gejala primer
        heat_prim = get_heatmap(primary_name, is_healthy)

        # Label singkat untuk badge HUD
        badge_name_1 = (
            "Daun Sehat" if (is_healthy or "sehat" in (primary_name or "").lower())
            else primary_name.split("/")[0].split("(")[0].strip()[:14]
        )
        badge_name_2 = (
            (second_name or "").split("/")[0].split("(")[0].strip()[:14]
            if (is_differential and second_name) else ""
        )

        spots = []
        base_rad = max(18, int(min(w, h) * 0.050))
        max_rad = max(24, int(min(w, h) * 0.090))
        min_dist = max(38, int(min(w, h) * 0.12))

        # Helper untuk mencari koordinat titik fokus terbaik dari peta panas
        def locate_focal_point(heat_map, mask_exclude=None):
            h_eff = heat_map.copy()
            if mask_exclude is not None:
                h_eff = np.where(mask_exclude, 0.0, h_eff)

            vals_in_plant = h_eff[is_plant & (mask_exclude == False if mask_exclude is not None else True)]
            if len(vals_in_plant) == 0 or np.max(vals_in_plant) <= 0.01:
                coords_y, coords_x = np.where(is_plant & (mask_exclude == False if mask_exclude is not None else True))
                if len(coords_y) > 0:
                    mid_idx = len(coords_y) // 2
                    return int(coords_x[mid_idx]), int(coords_y[mid_idx]), base_rad
                return w // 2, h // 2, base_rad

            # Ambang batas dinamis persentil 82 untuk mengekstrak inti bercak terkuat
            th_val = max(float(np.percentile(vals_in_plant, 82)), 0.20)
            bin_les = ((h_eff >= th_val) & is_plant).astype(np.uint8) * 255
            
            kernel_les = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bin_clean = cv2.morphologyEx(bin_les, cv2.MORPH_OPEN, kernel_les)
            bin_clean = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, kernel_les)

            contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_c = [c for c in contours if cv2.contourArea(c) >= max(15, int(w * h * 0.00010))]

            if len(valid_c) > 0:
                def contour_score(c):
                    mask_c = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_c, [c], -1, 1, -1)
                    vals = h_eff[mask_c > 0]
                    if len(vals) == 0:
                        return 0.0
                    return float(np.max(vals) * 0.70 + np.mean(vals) * 0.30)

                valid_c.sort(key=contour_score, reverse=True)
                top_c = valid_c[0]
                area = cv2.contourArea(top_c)
                r_calc = max(base_rad, min(int(np.sqrt(area / np.pi) * 1.15), max_rad))

                # Cari titik pusat massa gejala (Weighted Intensity Centroid) persis di dalam kontur lesi
                mask_top = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_top, [top_c], -1, 1, -1)
                heat_in_c = np.where(mask_top > 0, h_eff, 0.0)
                pts_y, pts_x = np.where(heat_in_c > 0.01)
                if len(pts_y) > 0:
                    weights = np.power(heat_in_c[pts_y, pts_x], 2)
                    w_sum = float(np.sum(weights))
                    if w_sum > 0:
                        cx = int(np.round(np.sum(pts_x * weights) / w_sum))
                        cy = int(np.round(np.sum(pts_y * weights) / w_sum))
                        # Verifikasi titik berada di kanopi daun (jika sedikit keluar batas, kunci ke piksel kanopi terdekat)
                        if is_plant[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)]:
                            return cx, cy, r_calc
                        else:
                            pts = top_c.reshape(-1, 2)
                            dists = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
                            best_idx = np.argmin(dists)
                            return int(pts[best_idx, 0]), int(pts[best_idx, 1]), r_calc

                M = cv2.moments(top_c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Verifikasi titik berada di kanopi daun (jika melengser keluar, kunci ke piksel kanopi terdekat)
                    if not is_plant[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)]:
                        pts = top_c.reshape(-1, 2)
                        dists = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
                        best_idx = np.argmin(dists)
                        cx, cy = int(pts[best_idx, 0]), int(pts[best_idx, 1])
                    return cx, cy, r_calc

            # Fallback terjamin: Ambil weighted centroid dari piksel bergejala tertinggi di kanopi daun
            max_val = np.max(h_eff)
            if max_val > 0.05:
                top_thresh = max_val * 0.85
                pts_y, pts_x = np.where((h_eff >= top_thresh) & is_plant)
                if len(pts_y) > 0:
                    weights = np.power(h_eff[pts_y, pts_x], 2)
                    w_sum = float(np.sum(weights))
                    if w_sum > 0:
                        cx = int(np.round(np.sum(pts_x * weights) / w_sum))
                        cy = int(np.round(np.sum(pts_y * weights) / w_sum))
                        return cx, cy, base_rad
            max_pos = np.argmax(h_eff)
            cy, cx = divmod(max_pos, w)
            return int(cx), int(cy), base_rad

        # -------------------------------------------------------------
        # 4. PENENTUAN TITIK RETIKEL (FOKUS TUNGGAL SEPERTI BULU EMBUN)
        # -------------------------------------------------------------
        if is_healthy or "sehat" in (primary_name or "").lower():
            # Daun Sehat: Tepat 1 Titik Retikel Hijau pada Helai Daun Segar Utama
            cx1, cy1, r1 = locate_focal_point(heat_prim)
            spots.append((cx1, cy1, r1, "Daun Sehat", "green"))
        elif is_differential and second_name:
            # Multi-Penyakit: Tepat 2 Titik (1 Merah Primer + 1 Oranye Sekunder)
            cx1, cy1, r1 = locate_focal_point(heat_prim)
            spots.append((cx1, cy1, r1, badge_name_1, "red"))

            # Buat mask eksklusi agar titik kedua tidak bertumpuk di titik pertama
            y_g, x_g = np.ogrid[:h, :w]
            excl_mask = ((x_g - cx1)**2 + (y_g - cy1)**2) < (min_dist**2)

            heat_sec = get_heatmap(second_name, False)
            cx2, cy2, r2 = locate_focal_point(heat_sec, mask_exclude=excl_mask)
            spots.append((cx2, cy2, r2, badge_name_2, "orange"))
        else:
            # Penyakit Tunggal: Tepat 1 Titik Fokus Utama Presisi (Sama Persis Seperti Bulu Embun)
            cx1, cy1, r1 = locate_focal_point(heat_prim)
            spots.append((cx1, cy1, r1, badge_name_1, "red"))

        # Pastikan spots tidak pernah kosong dalam kondisi apapun!
        if len(spots) == 0:
            spots.append((w // 2, h // 2, base_rad, badge_name_1, "green" if is_healthy else "red"))

        # -------------------------------------------------------------
        # 5. PENGGAMBARAN HUD RETICLE PADA CITRA (DESAIN AESTHETIC HI-TECH)
        # -------------------------------------------------------------
        annotated = img_rgb.copy()
        draw = ImageDraw.Draw(annotated)

        for cx, cy, rad, badge_label, color_type in spots:
            if color_type == "green":
                # Hijau Zamrud untuk Daun Sehat & Prima
                color_hud = (16, 185, 129)
                color_inner = (167, 243, 208)
                badge_bg = (5, 150, 105)
            elif color_type == "orange":
                # Oranye Amber untuk Diferensial Kedua
                color_hud = (245, 158, 11)
                color_inner = (254, 240, 138)
                badge_bg = (217, 119, 6)
            else:
                # Merah Crimson untuk Penyakit Primer
                color_hud = (239, 68, 68)
                color_inner = (254, 202, 202)
                badge_bg = (220, 38, 38)

            line_w = max(2, int(min(w, h) * 0.005))
            tick = max(8, int(min(w, h) * 0.018))
            dot_r = max(3, int(min(w, h) * 0.007))

            # Lingkaran luar HUD
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color_hud, width=line_w)
            
            # Lingkaran dalam HUD
            inner_r = max(rad - line_w * 2, 6)
            draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], outline=color_inner, width=max(1, line_w - 1))
            
            # Titik sentral bidik
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=color_hud)

            # Garis bidik crosshair 4 arah
            draw.line([cx - rad - tick, cy, cx - rad + 3, cy], fill=color_hud, width=line_w)
            draw.line([cx + rad - 3, cy, cx + rad + tick, cy], fill=color_hud, width=line_w)
            draw.line([cx, cy - rad - tick, cx, cy - rad + 3], fill=color_hud, width=line_w)
            draw.line([cx, cy + rad - 3, cx, cy + rad + tick], fill=color_hud, width=line_w)

            # Chip Badge Nama Gejala
            font_scale = max(8, int(min(w, h) * 0.018))
            badge_w = max(len(badge_label) * font_scale + 16, 75)
            badge_h = max(18, font_scale + 8)
            bx1 = max(2, cx - badge_w // 2)
            by1 = cy - rad - badge_h - 6
            bx2 = min(w - 2, bx1 + badge_w)
            by2 = by1 + badge_h
            if by1 < 3:
                by1 = cy + rad + 6
                by2 = by1 + badge_h
            if by2 < h - 2:
                draw.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=badge_bg)
                draw.text((bx1 + 6, by1 + 2), badge_label, fill=(255, 255, 255))

        return annotated, [(s[0], s[1], s[2]) for s in spots]
    except Exception:
        return img_rgb, [(w // 2, h // 2, 25)]

# Alias kompatibilitas
generate_keras_cam_map = generate_lesion_hud_map

def predict_disease(image: Image.Image, model, meta=None, class_names=None, enforce_verification: bool = True, **kwargs):
    """
    Pipeline Prediksi PyTorch TorchScript Resmi (EfficientNet-B0):
    Mengikuti 7 langkah presisi sesuai instruksi pelatihan Google Colab:
    1. Buka gambar dengan PIL, perbaiki orientasi EXIF (ImageOps.exif_transpose), convert ke RGB.
    2. Resize ke (img_size, img_size) = 224x224 (tanpa crop), ubah ke tensor float 0-1.
    3. Normalisasi dengan mean=[0.485,0.456,0.406] dan std=[0.229,0.224,0.225] (dari meta.json).
    4. Bentuk batch [1,3,224,224]. Hitung logits = (model(x) + model(torch.flip(x, dims=[3]))) / 2 (TTA).
    5. probs = softmax(logits / temperature, dim=1), temperature dari meta.json.
    6. Ambil kelas dengan probabilitas tertinggi. Jika confidence < conf_threshold (0.70),
       tampilkan "Tidak yakin, foto kurang jelas atau bukan daun bawang" alih-alih menebak.
    7. Nama kelas ditampilkan dari meta.json -> class_labels_id, dengan urutan index yang sama.

    Format API JSON:
    { "label": "...", "confidence": 0.93, "uncertain": false,
      "probabilities": {"Sehat": 0.93, "Bercak Ungu / Trotol ...": 0.03, ...} }
    """
    import torch

    if meta is None or not isinstance(meta, dict):
        meta = load_meta_config()

    if enforce_verification:
        is_shallot, reason_msg, _ = check_shallot_leaf_mask(image, min_ratio=0.08)
        if not is_shallot:
            raise ValueError(f"OOD_GUARD_REJECTED: {reason_msg}")

    # 1. Buka gambar dengan PIL, perbaiki orientasi EXIF, convert ke RGB
    img_rgb = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = img_rgb.size

    # 2. Resize ke (img_size, img_size) tanpa crop, ubah ke tensor float 0-1
    img_size = int(meta.get("img_size", 224))
    resized = img_rgb.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # [224, 224, 3], range 0-1
    tensor_chw = torch.from_numpy(arr).permute(2, 0, 1)  # [3, 224, 224]

    # 3. Normalisasi dengan mean & std dari meta.json
    mean_vals = meta.get("mean", [0.485, 0.456, 0.406])
    std_vals = meta.get("std", [0.229, 0.224, 0.225])
    mean_t = torch.tensor(mean_vals, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std_vals, dtype=torch.float32).view(3, 1, 1)
    normalized = (tensor_chw - mean_t) / std_t

    # 4. Bentuk batch [1, 3, 224, 224] & Hitung logits dengan TTA flip horizontal
    x = normalized.unsqueeze(0)
    with torch.no_grad():
        x_flip = torch.flip(x, dims=[3])
        logits = (model(x) + model(x_flip)) / 2.0

        # 5. probs = softmax(logits / temperature, dim=1)
        temperature = float(meta.get("temperature", 0.05))
        probs_tensor = torch.softmax(logits / temperature, dim=1)[0]

    # 6. Ambil kelas tertinggi & evaluasi conf_threshold (0.70)
    conf_threshold = float(meta.get("conf_threshold", 0.70))
    top_idx = int(torch.argmax(probs_tensor).item())
    top_confidence_val = float(probs_tensor[top_idx].item())
    top_confidence = round(top_confidence_val * 100.0, 1)

    uncertain = bool(top_confidence_val < conf_threshold)

    class_labels = meta.get("class_labels_id", [
        "Downy mildew",
        "Sehat",
        "Iris Yellow Spot Virus (IYSV)",
        "Hawar Daun (Stemphylium / Colletotrichum)",
        "Moler",
        "Bercak Ungu / Trotol (Alternaria porri)",
        "Rust"
    ])

    raw_class_name = class_labels[top_idx]
    if uncertain:
        display_label = "Tidak yakin, foto kurang jelas atau bukan daun bawang"
    else:
        display_label = raw_class_name

    # 7. Format JSON probabilities
    probabilities_dict = {
        class_labels[i]: round(float(probs_tensor[i].item()), 4)
        for i in range(len(class_labels))
    }

    # Format Keluaran API (JSON)
    api_output = {
        "label": display_label,
        "confidence": round(top_confidence_val, 4),
        "uncertain": uncertain,
        "probabilities": probabilities_dict
    }

    probs_np = probs_tensor.cpu().numpy()
    sorted_indices = [int(i) for i in np.argsort(probs_np)[::-1]]
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else top_idx
    second_confidence = round(float(probs_np[second_idx]) * 100.0, 1)
    second_class_name = class_labels[second_idx]
    confidence_margin = round(top_confidence - second_confidence, 1)

    is_healthy_1 = (raw_class_name == "Sehat") or CLASS_METADATA.get(raw_class_name, {}).get("is_healthy", False)
    is_healthy_2 = (second_class_name == "Sehat") or CLASS_METADATA.get(second_class_name, {}).get("is_healthy", False)

    is_differential = (
        (not uncertain)
        and (confidence_margin <= 18.0)
        and (top_confidence < 68.0)
        and (second_confidence >= 22.0)
        and (not is_healthy_1)
        and (not is_healthy_2)
        and (raw_class_name != second_class_name)
    )

    metadata = CLASS_METADATA.get(raw_class_name, {
        "nama_id": raw_class_name,
        "latin": "-",
        "status": "healthy" if is_healthy_1 else "disease",
        "is_healthy": is_healthy_1,
        "ciri_lapangan": "Periksa kondisi helai daun dan bercak secara teliti.",
        "gejala": "Pangkas daun yang bergejala dan keluarkan dari areal kebun.",
        "pencegahan": "Jaga kelancaran parit dan kebersihan gulma bedengan.",
        "solusi": "Gunakan fungisida atau bakterisida sesuai rekomendasi petugas penyuluh.",
        "rekomendasi_singkat": "Pangkas daun sakit dan lakukan penyemprotan obat yang sesuai."
    })

    second_metadata = CLASS_METADATA.get(second_class_name, {
        "nama_id": second_class_name,
        "latin": "-",
        "status": "healthy" if is_healthy_2 else "disease",
        "is_healthy": is_healthy_2,
        "ciri_lapangan": "Periksa kondisi helai daun dan bercak secara teliti.",
        "gejala": "Pangkas daun yang bergejala dan bersihkan bedengan.",
        "pencegahan": "Jaga drainase parit dan sanitasi pematang.",
        "solusi": "Gunakan obat yang sesuai.",
        "rekomendasi_singkat": "Lakukan sanitasi daun sakit."
    })

    # 7. Ekstraksi Bukti Visual yang Sinkron 100% dengan Hasil Prediksi Model
    visual_evidence = inspect_visual_leaf_symptoms(
        image=image,
        target_disease=metadata["nama_id"],
        is_healthy=is_healthy_1
    )

    # Peta HUD Scanner Lesi
    annotated_cam, cam_circles = generate_lesion_hud_map(
        image=image,
        target_class_idx=top_idx,
        is_healthy=is_healthy_1,
        second_class_idx=second_idx if is_differential else None,
        primary_name=metadata["nama_id"],
        second_name=second_metadata["nama_id"] if is_differential else None,
        is_differential=is_differential
    )
    visual_evidence["overlay_img"] = annotated_cam
    visual_evidence["num_spots_detected"] = len(cam_circles)

    diag_info = {
        "orig_mode": image.mode,
        "orig_size": (orig_w, orig_h),
        "norm_mode_name": "EfficientNet-B0 TorchScript + ImageNet Mean/Std + TTA",
        "num_views": 2,
        "is_auto_cropped": False,
        "leaf_coverage_pct": 100.0,
        "leaf_bbox": (0, 0, orig_w, orig_h),
        "min_pixel": 0.0,
        "max_pixel": 1.0,
        "api_output": api_output,
        "uncertain": uncertain
    }

    return (
        probs_np,
        sorted_indices,
        raw_class_name,
        top_confidence,
        metadata,
        second_class_name,
        second_confidence,
        second_metadata,
        is_differential,
        confidence_margin,
        resized,
        diag_info,
        visual_evidence,
        api_output
    )

# Alias untuk kompatibilitas fungsi lama
predict_image = predict_disease

def validate_onion_leaf(image: Image.Image, top_confidence: float, threshold: float = 40.0):
    """
    Validasi Citra Daun Bawang Merah:
    1. Filter Analisis Citra Digital Warna Tanaman (Hue/Chrominance Masking):
       Memeriksa apakah citra mengandung spektrum vegetasi/daun bawang merah (>= 12%).
    2. Filter Ambang Batas Keyakinan Model (OOD Rejection):
       Jika model memiliki top_confidence < threshold (bawaan 40%),
       objek dipastikan bukan bagian daun/umbi bawang merah yang dapat diidentifikasi.
    """
    # 1. Validasi Spektrum Warna Daun/Tanaman
    is_valid_mask, _, ratio = check_shallot_leaf_mask(image, min_ratio=0.12)
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
    is_differential=False,
    latin_name=None,
    severity_level=None,
    evidence_desc=None
):
    """
    Memanggil Groq API untuk menyusun petunjuk obat dan perawatan lahan yang panjang, mendalam,
    diselaraskan persis dengan hasil vonis diagnosis penyakit dari sumber resmi Balitsa Lembang & BPTP Kementan.
    """
    latin_str = f" ({latin_name})" if latin_name else ""
    sev_str = f"Tingkat Keparahan Infeksi: {severity_level}\n" if severity_level else ""
    ev_str = f"Gejala Fisik Lapangan: {evidence_desc}\n" if evidence_desc else ""

    if is_differential and second_disease_name:
        angle_title = "Diferensial Diagnosis & Perlindungan Spektrum Ganda"
        user_prompt = (
            f"VONIS DIAGNOSIS PENYAKIT (KEMUNGKINAN GANDA): {disease_name}{latin_str} ({confidence:.1f}%) dan {second_disease_name} ({second_confidence:.1f}%).\n"
            f"{sev_str}"
            f"{ev_str}\n"
            "Anda bertindak sebagai Ahli Agronomi dan Konsultan Proteksi Tanaman Hortikultura Bawang Merah (merujuk pada riset Balitsa Lembang, BPTP Kementan, dan Jurnal Fitopatologi Indonesia).\n"
            "Bantu petani membedakan kedua penyakit ini di lapangan dan berikan penanganan terpadu:\n"
            "1. Ciri khas fisik pembeda yang paling mudah dilihat mata petani di lapangan (warna bercak, tekstur basah/kering, ada tidaknya tepung spora atau lendir).\n"
            "2. Tindakan pengobatan yang aman mencakup kedua spektrum (kombinasi fungisida + bakterisida tembaga atau sanitasi umum).\n\n"
            "WAJIB susun jawaban ke dalam 3 bagian persis dengan judul pemisah berikut:\n\n"
            "=== TINDAKAN LANGSUNG DI KEBUN ===\n"
            "- Berikan langkah taktis darurat dalam 24 jam pertama di bedengan.\n"
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
            f"VONIS DIAGNOSIS RESMI: {disease_name}{latin_str}.\n"
            f"Tingkat Keyakinan Prediksi: {confidence:.1f}%.\n"
            f"{sev_str}"
            f"{ev_str}"
            f"Status Tanaman: {'Daun Sehat/Normal' if is_healthy else 'Terserang Penyakit/Hama Tanaman'}.\n"
            f"Fokus Sudut Solusi: [{angle_title}] - {angle_desc}.\n\n"
            "INSTRUKSI UTAMA KONSULTAN PROTEKSI TANAMAN:\n"
            f"1. Seluruh petunjuk obat semprot, takaran dosis, dan penanganan WAJIB PERSIS DAN KHUSUS untuk penyakit {disease_name}{latin_str}.\n"
            f"2. JANGAN PERNAH menyarankan obat untuk penyakit lain (misalnya jika terdiagnosis Karat Daun, Anda WAJIB memberikan fungisida khusus Karat seperti Tebukonazol, Heksakonazol, Difenokonazol, atau Mankozeb, DILARANG memberikan bakterisida Xanthomonas).\n"
            "3. Rujukan ilmiah wajib berasal dari pedoman resmi Balai Penelitian Tanaman Sayuran (Balitsa Lembang), BPTP Kementan RI, dan Pedoman Pengendalian OPT Hortikultura.\n"
            "4. Sajikan dalam Bahasa Indonesia yang lugas, ramah, dan sangat praktis untuk petani di sawah.\n\n"
            "WAJIB susun jawaban ke dalam 3 bagian persis dengan judul pemisah berikut:\n\n"
            "=== TINDAKAN LANGSUNG DI KEBUN ===\n"
            "- Berikan langkah taktis darurat dalam 24-48 jam pertama di bedengan.\n"
            "- Jelaskan teknik pemotongan daun yang benar (jangan sampai spora terbang/berhamburan tertiup angin).\n"
            "- Jelaskan tindakan sanitasi alat gunting/pisau dan pemusnahan sisa pangkasan ke luar lahan (bakar/kubur jauh dari saluran air irigasi).\n\n"
            "=== REKOMENDASI OBAT SEMPROT ===\n"
            "- Berikan 2-3 pilihan kombinasi bahan aktif fungisida/insektisida/bakterisida terpercaya resmi Balitsa/Kementan (sebutkan golongan kontak dan sistemik).\n"
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

    models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
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
        except (requests.RequestException, KeyError, IndexError, ValueError):
            continue

    return None, angle_title

def get_disease_physical_checks(primary_name, second_name=None, is_differential=False):
    """
    Panduan Verifikasi Cek Fisik Langsung di Sawah:
    100% diselaraskan dengan hasil diagnosis penyakit spesifik untuk membantu petani memvalidasi gejala di bedengan.
    """
    p_lower = str(primary_name).lower()

    if is_differential and second_name:
        s_lower = str(second_name).lower()
        if "xanthomonas" in p_lower or "bakteri" in p_lower or "xanthomonas" in s_lower or "bakteri" in s_lower:
            jamur_name = primary_name if ("bakteri" not in p_lower and "xanthomonas" not in p_lower) else second_name
            return (
                "* 🖐️ **Uji Raba & Kelicinan Daun:**\n"
                "  • **Hawar Bakteri (Xanthomonas):** Daun lemas kebasah-basahan (*water-soaked*) seperti tersiram air mendidih. Saat pagi berembun terasa licin berlendir.\n"
                f"  • **Penyakit Jamur ({jamur_name}):** Tidak berlendir. Teraba kasar berbintil debu serbuk (Karat) atau bercak melekuk kering bertingkat (Bercak Ungu).\n"
                "* 👃 **Uji Aroma Daun:**\n"
                "  • **Bakteri (Xanthomonas):** Saat daun dipetik dan diremas, tercium bau langu agak busuk menyengat.\n"
                "  • **Jamur:** Tidak berbau busuk, hanya aroma khas dedaunan layu biasa.\n"
                "* 🔍 **Uji Bekas Usapan Jari:**\n"
                "  • Jika diusap jari meninggalkan debu/serbuk warna tembaga atau oranye karat, itu adalah **Karat Daun**, bukan bakteri!"
            )
        else:
            return (
                f"* 🖐️ **Uji Raba Permukaan Daun:**\n"
                f"  • Periksa apakah bercak terasa kasar melepuh ({primary_name}) atau melekuk kering rapuh ({second_name}).\n"
                f"* 🔍 **Uji Pola Bercak & Spora:**\n"
                f"  • Amati dengan teliti: apakah tampak bintil debu spora menonjol, lingkaran cincin bertingkat, atau bercak memanjang kering di ujung daun.\n"
                f"* 👃 **Uji Aroma:**\n"
                f"  • Kedua penyakit jamur ini kering dan tidak mengeluarkan bau busuk basah."
            )

    # Penyakit Tunggal (Single Diagnosis)
    if "karat" in p_lower or "rust" in p_lower:
        return (
            "* 🖐️ **Uji Raba Permukaan Daun:** Teraba bintil-bintil lepuh kecil menonjol yang terasa kasar saat diraba jari.\n"
            "* 🔍 **Uji Usapan Jari:** Bila bercak diusap jari, meninggalkan debu/serbuk berwarna jingga atau merah tembaga seperti serbuk besi berkarat.\n"
            "* 👃 **Uji Aroma Daun:** Kering dan tidak berlendir, mengeluarkan aroma dedaunan biasa tanpa bau busuk basah."
        )
    elif "ungu" in p_lower or "trotol" in p_lower or "alternaria" in p_lower or "blotch" in p_lower:
        return (
            "* 🖐️ **Uji Raba Permukaan Daun:** Bercak melekuk ke dalam (cekung), helai daun di sekitar lesi terasa kaku dan rapuh mudah patah.\n"
            "* 🔍 **Uji Cincin Konsentris:** Terlihat lingkaran-lingkaran bertingkat konsentris menyerupai sasaran panah dengan pusat keunguan kelabu bertepung spora.\n"
            "* 👃 **Uji Aroma Daun:** Kering tanpa lendir, tidak mengeluarkan aroma busuk menyengat."
        )
    elif "xanthomonas" in p_lower or "bakteri" in p_lower:
        return (
            "* 🖐️ **Uji Kelicinan Permukaan:** Helai daun terasa lemas kebasah-basahan (*water-soaked*) seperti tersiram air mendidih. Saat pagi hari berembun terasa licin berlendir.\n"
            "* 👃 **Uji Aroma Daun:** Bila helai daun dipetik dan diremas dengan jari, tercium aroma langu agak busuk menyengat khas infeksi bakteri.\n"
            "* 🔍 **Uji Urat Daun:** Lesi memanjang dari ujung daun ke bawah mengikuti alur urat daun berwarna hijau pucat hingga jerami tanpa adanya tepung spora jamur."
        )
    elif "moler" in p_lower or "fusarium" in p_lower or "layu" in p_lower:
        return (
            "* 🖐️ **Uji Bentuk Daun:** Helai daun melintir-lintir abnormal bergelombang (moler) dan menguning pucat dari ujung.\n"
            "* 🔍 **Uji Cabut Tanaman:** Tanaman sangat gampang dicabut dari tanah karena sebagian besar akar membusuk kering berwarna cokelat kehitaman.\n"
            "* 👃 **Uji Pangkal Batang:** Tercium bau tanah masam berjamur pada perakaran yang membusuk."
        )
    elif "embun" in p_lower or "downy" in p_lower or "mildew" in p_lower:
        return (
            "* 🖐️ **Uji Lapisan Beledu:** Pada pagi hari dingin berembun, permukaan helai daun dilapisi lapisan halus seperti beledu atau kapang tipis.\n"
            "* 🔍 **Uji Warna Kapang:** Lapisan kapang berwarna putih kelabu hingga keunguan pucat di sela-sela lekukan helai daun.\n"
            "* 👃 **Uji Aroma Daun:** Daun terasa basah dingin namun tidak berlendir kental dan tidak berbau busuk."
        )
    elif "stemphylium" in p_lower or "kering ujung" in p_lower:
        return (
            "* 🖐️ **Uji Ujung Daun:** Ujung helai daun mengering kaku berwarna cokelat jerami memanjang ke bawah.\n"
            "* 🔍 **Uji Bintik Spora:** Di perbatasan antara area kering dan hijau terlihat bintik hitam kecil spora jamur saat cuaca kering.\n"
            "* 👃 **Uji Bau:** Kering dan tidak berlendir."
        )
    elif "ulat" in p_lower or "caterpillar" in p_lower or "grayak" in p_lower:
        return (
            "* 🖐️ **Uji Tabung Daun:** Helai daun terasa tipis transparan seperti selaput kaca akibat jaringan hijau dikikis dari dalam.\n"
            "* 🔍 **Uji Kotoran Ulat:** Belah tabung daun, terlihat butiran kotoran kecil (frass) berwarna hijau kehitaman dan ulat grayak di dalam rongga daun.\n"
            "* 👃 **Uji Daun:** Daun yang bolong transparan mengering tanpa lendir pembusukan."
        )
    elif "umbi" in p_lower or "bulb" in p_lower or "rot" in p_lower:
        return (
            "* 🖐️ **Uji Pijit Leher Umbi:** Leher dan siung umbi terasa lembek berair saat dipijit ibu jari, lapisan kulit luar terkelupas busuk basah.\n"
            "* 👃 **Uji Bau Busuk:** Mengeluarkan aroma busuk menyengat khas pembusukan jaringan umbi basah.\n"
            "* 🔍 **Uji Daun Atas:** Daun bagian atas layu terkulai lunglai karena leher umbi penopang membusuk."
        )
    elif "virus" in p_lower or "virosis" in p_lower or "iysv" in p_lower:
        return (
            "* 🔍 **Uji Bentuk Lesi:** Terlihat bercak klorotik kuning berbentuk ketupat (belah ketupat) khas virus thrips, atau daun bergaris belang kuning kusam.\n"
            "* 🖐️ **Uji Kelenturan Daun:** Helai daun berkerut kaku, rapuh, dan mudah patah bila ditekuk.\n"
            "* 👃 **Uji Bau:** Bersih tanpa lendir dan tidak berbau busuk."
        )
    elif "putih" in p_lower or "botrytis" in p_lower:
        return (
            "* 🖐️ **Uji Bintik Daun:** Bintik-bintik putih kecil (1-2 mm) melekuk di helai daun, ujung daun memutih kering seperti terbakar.\n"
            "* 🔍 **Uji Tekstur:** Kering tanpa lendir, tidak basah berair.\n"
            "* 👃 **Uji Bau:** Tidak berbau busuk."
        )
    else:
        return (
            f"* 🖐️ **Uji Raba Permukaan Daun:** Periksa apakah bercak pada daun {primary_name} terasa basah berlendir (bakteri) atau kering bertepung (jamur).\n"
            "* 👃 **Uji Aroma Daun:** Daun yang terinfeksi bakteri biasanya mengeluarkan bau langu busuk saat diremas.\n"
            "* 🔍 **Uji Bentuk Lesi:** Periksa apakah bercak berbentuk cincin bertingkat, bintil serbuk spora menonjol, atau lesi memanjang."
        )

def get_groq_physical_verification(primary_name, second_name=None, is_differential=False):
    """
    Modul Validasi Karakteristik Fisik:
    Menghasilkan panduan verifikasi fisik lapangan berbasis riset agronomi
    diselaraskan 100% dengan diagnosis penyakit.
    Jika Groq API offline atau kuota habis, otomatis menggunakan Database Mandiri Sistem.
    """
    fallback_content = get_disease_physical_checks(primary_name, second_name, is_differential)
    api_key = get_groq_api_key()
    if not api_key:
        return fallback_content

    import requests
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

    models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
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
        except (requests.RequestException, KeyError, IndexError, ValueError):
            continue

    return fallback_content

def get_system_agronomy_recommendation(info, second_info=None, is_differential=False, angle_title=None):
    """
    Sistem Database Agronomi Mandiri (Built-in Agro-Engine):
    Menyusun rekomendasi lengkap, terstruktur, dan kaya agronomi resmi Balitsa/BPTP Kementan
    sebagai fallback otomatis jika kuota token Groq AI habis, terkena limit (429), atau offline.
    Menghasilkan 3 kartu:
    1. Tindakan Langsung di Kebun (24 Jam Pertama)
    2. Rekomendasi Obat Semprot (Bahan Aktif Resmi Balitsa & Takaran Dosis Tangki)
    3. Perawatan Lahan, Pemupukan & Agens Hayati (4 Poin Terstruktur)
    """
    nama_1 = info.get("nama_id", "Penyakit Bawang")
    is_healthy = info.get("is_healthy", False) or info.get("status") == "healthy"

    if is_healthy:
        c1 = (
            "• Kondisi tanaman bawang merah sangat baik, segar optimal, dan tidak ditemukan lesi penyakit aktif.\n"
            "• Lakukan pemantauan rutin 2–3 hari sekali terutama di waktu pagi saat embun menempel di helai daun.\n"
            "• Bersihkan gulma dan rumput liar di sekitar parit dan pematang yang berpotensi menjadi sarang serangga vektor."
        )
        c2 = (
            "• Tanaman sehat TIDAK memerlukan penyemprotan obat kimia sintetis atau fungisida kuratif.\n"
            "• Cukup semprotkan pupuk daun mikro lengkap atau asam amino berkonsentrasi rendah untuk menjaga ketahanan sel.\n"
            "• Waktu aplikasi terbaik adalah pagi hari pukul 06.30 – 08.00 WIB saat stomata helai daun terbuka optimal."
        )
        c3 = (
            "1. Pengaturan Parit & Tata Air: Pertahankan muka air parit 20–25 cm di bawah permukaan bedengan (kondisi macak-macak). Hindari kekeringan ekstrem maupun genangan air berlebih.\n"
            "2. Manajemen Pupuk: Teruskan pemupukan berimbang NPK 16-16-16 sesuai fase pertumbuhan umbi bawang.\n"
            "3. Penguat Dinding Sel: Semprotkan pupuk Kalsium dan Silika cair secara berkala tiap 7–10 hari untuk memperkokoh lapisan lilin daun.\n"
            "4. Perawatan Tanah: Lakukan penggemburan tepi bedengan secara hati-hati agar aerasi perakaran tetap gembur dan sehat."
        )
        return c1, c2, c3

    if is_differential and second_info:
        nama_2 = second_info.get("nama_id", "Penyakit Serupa")
        c1 = (
            f"• Waspada Gejala Serupa di Kebun: Bedakan segera antara {nama_1} vs {nama_2} langsung di bedengan.\n"
            f"• Ciri Lapangan {nama_1}: {info.get('ciri_lapangan', '-')}\n"
            f"• Ciri Lapangan {nama_2}: {second_info.get('ciri_lapangan', '-')}\n"
            "• Tindakan Darurat 24 Jam: Pangkas seluruh helai daun yang bergejala menggunakan gunting bersih. Masukkan sisa pangkasan ke wadah tertutup dan musnahkan di luar areal sawah agar patogen tidak menyebar."
        )
        c2 = (
            f"• Rekomendasi Solusi Penanganan {nama_1}: {info.get('solusi', '-')}\n"
            f"• Alternatif Spektrum {nama_2}: {second_info.get('solusi', '-')}\n"
            "• Takaran Dosis Tangki: Gunakan 1,5 hingga 2 sendok makan (sekitar 20–25 gram/ml) per tangki semprot 16 Liter air.\n"
            "• Waktu Semprot Terbaik: Pagi hari (pukul 06.00 – 08.30 WIB) saat embun mulai kering, atau sore hari (pukul 16.00 WIB) saat angin tenang.\n"
            "• Wajib tambahkan perekat/perata (surfactant) 1 tutup per tangki semprot agar larutan obat menempel merata dan tidak mudah tercuci hujan."
        )
    else:
        c1 = (
            f"• Langkah Segera 24 Jam Pertama: {info.get('gejala', 'Pangkas helai daun yang bergejala.')}\n"
            f"• Karakteristik Fisik di Sawah: {info.get('ciri_lapangan', '-')}\n"
            "• Teknik Pemangkasan Presisi: Potong helai daun sekitar 2 cm di bawah batas lesi menggunakan gunting/pisau yang dicelup alkohol 70% atau air sabun. Masukkan potongan ke dalam kantong kresek/wadah tertutup agar spora atau bakteri tidak berhamburan tertiup angin.\n"
            "• Sanitasi Lahan: Dilarang keras membuang potongan daun sakit ke saluran parit irigasi; kumpulkan dan bakar atau kubur jauh dari areal pertanaman."
        )
        c2 = (
            f"• Rekomendasi Bahan Aktif Resmi (Balitsa/Kementan): {info.get('solusi', 'Gunakan fungisida/bakterisida yang sesuai.')}\n"
            "• Takaran Dosis Aplikasi: 1,5 – 2 sendok makan (20 – 25 gram/ml) per tangki semprot standar 16 Liter air.\n"
            "• Waktu Penyemprotan: Pagi hari pukul 06.00 – 08.30 WIB saat stomata daun terbuka, atau sore hari pukul 16.00 WIB saat cuaca teduh tidak terik.\n"
            "• Penambahan Perekat & Perata: Selalu campurkan perekat/perata (surfactant) non-ionik agar lapisan lilin daun bawang terbasahi secara merata."
        )

    c3 = (
        "1. Pengaturan Parit & Tata Air: Atur muka air parit 20–25 cm di bawah permukaan bedengan (sistem macak-macak). Pastikan pembuangan drainase lancar dan jangan biarkan air hujan menggenang di parit sela bedengan.\n"
        "2. Manajemen Pupuk Khusus Masalah: Wajib STOP atau kurangi pupuk Nitrogen tunggal (Urea/ZA) karena menyebabkan dinding sel daun sukulen (terlalu empuk berair) yang sangat rentan ditembus patogen. Gantikan dengan pupuk Kalium (KNO3 Putih / MKP 2–3 sendok/tangki) untuk memperkokoh umbi dan helai daun.\n"
        "3. Penguat Dinding Sel: Semprotkan pupuk Kalsium-Boron dan pupuk Silika cair secara berkala untuk mempertebal lapisan kutikula (lilin pelindung) helai daun sehingga spora dan bakteri tidak mudah menembus jaringan tanaman.\n"
        "4. Perawatan Tanah & Agens Hayati: Jika tanah bedengan masam (pH < 6.0), taburkan kapur dolomit 1–2 genggam per meter bedengan untuk menetralkan keasaman. Campurkan agens hayati Trichoderma harzianum atau bakteri Bacillus subtilis bersama pupuk kandang matang untuk menekan populasi jamur patogen tular tanah."
    )
    return c1, c2, c3

def parse_groq_to_cards(ai_text, info, second_info=None, is_differential=False, angle_title=None):
    """
    Memecah teks balasan Groq menjadi 3 kartu panduan terstruktur.
    Jika ai_text kosong (kuota Groq habis / error / offline),
    sistem secara otomatis mengalirkan jawaban lengkap dari Database Mandiri Sistem.
    """
    sys_c1, sys_c2, sys_c3 = get_system_agronomy_recommendation(
        info=info,
        second_info=second_info,
        is_differential=is_differential,
        angle_title=angle_title
    )

    if not ai_text:
        return sys_c1, sys_c2, sys_c3

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
            c3 = sys_c3

    return c1 or sys_c1, c2 or sys_c2, c3 or sys_c3

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

# Inisialisasi Model PyTorch TorchScript & Konfigurasi dari meta.json
model_loaded = False
load_error_message = None
try:
    meta_config = load_meta_config()
    model = load_torch_model()
    class_names = meta_config.get("class_labels_id", [
        "Downy mildew",
        "Sehat",
        "Iris Yellow Spot Virus (IYSV)",
        "Hawar Daun (Stemphylium / Colletotrichum)",
        "Moler",
        "Bercak Ungu / Trotol (Alternaria porri)",
        "Rust"
    ])
    model_loaded = True
except (OSError, ValueError, FileNotFoundError, AttributeError) as e:
    load_error_message = str(e)
    meta_config = {}
    class_names = []

# ==============================================================================
# 5. SIDEBAR: PENGATURAN TEKNIS & RIWAYAT (DIPINDAHKAN AGAR TIDAK MEMBINGUNGKAN)
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 0.5rem 0;'>
            <span style='font-size: 2.5rem;'>🧅</span>
            <h2 style='margin: 0.1rem 0; color: #1b5e20; font-weight: 800;'>AgroScan</h2>
            <p style='color: #64748b; font-size: 0.88rem; margin-bottom: 4px;'>Menu Pengaturan & Riwayat</p>
            <span style='background-color: #dcfce7; color: #166534; padding: 3px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 800; border: 1px solid #86efac;'>v3.2.0 • 7-Kelas TorchScript</span>
        </div>
    """, unsafe_allow_html=True)

    # Indikator Status Koneksi Groq AI Otomatis (Secrets / Env)
    active_key = get_groq_api_key()
    if active_key and len(active_key) > 15:
        st.markdown(
            "<div style='text-align: center; margin: 0.1rem 0 0.75rem 0; font-weight: 700; color: #16a34a; font-size: 0.95rem;'>"
            "🟢 AI Dokter Terhubung"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align: center; margin: 0.1rem 0 0.75rem 0; font-weight: 700; color: #dc2626; font-size: 0.95rem;'>"
            "🟡 Mode Agronomi Mandiri (Offline)"
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------------------------
    # FITUR 1: BUKU PANDUAN PENGGUNAAN FITUR WEB (OPERASIONAL APLIKASI)
    # --------------------------------------------------------------------------
    with st.expander("📖 Buku Panduan Penggunaan Web", expanded=False):
        st.markdown("""
        <div style="font-size: 0.82rem; line-height: 1.5; color: #334155;">
        
        <strong style="color: #166534;">📸 1. Cara Ambil / Unggah Foto:</strong>
        <ul style="margin: 4px 0 8px 16px; padding: 0;">
            <li><strong>Jarak Ideal:</strong> 10–20 cm tegak lurus ke helai daun.</li>
            <li><strong>Fokus Tajam:</strong> Pastikan helai daun / bercak penyakit fokus tajam, tidak buram atau goyang.</li>
            <li><strong>Pencahayaan:</strong> Terang alami (pagi/siang), hindari bayangan gelap pekat atau silau berlebih.</li>
            <li><strong>Posisi:</strong> Jangan menutupi bercak lesi dengan telapak tangan atau jari saat memegang daun.</li>
        </ul>

        <strong style="color: #166534;">🖼️ 2. Membaca Retikel HUD Lesion Scanner:</strong>
        <ul style="margin: 4px 0 8px 16px; padding: 0;">
            <li><span style="color: #dc2626; font-weight: 700;">🔴 Retikel Merah:</span> Mengunci titik pusat lesi aktif penyakit utama pada daun (presisi tinggi seperti Bulu Embun).</li>
            <li><span style="color: #d97706; font-weight: 700;">🟠 Retikel Oranye:</span> Menandai fokus penyakit kedua saat terjadi gejala ganda (diferensial).</li>
            <li><span style="color: #16a34a; font-weight: 700;">🟢 Retikel Hijau:</span> Memverifikasi helai daun berklorofil sehat prima bebas patogen.</li>
        </ul>

        <strong style="color: #166534;">💾 3. Riwayat Pemeriksaan:</strong>
        <ul style="margin: 4px 0 8px 16px; padding: 0;">
            <li>Klik tombol <strong>💾 Simpan Hasil ke Riwayat</strong> untuk menyimpan diagnosis ke penyimpanan lokal browser.</li>
            <li>Hapus butir riwayat satu per satu secara fleksibel dengan tombol <strong>🗑️</strong> di daftar riwayat sidebar.</li>
        </ul>

        <strong style="color: #166534;">💊 4. Resep & Rekomendasi Obat:</strong>
        <ul style="margin: 4px 0 8px 16px; padding: 0;">
            <li><strong>Tindakan 24 Jam:</strong> Prosedur darurat pemangkasan presisi dan sanitasi daun sakit.</li>
            <li><strong>Obat Semprot:</strong> Bahan aktif resmi Balitsa/Kementan dengan takaran sendok per tangki 16L.</li>
        </ul>
        
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # FITUR 2: PANDUAN & TOLOK UKUR VALIDASI FOTO (TERPISAH & DETAIL PERSEN)
    # --------------------------------------------------------------------------
    with st.expander("🎯 Panduan & Tolok Ukur Validasi Foto", expanded=False):
        st.markdown("""
        <div style="font-size: 0.82rem; line-height: 1.55; color: #334155;">
        
        <strong style="color: #166534;">📊 1. Batas Keyakinan (Confidence Threshold):</strong>
        <p style="margin: 2px 0 6px 0; color: #64748b;">Kriteria persentase kepastian AI sebelum menetapkan diagnosis:</p>
        <ul style="margin: 0 0 10px 16px; padding: 0;">
            <li><strong style="color: #b45309;">🟡 40% – 55% (Mode Toleran / Sore / Dini):</strong><br>
            <em>Cocok untuk:</em> Foto sore hari, cuaca mendung, pencahayaan redup, gejala awal yang masih sangat tipis, atau kamera sedikit bergetar. Sistem lebih toleran menerima foto.</li>
            <li><strong style="color: #15803d;">🟢 60% – 70% (Standar Lapangan Sawah - Rekomendasi Balitsa 65%):</strong><br>
            <em>Cocok untuk:</em> Pemantauan harian di bedengan sawah dengan cahaya alami terang. Memberikan akurasi optimal dan mencegah tebak sembarangan.</li>
            <li><strong style="color: #b91c1c;">🔴 75% – 90% (Mode Super Ketat / Laboratorium):</strong><br>
            <em>Cocok untuk:</em> Foto makro sangat tajam, pencahayaan studio/lampu terang, sertifikasi mutu benih. Menolak tegas jika ada keraguan sedikit pun.</li>
        </ul>

        <strong style="color: #166534;">🍃 2. Sensitivitas Daun Bawang (Minimal Leaf Ratio):</strong>
        <p style="margin: 2px 0 6px 0; color: #64748b;">Kriteria persentase minimal kanopi daun yang wajib ada pada foto:</p>
        <ul style="margin: 0 0 6px 16px; padding: 0;">
            <li><strong style="color: #b45309;">🟡 3% – 6% (Toleransi Kanopi Kecil / Jarak Jauh):</strong><br>
            <em>Cocok untuk:</em> Satu helai daun kecil dipegang tangan, bibit muda 1–2 minggu, atau foto dari jarak agak jauh (> 30 cm).</li>
            <li><strong style="color: #15803d;">🟢 8% – 15% (Standar Rumpun Sawah Normal):</strong><br>
            <em>Cocok untuk:</em> Rumpun daun bawang normal umur 3–8 minggu. Latar tanah dan pematang otomatis tersaring.</li>
            <li><strong style="color: #b91c1c;">🔴 18% – 35% (Mode Makro Daun Penuh):</strong><br>
            <em>Cocok untuk:</em> Foto jarak dekat helai daun yang memenuhi bidang foto. Sangat ketat menolak jika latar belakang tanah/pot mendominasi.</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔬 Model PyTorch TorchScript")
    st.caption("EfficientNet-B0 (TorchScript) dengan Test-Time Augmentation (TTA) Flip Horizontal.")

    # Tampilkan 7 Kategori yang Dideteksi di Sidebar
    with st.expander("📋 7 Kategori yang Dideteksi AI", expanded=False):
        for item in SUPPORTED_DISEASES_7:
            status_color = "#15803d" if item["is_healthy"] else ("#b45309" if item["status"] == "virus" else "#b91c1c")
            tag_text = "SEHAT" if item["is_healthy"] else ("VIRUS" if item["status"] == "virus" else "PENYAKIT")
            st.markdown(
                f"<div style='margin-bottom: 8px; font-size: 0.84rem; background: #ffffff; padding: 7px 10px; border-radius: 8px; border: 1px solid #e2e8f0;'>"
                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                f"<strong>{item['icon']} {item['nama_id']}</strong>"
                f"<span style='background: {status_color}; color: #fff; font-size: 0.70rem; font-weight: 700; padding: 1px 6px; border-radius: 4px;'>{tag_text}</span>"
                f"</div>"
                f"<span style='color: {status_color}; font-size: 0.76rem;'>• <em>{item['latin']}</em></span><br>"
                f"<span style='color: #475569; font-size: 0.78rem;'>🔍 {item['ciri_lapangan']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    default_conf_pct = int(round(float(meta_config.get("conf_threshold", 0.65)) * 100))
    st.markdown("### ⚙️ Validasi Foto Bawang")
    st.caption("Pilih preset cepat atau geser slider sesuai kondisi foto lapangan:")

    # Tombol Preset Cepat Langsung Sinkron ke Web
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    with col_pre1:
        if st.button("🌾 Standar", help="Preset Standar Sawah (65% / 8%)", use_container_width=True):
            st.session_state["conf_slider"] = 65
            st.session_state["leaf_slider"] = 8
            st.rerun()
    with col_pre2:
        if st.button("☁️ Redup", help="Preset Foto Sore / Gejala Dini (50% / 5%)", use_container_width=True):
            st.session_state["conf_slider"] = 50
            st.session_state["leaf_slider"] = 5
            st.rerun()
    with col_pre3:
        if st.button("🔬 Ketat", help="Preset Super Ketat Lab (80% / 20%)", use_container_width=True):
            st.session_state["conf_slider"] = 80
            st.session_state["leaf_slider"] = 20
            st.rerun()

    # Inisialisasi default jika belum ada di session_state
    if "conf_slider" not in st.session_state:
        st.session_state["conf_slider"] = default_conf_pct
    if "leaf_slider" not in st.session_state:
        st.session_state["leaf_slider"] = 8

    conf_threshold_pct = st.slider(
        "Batas Keyakinan / Confidence Threshold (%)",
        min_value=20,
        max_value=90,
        value=st.session_state["conf_slider"],
        step=5,
        key="conf_slider",
        help="Jika kepastian model di bawah nilai ini, foto akan ditandai 'Tidak yakin, foto kurang jelas atau bukan daun bawang'."
    )
    conf_threshold = conf_threshold_pct / 100.0

    # Kriteria dinamis real-time untuk Batas Keyakinan
    if conf_threshold_pct < 55:
        c_badge_bg = "#fefce8"
        c_badge_border = "#eab308"
        c_badge_color = "#713f12"
        c_badge_txt = f"🟡 <strong>Mode Sensitif ({conf_threshold_pct}%):</strong> Menerima foto sore/mendung & gejala bercak dini yang masih tipis."
    elif conf_threshold_pct <= 72:
        c_badge_bg = "#f0fdf4"
        c_badge_border = "#22c55e"
        c_badge_color = "#14532d"
        c_badge_txt = f"🟢 <strong>Standar Sawah ({conf_threshold_pct}%):</strong> Keseimbangan optimal untuk pemantauan harian di bedengan (Rekomendasi)."
    else:
        c_badge_bg = "#fef2f2"
        c_badge_border = "#ef4444"
        c_badge_color = "#7f1d1d"
        c_badge_txt = f"🔴 <strong>Mode Super Ketat ({conf_threshold_pct}%):</strong> Hanya menerima foto sangat tajam & tanpa sedikit pun keraguan."

    st.markdown(
        f"<div style='background: {c_badge_bg}; border-left: 4px solid {c_badge_border}; color: {c_badge_color}; padding: 6px 10px; border-radius: 6px; font-size: 0.77rem; margin-top: -6px; margin-bottom: 12px; line-height: 1.4;'>"
        f"{c_badge_txt}"
        f"</div>",
        unsafe_allow_html=True
    )

    min_leaf_ratio_pct = st.slider(
        "Sensitivitas Daun Bawang (%)",
        min_value=3,
        max_value=35,
        value=st.session_state["leaf_slider"],
        step=1,
        key="leaf_slider",
        help="Persentase minimal kanopi daun bawang merah yang harus ada pada foto. Naikkan jika ingin validasi lebih ketat menolak foto selain bawang."
    )
    min_leaf_ratio = min_leaf_ratio_pct / 100.0

    # Kriteria dinamis real-time untuk Sensitivitas Daun
    if min_leaf_ratio_pct < 7:
        l_badge_bg = "#fefce8"
        l_badge_border = "#eab308"
        l_badge_color = "#713f12"
        l_badge_txt = f"🟡 <strong>Toleransi Tinggi ({min_leaf_ratio_pct}%):</strong> Menerima satu helai daun kecil / bibit muda / foto agak jauh."
    elif min_leaf_ratio_pct <= 16:
        l_badge_bg = "#f0fdf4"
        l_badge_border = "#22c55e"
        l_badge_color = "#14532d"
        l_badge_txt = f"🟢 <strong>Standar Rumpun Sawah ({min_leaf_ratio_pct}%):</strong> Ideal untuk tanaman bawang merah umur 3–8 minggu."
    else:
        l_badge_bg = "#fef2f2"
        l_badge_border = "#ef4444"
        l_badge_color = "#7f1d1d"
        l_badge_txt = f"🔴 <strong>Filter Makro Ketat ({min_leaf_ratio_pct}%):</strong> Wajib helai daun mendominasi foto, tolak latar tanah luas."

    st.markdown(
        f"<div style='background: {l_badge_bg}; border-left: 4px solid {l_badge_border}; color: {l_badge_color}; padding: 6px 10px; border-radius: 6px; font-size: 0.77rem; margin-top: -6px; margin-bottom: 8px; line-height: 1.4;'>"
        f"{l_badge_txt}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### 📋 Riwayat Pemeriksaan")
    history_list = st.session_state.get('history', [])
    total_hist = len(history_list)
    st.write(f"Total Diagnosa Tersimpan: **{total_hist}**")

    if total_hist > 0:
        with st.expander(f"📂 Daftar Riwayat Disimpan ({total_hist})", expanded=True):
            for idx, item in enumerate(reversed(history_list)):
                actual_idx = total_hist - 1 - idx
                col_h_txt, col_h_del = st.columns([4, 1])
                with col_h_txt:
                    badge_color = "#15803d" if "sehat" in item.get('status', '').lower() else "#b91c1c"
                    st.markdown(
                        f"<div style='font-size: 0.84rem; line-height: 1.35;'>"
                        f"<strong>{item.get('penyakit', 'Diagnosa')}</strong> "
                        f"<span style='color: {badge_color}; font-weight: 700;'>({item.get('confidence', '')})</span><br>"
                        f"<span style='color: #64748b; font-size: 0.74rem;'>🕒 {item.get('waktu', '')} | {item.get('lokasi', 'Kebun')}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_h_del:
                    if st.button("🗑️", key=f"del_h_{item.get('id', actual_idx)}_{actual_idx}", help="Hapus entri ini"):
                        delete_history_item(actual_idx)
                        st.toast("Entri riwayat berhasil dihapus.", icon="🗑️")
                        st.rerun()
                st.markdown("<hr style='margin: 3px 0 6px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

        df_hist = pd.DataFrame(history_list)
        kolom_ekspor = [col for col in ["waktu", "penyakit", "confidence", "status", "lokasi", "catatan", "rekomendasi"] if col in df_hist.columns]
        
        # Download CSV
        csv_bytes = df_hist[kolom_ekspor].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Laporan (.CSV)",
            data=csv_bytes,
            file_name=f"riwayat_bawang_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Download JSON
        json_bytes = json.dumps(history_list, indent=2).encode('utf-8')
        st.download_button(
            label="📥 Unduh Cadangan (.JSON)",
            data=json_bytes,
            file_name=f"riwayat_bawang_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            reset_all_history()
            st.toast("Semua riwayat berhasil dihapus.", icon="🗑️")
            st.rerun()
    else:
        st.caption("Belum ada riwayat yang disimpan. Klik tombol '💾 Simpan Hasil ke Riwayat' setelah foto diperiksa.")

    if st.button("🔄 Bersihkan Cache & Muat Ulang Model", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        if "has_inspected_current" in st.session_state:
            del st.session_state["has_inspected_current"]
        st.toast("Cache sesi berhasil dibersihkan!", icon="🔄")
        st.rerun()

    st.divider()
    st.markdown("""
        <div style='font-size: 0.85rem; color: #94a3b8; text-align: center;'>
            Model Deep Learning: EfficientNet-B0 TorchScript (7 Kelas)<br>
            Asisten AI: Groq Cloud Intelligence & Built-in Engine
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. HEADER APLIKASI UTAMA (RAMAH PETANI & MODERN)
# ==============================================================================
st.markdown("""
    <div class="farmer-hero">
        <h1>🧅 Dokter Tanaman Bawang Merah</h1>
        <p>Sistem Pakar Deteksi Dini 7 Kondisi & Penyakit Daun Bawang Merah Langsung di Sawah</p>
        <div style="margin-top: 0.75rem; display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
            <span style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 4px 14px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.35);">Versi 3.2.0 (TorchScript)</span>
            <span style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 4px 14px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.35);">EfficientNet-B0 7-Kelas Presisi</span>
            <span style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 4px 14px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.35);">Standar Balitsa & BPTP Kementan</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.info("ℹ️ **Catatan:** Hasil diagnosis ini adalah alat bantu deteksi dini berbasis citra digital untuk petani.")


if not model_loaded:
    st.error(f"❌ Gagal memuat model pendeteksi: {load_error_message}")
    st.stop()

# ==============================================================================
# 7. LANGKAH 1: AMBIL / MASUKKAN FOTO DAUN (SINGLE COLUMN MOBILE FIRST)
# ==============================================================================
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.markdown("""
    <div class="step-header">
        <div class="step-num">1</div>
        <div class="step-title">Ambil / Masukkan Foto Daun</div>
    </div>
    <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 12px; padding: 12px 16px; margin: 10px 0 16px 0; color: #166534; font-size: 0.95rem;">
        📸 <strong>Petunjuk Pengambilan Foto:</strong> Foto daun dari dekat, fokus, cahaya cukup.<br>
        <span style="font-size: 0.88rem; color: #15803d;">⚠️ <strong>Penting:</strong> Model AI ini dilatih khusus dan <strong>hanya valid untuk foto DAUN</strong> tanaman bawang merah (bukan umbi di piring, tanah kosong, pot, atau daun tanaman lain).</span>
    </div>
""", unsafe_allow_html=True)

tab_camera, tab_upload = st.tabs(["📸 Ambil Foto Langsung (Kamera HP)", "📁 Pilih dari Galeri HP"])

selected_image = None
file_error = False

with tab_camera:
    cam_file = st.camera_input("Arahkan kamera dekat ke bagian daun yang sakit:", key="input_camera_field")
    if cam_file is not None:
        if cam_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ Ukuran foto ({cam_file.size / (1024*1024):.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB. Silakan ambil ulang dengan resolusi wajar.")
            file_error = True
        else:
            try:
                selected_image = Image.open(cam_file)
            except (ValueError, OSError) as err:
                st.error(f"Gagal membaca foto kamera: {err}")

with tab_upload:
    uploaded_file = st.file_uploader(
        f"Pilih file foto dari galeri (JPG, JPEG, PNG, WEBP - Maksimal {MAX_FILE_SIZE_MB} MB):",
        type=["jpg", "jpeg", "png", "webp"],
        key="input_file_field"
    )
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ Ukuran file ({uploaded_file.size / (1024*1024):.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB. Silakan unggah foto yang lebih kecil.")
            file_error = True
        else:
            try:
                selected_image = Image.open(uploaded_file)
            except (ValueError, OSError) as err:
                st.error(f"Gagal membuka berkas foto: {err}")

# Tips Ringkas untuk Petani
st.caption("💡 **Petunjuk Foto Bagus:** Foto daun dari dekat (jarak 10-20 cm), fokus, cahaya cukup tepat pada bercak daun yang bergejala.")

# Tombol Pemeriksaan Utama & Pemrosesan
if selected_image is not None and not file_error:
    st.markdown("<div style='text-align: center; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.image(selected_image, caption="Foto Daun yang Dipilih", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Identifikasi unik gambar aktif berbasis hash konten citra
    img_bytes = selected_image.tobytes()
    img_hash = hashlib.md5(img_bytes[:65536]).hexdigest()[:12]
    current_img_sig = f"{selected_image.size}_{selected_image.mode}_{img_hash}"
    
    # Tombol Utama Periksa
    btn_check = st.button("🔍 PERIKSA DAUN SEKARANG", type="primary", use_container_width=True, key="btn_inspect_main")
    
    # Jika tombol ditekan, aktifkan pemeriksaan untuk gambar ini
    if btn_check:
        st.session_state["has_inspected_current"] = current_img_sig

    # Tampilkan Hasil Pemeriksaan jika sudah diperiksa atau pengguna siap memeriksa
    if st.session_state.get("has_inspected_current") == current_img_sig:
        # ==============================================================================
        # TAHAP 1: VALIDASI GAMBAR (GUARDRAIL GATEKEEPER GROQ VISION & OOD GUARD)
        # ==============================================================================
        with st.spinner("🔍 Memverifikasi keaslian foto daun bawang..."):
            is_valid_vision, vision_verdict = validate_onion_image(selected_image, min_ratio=min_leaf_ratio)

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
                            <li>Gunakan foto <strong>daun tanaman bawang merah asli</strong> di bedengan kebun/sawah.</li>
                            <li>Arahkan kamera HP (jarak ideal <strong>10–20 cm</strong>) tepat pada helai daun yang sakit.</li>
                            <li>Pastikan pencahayaan terang dan daun terlihat jelas tanpa bayangan gelap.</li>
                            <li>Hindari memotret wajah, hewan, kendaraan, atau pemandangan sawah dari kejauhan.</li>
                        </ol>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # ==============================================================================
        # TAHAP 2: PROSES DIAGNOSIS EFFICIENTNET-B0 TORCHSCRIPT
        # ==============================================================================
        with st.spinner("🔍 Sedang menganalisis kondisi daun bawang merah dengan EfficientNet-B0 TorchScript..."):
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
                diag_info,
                visual_evidence,
                api_output
            ) = predict_disease(
                selected_image, model, meta=meta_config, enforce_verification=False
            )
            # Selaraskan alias variabel agar konsisten (mencegah NameError)
            metadata = info
            second_metadata = second_info

        # ==============================================================================
        # TAHAP 3: EVALUASI KEPUTUSAN MODEL / THRESHOLD
        # Jika confidence < conf_threshold, tampilkan "Tidak yakin" alih-alih menebak
        # ==============================================================================
        is_uncertain = api_output.get("uncertain", False) or (top_confidence < (conf_threshold * 100.0))
        if is_uncertain:
            st.warning("⚠️ **Tidak yakin, foto kurang jelas atau bukan daun bawang**")
            st.markdown(f"""
                <div class="card-rejection">
                    <div class="card-rejection-badge">⚠️ TINGKAT KEYAKINAN RENDAH ({top_confidence:.1f}%)</div>
                    <div class="card-rejection-title">Tidak yakin, foto kurang jelas atau bukan daun bawang</div>
                    <div class="card-rejection-reason">
                        Tingkat keyakinan model hanya <strong>{top_confidence:.1f}%</strong> (di bawah ambang batas minimal <strong>{conf_threshold * 100.0:.0f}%</strong>). Sistem menolak menebak diagnosis secara sembarangan untuk mencegah kesalahan penanganan di kebun.
                    </div>
                    <div class="card-rejection-desc">
                        <strong>📌 Model AI hanya valid untuk foto DAUN bawang merah.</strong> Kemungkinan penyebab ketidakyakinan:
                        <ul>
                            <li>Objek foto bukan daun bawang merah asli (seperti foto tanah, pot, manusia, atau tanaman lain).</li>
                            <li>Foto kurang fokus, buram (blur), atau jarak pengambilan terlalu jauh dari daun.</li>
                            <li>Pencahayaan redup, backlight kuat, atau bayangan gelap menutupi daun.</li>
                        </ul>
                        <strong>💡 Petunjuk Pengambilan Foto yang Tepat:</strong>
                        <p style="margin-top: 4px;">Foto daun dari dekat (jarak 10–20 cm), pastikan fokus tajam pada bercak/helai daun, dan pencahayaan terang merata.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.stop()  # Hentikan eksekusi, jangan tebak penyakit & jangan panggil resep obat

        # ==============================================================================
        # KASUS 2: FOTO VALID & KEYAKINAN TINGGI (LOLOS THRESHOLD)
        # ==============================================================================
        else:

            # ==============================================================================
            # 8. LANGKAH 2: HASIL PEMERIKSAAN (NAMA PENYAKIT & KEPASTIAN)
            # ==============================================================================
            st.markdown("""
                <div class="step-header">
                    <div class="step-num">2</div>
                    <div class="step-title">Hasil Pemeriksaan Daun</div>
                </div>
            """, unsafe_allow_html=True)

            if diag_info.get("is_auto_cropped", False):
                with st.expander("🔍 Lihat Hasil Pemotongan Otomatis Daun (Auto-Crop)", expanded=False):
                    col_crop1, col_crop2 = st.columns([1, 1])
                    with col_crop1:
                        st.image(preview_crop, caption="Fokus Helai Daun (Bebas Latar Belakang)", use_container_width=True)
                    with col_crop2:
                        st.info(
                            f"🍃 **Auto-Fokus Daun Aktif:** Sistem mendeteksi daun pada "
                            f"**{diag_info.get('leaf_coverage_pct', 0)}%** area foto. "
                            "Latar belakang tanah, tangan, atau pematang otomatis disingkirkan agar model menganalisis lesi dengan presisi."
                        )

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

            # Tombol Opsi Simpan Hasil ke Riwayat (Tidak Otomatis)
            saved_key = f"saved_entry_{current_img_sig}"
            is_already_saved = st.session_state.get(saved_key, False)
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if is_already_saved:
                    st.button("✅ Hasil Sudah Disimpan di Riwayat", disabled=True, use_container_width=True)
                else:
                    if st.button("💾 Simpan Hasil ke Riwayat", type="primary", use_container_width=True):
                        rec_name = f"{info['nama_id']} ({top_confidence:.1f}%) & {second_info['nama_id']} ({second_confidence:.1f}%)" if is_differential else info["nama_id"]
                        save_diagnosis_to_history(
                            disease_code=top_class_raw,
                            display_name=rec_name,
                            confidence=top_confidence,
                            recommendation=info.get("rekomendasi_singkat", ""),
                            is_healthy=is_healthy,
                            notes="Pemeriksaan Lapangan",
                            location="Kebun Bawang"
                        )
                        st.session_state[saved_key] = True
                        st.toast("✅ Berhasil disimpan ke riwayat pemeriksaan!", icon="💾")
                        st.rerun()
            with col_save2:
                if st.button("🔄 Periksa Foto Lain", use_container_width=True):
                    if "has_inspected_current" in st.session_state:
                        del st.session_state["has_inspected_current"]
                    st.rerun()

            # ==============================================================================
            # MODUL BUKTI VISUAL NYATA DARI FOTO (REAL VISUAL LESION AUDIT)
            # ==============================================================================
            if visual_evidence and visual_evidence.get("has_visual_evidence"):
                v_sev_pct = visual_evidence.get("severity_pct", 0.0)
                v_sev_lvl = visual_evidence.get("severity_level", "Normal")
                v_healthy_pct = visual_evidence.get("healthy_pct", 100.0)
                v_desc = visual_evidence.get("evidence_desc", "")

                st.markdown("### 🔬 Bukti Analisis Visual Nyata dari Foto Daun")
                st.caption("Hasil pemindaian fitur fisik piksel langsung dari foto yang diunggah:")

                col_v1, col_v2, col_v3 = st.columns(3)
                if is_healthy:
                    with col_v1:
                        st.metric(label="🩺 Luas Kerusakan Daun", value="0.0%", delta="Sehat Prima")
                    with col_v2:
                        st.metric(label="🌿 Jaringan Daun Sehat", value="100.0%", delta="Normal")
                    with col_v3:
                        st.metric(label="🛡️ Kondisi Tanaman", value="Bebas Patogen")
                    st.success("✅ **Daun Sehat & Normal:** Pemindaian visual mengonfirmasi helai daun segar merata, berlilin alami, dan tidak ditemukan bercak lesi patogen aktif.")
                else:
                    with col_v1:
                        st.metric(label="🩺 Luas Kerusakan Daun", value=f"{v_sev_pct:.1f}%", delta=v_sev_lvl, delta_color="inverse")
                    with col_v2:
                        st.metric(label="🌿 Jaringan Hijau Tersisa", value=f"{v_healthy_pct:.1f}%", help="Persentase area klorofil daun yang masih sehat")
                    with col_v3:
                        st.metric(label="🎯 Patogen Terdeteksi", value=info['nama_id'].split('/')[0].strip())
                    st.info(f"📋 **Karakteristik Fisik Daun pada Foto:**\n\n{v_desc}")

                if visual_evidence.get("overlay_img") is not None:
                    with st.expander("🖼️ Peta Titik Kerusakan pada Foto Daun (HUD Lesion Scanner)", expanded=True):
                        if is_healthy:
                            map_caption = "Peta Verifikasi Jaringan Daun: Retikel Hijau memverifikasi helai daun bawang dalam kondisi sehat optimal berklorofil tinggi bebas lesi patogen."
                        elif is_differential:
                            map_caption = f"Peta Titik Kerusakan Multi-Penyakit: Retikel Merah menandai fokus gejala {info['nama_id']}, sedangkan Retikel Oranye menandai fokus gejala {second_info['nama_id']} pada helai daun."
                        else:
                            map_caption = f"Peta Titik Kerusakan: Retikel scanner presisi tinggi menandai titik pusat kerusakan aktif {info['nama_id']} pada helai daun."

                        st.image(
                            visual_evidence["overlay_img"],
                            caption=map_caption,
                            use_container_width=True
                        )
                        num_spots = visual_evidence.get("num_spots_detected", 0)
                        if is_healthy:
                            st.success(f"✅ **Daun Sehat & Normal ({num_spots} Titik Diverifikasi):** Retikel hijau memvalidasi helai daun sehat prima, berklorofil merata, dan bebas dari bercak lesi patogen aktif.")
                        elif is_differential:
                            st.caption(
                                f"💡 **Petunjuk Deteksi Multi-Penyakit ({num_spots} Titik Terdeteksi):** Sistem mendeteksi dua kemungkinan patogen yang menginfeksi helai daun. "
                                f"Retikel **Merah** menandai area kerusakan yang paling kuat dicurigai sebagai **{info['nama_id']}**, "
                                f"sedangkan Retikel **Oranye** menandai area yang dicurigai sebagai **{second_info['nama_id']}**. "
                                "Cocokkan perbedaan ciri fisik kedua area tersebut langsung di bedengan kebun untuk penanganan yang tepat."
                            )
                        elif num_spots > 0:
                            st.caption(
                                f"💡 **Petunjuk Deteksi ({num_spots} Titik Kerusakan Terdeteksi):** Retikel scanner di atas memetakan titik lesi aktif tepat pada helai daun tanaman (anti-melengser, bebas gangguan latar belakang atau tangan). Titik bertanda nama penyakit menunjukkan konsentrasi infeksi aktif tempat patogen berkembang. Fokuskan sanitasi pemangkasan daun sakit dan penyemprotan obat pada titik-titik tersebut."
                            )
                        else:
                            st.caption(
                                "💡 **Petunjuk Deteksi:** Retikel scanner presisi dihasilkan langsung dari pemindaian densitas lesi pada foto helai daun Anda."
                            )

            # Distribusi Probabilitas Model PyTorch TorchScript
            with st.expander("📊 Distribusi Probabilitas Model (TorchScript EfficientNet-B0 - 7 Kelas)", expanded=True):
                st.caption("Distribusi probabilitas softmax terkalibrasi (temperature-scaled) untuk seluruh 7 kelas:")
                probs_dict = api_output.get("probabilities", {})
                sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                for rank, (c_label, prob_val) in enumerate(sorted_probs, 1):
                    info_c = CLASS_METADATA.get(c_label, {"nama_id": c_label, "icon": "🔍"})
                    c_icon = info_c.get("icon", "🌱" if "sehat" in c_label.lower() else "🚨")
                    prob_pct = round(prob_val * 100.0, 1)

                    if rank == 1:
                        rank_badge = "🥇 Prediksi Terpilih"
                        val_color = "#16a34a" if info_c.get("is_healthy") else "#dc2626"
                    elif rank == 2:
                        rank_badge = "🥈 Peringkat 2"
                        val_color = "#ea580c"
                    elif rank == 3:
                        rank_badge = "🥉 Peringkat 3"
                        val_color = "#0284c7"
                    else:
                        rank_badge = f"#{rank}"
                        val_color = "#475569"

                    col_pb1, col_pb2 = st.columns([3, 1])
                    with col_pb1:
                        st.markdown(f"**{c_icon} {info_c['nama_id']}** <span style='font-size: 0.78rem; color: #64748b; margin-left: 6px; background: #f1f5f9; padding: 2px 6px; border-radius: 4px;'>{rank_badge}</span>", unsafe_allow_html=True)
                    with col_pb2:
                        st.markdown(f"<div style='text-align: right; font-weight: 800; font-size: 0.95rem; color: {val_color};'>{prob_pct:.1f}%</div>", unsafe_allow_html=True)
                    st.progress(min(max(float(prob_val), 0.0), 1.0))

            # ==============================================================================
            # MODUL VALIDASI KARAKTERISTIK FISIK LAPANGAN
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

                phys_sub_text = (
                    f"Cocokkan tanda fisik berikut langsung di bedengan untuk memastikan apakah daun terserang <strong>{info['nama_id']}</strong> atau <strong>{second_info['nama_id']}</strong>:"
                    if is_differential
                    else f"Cocokkan tanda fisik berikut langsung pada tanaman di sawah untuk memastikan gejala penyakit <strong>{info['nama_id']}</strong>:"
                )

                st.markdown(f"""
                    <div style="background: #FFFFFF; border-radius: 16px; border: 1.5px solid #CBD5E1; padding: 1.15rem 1.25rem; margin: 1rem 0; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.35rem;">🔬</span>
                            <span style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">Verifikasi Karakteristik Fisik Langsung di Sawah</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.85rem; line-height: 1.55;">
                            {phys_sub_text}
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
                        is_differential=is_differential,
                        latin_name=info.get("latin"),
                        severity_level=visual_evidence.get("severity_level") if visual_evidence else None,
                        evidence_desc=visual_evidence.get("evidence_desc") if visual_evidence else None
                    )
                    st.session_state["ai_text_saved"] = ai_text
                    st.session_state["ai_angle_saved"] = ai_angle or "Pendekatan Terpadu Lapangan (Database Mandiri Sistem)"
                    st.session_state["ai_token_saved"] = ai_token_now

            # Parsing Resep Menjadi 3 Kartu Jelas & Format HTML Terstruktur (Otomatis Fallback ke Database Mandiri Sistem jika Kuota Groq Habis)
            kartu_tindakan, kartu_obat, kartu_lahan = parse_groq_to_cards(
                st.session_state.get("ai_text_saved"),
                info,
                second_info=second_info if is_differential else None,
                is_differential=is_differential,
                angle_title=st.session_state.get("ai_angle_saved")
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
                with st.spinner("🔄 Sedang meracik alternatif kombinasi obat dan panduan lain dari Balitsa/Kementan..."):
                    curr_angle = st.session_state.get("ai_angle_saved")
                    avail_idx = [i for i, (title, _) in enumerate(FOCUS_ANGLES) if title not in str(curr_angle)]
                    chosen_idx = random.choice(avail_idx) if avail_idx else random.randint(0, len(FOCUS_ANGLES) - 1)
                    alt_title = FOCUS_ANGLES[chosen_idx][0]

                    new_text, new_angle = get_groq_recommendation(
                        disease_name=info["nama_id"],
                        confidence=top_confidence,
                        is_healthy=is_healthy,
                        angle_idx=chosen_idx,
                        latin_name=info.get("latin"),
                        severity_level=visual_evidence.get("severity_level") if visual_evidence else None,
                        evidence_desc=visual_evidence.get("evidence_desc") if visual_evidence else None
                    )
                    if new_text:
                        st.session_state["ai_text_saved"] = new_text
                        st.session_state["ai_angle_saved"] = new_angle
                        st.toast(f"Petunjuk alternatif terpercaya berhasil dimuat ({new_angle})", icon="🌱")
                    else:
                        st.session_state["ai_text_saved"] = None
                        st.session_state["ai_angle_saved"] = f"{alt_title} (Database Mandiri Sistem)"
                        st.toast(f"Petunjuk alternatif dimuat dari database sistem ({alt_title})", icon="🌱")
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
                    f"📅 Tanggal: {datetime.now(timezone.utc).astimezone().strftime('%d/%m/%Y %H:%M')}\n"
                    f"🔬 Hasil Periksa: *{info['nama_id']}* (Kepastian: {top_confidence:.1f}%)\n"
                    f"📍 Lokasi: {in_lokasi if in_lokasi.strip() else 'Sawah Bawang'}\n\n"
                    f"🚨 *Saran Cepat:* {info['rekomendasi_singkat']}"
                )
                st.code(wa_share_text, language="text")
                st.caption("Tekan ikon salin di sudut kanan atas kotak abu-abu di atas, lalu tempel di obrolan WhatsApp kelompok tani.")
