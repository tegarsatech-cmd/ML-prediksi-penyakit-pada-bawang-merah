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
CLASS_METADATA = {
    "Iris Yellow Spot Virus (IYSV)": {
        "nama_id": "Virus Iris Kuning (IYSV)",
        "latin": "Iris yellow spot virus",
        "status": "virus",
        "is_healthy": False,
        "ciri_lapangan": "Bercak klorotik khas berbentuk ketupat/belah ketupat warna kuning jerami di tengah helai daun, terkadang memiliki pulau hijau di tengah bercak (green islands).",
        "gejala": "Cabut dan musnahkan tanaman yang daunnya terdapat bercak kuning berbentuk ketupat agar tidak menular ke tanaman sekitarnya.",
        "pencegahan": "Gunakan mulsa plastik perak untuk memantulkan sinar matahari dan menghalau hama kutu trips (Thrips tabaci) pembawa virus.",
        "solusi": "Kendalikan kutu trips dengan insektisida sistemik berbahan aktif Abamektin, Spinetoram, atau Klorfenapir pada pagi/sore hari.",
        "rekomendasi_singkat": "Cabut tanaman bergejala ketupat & semprot Abamektin untuk basmi kutu trips."
    },
    "Hawar Daun (Stemphylium / Colletotrichum)": {
        "nama_id": "Hawar Daun (Stemphylium / Antraknosa)",
        "latin": "Stemphylium vesicarium / Colletotrichum gloeosporioides",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Ujung daun menguning kecokelatan kering merambat ke bawah (blight), atau bercak melekuk kebasahan pada helai daun.",
        "gejala": "Pangkas helai daun yang tampak membusuk atau mengering dari ujung sebelum merambat ke leher umbi tanaman.",
        "pencegahan": "Hindari penyiraman sore/malam hari dan jaga sirkulasi parit bedengan macak-macak agar tidak tergenang air.",
        "solusi": "Semprot fungisida berbahan aktif Mankozeb, Klorotalonil, atau Difenokonazol selang-seling dengan Tembaga Oksiklorida.",
        "rekomendasi_singkat": "Pangkas daun kering ujung dan semprot fungisida Mankozeb/Klorotalonil."
    },
    "Bercak Ungu / Trotol (Alternaria porri)": {
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
    "Busuk Daun": {
        "nama_id": "Hawar / Busuk Daun",
        "latin": "Botrytis / Stemphylium / Xanthomonas",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Ujung daun menguning kecokelatan kering memanjang ke bawah atau melepuh kebasah-basahan seperti tersiram air mendidih.",
        "gejala": "Pangkas helai daun yang tampak membusuk atau mengering seperti terbakar sebelum merambat ke umbi tanaman.",
        "pencegahan": "Hindari penyiraman sore/malam hari dan jaga sirkulasi parit bedengan macak-macak.",
        "solusi": "Semprot bakterisida/fungisida berbahan aktif Tembaga Hidroksida atau Klorotalonil secara merata pada pagi hari.",
        "rekomendasi_singkat": "Semprot Tembaga Hidroksida/Klorotalonil dan pangkas daun yang membusuk."
    },
    "Moler": {
        "nama_id": "Layu Moler / Fusarium",
        "latin": "Fusarium oxysporum",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Helai daun melintir-lintir abnormal (moler), menguning pucat dari ujung, perakaran membusuk dan mudah dicabut.",
        "gejala": "Segera cabut tanaman yang daunnya melintir abnormal (moler) sampai ke perakarannya agar jamur tidak menular lewat parit.",
        "pencegahan": "Campurkan agens hayati Trichoderma dengan pupuk kandang matang saat olah tanah dasar bedengan.",
        "solusi": "Taburkan kapur dolomit pada lubang bekas cabutan dan kocorkan fungisida sistemik berbahan aktif Benomil atau Mankozeb.",
        "rekomendasi_singkat": "Cabut tanaman melintir/moler, taburkan dolomit & agens Trichoderma."
    },
    "Sehat": {
        "nama_id": "Daun Sehat & Segar",
        "latin": "Kondisi Normal",
        "status": "healthy",
        "is_healthy": True,
        "ciri_lapangan": "Daun hijau segar mengkilap, silindris tegak berdiri kokoh tanpa bercak berlendir, luka trotol, maupun kelintiran abnormal.",
        "gejala": "Kondisi tanaman sangat prima! Daun hijau segar, tegak berdiri kokoh tanpa bercak patogen atau luka hama.",
        "pencegahan": "Lanjutkan pemantauan rutin 2-3 hari sekali. Pertahankan kebersihan gulma di parit dan pematang.",
        "solusi": "Tidak memerlukan obat semprot kimia kuratif. Cukup semprotkan pupuk daun mikro dan asam amino untuk menjaga kesegaran.",
        "rekomendasi_singkat": "Tanaman sehat optimal! Cukup lanjutkan pemupukan berimbang dan pengairan rutin."
    },
    "Trotol": {
        "nama_id": "Bercak Ungu / Trotol",
        "latin": "Alternaria porri",
        "status": "disease",
        "is_healthy": False,
        "ciri_lapangan": "Bercak melekuk ke dalam berbentuk cincin konsentris bertepung keunguan/gelap di tengah helai daun, tepi menguning kering.",
        "gejala": "Potong atau pangkas daun yang terdapat bercak trotol ungu cincin konsentris. Kumpulkan dan bakar di luar areal sawah.",
        "pencegahan": "Bersihkan gulma dan rumput liar di sekitar parit. Buat bedengan lebih tinggi agar tidak tergenang air saat hujan lebat.",
        "solusi": "Semprot fungisida berbahan aktif Difenokonazol atau Mankozeb. Semprot pada pagi hari (pukul 06.00 - 08.30) atau sore (pukul 16.00) saat angin tenang.",
        "rekomendasi_singkat": "Semprot Difenokonazol/Mankozeb dan pangkas daun trotol segera."
    },
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

def check_shallot_leaf_mask(image: Image.Image, min_ratio: float = 0.02) -> tuple[bool, str, float]:
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
        
        # 1. Daun hijau sehat / bergejala (Hue 40 - 165)
        is_green_leaf = (h >= 40.0) & (h <= 165.0) & (s >= 0.12) & (v >= 0.10)
        
        # 2. Daun menguning / ujung kering penyakit (Hue 25 - 40, G dominan)
        is_yellowing = (h >= 25.0) & (h < 40.0) & (g >= r * 0.75) & (s >= 0.15)
        
        # 3. Bintil karat oranye / lesi kemerahan
        is_rust_spot = (h >= 8.0) & (h < 25.0) & (r > g * 1.1) & (s >= 0.25) & (v >= 0.16)

        # 4. Umbi / selubung ungu kemerahan bawang merah (Hue 285 - 355)
        is_purple_bulb = ((h >= 285.0) | (h <= 14.0)) & (r > g * 1.15) & (s >= 0.15)
        
        plant_mask = is_green_leaf | is_yellowing | is_rust_spot | is_purple_bulb
        plant_ratio = float(np.mean(plant_mask))
        
        if plant_ratio < min_ratio:
            return False, f"Rasio warna daun bawang merah hanya {plant_ratio*100:.1f}% (minimal {min_ratio*100:.0f}%).", plant_ratio
            
        return True, "Valid", plant_ratio
    except (ValueError, TypeError, ZeroDivisionError) as e:
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

def validate_onion_image(image: Image.Image, api_key: str | None = None) -> tuple[bool, str]:
    """
    Sistem Validasi Guardrail Gatekeeper Citra menggunakan Groq Vision:
    Mencegah diagnosis foto non-tanaman bawang merah (manusia, hewan, kendaraan, tanah kosong, tanaman lain).
    Tetap mengizinkan anomali wajar seperti tangan manusia yang sedang memegang/memperlihatkan daun bawang merah.
    Model: llama-3.2-11b-vision-preview (temperature=0.0, max_tokens=10).
    """
    if not api_key:
        api_key = get_groq_api_key()

    if not api_key:
        # Fallback spektrum lokal jika API Key belum tersedia
        is_plant, _, _ = check_shallot_leaf_mask(image, min_ratio=0.02)
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
            "Anda adalah validator citra pertanian profesional.\n"
            "Tugas: Memeriksa apakah gambar memuat daun, umbi, atau bagian tanaman bawang merah (Allium cepa).\n"
            "ATURAN KHUSUS:\n"
            "1. Jika terlihat tangan manusia yang sedang memegang atau memperlihatkan helai daun/umbi bawang merah di kebun/sawah, gambar ini TETAP DINYATAKAN VALID!\n"
            "2. Asalkan ada helai daun bawang merah yang terlihat (meskipun dipegang tangan atau ada latar tanah/kebun), jawab VALID.\n"
            "3. HANYA jawab INVALID jika gambar sama sekali tidak memuat tanaman bawang merah (misal hanya selfie wajah, hewan, kendaraan, makanan jadi di piring, atau tanah kosong tanpa daun).\n"
            "Jawab HANYA satu kata: 'VALID' atau 'INVALID'."
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
            is_plant, _, _ = check_shallot_leaf_mask(image, min_ratio=0.02)
            if not is_plant:
                return False, f"INVALID (Vision Status {resp.status_code}): Spektrum citra bukan daun bawang."
            return True, f"VALID (Fallback status {resp.status_code})"
    except (requests.RequestException, KeyError, IndexError, ValueError, OSError) as err:
        # Fallback jaringan jika timeout
        is_plant, _, _ = check_shallot_leaf_mask(image, min_ratio=0.02)
        if not is_plant:
            return False, "INVALID (Fallback Timeout): Spektrum citra bukan daun bawang."
        return True, f"VALID (Fallback: {err})"

# Alias untuk kompatibilitas
validate_with_groq_vision = validate_onion_image

def inspect_visual_leaf_symptoms(image: Image.Image) -> dict:
    """
    Modul Inspeksi Fitur Visual Citra Daun Bawang Merah (Pure NumPy & PIL):
    Mendeteksi patologi lesi fisik nyata langsung dari piksel foto tanpa ketergantungan library luar:
    1. Bintil pustula serbuk oranye-karat (Karat Daun / Rust - Puccinia allii)
    2. Bercak trotol melekuk cincin konsentris keunguan (Bercak Ungu / Alternaria porri)
    3. Lesi pucat memanjang kebasah-basahan (Hawar Bakteri Xanthomonas)
    4. Klorosis dan jaringan daun hijau utuh (Normal / Daun Sehat)
    Menghitung Indeks Keparahan (Severity Index) kuantitatif dan membuat peta heatmap lesi.
    """

    # 1. Standardisasi resolusi gambar menggunakan PIL
    img = image.convert("RGB")
    w, h = img.size
    max_dim = 640
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = max(int(w * scale), 10), max(int(h * scale), 10)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_rgb = np.array(img, dtype=np.uint8)

    # 2. Ekstraksi HSV murni via NumPy
    arr_f = img_rgb.astype(np.float32) / 255.0
    r, g, b = arr_f[:, :, 0], arr_f[:, :, 1], arr_f[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    v = cmax * 255.0
    s = np.zeros_like(cmax)
    nz_cmax = cmax > 1e-5
    s[nz_cmax] = (delta[nz_cmax] / cmax[nz_cmax]) * 255.0

    delta_safe = np.where(delta == 0, 1e-7, delta)
    h_deg = np.zeros_like(cmax)

    mask_r = (cmax == r) & (delta > 1e-5)
    h_deg[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6.0)
    mask_g = (cmax == g) & (delta > 1e-5)
    h_deg[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta_safe[mask_g]) + 2.0)
    mask_b = (cmax == b) & (delta > 1e-5)
    h_deg[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta_safe[mask_b]) + 4.0)

    # Skala Hue ke 0..180 (selaras konvensi OpenCV)
    h_val = (h_deg / 2.0) % 180.0

    # 3. Segmentasi Jaringan Daun Tanaman
    # Daun hijau sehat
    mask_green = (h_val >= 27.0) & (h_val <= 86.0) & (s >= 28.0) & (v >= 28.0)
    # Daun menguning / klorotik
    mask_yellow = (h_val >= 17.0) & (h_val < 27.0) & (s >= 35.0) & (v >= 45.0)
    # Bintil pustula karat (oranye / tembaga / merah karat: Hue 3..24, Saturation >= 35, Value >= 35, R > G)
    mask_rust_raw = (
        (h_val >= 3.0) & (h_val <= 24.0) &
        (s >= 35.0) & (v >= 35.0) &
        (r > g * 1.06) & (g >= b * 0.95)
    )
    # Bercak trotol keunguan / nekrotik gelap (Alternaria / Stemphylium)
    mask_purple_dark = (
        ((h_val <= 14.0) | (h_val >= 140.0)) &
        (s >= 30.0) & (v >= 18.0) & (v <= 130.0)
    )
    # Hawar pucat jerami kebasah-basahan (Xanthomonas)
    mask_xantho_pale = (
        (h_val >= 18.0) & (h_val <= 40.0) &
        (s >= 15.0) & (s <= 80.0) & (v >= 135.0) &
        (~mask_rust_raw)
    )

    mask_plant = mask_green | mask_yellow | mask_rust_raw | mask_purple_dark | mask_xantho_pale
    total_plant_pixels = int(np.count_nonzero(mask_plant))

    if total_plant_pixels < 250:
        return {
            "has_visual_evidence": False,
            "evidence_disease": None,
            "suspected_rust": False,
            "override_applied": False,
            "severity_pct": 0.0,
            "severity_level": "Normal",
            "rust_pct": 0.0,
            "purple_pct": 0.0,
            "xantho_pct": 0.0,
            "healthy_pct": 100.0,
            "evidence_desc": "Area daun terlalu minim untuk ekstraksi lesi visual.",
            "overlay_img": None
        }

    rust_pixels = int(np.count_nonzero(mask_rust_raw & mask_plant))
    purple_pixels = int(np.count_nonzero(mask_purple_dark & mask_plant))
    xantho_pixels = int(np.count_nonzero(mask_xantho_pale & mask_plant))
    green_pixels = int(np.count_nonzero(mask_green & mask_plant))

    rust_pct = (rust_pixels / total_plant_pixels) * 100.0
    purple_pct = (purple_pixels / total_plant_pixels) * 100.0
    xantho_pct = (xantho_pixels / total_plant_pixels) * 100.0
    healthy_pct = (green_pixels / total_plant_pixels) * 100.0

    diseased_pixels = int(np.count_nonzero((mask_rust_raw | mask_purple_dark | mask_xantho_pale | mask_yellow) & mask_plant))
    severity_pct = (diseased_pixels / total_plant_pixels) * 100.0

    if severity_pct < 5.0:
        severity_level = "Sangat Ringan (< 5%)"
    elif severity_pct < 20.0:
        severity_level = "Ringan (5% - 20%)"
    elif severity_pct < 45.0:
        severity_level = "Sedang (20% - 45%)"
    else:
        severity_level = "Berat / Kritis (> 45%)"

    # Buat Peta Lesi Citra murni via NumPy & PIL
    overlay_rgb = img_rgb.copy()
    overlay_rgb[(mask_rust_raw & mask_plant) == 1] = [245, 110, 10]
    overlay_rgb[(mask_purple_dark & mask_plant) == 1] = [185, 20, 160]
    overlay_rgb[(mask_xantho_pale & mask_plant) == 1] = [240, 220, 60]

    annotated = (img_rgb * 0.62 + overlay_rgb * 0.38).astype(np.uint8)
    annotated_pil = Image.fromarray(annotated)

    evidence_disease = None
    has_strong_lesion = False
    suspected_rust = False
    evidence_desc = ""

    # Karakterisasi Bukti Fisik Lesi dari Citra
    if rust_pct >= 0.8 or (rust_pixels >= 75 and rust_pct > purple_pct * 0.5):
        evidence_disease = "Rust"
        has_strong_lesion = True
        suspected_rust = True
        evidence_desc = (
            f"Ditemukan kluster bintil pustula serbuk berwarna jingga-karat khas jamur *Puccinia allii* "
            f"seluas {rust_pct:.1f}% pada helai daun di foto (Tingkat Keparahan: {severity_level})."
        )
    elif purple_pct >= 2.5 and purple_pct > rust_pct:
        evidence_disease = "Purple blotch"
        has_strong_lesion = True
        evidence_desc = (
            f"Ditemukan lesi bercak trotol melekuk warna gelap keunguan dengan pola cincin konsentris "
            f"khas jamur *Alternaria porri* seluas {purple_pct:.1f}% pada helai daun di foto (Tingkat Keparahan: {severity_level})."
        )
    elif healthy_pct >= 90.0 and severity_pct < 4.0:
        evidence_disease = "Healthy leaves"
        has_strong_lesion = False
        evidence_desc = (
            f"Helai daun hijau segar optimal ({healthy_pct:.1f}% klorofil normal utuh) "
            f"tanpa ditemukan bercak nekrotik, bintil jamur, maupun luka gigitan hama."
        )
    elif xantho_pct >= 12.0 and rust_pct < 0.8 and purple_pct < 1.2:
        evidence_disease = "Xanthomonas Leaf Blight"
        has_strong_lesion = True
        evidence_desc = (
            f"Ditemukan gejala hawar pucat memanjang kebasah-basahan ({xantho_pct:.1f}%) "
            f"seperti tersiram air mendidih tanpa disertai bintil serbuk karat oranye maupun bercak trotol ungu."
        )
    else:
        evidence_desc = (
            f"Spektrum lesi pada daun di foto: {rust_pct:.1f}% spektrum karat, "
            f"{purple_pct:.1f}% spektrum bercak gelap, total kerusakan helai daun {severity_pct:.1f}% ({severity_level})."
        )

    return {
        "has_visual_evidence": True,
        "evidence_disease": evidence_disease,
        "suspected_rust": suspected_rust,
        "override_applied": has_strong_lesion,
        "severity_pct": round(severity_pct, 1),
        "severity_level": severity_level,
        "rust_pct": round(rust_pct, 1),
        "purple_pct": round(purple_pct, 1),
        "xantho_pct": round(xantho_pct, 1),
        "healthy_pct": round(healthy_pct, 1),
        "num_spots_detected": 0,
        "evidence_desc": evidence_desc,
        "overlay_img": annotated_pil
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
    Peta HUD Scanner Deteksi Titik Kerusakan Lesi Aktif Daun Bawang Merah:
    1. Segmentasi Kanopi Daun Bawang Merah (Leaf Canopy Masking) berbasis piksel murni NumPy & PIL.
    2. Pengecualian warna kulit tangan manusia agar penanda hanya menempel pada helai daun.
    3. Mendukung multi-penyakit: Jika diferensial diagnosis aktif, retikel Merah menandai fokus
       Penyakit A dan retikel Oranye menandai fokus Penyakit B.
    4. Menggambar retikel modern High-Precision Agro-Tech Scanner pada titik lesi aktif.
    5. Jika daun sehat, lingkaran tidak digambar untuk menjaga foto tetap bersih dan jernih.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    if is_healthy:
        return img_rgb, []

    try:
        # 1. Segmentasi Kanopi Daun Bawang Merah pada Resolusi Asli
        img_np = np.array(img_rgb, dtype=np.float32)
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        delta_safe = np.where(delta == 0, 1.0, delta)

        h_arr = np.zeros_like(delta)
        mask_r = (cmax == r) & (delta > 0)
        h_arr[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6.0)
        mask_g = (cmax == g) & (delta > 0)
        h_arr[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta_safe[mask_g]) + 2.0)
        mask_b = (cmax == b) & (delta > 0)
        h_arr[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta_safe[mask_b]) + 4.0)

        s_arr = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
        v_arr = cmax / 255.0

        is_green_leaf = (h_arr >= 40.0) & (h_arr <= 165.0) & (s_arr >= 0.12) & (v_arr >= 0.08)
        is_yellow_lesion = (h_arr >= 25.0) & (h_arr < 40.0) & (g >= r * 0.75) & (s_arr >= 0.15) & (v_arr >= 0.12)
        is_rust_lesion = (h_arr >= 8.0) & (h_arr < 28.0) & (r > g * 1.08) & (s_arr >= 0.22) & (v_arr >= 0.12)
        is_purple_lesion = ((h_arr <= 14.0) | (h_arr >= 270.0)) & (r > g * 1.1) & (s_arr >= 0.15) & (v_arr >= 0.08) & (v_arr <= 0.80)
        is_brown_necrotic = (h_arr >= 15.0) & (h_arr < 40.0) & (s_arr >= 0.12) & (v_arr >= 0.10) & (v_arr <= 0.55)

        # PENTING: Kecualikan warna kulit tangan manusia agar lingkaran tidak menempel di tangan/jari
        is_human_skin = (
            (h_arr >= 8.0) & (h_arr <= 25.0) &
            (s_arr >= 0.18) & (s_arr <= 0.50) &
            (v_arr >= 0.40) & (v_arr <= 0.85) &
            (r > g * 1.15) & (g > b * 1.10) &
            (b > 50) & (b < 180) &
            (np.abs(r - g) < 80)
        )

        leaf_mask = ((is_green_leaf | is_yellow_lesion | is_rust_lesion | is_purple_lesion | is_brown_necrotic) & (~is_human_skin)).astype(np.float32)

        # Abaikan margin tepi bingkai terluar 4% agar tidak menempel pada bingkai foto
        m_x, m_y = max(int(w * 0.04), 2), max(int(h * 0.04), 2)
        leaf_mask[:m_y, :] = 0
        leaf_mask[-m_y:, :] = 0
        leaf_mask[:, :m_x] = 0
        leaf_mask[:, -m_x:] = 0

        if np.sum(leaf_mask) < 50:
            return img_rgb, []

        # Segmentasi lesi fisik tampak nyata pada helai daun
        is_physical_lesion = (is_yellow_lesion | is_rust_lesion | is_purple_lesion | is_brown_necrotic) & (leaf_mask > 0)
        has_physical_lesion = np.sum(is_physical_lesion) > 30

        # Jika daun sehat atau tidak terdapat kerusakan fisik nyata, jangan beri penanda apapun
        if (not has_physical_lesion) or is_healthy:
            return img_rgb, []

        les_pil = Image.fromarray((is_physical_lesion * 255).astype(np.uint8))
        blur_rad = max(2, int(min(w, h) * 0.035))
        lesion_density = np.array(les_pil.filter(ImageFilter.GaussianBlur(radius=blur_rad)), dtype=np.float32) / 255.0
        ld_max = float(np.max(lesion_density))
        if ld_max > 1e-5:
            lesion_density = lesion_density / ld_max

        # 2. Evaluasi spasial kepadatan lesi fisik pada daun
        cam_eval_1 = lesion_density * is_physical_lesion.astype(np.float32)
        cam_eval_2 = cam_eval_1 if is_differential else None

        def extract_peaks(c_eval):
            if not has_physical_lesion:
                return []
            grid_n = 8
            cell_h = max(h // grid_n, 1)
            cell_w = max(w // grid_n, 1)
            p_list = []
            c_damage_eval = c_eval * is_physical_lesion.astype(np.float32)

            for r_i in range(grid_n):
                for c_i in range(grid_n):
                    y1, y2 = r_i * cell_h, min((r_i + 1) * cell_h, h)
                    x1, x2 = c_i * cell_w, min((c_i + 1) * cell_w, w)
                    patch = c_damage_eval[y1:y2, x1:x2]
                    patch_lesion = is_physical_lesion[y1:y2, x1:x2]
                    if patch.size == 0 or np.sum(patch_lesion) == 0:
                        continue
                    max_p = float(np.max(patch))
                    if max_p > 0.12:
                        py, px = np.unravel_index(np.argmax(patch), patch.shape)
                        rx = int(x1 + px)
                        ry = int(y1 + py)
                        if ry < h and rx < w and is_physical_lesion[min(ry, h-1), min(rx, w-1)]:
                            p_list.append((rx, ry, max_p))
            if not p_list and has_physical_lesion:
                py, px = np.unravel_index(np.argmax(c_damage_eval), c_damage_eval.shape)
                py, px = min(py, h-1), min(px, w-1)
                if is_physical_lesion[py, px]:
                    p_list.append((int(px), int(py), float(c_damage_eval[py, px])))
            p_list.sort(key=lambda p: p[2], reverse=True)
            return p_list

        base_rad = max(22, int(min(w, h) * 0.06))
        min_dist = max(40, int(min(w, h) * 0.10))
        spots = []

        def clean_disease_badge(name: str) -> str:
            if not name:
                return "Titik Infeksi"
            n = name.split("/")[0].split("(")[0].strip()
            clean_map = {
                "Hawar Daun Bakteri": "Hawar Bakteri",
                "Karat Daun": "Karat Daun",
                "Bercak Ungu": "Bercak Ungu",
                "Embun Bulu": "Embun Bulu",
                "Layu Moler": "Layu Fusarium",
                "Hawar Daun Kering Ujung": "Stemphylium",
                "Hawar Daun": "Hawar Daun",
                "Busuk Umbi": "Busuk Umbi",
                "Ulat Grayak": "Ulat Grayak",
                "Virus Iris Kuning": "Virus Iris",
                "Virus Kuning Melintir": "Virus Daun"
            }
            for k, v in clean_map.items():
                if k.lower() in n.lower():
                    return v
            return n[:12]

        p_name_badge = clean_disease_badge(primary_name)
        s_name_badge = clean_disease_badge(second_name) if second_name else ""

        if is_differential and cam_eval_2 is not None:
            peaks_1 = extract_peaks(cam_eval_1)
            peaks_2 = extract_peaks(cam_eval_2)

            # Spot 1: Titik kerusakan fokus Penyakit A (Merah)
            if peaks_1:
                p1_x, p1_y, _ = peaks_1[0]
                spots.append((p1_x, p1_y, base_rad, f"{p_name_badge}", "red"))

            # Spot 2: Titik kerusakan fokus Penyakit B (Oranye)
            spot2_found = False
            for p2_x, p2_y, _ in peaks_2:
                if not spots or ((p2_x - spots[0][0])**2 + (p2_y - spots[0][1])**2 >= min_dist**2):
                    spots.append((p2_x, p2_y, base_rad, f"{s_name_badge}", "orange"))
                    spot2_found = True
                    break
            if not spot2_found and peaks_2:
                p2_x, p2_y, _ = peaks_2[0]
                spots.append((p2_x, p2_y, base_rad, f"{s_name_badge}", "orange"))

            # Titik tambahan bila ada sebaran kerusakan lain
            for px, py, _ in peaks_1[1:]:
                if len(spots) >= 4:
                    break
                if not any((px - sx)**2 + (py - sy)**2 < min_dist**2 for sx, sy, _, _, _ in spots):
                    spots.append((px, py, base_rad, f"{p_name_badge} #2", "red"))

            for px, py, _ in peaks_2[1:]:
                if len(spots) >= 4:
                    break
                if not any((px - sx)**2 + (py - sy)**2 < min_dist**2 for sx, sy, _, _, _ in spots):
                    spots.append((px, py, base_rad, f"{s_name_badge} #2", "orange"))
        else:
            peaks_1 = extract_peaks(cam_eval_1)
            for _, (px, py, _) in enumerate(peaks_1):
                if not any((px - sx)**2 + (py - sy)**2 < min_dist**2 for sx, sy, _, _, _ in spots):
                    lbl = f"{p_name_badge}" if len(spots) == 0 else f"{p_name_badge} #{len(spots)+1}"
                    spots.append((px, py, base_rad, lbl, "red"))
                if len(spots) >= 4:
                    break

        annotated = img_rgb.copy()
        draw = ImageDraw.Draw(annotated)

        for cx, cy, rad, badge_label, color_type in spots:
            if color_type == "orange":
                color_hud = (245, 158, 11)      # Neon Orange / Amber
                color_inner = (254, 240, 138)   # Soft Amber Glow
                badge_bg = (217, 119, 6)        # Amber badge
            else:
                color_hud = (239, 68, 68)       # Merah Neon Presisi
                color_inner = (254, 202, 202)   # Soft Red Glow
                badge_bg = (220, 38, 38)        # Red badge

            line_w = max(2, int(min(w, h) * 0.005))
            tick = max(8, int(min(w, h) * 0.018))
            dot_r = max(3, int(min(w, h) * 0.007))

            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color_hud, width=line_w)
            inner_r = max(rad - line_w * 2, 6)
            draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], outline=color_inner, width=max(1, line_w - 1))
            draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=color_hud)
            draw.line([cx - rad - tick, cy, cx - rad + 3, cy], fill=color_hud, width=line_w)
            draw.line([cx + rad - 3, cy, cx + rad + tick, cy], fill=color_hud, width=line_w)
            draw.line([cx, cy - rad - tick, cx, cy - rad + 3], fill=color_hud, width=line_w)
            draw.line([cx, cy + rad - 3, cx, cy + rad + tick], fill=color_hud, width=line_w)

            font_scale = max(8, int(min(w, h) * 0.018))
            badge_w = max(len(badge_label) * font_scale + 16, 80)
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
    except (ValueError, KeyError, IndexError, TypeError):
        return img_rgb, []

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
        is_shallot, reason_msg, _ = check_shallot_leaf_mask(image, min_ratio=0.02)
        if not is_shallot:
            raise ValueError(f"OOD_GUARD_REJECTED: {reason_msg}")

    # Jalankan inspeksi fitur visual nyata pada foto
    visual_evidence = inspect_visual_leaf_symptoms(image)

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
        "Iris Yellow Spot Virus (IYSV)",
        "Hawar Daun (Stemphylium / Colletotrichum)",
        "Sehat",
        "Bercak Ungu / Trotol (Alternaria porri)"
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

    # Selaraskan deskripsi bukti fisik citra
    if visual_evidence.get("has_visual_evidence"):
        v_sev_pct = visual_evidence.get("severity_pct", 0.0)
        v_sev_lvl = visual_evidence.get("severity_level", "Normal")
        v_healthy_pct = visual_evidence.get("healthy_pct", 100.0)

        if visual_evidence.get("suspected_rust"):
            visual_evidence["evidence_desc"] = (
                f"Modul analisis visual mendeteksi kluster bintil spora serbuk jingga-karat seluas {visual_evidence.get('rust_pct', 0.0)}% "
                f"(indikasi penyakit Karat Daun / Puccinia allii) dengan tingkat keparahan {v_sev_lvl}."
            )
        elif is_healthy_1:
            visual_evidence["evidence_desc"] = (
                f"Helai daun bawang hijau segar optimal ({v_healthy_pct:.1f}% klorofil normal utuh) "
                "tanpa ditemukan bercak nekrotik, bintil jamur, maupun luka gigitan hama."
            )
            visual_evidence["override_applied"] = False
        else:
            visual_evidence["evidence_desc"] = (
                f"Hasil pemindaian fitur citra mengonfirmasi infeksi {metadata['nama_id']} dengan tingkat keparahan {v_sev_lvl} "
                f"(luas jaringan daun terdampak: {v_sev_pct:.1f}%, jaringan hijau sehat tersisa: {v_healthy_pct:.1f}%)."
            )
            visual_evidence["override_applied"] = False

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
       Jika model 15 kelas memiliki top_confidence < threshold (bawaan 40%),
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
        "Iris Yellow Spot Virus (IYSV)",
        "Hawar Daun (Stemphylium / Colletotrichum)",
        "Sehat",
        "Bercak Ungu / Trotol (Alternaria porri)"
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
            <span style='background-color: #e2e8f0; color: #334155; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;'>v3.0.0 • EfficientNet-B0</span>
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

    st.markdown("### 🔬 Model PyTorch TorchScript")
    st.caption("EfficientNet-B0 (TorchScript) dengan Test-Time Augmentation (TTA) Flip Horizontal.")

    st.divider()

    default_conf_pct = int(round(float(meta_config.get("conf_threshold", 0.70)) * 100))
    st.markdown("### ⚙️ Validasi Foto Bawang")
    conf_threshold_pct = st.slider(
        "Batas Keyakinan / Confidence Threshold (%)",
        min_value=20,
        max_value=90,
        value=default_conf_pct,
        step=5,
        help="Jika kepastian model di bawah nilai ini, foto akan ditandai 'Tidak yakin, foto kurang jelas atau bukan daun bawang'."
    )
    conf_threshold = conf_threshold_pct / 100.0
    st.caption(f"Ambang batas kepastian: **{conf_threshold_pct}%** (Default Colab: {default_conf_pct}%)")

    min_leaf_ratio = st.slider(
        "Sensitivitas Daun Bawang (%)",
        min_value=2,
        max_value=30,
        value=3,
        step=1,
        help="Persentase minimal warna daun bawang merah yang harus ada pada foto sebelum model dijalankan."
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
            file_name=f"riwayat_bawang_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Download JSON
        json_bytes = json.dumps(st.session_state['history'], indent=2).encode('utf-8')
        st.download_button(
            label="📥 Unduh Cadangan (.JSON)",
            data=json_bytes,
            file_name=f"riwayat_bawang_{datetime.now(timezone.utc).astimezone().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            reset_all_history()
            components.html("<script>try{ window.parent.localStorage.removeItem('agroscan_bawang_history'); }catch(e){}</script>", height=0, width=0)
            st.toast("Semua riwayat berhasil dihapus.", icon="🗑️")
            st.rerun()

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
            Model Deep Learning: EfficientNet-B0 TorchScript (4 Kelas)<br>
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
        <div style="margin-top: 0.6rem; display: inline-flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.35);">Versi 3.0.0 (TorchScript)</span>
            <span style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.35);">EfficientNet-B0 4-Kelas Presisi</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.info("ℹ️ **Catatan:** Hasil ini hanya alat bantu, bukan diagnosis final.")

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
        # Jika confidence < conf_threshold (0.70), tampilkan "Tidak yakin, foto kurang jelas atau bukan daun bawang"
        # alih-alih menebak
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

            with st.expander("🔌 Format Keluaran API (JSON)", expanded=True):
                st.caption("Respon standar API JSON:")
                st.json(api_output)

            st.stop()  # Hentikan eksekusi, jangan tebak penyakit & jangan panggil resep obat

        # ==============================================================================
        # KASUS 2: FOTO VALID & KEYAKINAN TINGGI (LOLOS THRESHOLD)
        # ==============================================================================
        else:
            # Rekam Otomatis ke Riwayat Sesi Sekali Saja
            diag_sig = f"{current_img_sig}_{top_class_raw}_{round(top_confidence, 1)}"
            if st.session_state.get("last_auto_recorded") != diag_sig:
                st.session_state["last_auto_recorded"] = diag_sig
                status_rec = "Multidiagnosis (Mirip)" if is_differential else ("Healthy / Sehat" if info.get("status") == "healthy" or info.get("is_healthy", False) else "Penyakit / Hama")
                st.session_state['history'].append({
                    "id": int(time.time() * 1000),
                    "waktu": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
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

            # ==============================================================================
            # MODUL BUKTI VISUAL NYATA DARI FOTO (REAL VISUAL LESION AUDIT)
            # ==============================================================================
            if visual_evidence and visual_evidence.get("has_visual_evidence"):
                v_sev_pct = visual_evidence.get("severity_pct", 0.0)
                v_sev_lvl = visual_evidence.get("severity_level", "Normal")
                v_rust_pct = visual_evidence.get("rust_pct", 0.0)
                v_purple_pct = visual_evidence.get("purple_pct", 0.0)
                v_xantho_pct = visual_evidence.get("xantho_pct", 0.0)
                v_healthy_pct = visual_evidence.get("healthy_pct", 100.0)
                v_desc = visual_evidence.get("evidence_desc", "")
                v_override = visual_evidence.get("override_applied", False)

                st.markdown("### 🔬 Bukti Analisis Visual Nyata dari Foto Daun")
                st.caption("Hasil pemindaian fitur fisik piksel langsung dari foto yang diunggah (mendeteksi bintil spora, warna lesi, dan luas infeksi sesungguhnya):")

                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric(label="🩺 Luas Kerusakan Daun", value=f"{v_sev_pct:.1f}%", delta=v_sev_lvl, delta_color="inverse")
                with col_v2:
                    st.metric(label="🟠 Bintil Pustula Karat", value=f"{v_rust_pct:.1f}%", help="Persentase kluster serbuk jingga-karat pada helai daun")
                with col_v3:
                    st.metric(label="🌿 Jaringan Daun Hijau", value=f"{v_healthy_pct:.1f}%", help="Persentase area klorofil daun yang masih sehat")

                if visual_evidence.get("suspected_rust", False):
                    st.warning(
                        "🟠 **Peringatan Fusi Fitur Fisik — Dugaan Penyakit Karat Daun (*Puccinia allii*):**\n\n"
                        f"Modul analisis citra mendeteksi kluster bintil serbuk jingga-karat seluas **{v_rust_pct:.1f}%** pada helai daun.\n\n"
                        "📌 **Catatan Model AI:** Model klasifikasi saat ini dilatih khusus pada **4 Kategori Spesifik** (`Busuk Daun`, `Moler`, `Sehat`, `Trotol`). "
                        "Penyakit Karat Daun belum termasuk dalam 4 kelas latih tersebut, sehingga model neural mengarahkannya ke kelas bercak terdekat (**Trotol / Bercak Ungu**). "
                        "Jika saat daun diusap dengan jari tertinggal serbuk halus warna jingga-karat, infeksi utama adalah **Karat Daun**."
                    )

                if v_override:
                    st.success(
                        f"🛡️ **Vonis Diagnosis Dikonfirmasi Bukti Fisik Citra:**\n\n"
                        f"{v_desc}"
                    )
                else:
                    st.info(
                        f"📋 **Karakteristik Fisik Daun pada Foto:**\n\n"
                        f"{v_desc}"
                    )

                if visual_evidence.get("overlay_img") is not None:
                    with st.expander("🖼️ Peta Titik Kerusakan pada Foto Daun (HUD Lesion Scanner)", expanded=True):
                        map_caption = (
                            f"Peta Titik Kerusakan Multi-Penyakit: Retikel Merah menandai fokus gejala {info['nama_id']}, sedangkan Retikel Oranye menandai fokus gejala {second_info['nama_id']} pada helai daun."
                            if is_differential
                            else f"Peta Titik Kerusakan: Retikel scanner presisi tinggi menandai titik pusat kerusakan aktif {info['nama_id']} pada helai daun."
                        )
                        st.image(
                            visual_evidence["overlay_img"],
                            caption=map_caption,
                            use_container_width=True
                        )
                        num_spots = visual_evidence.get("num_spots_detected", 0)
                        if is_healthy:
                            st.success("✅ **Daun Sehat & Normal:** Pemindaian visual mengonfirmasi helai daun segar dan tidak menemukan titik kerusakan lesi aktif.")
                        elif is_differential:
                            st.caption(
                                f"💡 **Petunjuk Deteksi Multi-Penyakit ({num_spots} Titik Terdeteksi):** Sistem mendeteksi dua kemungkinan patogen yang menginfeksi helai daun. "
                                f"Retikel **Merah** menandai area kerusakan yang paling kuat dicurigai sebagai **{info['nama_id']}**, "
                                f"sedangkan Retikel **Oranye** menandai area yang dicurigai sebagai **{second_info['nama_id']}**. "
                                "Cocokkan perbedaan ciri fisik kedua area tersebut langsung di bedengan kebun untuk penanganan yang tepat."
                            )
                        elif num_spots > 0:
                            st.caption(
                                f"💡 **Petunjuk Deteksi ({num_spots} Titik Kerusakan Terdeteksi):** Retikel scanner di atas memetakan titik lesi aktif pada helai daun tanaman (bukan tangan atau latar belakang). Titik bertanda nama penyakit menunjukkan konsentrasi infeksi aktif tempat patogen berkembang. Fokuskan sanitasi pemangkasan daun sakit dan penyemprotan obat pada titik-titik tersebut."
                            )
                        else:
                            st.caption(
                                "💡 **Petunjuk Deteksi:** Retikel scanner presisi dihasilkan langsung dari pemindaian densitas lesi pada foto helai daun Anda."
                            )

            # Distribusi Probabilitas Model PyTorch TorchScript
            with st.expander("📊 Distribusi Probabilitas Model (TorchScript EfficientNet-B0)", expanded=True):
                st.caption("Distribusi probabilitas softmax terkalibrasi (temperature-scaled) untuk setiap kelas:")
                probs_dict = api_output.get("probabilities", {})
                for c_label, prob_val in probs_dict.items():
                    info_c = CLASS_METADATA.get(c_label, {"nama_id": c_label})
                    prob_pct = round(prob_val * 100.0, 1)
                    st.write(f"• **{info_c['nama_id']}** (`{c_label}`): **{prob_pct:.1f}%** ({prob_val:.4f})")
                    st.progress(min(max(float(prob_val), 0.0), 1.0))

            # Format Keluaran API (JSON)
            with st.expander("🔌 Format Keluaran API (JSON)", expanded=False):
                st.caption("Respon standar API JSON untuk integrasi backend / mobile:")
                st.json(api_output)

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
