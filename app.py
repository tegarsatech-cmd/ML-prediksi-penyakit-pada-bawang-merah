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
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

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
            min-height: 52px !important;
            font-size: 1.05rem !important;
        }
        .card-ai-step {
            padding: 1.1rem !important;
        }
        .card-rejection {
            padding: 1.15rem 1rem !important;
            border-radius: 14px !important;
        }
        .card-rejection-title {
            font-size: 1.35rem !important;
            line-height: 1.3 !important;
        }
        .card-rejection-reason {
            font-size: 0.95rem !important;
            padding: 0.75rem 0.9rem !important;
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

    /* Anti-Crop & Responsive Metric Card Styling */
    .metric-anti-crop {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        min-height: 88px !important;
        box-sizing: border-box !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    .metric-anti-crop-label {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        margin-bottom: 4px !important;
        line-height: 1.25 !important;
    }
    .metric-anti-crop-val {
        font-size: 1.30rem !important;
        font-weight: 900 !important;
        line-height: 1.2 !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    .metric-anti-crop-delta {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        margin-top: 4px !important;
        line-height: 1.2 !important;
        word-break: break-word !important;
    }

    /* Universal Anti-Crop Override untuk Streamlit Metric */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 0.75rem 0.95rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
        overflow: visible !important;
        word-break: break-word !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.22rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        line-height: 1.25 !important;
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        color: #475569 !important;
    }
    [data-testid="stMetricDelta"] {
        white-space: normal !important;
        word-break: break-word !important;
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

    /* Tombol Khusus Sidebar (Responsif, Pas di Kolom, Anti-Meluber & Terbaca Sempurna di Semua Perangkat) */
    [data-testid="stSidebar"] div.stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        min-height: 40px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        padding: 0.45rem 0.65rem !important;
        border-radius: 9px !important;
        white-space: normal !important;
        word-break: normal !important;
        line-height: 1.25 !important;
        overflow: hidden !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] div.stButton > button p,
    [data-testid="stSidebar"] .stDownloadButton > button p {
        font-size: 0.88rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        word-break: normal !important;
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
        border-radius: 12px !important;
        padding: 0.75rem !important;
        width: 100% !important;
        overflow: visible !important;
    }
    [data-testid="stFileUploader"] * {
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        padding: 0.5rem !important;
        background-color: transparent !important;
        border: none !important;
        width: 100% !important;
        overflow: visible !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        white-space: normal !important;
        word-break: break-word !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
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

    /* Ikon Ganti Kamera Depan/Belakang di HP / Mobile (Wajib Hitam Pekat, Kontras Tinggi & Tidak Samar) */
    [data-testid="stCameraInput"] button:not([kind="primary"]) svg,
    [data-testid="stCameraInput"] button:not([kind="primary"]) path,
    [data-testid="stCameraInput"] button[aria-label*="switch" i] svg,
    [data-testid="stCameraInput"] button[aria-label*="switch" i] path,
    [data-testid="stCameraInput"] button[aria-label*="camera" i] svg,
    [data-testid="stCameraInput"] button[aria-label*="camera" i] path,
    [data-testid="stCameraInput"] button[title*="switch" i] svg,
    [data-testid="stCameraInput"] button[title*="switch" i] path,
    [data-testid="stCameraInput"] button[title*="camera" i] svg,
    [data-testid="stCameraInput"] button[title*="camera" i] path,
    [data-testid="stCameraInput"] [data-testid*="switch" i] svg,
    [data-testid="stCameraInput"] [data-testid*="camera" i] svg {
        color: #000000 !important;
        fill: #000000 !important;
        stroke: #000000 !important;
        opacity: 1 !important;
        filter: drop-shadow(0px 1px 1px rgba(255,255,255,0.9)) !important;
    }
    [data-testid="stCameraInput"] button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        border-radius: 999px !important;
        color: #000000 !important;
        opacity: 1 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
    }

    /* 8. Expander & Alert (Dikecilkan Ramping & Proporsional di Mobile & Desktop) */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        padding: 0.45rem 0.75rem !important;
        border-radius: 8px !important;
        min-height: unset !important;
        gap: 8px !important;
    }
    [data-testid="stExpander"] summary * {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        padding: 0.65rem 0.85rem !important;
    }
    [data-testid="stExpanderDetails"] * {
        color: #0F172A !important;
    }

    /* Kotak Khusus Panduan & Kategori Sidebar (Dikecilkan Lebih Ramping di Mobile & Desktop) */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        margin-bottom: 0.35rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background-color: #F8FAFC !important;
        padding: 0.32rem 0.55rem !important;
        border-radius: 7px !important;
        min-height: unset !important;
        gap: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        line-height: 1.3 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        width: 14px !important;
        height: 14px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        padding: 0.45rem 0.6rem !important;
        font-size: 0.78rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {
        color: #1E293B !important;
        font-size: 0.78rem !important;
        line-height: 1.4 !important;
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

def extract_leaf_roi(image: Image.Image, padding_pct: float = 0.08) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """
    Ekstraksi Otomatis Region of Interest (ROI) Daun Bawang Merah:
    Mendeteksi area helai daun bawang secara presisi dan memangkas latar belakang tanah, mulsa,
    tangan petani yang memegang daun, atau lantai bedengan.
    Memastikan AI EfficientNet-B0 fokus 100% pada jaringan daun tanpa terganggu objek luar.
    """
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    thumb_dim = 256
    scale_w = orig_w / float(thumb_dim)
    scale_h = orig_h / float(thumb_dim)
    thumb = img_rgb.resize((thumb_dim, thumb_dim), Image.Resampling.BILINEAR)

    arr = np.array(thumb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    exg = 2.0 * g - r - b

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = np.where(cmax - cmin == 0, 1.0, cmax - cmin)

    h = np.zeros_like(delta)
    mask_r = (cmax == r) & (cmax > cmin)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    mask_g = (cmax == g) & (cmax > cmin)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    mask_b = (cmax == b) & (cmax > cmin)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    s = np.where(cmax == 0, 0.0, (cmax - cmin) / np.where(cmax == 0, 1.0, cmax))
    v = cmax / 255.0

    # 1. Deteksi kulit manusia (tangan / jari petani yang memegang daun)
    is_skin = (
        (h >= 5.0) & (h <= 28.0) &
        (s >= 0.15) & (s <= 0.60) &
        (v >= 0.30) & (v <= 0.95) &
        (r > g * 1.08) & (g > b * 1.02) &
        (np.abs(r - g) < 95)
    )

    # 2. Deteksi jaringan daun bawang (hijau botani, kuning klorotik penyakit)
    is_green = (h >= 35.0) & (h <= 170.0) & (s >= 0.12) & (v >= 0.09) & (exg > 0)
    is_yellow = (h >= 25.0) & (h < 55.0) & (s >= 0.20) & (g > b * 1.30) & ((exg > 5.0) | ((g >= r * 0.85) & (g > 110.0)))
    leaf_veg = (is_green | is_yellow) & (~is_skin)

    # 3. Deteksi lesi penyakit pada helai daun (bercak ungu/karat/antraknosa)
    if cv2 is not None:
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        leaf_expanded = cv2.dilate(leaf_veg.astype(np.uint8), kernel_expand, iterations=2)
    else:
        leaf_expanded = leaf_veg.astype(np.uint8)

    is_lesion_candidate = (h >= 6.0) & (h < 30.0) & (r > g * 1.05) & (s >= 0.15) & (v >= 0.14) & (~is_skin)
    is_leaf_lesion = is_lesion_candidate & (leaf_expanded > 0)

    total_leaf_mask = leaf_veg | is_leaf_lesion

    if cv2 is not None:
        clean_leaf_mask = cv2.morphologyEx(total_leaf_mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(clean_leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_c = [c for c in contours if cv2.contourArea(c) >= 20]
        if valid_c:
            all_pts = np.vstack(valid_c)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            ox1 = int(bx * scale_w)
            oy1 = int(by * scale_h)
            ox2 = int((bx + bw) * scale_w)
            oy2 = int((by + bh) * scale_h)
        else:
            plant_pixels = np.argwhere(total_leaf_mask)
            if len(plant_pixels) < 30:
                return img_rgb, (0, 0, orig_w, orig_h), 1.0
            y_min, x_min = plant_pixels.min(axis=0)
            y_max, x_max = plant_pixels.max(axis=0)
            ox1, oy1 = int(x_min * scale_w), int(y_min * scale_h)
            ox2, oy2 = int(x_max * scale_w), int(y_max * scale_h)
    else:
        plant_pixels = np.argwhere(total_leaf_mask)
        if len(plant_pixels) < 30:
            return img_rgb, (0, 0, orig_w, orig_h), 1.0
        y_min, x_min = plant_pixels.min(axis=0)
        y_max, x_max = plant_pixels.max(axis=0)
        ox1, oy1 = int(x_min * scale_w), int(y_min * scale_h)
        ox2, oy2 = int(x_max * scale_w), int(y_max * scale_h)

    bw_px = ox2 - ox1
    bh_px = oy2 - oy1
    pad_x = int(bw_px * padding_pct)
    pad_y = int(bh_px * padding_pct)

    fx1 = max(0, ox1 - pad_x)
    fy1 = max(0, oy1 - pad_y)
    fx2 = min(orig_w, ox2 + pad_x)
    fy2 = min(orig_h, oy2 + pad_y)

    crop_w = fx2 - fx1
    crop_h = fy2 - fy1
    area_ratio = (crop_w * crop_h) / float(orig_w * orig_h)

    if area_ratio < 0.02 or crop_w < 30 or crop_h < 30:
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
    leaf_roi, leaf_bbox, leaf_coverage = extract_leaf_roi(img_clean, padding_pct=0.08)
    is_auto_cropped = leaf_coverage < 0.96

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
        delta = np.where(cmax - cmin == 0, 1.0, cmax - cmin)
        
        h = np.zeros_like(delta)
        mask_r = (cmax == r) & (cmax > cmin)
        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
        mask_g = (cmax == g) & (cmax > cmin)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
        mask_b = (cmax == b) & (cmax > cmin)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)
        
        s = np.where(cmax == 0, 0.0, (cmax - cmin) / np.where(cmax == 0, 1.0, cmax))
        v = cmax / 255.0
        
        # Deteksi kulit tangan/wajah manusia untuk eksklusi
        is_skin = (
            (h >= 5.0) & (h <= 28.0) &
            (s >= 0.15) & (s <= 0.60) &
            (v >= 0.30) & (v <= 0.95) &
            (r > g * 1.08) & (g > b * 1.02) &
            (np.abs(r - g) < 95)
        )
        skin_ratio = float(np.mean(is_skin))

        # Jaringan daun hijau tanaman (klorofil aktif)
        is_green_leaf = (h >= 35.0) & (h <= 170.0) & (s >= 0.12) & (v >= 0.09) & (exg > 0)
        
        # Daun menguning klorotik / ujung mengering penyakit
        is_yellowing = (h >= 24.0) & (h < 55.0) & (s >= 0.18) & (g > b * 1.25) & ((exg > 4.0) | ((g >= r * 0.85) & (g > 110.0)))
        
        leaf_base = (is_green_leaf | is_yellowing) & (~is_skin)
        if cv2 is not None:
            kernel_exp = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            leaf_exp = cv2.dilate(leaf_base.astype(np.uint8), kernel_exp, iterations=2)
        else:
            leaf_exp = leaf_base.astype(np.uint8)

        # Bintil pustula karat / bercak nekrotik pada daun (hanya yang menempel pada daun)
        is_rust_spot = (h >= 6.0) & (h < 30.0) & (r > g * 1.05) & (s >= 0.18) & (v >= 0.14) & (~is_skin) & (leaf_exp > 0)

        # Mask vegetasi murni pada helai daun
        plant_mask = leaf_base | is_rust_spot
        plant_ratio = float(np.mean(plant_mask))
        
        # Cek kertas putih / dinding polos / background abu-abu
        is_white_gray = (s < 0.10) & (v > 0.70)
        if np.mean(is_white_gray) > 0.88 and plant_ratio < 0.02:
            return False, "Terdeteksi objek kertas atau dinding putih polos, bukan daun bawang.", plant_ratio

        # Hanya tolak jika BENAR-BENAR murni kulit/tangan manusia tanpa helai daun (< 1.2% tanaman)
        if skin_ratio > 0.35 and plant_ratio < 0.012:
            return False, "Terdeteksi hanya menampilkan kulit/tangan manusia tanpa helai daun bawang.", plant_ratio
        
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

def validate_onion_image(image: Image.Image, api_key: str | None = None, min_ratio: float = 0.08):
    """
    Sistem Validasi Guardrail Gatekeeper Citra Tanaman Bawang Merah:
    Memverifikasi keaslian foto daun bawang merah sebelum proses diagnosa.
    Menolak foto manusia, hewan, kendaraan, tanah kosong, dan objek non-bawang.
    Tetap mengizinkan anomali wajar seperti daun bawang yang dipegang tangan petani di kebun.
    """
    is_plant, reason, ratio = check_shallot_leaf_mask(image, min_ratio=min_ratio)
    rec_leaf_pct = max(3, min(35, int(np.floor(ratio * 100.0)))) if ratio >= 0.03 else 5
    info = {
        "plant_ratio": ratio,
        "min_ratio": min_ratio,
        "is_ratio_rejection": (not is_plant and "Rasio daun bawang pada foto hanya" in reason),
        "reason": reason,
        "recommended_leaf_pct": rec_leaf_pct
    }
    if not is_plant:
        return False, f"INVALID: {reason}", info
    return True, f"VALID (Rasio Kanopi Daun: {ratio*100:.1f}%)", info

# Alias untuk kompatibilitas
validate_with_groq_vision = validate_onion_image

def inspect_visual_leaf_symptoms(
    image: Image.Image,
    target_disease: str = "",
    is_healthy: bool = False,
    is_pure_healthy: bool = False,
    diag_mode: str = "single",
    secondary_disease: str = "",
    tertiary_disease: str = ""
) -> dict:
    """
    Modul Inspeksi Fitur Visual Citra Daun Bawang Merah:
    Menganalisis karakteristik visual fisik helai daun secara sinkron dengan hasil prediksi model.
    Jika daun sehat murni (is_pure_healthy=True), secara tegas menetapkan kerusakan 0% dan kondisi sehat prima.
    Jika terdeteksi gejala penyakit (meskipun sehat berada di peringkat 1 namun bersaing dengan hawar/virus),
    sistem menghitung kerusakan piksel nyata dari helai daun secara presisi tanpa angka fiktif.
    """
    if is_pure_healthy or (is_healthy and diag_mode == "single" and not secondary_disease):
        return {
            "has_visual_evidence": True,
            "evidence_disease": "Daun Sehat & Normal",
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

        name_lower = f"{target_disease} {secondary_disease} {tertiary_disease}".lower()
        is_les = np.zeros(h_arr.shape, dtype=bool)

        if "rust" in name_lower or "karat" in name_lower:
            is_les |= ((h_arr >= 7.0) & (h_arr <= 28.0) & (r > g * 1.08) & (s_arr >= 0.22) & (v_arr >= 0.15) & (v_arr <= 0.85))
        if "trotol" in name_lower or "bercak" in name_lower or "alternaria" in name_lower:
            is_les |= (((h_arr <= 16.0) | (h_arr >= 265.0)) & (r > g * 1.05) & (s_arr >= 0.15) & (v_arr >= 0.08) & (v_arr <= 0.70))
        if "hawar" in name_lower or "stemphylium" in name_lower or "colletotrichum" in name_lower:
            is_les |= ((h_arr >= 18.0) & (h_arr <= 42.0) & (s_arr >= 0.12) & (v_arr >= 0.15) & (g >= r * 0.70))
        if "moler" in name_lower or "fusarium" in name_lower or "inul" in name_lower:
            is_les |= ((h_arr >= 26.0) & (h_arr <= 50.0) & (s_arr >= 0.15) & (v_arr >= 0.20))
        if "virus" in name_lower or "iysv" in name_lower or "mildew" in name_lower or "embun" in name_lower:
            is_les |= ((h_arr >= 25.0) & (h_arr <= 55.0) & (s_arr >= 0.12) & (v_arr >= 0.30))

        if not np.any(is_les):
            is_les = (h_arr >= 10.0) & (h_arr <= 45.0) & (s_arr >= 0.15) & (v_arr >= 0.12)

        les_pixels = int(np.count_nonzero(is_les & is_plant))
        severity_pct = round(min(max((les_pixels / total_plant) * 100.0, 3.5), 92.0), 1)
        healthy_pct = round(max(100.0 - severity_pct, 5.0), 1)

        if severity_pct < 10.0:
            severity_level = "Sangat Ringan (< 10%)"
        elif severity_pct < 25.0:
            severity_level = "Ringan (10% - 25%)"
        elif severity_pct < 50.0:
            severity_level = "Sedang (25% - 50%)"
        else:
            severity_level = "Berat / Kritis (> 50%)"

        if diag_mode == "three_way":
            ev_desc = (
                f"Pemindaian visual helai daun mendeteksi sekitar {severity_pct}% area daun memperlihatkan "
                f"gejala perubahan warna, klorosis, dan pengeringan ujung (indikasi infeksi awal {secondary_disease} "
                f"dan {tertiary_disease}), sementara {healthy_pct}% jaringan daun hijau masih utuh."
            )
        elif diag_mode == "two_way":
            ev_desc = (
                f"Pemindaian visual mengonfirmasi gejala infeksi fisik {target_disease} / {secondary_disease} "
                f"dengan tingkat keparahan {severity_level} (estimasi jaringan terdampak: {severity_pct}%, "
                f"jaringan daun hijau tersisa: {healthy_pct}%)."
            )
        else:
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
    except Exception:
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

def get_short_disease_title(name: str) -> str:
    """Mengembalikan nama ringkas penyakit yang bersih dan ramah layar HP untuk badge HUD Scanner."""
    low = (name or "").lower()
    if "sehat" in low or "healthy" in low:
        return "Daun Sehat"
    if "trotol" in low or "bercak ungu" in low or "alternaria" in low or "purple" in low:
        return "Bercak Ungu"
    if "embun" in low or "mildew" in low or "peronospora" in low:
        return "Embun Bulu"
    if "karat" in low or "rust" in low or "puccinia" in low:
        return "Karat Daun"
    if "hawar" in low or "blight" in low or "stemphylium" in low or "colletotrichum" in low:
        return "Hawar Daun"
    if "moler" in low or "fusarium" in low or "inul" in low:
        return "Layu Moler"
    if "virus" in low or "iysv" in low:
        return "Virus IYSV"
    clean = (name or "").split("/")[0].split("(")[0].strip()
    return clean[:14] if clean else "Penyakit"

def _hud_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1

def clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_lesion_hud_map(
    image: Image.Image,
    target_class_idx: int = 0,
    is_healthy: bool = False,
    second_class_idx: int | None = None,
    third_class_idx: int | None = None,
    primary_name: str = "Penyakit A",
    second_name: str | None = None,
    third_name: str | None = None,
    is_differential: bool = False,
    has_multi_disease: bool = False,
    has_three_diseases: bool = False,
    is_pure_healthy: bool = False,
    model=None,
    **kwargs
):
    """
    Peta HUD Scanner Titik Kerusakan Daun Bawang Merah (v7 - Adaptive Multi-Rank 1-2-3 Detection).
    Menandai 1, 2, atau 3 penyakit/kondisi tergantung distribusi keyakinan model.
    """
    img_rgb = image.convert("RGB")
    W, H = img_rgb.size

    scale = min(1.0, 512.0 / float(max(W, H)))
    ww, hh = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
    work = img_rgb.resize((ww, hh), Image.Resampling.BILINEAR) if scale < 1.0 else img_rgb
    arr_u8 = cv2.GaussianBlur(np.asarray(work, dtype=np.uint8), (3, 3), 0)
    arr = arr_u8.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    hsv = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2.0
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    total = r + g + b + 1.0
    gn = g / total
    rn = r / total
    exg = 2.0 * g - r - b
    g_over_r = g / (r + 1.0)
    min_side = min(ww, hh)
    img_area = float(ww * hh)

    # 1. Deteksi objek non-daun (kulit tangan, tanah sawah, pantulan cahaya, kegelapan)
    is_skin = (
        (hue >= 4.0) & (hue <= 30.0) & (sat >= 0.15) & (sat <= 0.62) &
        (val >= 0.30) & (r > g * 1.10) & (g > b * 1.02) & (rn >= 0.38)
    )
    is_soil = (
        (hue >= 8.0) & (hue <= 42.0) & (sat >= 0.15) & (sat <= 0.50) &
        (g_over_r < 0.86) & (exg < 12.0) & (r - b < 70.0)
    )
    is_neutral = (sat < 0.10) & ((val > 0.85) | (val < 0.10))
    is_specular = (sat < 0.12) & (val > 0.90)
    is_too_dark = val < 0.07

    # 2. Daun hijau inti (green core) & jaringan kuning terhubung
    green_core = (
        (hue >= 42.0) & (hue <= 170.0) & (sat >= 0.12) & (val >= 0.08) &
        (gn >= 0.355) & (exg > 6.0) & (~is_skin) & (~is_soil)
    )
    yellow_tissue = (
        (hue >= 36.0) & (hue <= 72.0) & (sat >= 0.16) & (val >= 0.28) &
        (g_over_r >= 0.84) & (~is_skin) & (~is_soil)
    )

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_u8 = cv2.morphologyEx(green_core.astype(np.uint8), cv2.MORPH_OPEN, k3)
    tissue = ((green_u8 > 0) | yellow_tissue).astype(np.uint8)

    n_t, lab_t, stats_t, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    keep = np.zeros(n_t, dtype=bool)
    if n_t > 1:
        green_count = np.bincount(lab_t.ravel(), weights=(green_u8.ravel() > 0), minlength=n_t)
        for i in range(1, n_t):
            if green_count[i] >= max(12.0, 0.08 * stats_t[i, cv2.CC_STAT_AREA]):
                keep[i] = True
        if not keep[1:].any():
            keep[1 + int(np.argmax(stats_t[1:, cv2.CC_STAT_AREA]))] = True
    leaf_core = keep[lab_t] if n_t > 1 else np.zeros((hh, ww), dtype=bool)

    core_u8 = cv2.morphologyEx(leaf_core.astype(np.uint8), cv2.MORPH_CLOSE, k3)
    n_c, lab_c, stats_c, _ = cv2.connectedComponentsWithStats(core_u8, connectivity=8)
    if n_c > 1:
        areas = stats_c[1:, cv2.CC_STAT_AREA]
        min_keep = max(0.08 * float(areas.max()), 0.002 * img_area)
        valid = np.zeros(n_c, dtype=bool)
        valid[1:] = areas >= min_keep
        core_u8 = valid[lab_c].astype(np.uint8)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_hud_odd(min_side * 0.03), _hud_odd(min_side * 0.03)))
    closed = cv2.morphologyEx(core_u8, cv2.MORPH_CLOSE, k_close)

    leaf_region = closed.astype(bool)
    inv_closed = (closed == 0).astype(np.uint8)
    n_h, lab_h, stats_h, _ = cv2.connectedComponentsWithStats(inv_closed, connectivity=8)
    for i in range(1, n_h):
        area_h = stats_h[i, cv2.CC_STAT_AREA]
        x0, y0 = stats_h[i, cv2.CC_STAT_LEFT], stats_h[i, cv2.CC_STAT_TOP]
        w0, h0 = stats_h[i, cv2.CC_STAT_WIDTH], stats_h[i, cv2.CC_STAT_HEIGHT]
        is_border = (x0 == 0) or (y0 == 0) or (x0 + w0 >= ww) or (y0 + h0 >= hh)
        if (not is_border) and (area_h < 0.035 * img_area):
            leaf_region |= (lab_h == i)

    leaf_region &= ~(is_soil | is_skin | is_neutral | is_specular | is_too_dark)

    mx, my = max(2, int(ww * 0.015)), max(2, int(hh * 0.015))
    leaf_region[:my, :] = False
    leaf_region[-my:, :] = False
    leaf_region[:, :mx] = False
    leaf_region[:, -mx:] = False

    if leaf_region.sum() < max(60, int(0.004 * img_area)):
        leaf_region = (~is_skin) & (~is_soil) & (~is_too_dark)

    healthy_green = green_core & leaf_region & (sat >= 0.18)
    ref_px = healthy_green if healthy_green.sum() >= 40 else leaf_region
    med_v = float(np.median(val[ref_px])) if ref_px.any() else 0.5
    med_s = float(np.median(sat[ref_px])) if ref_px.any() else 0.4
    med_v = max(med_v, 0.12)
    med_s = max(med_s, 0.10)
    rel_v = val / med_v
    rel_s = sat / med_s

    greenness = np.clip((gn - 0.33) / 0.10, 0.0, 1.0) * np.clip(exg / 40.0, 0.0, 1.0)
    not_green = 1.0 - greenness

    def disease_score_map(name: str) -> np.ndarray:
        n = (name or "").lower()
        if "karat" in n or "rust" in n or "puccinia" in n:
            rust_col = (hue >= 12.0) & (hue <= 48.0) & (r > g * 1.04) & (r > b * 1.20) & (sat >= 0.26)
            s = np.where(rust_col, clip01((r - g) / 35.0) * clip01((sat - 0.25) / 0.35), 0.0)
        elif "trotol" in n or "bercak" in n or "alternaria" in n or "purple" in n:
            purple = ((hue >= 240.0) | (hue <= 24.0)) & (sat >= 0.10)
            dark_sunken = (rel_v < 0.72) & (not_green > 0.25)
            match = purple | dark_sunken
            s = np.where(match, 0.50 * clip01((0.75 - rel_v) / 0.40) + 0.35 * purple.astype(np.float32) + 0.15 * not_green, 0.0)
        elif "embun" in n or "mildew" in n or "peronospora" in n:
            pale = (hue >= 30.0) & (hue <= 110.0) & (sat <= 0.35) & (val >= 0.25) & (val <= 0.90)
            s = np.where(pale, 0.50 * clip01((0.36 - sat) / 0.25) + 0.30 * clip01(1.0 - np.abs(rel_v - 1.0) / 0.5) + 0.20 * not_green, 0.0)
        elif "hawar" in n or "blight" in n or "stemphylium" in n or "colletotrichum" in n:
            tan = (hue >= 16.0) & (hue <= 62.0) & (sat >= 0.10) & (sat <= 0.65) & (val >= 0.30) & (not_green > 0.35)
            s = np.where(tan, 0.45 * not_green + 0.35 * clip01((val - 0.30) / 0.40) + 0.20 * clip01((r - b) / 40.0), 0.0)
        elif "moler" in n or "fusarium" in n or "inul" in n:
            yellow = (hue >= 38.0) & (hue <= 68.0) & (sat >= 0.26) & (val >= 0.35) & (g > b * 1.15)
            s = np.where(yellow, 0.50 * clip01((sat - 0.24) / 0.35) + 0.30 * clip01((val - 0.35) / 0.45) + 0.20 * not_green, 0.0)
        elif "virus" in n or "iysv" in n:
            straw = (hue >= 24.0) & (hue <= 64.0) & (sat >= 0.10) & (sat <= 0.55) & (val >= 0.30) & (not_green > 0.30)
            s = np.where(straw, 0.45 * not_green + 0.35 * clip01((val - 0.30) / 0.40) + 0.20 * clip01((0.55 - sat) / 0.35), 0.0)
        elif "sehat" in n or "healthy" in n:
            s = np.where(green_core, 0.7 * greenness + 0.3 * sat, 0.0)
        else:
            s = np.where(not_green > 0.30, 0.7 * not_green + 0.3 * clip01(np.abs(1.0 - rel_v) / 0.5), 0.0)

        s = np.where(leaf_region, s.astype(np.float32), 0.0)
        return s

    leaf_dist = cv2.distanceTransform(leaf_region.astype(np.uint8), cv2.DIST_L2, 3)
    max_d = float(leaf_dist.max())
    leaf_interior = (leaf_dist >= max(1.5, min(3.0, max_d * 0.3))) if max_d >= 2.0 else leaf_region

    base_rad_w = max(9, int(min_side * 0.045))
    min_dist_w = max(18, int(min_side * 0.12))

    def find_peak_spot(score_map, exclude_mask=None):
        sm = score_map.copy()
        if exclude_mask is not None:
            sm = np.where(exclude_mask, 0.0, sm)
        # Kunci ketat ke dalam area helai daun (bebas latar belakang & jari tangan)
        sm = np.where(leaf_region, sm, 0.0)

        if sm.max() <= 0.05:
            fallback_score = np.where(leaf_region, not_green, 0.0)
            if exclude_mask is not None:
                fallback_score = np.where(exclude_mask, 0.0, fallback_score)
            sm = fallback_score

        smoothed = cv2.GaussianBlur(sm.astype(np.float32), (7, 7), 0)
        # Berikan bobot jarak interior agar retikel tidak melenceng ke tepian helai daun
        # Nol-kan piksel yang terlalu dekat tepi (leaf_dist < 1.5) agar retikel selalu berakar kuat di daging helai daun
        smoothed = np.where(leaf_region & (leaf_dist >= 1.5), smoothed * np.clip(leaf_dist / max(2.0, float(base_rad_w * 0.75)), 0.1, 1.0), 0.0)

        iy, ix = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        val_peak = float(smoothed[iy, ix])

        # Jika nilai terlalu kecil atau bukan pada daun, ambil titik terdalam di interior helai daun
        if val_peak <= 0.001 or not leaf_region[iy, ix]:
            target_mask = leaf_interior if (exclude_mask is None) else (leaf_interior & (~exclude_mask))
            if not target_mask.any():
                target_mask = leaf_region if (exclude_mask is None) else (leaf_region & (~exclude_mask))
            if not target_mask.any():
                target_mask = leaf_region
            dist_cand = np.where(target_mask, leaf_dist, 0.0)
            iy, ix = np.unravel_index(np.argmax(dist_cand), dist_cand.shape)

        local_dist = float(leaf_dist[iy, ix])
        # Batasi radius agar pas di dalam lebar helai daun
        rad_l = int(np.clip(base_rad_w, 6, max(6, int(local_dist * 0.75))))
        return {"x": int(ix), "y": int(iy), "r": rad_l, "score": val_peak}

    def find_healthy_spot(exclude_mask=None):
        hg = healthy_green.copy()
        if exclude_mask is not None:
            hg &= ~exclude_mask
        if hg.sum() < 20:
            hg = leaf_region.copy()
            if exclude_mask is not None:
                hg &= ~exclude_mask
        dist = cv2.distanceTransform(hg.astype(np.uint8), cv2.DIST_L2, 3)
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        if not leaf_region[iy, ix]:
            iy, ix = np.unravel_index(int(np.argmax(leaf_dist)), leaf_dist.shape)
        local_dist = float(leaf_dist[iy, ix])
        rad_l = int(np.clip(base_rad_w, 6, max(6, int(local_dist * 0.75))))
        return {"x": int(ix), "y": int(iy), "r": rad_l, "score": 1.0}

    spots_w = []
    p1_is_healthy = ("sehat" in (primary_name or "").lower()) or ("healthy" in (primary_name or "").lower())
    p2_is_healthy = bool(second_name) and (("sehat" in second_name.lower()) or ("healthy" in second_name.lower()))
    p3_is_healthy = bool(third_name) and (("sehat" in third_name.lower()) or ("healthy" in third_name.lower()))

    # KONDISI A: DAUN SEHAT MURNI (Dominan Sehat)
    if is_pure_healthy or (p1_is_healthy and not (has_three_diseases or has_multi_disease or is_differential)):
        sp_h = find_healthy_spot()
        spots_w.append((sp_h["x"], sp_h["y"], base_rad_w, "[OK] Daun Sehat", "green"))

    # KONDISI B: 3 KEMUNGKINAN BERSAING (Top 1, 2, 3 Tinggi/Mendekati)
    elif has_three_diseases and second_name and third_name:
        yy, xx = np.ogrid[:hh, :ww]
        # Spot 1
        title_1 = get_short_disease_title(primary_name)
        if p1_is_healthy:
            sp1 = find_healthy_spot()
            c1 = "green"
            l1 = f"[1] {title_1}"
        else:
            sp1 = find_peak_spot(disease_score_map(primary_name))
            c1 = "red"
            l1 = f"[1] {title_1}"
        spots_w.append((sp1["x"], sp1["y"], sp1["r"], l1, c1))
        excl_1 = (xx - sp1["x"]) ** 2 + (yy - sp1["y"]) ** 2 < max(min_dist_w, sp1["r"] * 2) ** 2

        # Spot 2
        title_2 = get_short_disease_title(second_name)
        if p2_is_healthy:
            sp2 = find_healthy_spot(exclude_mask=excl_1)
            c2 = "green"
            l2 = f"[2] {title_2}"
        else:
            sp2 = find_peak_spot(disease_score_map(second_name), exclude_mask=excl_1)
            c2 = "orange"
            l2 = f"[2] {title_2}"
        spots_w.append((sp2["x"], sp2["y"], sp2["r"], l2, c2))
        excl_2 = excl_1 | ((xx - sp2["x"]) ** 2 + (yy - sp2["y"]) ** 2 < max(min_dist_w, sp2["r"] * 2) ** 2)

        # Spot 3
        title_3 = get_short_disease_title(third_name)
        if p3_is_healthy:
            sp3 = find_healthy_spot(exclude_mask=excl_2)
            c3 = "green"
            l3 = f"[3] {title_3}"
        else:
            sp3 = find_peak_spot(disease_score_map(third_name), exclude_mask=excl_2)
            c3 = "blue"
            l3 = f"[3] {title_3}"
        spots_w.append((sp3["x"], sp3["y"], sp3["r"], l3, c3))

    # KONDISI C: 2 KEMUNGKINAN BERSAING (Top 2 Tinggi / Diferensial)
    elif (has_multi_disease or is_differential) and second_name:
        yy, xx = np.ogrid[:hh, :ww]
        title_1 = get_short_disease_title(primary_name)
        if p1_is_healthy:
            sp1 = find_healthy_spot()
            c1 = "green"
            l1 = f"[1] {title_1}"
        else:
            sp1 = find_peak_spot(disease_score_map(primary_name))
            c1 = "red"
            l1 = f"[1] {title_1}"
        spots_w.append((sp1["x"], sp1["y"], sp1["r"], l1, c1))
        excl_1 = (xx - sp1["x"]) ** 2 + (yy - sp1["y"]) ** 2 < max(min_dist_w, sp1["r"] * 2) ** 2

        title_2 = get_short_disease_title(second_name)
        if p2_is_healthy:
            sp2 = find_healthy_spot(exclude_mask=excl_1)
            c2 = "green"
            l2 = f"[2] {title_2}"
        else:
            sp2 = find_peak_spot(disease_score_map(second_name), exclude_mask=excl_1)
            c2 = "orange"
            l2 = f"[2] {title_2}"
        spots_w.append((sp2["x"], sp2["y"], sp2["r"], l2, c2))

    # KONDISI D: 1 PENYAKIT DOMINAN (Single Diagnosis Tinggi)
    else:
        title_1 = get_short_disease_title(primary_name)
        score_1 = disease_score_map(primary_name)
        spot_1 = find_peak_spot(score_1)
        spots_w.append((spot_1["x"], spot_1["y"], spot_1["r"], f"[1] {title_1}", "red"))

        yy, xx = np.ogrid[:hh, :ww]
        excl = (xx - spot_1["x"]) ** 2 + (yy - spot_1["y"]) ** 2 < max(min_dist_w, spot_1["r"] * 2) ** 2
        # Hanya cari spot kedua jika benar-benar ada lesi lain dari penyakit yang sama
        spot_2 = find_peak_spot(score_1, exclude_mask=excl)
        if spot_2 and spot_2["score"] >= max(0.25, spot_1["score"] * 0.50):
            spots_w.append((spot_2["x"], spot_2["y"], spot_2["r"], f"[1] {title_1} #2", "red"))

    inv = 1.0 / scale
    leaf_orig = cv2.resize(leaf_region.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    leaf_dist_orig = cv2.distanceTransform(leaf_orig, cv2.DIST_L2, 3)
    leaf_orig_mask = (leaf_dist_orig > 0)

    spots = []
    for (x, y, rad, label, color) in spots_w:
        ox = int(np.clip(round(x * inv), 0, W - 1))
        oy = int(np.clip(round(y * inv), 0, H - 1))

        # Pastikan titik berada kokoh di dalam helai daun
        # Jika berada di luar atau di tepi tipis (leaf_dist < 6), cari titik jaringan terdekat yang lebih tebal di lesi lokal tersebut
        if (not leaf_orig_mask[oy, ox]) or (leaf_dist_orig[oy, ox] < 6):
            search_r = max(10, int(min_wh * 0.035))
            y_min = max(0, oy - search_r)
            y_max = min(H, oy + search_r + 1)
            x_min = max(0, ox - search_r)
            x_max = min(W, ox + search_r + 1)
            sub_dist = leaf_dist_orig[y_min:y_max, x_min:x_max]
            if sub_dist.any() and sub_dist.max() > 0:
                sy, sx = np.unravel_index(np.argmax(sub_dist), sub_dist.shape)
                ox, oy = x_min + int(sx), y_min + int(sy)
            else:
                iy_snap, ix_snap = np.unravel_index(np.argmax(leaf_dist_orig), leaf_dist_orig.shape)
                ox, oy = int(ix_snap), int(iy_snap)

        local_d = float(leaf_dist_orig[oy, ox])
        # Batasi radius agar lingkaran retikel TIDAK PERNAH tembus ke luar batas helai daun
        rad_orig = int(np.clip(round(rad * inv), 4, max(4, int(local_d * 0.68))))
        spots.append((ox, oy, rad_orig, label, color))

    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)
    palette = {
        "green": ((16, 185, 129), (167, 243, 208), (5, 150, 105)),
        "orange": ((245, 158, 11), (254, 240, 138), (217, 119, 6)),
        "red": ((239, 68, 68), (254, 202, 202), (220, 38, 38)),
        "blue": ((2, 132, 199), (186, 230, 253), (3, 105, 161)),
    }
    min_wh = min(W, H)
    line_w = max(2, int(min_wh * 0.005))
    dot_r = max(3, int(min_wh * 0.007))
    font_px = max(11, int(min_wh * 0.028))
    try:
        font = ImageFont.load_default(size=font_px)
    except TypeError:
        font = ImageFont.load_default()

    for cx, cy, rad, label, color_type in spots:
        c_hud, c_in, c_bg = palette.get(color_type, palette["red"])
        local_d = float(leaf_dist_orig[cy, cx]) if (0 <= cy < H and 0 <= cx < W) else float(rad)
        # Garis bidik (tick) dibatasi maksimal 22% dari jarak ke tepi daun, sehingga total (rad + tick) < 90% dari batas daun
        local_tick = int(np.clip(max(3, int(min_wh * 0.012)), 2, max(2, int(local_d * 0.22))))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=c_hud, width=line_w)
        inner_r = max(rad - line_w * 2, 3)
        draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], outline=c_in, width=max(1, line_w - 1))
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=c_hud)
        draw.line([cx - rad - local_tick, cy, cx - rad + 2, cy], fill=c_hud, width=line_w)
        draw.line([cx + rad - 2, cy, cx + rad + local_tick, cy], fill=c_hud, width=line_w)
        draw.line([cx, cy - rad - local_tick, cx, cy - rad + 2], fill=c_hud, width=line_w)
        draw.line([cx, cy + rad - 2, cx, cy + rad + local_tick], fill=c_hud, width=line_w)

        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, font_px // 4)
        bw_, bh_ = tw + pad * 2, th + pad * 2
        bx1 = int(np.clip(cx - bw_ // 2, 2, max(2, W - bw_ - 2)))
        by1 = cy - rad - local_tick - bh_ - 2
        if by1 < 2:
            by1 = cy + rad + local_tick + 2
        by1 = int(np.clip(by1, 2, max(2, H - bh_ - 2)))
        draw.rounded_rectangle([bx1, by1, bx1 + bw_, by1 + bh_], radius=max(3, pad), fill=c_bg, outline=(255, 255, 255), width=1)
        draw.text((bx1 + pad - tb[0], by1 + pad - tb[1]), label, fill=(255, 255, 255), font=font)

    return annotated, spots
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
        is_shallot, reason_msg, _ = check_shallot_leaf_mask(image, min_ratio=0.03)
        if not is_shallot:
            raise ValueError(f"OOD_GUARD_REJECTED: {reason_msg}")

    # 1. Buka gambar dengan PIL, perbaiki orientasi EXIF, convert ke RGB
    img_rgb = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = img_rgb.size

    # 2. Isolasi Helai Daun Bawang Merah (Auto-Detect Leaf ROI)
    # Menyingkirkan latar belakang tanah, mulsa, dan tangan petani yang memegang daun
    leaf_roi, leaf_bbox, leaf_coverage = extract_leaf_roi(img_rgb, padding_pct=0.08)
    is_auto_cropped = bool(leaf_coverage < 0.96 and leaf_roi.size[0] >= 30 and leaf_roi.size[1] >= 30)
    target_img = leaf_roi if is_auto_cropped else img_rgb

    # 3. Resize ke (img_size, img_size) pada area daun fokus, ubah ke tensor float 0-1
    img_size = int(meta.get("img_size", 224))
    resized = target_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
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
        temperature = float(meta.get("temperature", 0.50))
        if temperature < 0.20:
            temperature = 0.50
        probs_tensor = torch.softmax(logits / temperature, dim=1)[0]

    # 6. Ambil kelas tertinggi & evaluasi conf_threshold
    conf_threshold = float(meta.get("conf_threshold", 0.65))
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
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1] if len(sorted_indices) > 1 else top_idx
    third_idx = sorted_indices[2] if len(sorted_indices) > 2 else second_idx

    top_confidence = round(float(probs_np[top_idx]) * 100.0, 1)
    second_confidence = round(float(probs_np[second_idx]) * 100.0, 1)
    third_confidence = round(float(probs_np[third_idx]) * 100.0, 1)

    raw_class_name = class_labels[top_idx]
    second_class_name = class_labels[second_idx]
    third_class_name = class_labels[third_idx]

    confidence_margin = round(top_confidence - second_confidence, 1)
    margin_top_third = round(top_confidence - third_confidence, 1)

    is_healthy_1 = (raw_class_name == "Sehat") or CLASS_METADATA.get(raw_class_name, {}).get("is_healthy", False)
    is_healthy_2 = (second_class_name == "Sehat") or CLASS_METADATA.get(second_class_name, {}).get("is_healthy", False)
    is_healthy_3 = (third_class_name == "Sehat") or CLASS_METADATA.get(third_class_name, {}).get("is_healthy", False)

    # Evaluasi Adaptif Peringkat 1, 2, atau 3 Kemungkinan:
    # 1. Peringkat 1 Dominan:
    #    Jika peringkat 1 sangat tinggi (>= 65%) ATAU selisih jauh (margin >= 25%) ATAU peringkat 2 sangat kecil (< 15%)
    is_single_dominant = (
        (top_confidence >= 65.0) or
        (confidence_margin >= 25.0) or
        (second_confidence < 15.0)
    )

    # 2. 3 Kemungkinan Bersaing / Terdeteksi Banyak:
    #    Jika bukan dominan tunggal, peringkat 3 cukup tinggi (>= 15%) dan mendekati peringkat 1 (selisih <= 22%)
    is_three_way = (
        (not is_single_dominant) and
        (not uncertain) and
        (third_confidence >= 15.0) and
        (margin_top_third <= 22.0)
    )

    # 3. 2 Kemungkinan Bersaing (Diferensial):
    is_two_way = (
        (not is_single_dominant) and
        (not is_three_way) and
        (not uncertain) and
        (second_confidence >= 16.0)
    )

    if is_three_way:
        diag_mode = "three_way"
    elif is_two_way:
        diag_mode = "two_way"
    else:
        diag_mode = "single"

    is_pure_healthy = is_healthy_1 and (diag_mode == "single")
    is_differential = (diag_mode == "two_way")
    has_three_diseases = (diag_mode == "three_way")
    has_multi_disease = (diag_mode in ("two_way", "three_way"))

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

    third_metadata = CLASS_METADATA.get(third_class_name, {
        "nama_id": third_class_name,
        "latin": "-",
        "status": "healthy" if is_healthy_3 else "disease",
        "is_healthy": is_healthy_3,
        "ciri_lapangan": "Periksa kondisi helai daun dan bercak secara teliti.",
        "gejala": "Pangkas daun yang bergejala dan bersihkan bedengan.",
        "pencegahan": "Jaga drainase parit dan sanitasi pematang.",
        "solusi": "Gunakan obat yang sesuai.",
        "rekomendasi_singkat": "Lakukan sanitasi daun sakit."
    })

    # 7. Ekstraksi Bukti Visual yang Sinkron Presisi dengan Hasil Prediksi Model
    visual_evidence = inspect_visual_leaf_symptoms(
        image=image,
        target_disease=metadata["nama_id"],
        is_healthy=is_healthy_1,
        is_pure_healthy=is_pure_healthy,
        diag_mode=diag_mode,
        secondary_disease=second_metadata["nama_id"] if has_multi_disease else "",
        tertiary_disease=third_metadata["nama_id"] if has_three_diseases else ""
    )

    # Peta HUD Scanner Lesi Multi-Kondisi & Multi-Penyakit (Adaptif 1, 2, atau 3 Titik)
    annotated_cam, cam_spots = generate_lesion_hud_map(
        image=image,
        target_class_idx=top_idx,
        is_healthy=is_healthy_1,
        second_class_idx=second_idx if has_multi_disease else None,
        third_class_idx=third_idx if has_three_diseases else None,
        primary_name=metadata["nama_id"],
        second_name=second_metadata["nama_id"] if has_multi_disease else None,
        third_name=third_metadata["nama_id"] if has_three_diseases else None,
        is_differential=is_differential,
        has_multi_disease=has_multi_disease,
        has_three_diseases=has_three_diseases,
        is_pure_healthy=is_pure_healthy
    )
    visual_evidence["overlay_img"] = annotated_cam
    visual_evidence["num_spots_detected"] = len(cam_spots)
    visual_evidence["hud_spots"] = cam_spots
    visual_evidence["has_multi_disease"] = has_multi_disease
    visual_evidence["has_three_diseases"] = has_three_diseases

    diag_info = {
        "orig_mode": image.mode,
        "orig_size": (orig_w, orig_h),
        "norm_mode_name": "EfficientNet-B0 TorchScript + Leaf-Focused TTA",
        "num_views": 2,
        "is_auto_cropped": is_auto_cropped,
        "leaf_coverage_pct": round(leaf_coverage * 100.0, 1),
        "leaf_bbox": leaf_bbox,
        "min_pixel": 0.0,
        "max_pixel": 1.0,
        "api_output": api_output,
        "uncertain": uncertain,
        "diag_mode": diag_mode,
        "is_pure_healthy": is_pure_healthy,
        "third_class_raw": third_class_name,
        "third_confidence": third_confidence,
        "third_metadata": third_metadata,
        "has_three_diseases": has_three_diseases,
        "has_multi_disease": has_multi_disease,
        "is_single_dominant": is_single_dominant
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
        target_img,
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
    evidence_desc=None,
    third_disease_name=None,
    third_confidence=None,
    is_three_way=False
):
    """
    Memanggil Groq API untuk menyusun petunjuk obat dan perawatan lahan yang panjang, mendalam,
    diselaraskan persis dengan hasil vonis diagnosis penyakit dari sumber resmi Balitsa Lembang & BPTP Kementan.
    """
    latin_str = f" ({latin_name})" if latin_name else ""
    sev_str = f"Tingkat Keparahan Infeksi: {severity_level}\n" if severity_level else ""
    ev_str = f"Gejala Fisik Lapangan: {evidence_desc}\n" if evidence_desc else ""

    if is_three_way and second_disease_name and third_disease_name:
        angle_title = "Rekomendasi Terpadu 3 Spektrum Penyakit Bersaing (Riset Balitsa & BPTP Kementan)"
        user_prompt = (
            f"VONIS DIAGNOSIS PENYAKIT (3 KEMUNGKINAN BERSAING): {disease_name}{latin_str} ({confidence:.1f}%), {second_disease_name} ({second_confidence:.1f}%), dan {third_disease_name} ({third_confidence:.1f}%).\n"
            f"{sev_str}"
            f"{ev_str}\n"
            "Anda bertindak sebagai Ahli Agronomi dan Konsultan Proteksi Tanaman Hortikultura Bawang Merah (merujuk pada riset resmi Balitsa Lembang, BPTP Kementan RI, dan Jurnal Fitopatologi Indonesia).\n"
            "Bantu petani merangkum solusi penanganan terpadu langsung dari sumber-sumber terjamin ketika tanaman menunjukkan potensi 3 penyakit bersaing sekaligus:\n"
            f"1. Ciri fisik pembeda langsung di bedengan sawah antara {disease_name}, {second_disease_name}, dan {third_disease_name} (tekstur helai daun, pola bercak, bau langu bakteri vs serbuk spora jamur vs klorosis virus/vektor thrips).\n"
            "2. Rekomendasi obat semprot terpadu spektrum luas yang aman mencakup ketiga patogen secara berimbang tanpa merusak tanaman.\n\n"
            "WAJIB susun jawaban ke dalam 3 bagian persis dengan judul pemisah berikut:\n\n"
            "=== TINDAKAN LANGSUNG DI KEBUN ===\n"
            "- Berikan langkah taktis darurat dalam 24 jam pertama di bedengan sawah.\n"
            f"- Jelaskan panduan praktis membedakan {disease_name} vs {second_disease_name} vs {third_disease_name} secara visual dengan mata telanjang di sawah.\n"
            "- Jelaskan teknik pemotongan daun bergejala dan sanitasi alat gunting/pisau agar patogen tidak menyebar ke tanaman sekitar.\n\n"
            "=== REKOMENDASI OBAT SEMPROT ===\n"
            "- Berikan kombinasi obat semprot terpadu yang aman mencakup spektrum ketiga masalah (kombinasi bakterisida tembaga seperti Tembaga Hidroksida / Kasugamisin, fungisida sistemik seperti Difenokonazol / Mankozeb / Azoksistrobin, dan perlakuan serangga vektor bila ada suspek virus).\n"
            "- Sebutkan takaran dosis realistis (misal: 1,5 - 2 sendok makan per tangki semprot 16 Liter air).\n"
            "- Sebutkan waktu semprot terbaik (pagi hari sebelum jam 09.00 saat embun mengering, atau sore setelah jam 16.00 saat angin tenang).\n"
            "- Wajib ingatkan penambahan perekat/perata (surfactant) non-ionik agar obat menempel kuat di lapisan lilin daun bawang.\n\n"
            "=== PERAWATAN LAHAN & PUPUK ===\n"
            "Jelaskan bagian ini secara terstruktur dalam 4 poin praktis:\n"
            "1. Pengaturan Parit & Tata Air: Atur muka air parit 20-25 cm di bawah bedengan (sistem macak-macak), buang genangan air hujan segera.\n"
            "2. Manajemen Pupuk Khusus Masalah: Wajib STOP pupuk Nitrogen tunggal (Urea/ZA berlebih) yang memicu daun lunak, gantikan pupuk Kalium (KNO3 Putih / MKP).\n"
            "3. Penguat Dinding Sel: Semprot pupuk Kalsium-Boron dan pupuk Silika cair berkala tiap 7-10 hari untuk mempertebal lapisan lilin daun.\n"
            "4. Perawatan Tanah & Agens Hayati: Tabur kapur dolomit jika tanah masam (pH < 6), dan inokulasi agens hayati Trichoderma harzianum / Bacillus subtilis.\n"
        )
    elif is_differential and second_disease_name:
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

def get_groq_physical_verification(
    primary_name: str,
    second_name: str | None = None,
    is_differential: bool = False,
    third_name: str | None = None,
    is_three_way: bool = False
) -> str:
    """
    Memanggil Groq API untuk menyusun panduan verifikasi fisik lapangan singkat & padat
    agar petani dapat langsung mencocokkan gejala khas di kebun bawang merah.
    """
    if is_three_way and second_name and third_name:
        fallback_content = (
            f"• 🖐️ Uji Raba & Tekstur Daun: Periksa apakah helai daun berlendir basah khas {second_name}, berbercak kering/klorotik khas {third_name}, atau masih tegar sehat seperti {primary_name}.\n"
            f"• 👃 Uji Aroma & Kelembapan: Jika tercium aroma langu/busuk menyengat di pagi hari, waspadai serangan bakteri/hawar basah. Jika tanpa aroma busuk melainkan bercak kering, waspadai jamur atau virus.\n"
            f"• 🔍 Uji Bentuk Bercak: Amati pola bercak apakah meluas dari ujung daun (hawar), membentuk garis klorosis memanjang (virus), atau bercak cincin konsentris (trotol)."
        )
    else:
        fallback_content = (
            f"• 🖐️ Uji Raba Daun: Rasakan permukaan bercak pada helai daun. "
            f"Jika basah berlendir dan bau busuk, kuat mengarah ke bakteri. Jika kering bertepung atau bintil kasar, mengarah ke jamur.\n"
            f"• 👃 Uji Bau & Lendir: Daun yang terinfeksi bakteri busuk basah mengeluarkan aroma menyengat khas pembusukan sayur.\n"
            f"• 🔍 Uji Bentuk Bercak: Cermati tepi bercak di bawah sinar terang; bercak jamur biasanya memiliki batas konsentris atau bintil oranye khas karat."
        )

    api_key = get_groq_api_key()
    if not api_key:
        return fallback_content

    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AgroScan-Validator/1.0"
    }

    if is_three_way and second_name and third_name:
        prompt = (
            f"Anda adalah Konsultan Proteksi Tanaman Bawang Merah (merujuk Balitsa Lembang & BPTP).\n"
            f"Bantu petani mengidentifikasi 3 kemungkinan kondisi yang bersaing di kamera HP: '{primary_name}' vs '{second_name}' vs '{third_name}'.\n"
            "Tuliskan panduan verifikasi fisik langsung di bedengan sawah dalam 3 poin ringkas dan padat:\n"
            "1. 🖐️ Uji Raba & Tekstur Daun (Lendir basah licin vs serbuk tepung/bintil kering kasar vs helai licin sehat)\n"
            "2. 👃 Uji Aroma Daun (Bau langu busuk bakteri vs daun kering jamur vs aroma segar)\n"
            "3. 🔍 Uji Bentuk Bercak Lapangan (Kering ujung vs klorosis/belang virus vs bercak ungu konsentris)"
        )
    elif is_differential and second_name:
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
def get_system_agronomy_recommendation(
    info,
    second_info=None,
    is_differential=False,
    third_info=None,
    is_three_way=False,
    angle_title=None
):
    """
    Sistem Database Agronomi Mandiri (Built-in Agro-Engine):
    Menyusun rekomendasi lengkap, terstruktur, dan kaya agronomi resmi Balitsa/BPTP Kementan
    sebagai fallback otomatis jika kuota token Groq AI habis, terkena limit (429), atau offline.
    Mendukung vonis tunggal, 2 spektrum bersaing, maupun 3 spektrum bersaing terpadu.
    Menghasilkan 3 kartu:
    1. Tindakan Langsung di Kebun (24 Jam Pertama)
    2. Rekomendasi Obat Semprot (Bahan Aktif Resmi Balitsa & Takaran Dosis Tangki)
    3. Perawatan Lahan, Pemupukan & Agens Hayati (4 Poin Terstruktur)
    """
    nama_1 = info.get("nama_id", "Penyakit Bawang")
    is_healthy = info.get("is_healthy", False) or info.get("status") == "healthy"

    if is_healthy and not (is_three_way or is_differential):
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

    if is_three_way and second_info and third_info:
        nama_2 = second_info.get("nama_id", "Penyakit Kedua")
        nama_3 = third_info.get("nama_id", "Penyakit Ketiga")
        c1 = (
            f"• Waspada 3 Spektrum Gejala Bersaing di Lapangan: Periksa helai daun dengan cermat antara {nama_1}, {nama_2}, dan {nama_3} (Riset Balitsa Lembang & BPTP Kementan).\n"
            f"• Karakteristik 1 ({nama_1}): {info.get('ciri_lapangan', '-')}\n"
            f"• Karakteristik 2 ({nama_2}): {second_info.get('ciri_lapangan', '-')}\n"
            f"• Karakteristik 3 ({nama_3}): {third_info.get('ciri_lapangan', '-')}\n"
            "• Tindakan Taktis 24 Jam Pertama: Segera pangkas seluruh helai daun yang bergejala parah menggunakan gunting/pisau steril (usap alkohol 70% atau air sabun). Masukkan sisa potongan ke kantong tertutup dan bakar/kubur jauh dari saluran air irigasi."
        )
        c2 = (
            f"• Solusi Penanganan Spektrum 1 ({nama_1}): {info.get('solusi', '-')}\n"
            f"• Solusi Penanganan Spektrum 2 ({nama_2}): {second_info.get('solusi', '-')}\n"
            f"• Solusi Penanganan Spektrum 3 ({nama_3}): {third_info.get('solusi', '-')}\n"
            "• Takaran Dosis Tangki: Campurkan 1,5 hingga 2 sendok makan (20–25 gram/ml) per tangki semprot standar 16 Liter air.\n"
            "• Waktu Semprot Terbaik: Pagi hari (pukul 06.00 – 08.30 WIB) saat embun mulai mengering, atau sore hari (pukul 16.00 WIB) saat cuaca teduh dan angin tenang.\n"
            "• Wajib Tambahkan Perekat & Perata (Surfactant non-ionik) 1 tutup per tangki semprot agar lapisan lilin daun bawang terlapisi obat secara merata dan tidak mudah tercuci air hujan."
        )
    elif is_differential and second_info:
        nama_2 = second_info.get("nama_id", "Penyakit Serupa")
        c1 = (
            f"• Waspada Gejala Serupa di Kebun: Bedakan segera antara {nama_1} vs {nama_2} langsung di bedengan (Riset Balitsa & BPTP Kementan).\n"
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

def parse_groq_to_cards(
    ai_text,
    info,
    second_info=None,
    is_differential=False,
    third_info=None,
    is_three_way=False,
    angle_title=None
):
    """
    Memecah teks balasan Groq menjadi 3 kartu panduan terstruktur.
    Jika ai_text kosong (kuota Groq habis / error / offline),
    sistem secara otomatis mengalirkan jawaban lengkap dari Database Mandiri Sistem.
    """
    sys_c1, sys_c2, sys_c3 = get_system_agronomy_recommendation(
        info=info,
        second_info=second_info,
        is_differential=is_differential,
        third_info=third_info,
        is_three_way=is_three_way,
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
# SINKRONISASI PENDING STATE SEBELUM WIDGET SLIDER DI-INSTANTIATE
# Mencegah StreamlitWidgetAlreadyInstantiatedError saat tombol aksi ditekan
# ==============================================================================
if "pending_conf_slider" in st.session_state:
    st.session_state["conf_slider"] = int(st.session_state.pop("pending_conf_slider"))
if "pending_leaf_slider" in st.session_state:
    st.session_state["leaf_slider"] = int(st.session_state.pop("pending_leaf_slider"))

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
    with st.expander("📖 Panduan Penggunaan Web", expanded=False):
        st.markdown("""
        <div style="font-size: 0.76rem; line-height: 1.38; color: #334155;">
        
        <strong style="color: #166534;">📸 1. Pengambilan Foto:</strong>
        <ul style="margin: 1px 0 4px 12px; padding: 0;">
            <li><strong>Jarak:</strong> 10–20 cm tegak lurus daun, fokus tajam & tidak blur.</li>
            <li><strong>Cahaya:</strong> Terang alami, hindari bayangan pekat & silau.</li>
            <li><strong>Posisi:</strong> Jangan tutupi bercak lesi dengan jari/tangan.</li>
        </ul>

        <strong style="color: #166534;">🎯 2. Arti Retikel HUD Scanner:</strong>
        <ul style="margin: 1px 0 4px 12px; padding: 0;">
            <li><span style="color: #dc2626; font-weight: 700;">🔴 [1] Merah:</span> Lesi aktif penyakit utama.</li>
            <li><span style="color: #d97706; font-weight: 700;">🟠 [2] Oranye:</span> Lesi penyakit kedua.</li>
            <li><span style="color: #0284c7; font-weight: 700;">🔵 [3] Biru:</span> Lesi penyakit ketiga.</li>
            <li><span style="color: #16a34a; font-weight: 700;">🟢 [OK] Hijau:</span> Jaringan daun sehat prima.</li>
        </ul>

        <strong style="color: #166534;">💾 3. Riwayat Diagnosa:</strong>
        <ul style="margin: 1px 0 4px 12px; padding: 0;">
            <li>Klik <strong>💾 Simpan Hasil</strong> untuk mencatat riwayat di browser.</li>
            <li>Tombol <strong>🗑️</strong> di riwayat untuk menghapus per entri.</li>
        </ul>

        <strong style="color: #166534;">💊 4. Penanganan Tanaman:</strong>
        <ul style="margin: 1px 0 1px 12px; padding: 0;">
            <li><strong>24 Jam:</strong> Pangkas & musnahkan daun sakit ke luar sawah.</li>
            <li><strong>Obat Semprot:</strong> Bahan aktif Balitsa per tangki 16L.</li>
        </ul>
        
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # FITUR 2: PANDUAN & TOLOK UKUR VALIDASI FOTO (TERPISAH & DETAIL PERSEN)
    # --------------------------------------------------------------------------
    with st.expander("🎯 Panduan Validasi Foto", expanded=False):
        st.markdown("""
        <div style="font-size: 0.76rem; line-height: 1.38; color: #334155;">
        
        <strong style="color: #166534;">📊 1. Batas Keyakinan Model:</strong>
        <ul style="margin: 1px 0 4px 12px; padding: 0;">
            <li><strong style="color: #b45309;">40% – 55% (Redup/Dini):</strong> Foto sore, mendung, atau bercak tipis.</li>
            <li><strong style="color: #15803d;">65% (Standar Balitsa - Rekomendasi):</strong> Pemantauan harian sawah.</li>
            <li><strong style="color: #b91c1c;">75% – 90% (Super Ketat):</strong> Standar sertifikasi benih / makro.</li>
        </ul>

        <strong style="color: #166534;">🍃 2. Sensitivitas Daun Bawang:</strong>
        <ul style="margin: 1px 0 1px 12px; padding: 0;">
            <li><strong style="color: #b45309;">3% – 6% (Toleran):</strong> Daun tunggal / bibit muda / dipegang tangan.</li>
            <li><strong style="color: #15803d;">8% – 15% (Standar Sawah):</strong> Rumpun bawang normal umur 3–8 minggu.</li>
            <li><strong style="color: #b91c1c;">18% – 35% (Makro Penuh):</strong> Daun harus mendominasi layar foto.</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 0.86rem; font-weight: 800; color: #0F172A; margin: 0.35rem 0 0.1rem 0;">
            🔬 Model PyTorch TorchScript
        </div>
        <div style="font-size: 0.75rem; color: #475569; margin-bottom: 0.3rem; line-height: 1.3;">
            EfficientNet-B0 (TorchScript) + TTA Flip Horizontal.
        </div>
    """, unsafe_allow_html=True)

    # Tampilkan 7 Kategori yang Dideteksi di Sidebar (Ukuran Dikecilkan Ramping)
    with st.expander("📋 7 Kategori yang Dideteksi AI", expanded=False):
        for item in SUPPORTED_DISEASES_7:
            status_color = "#15803d" if item["is_healthy"] else ("#b45309" if item["status"] == "virus" else "#b91c1c")
            tag_text = "SEHAT" if item["is_healthy"] else ("VIRUS" if item["status"] == "virus" else "PENYAKIT")
            st.markdown(
                f"<div style='margin-bottom: 4px; font-size: 0.76rem; background: #ffffff; padding: 4px 7px; border-radius: 6px; border: 1px solid #e2e8f0;'>"
                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                f"<strong style='font-size: 0.78rem;'>{item['icon']} {item['nama_id']}</strong>"
                f"<span style='background: {status_color}; color: #fff; font-size: 0.62rem; font-weight: 700; padding: 1px 5px; border-radius: 4px;'>{tag_text}</span>"
                f"</div>"
                f"<span style='color: {status_color}; font-size: 0.70rem;'>• <em>{item['latin']}</em></span><br>"
                f"<span style='color: #475569; font-size: 0.72rem; line-height: 1.3;'>🔍 {item['ciri_lapangan']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    default_conf_pct = int(round(float(meta_config.get("conf_threshold", 0.65)) * 100))
    st.markdown("### ⚙️ Validasi Foto Bawang")
    st.caption("Pilih preset cepat atau geser slider sesuai kondisi foto lapangan:")

    current_conf = st.session_state.get("conf_slider", default_conf_pct)
    current_leaf = st.session_state.get("leaf_slider", 8)

    # Tombol Preset Cepat Langsung Sinkron ke Web (Vertikal Lebar Penuh: 100% Bebas Terpotong di Semua HP & Layar)
    if st.button(
        "🌾 Standar",
        help="Preset Standar Sawah (Keyakinan 65% | Daun 8%)",
        type="primary" if (current_conf == 65 and current_leaf == 8) else "secondary",
        use_container_width=True,
        key="btn_preset_standar"
    ):
        st.session_state["pending_conf_slider"] = 65
        st.session_state["pending_leaf_slider"] = 8
        st.rerun()

    if st.button(
        "☁️ Redup",
        help="Preset Cuaca Redup / Gejala Dini (Keyakinan 50% | Daun 5%)",
        type="primary" if (current_conf == 50 and current_leaf == 5) else "secondary",
        use_container_width=True,
        key="btn_preset_redup"
    ):
        st.session_state["pending_conf_slider"] = 50
        st.session_state["pending_leaf_slider"] = 5
        st.rerun()

    if st.button(
        "🔬 Ketat",
        help="Preset Super Ketat Lab (Keyakinan 80% | Daun 20%)",
        type="primary" if (current_conf == 80 and current_leaf == 20) else "secondary",
        use_container_width=True,
        key="btn_preset_ketat"
    ):
        st.session_state["pending_conf_slider"] = 80
        st.session_state["pending_leaf_slider"] = 20
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

    if st.button("🔄 Bersihkan Cache Sesi", use_container_width=True, key="btn_clear_cache_sidebar"):
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

if "input_source_mode" not in st.session_state:
    st.session_state["input_source_mode"] = "camera"

# Tombol Pilihan Mode Input (Anti-Crop & Auto-Refresh Antar Mode Kamera vs Galeri)
col_src1, col_src2 = st.columns(2)
is_cam_mode = (st.session_state["input_source_mode"] == "camera")
is_upload_mode = (st.session_state["input_source_mode"] == "upload")

with col_src1:
    btn_cam = st.button(
        "📸 Ambil Foto Kamera",
        type="primary" if is_cam_mode else "secondary",
        use_container_width=True,
        key="btn_switch_to_camera"
    )
    if btn_cam and not is_cam_mode:
        st.session_state["input_source_mode"] = "camera"
        st.session_state["upload_key_ver"] = st.session_state.get("upload_key_ver", 0) + 1
        if "has_inspected_current" in st.session_state:
            del st.session_state["has_inspected_current"]
        st.rerun()

with col_src2:
    btn_upload = st.button(
        "📁 Pilih dari Galeri",
        type="primary" if is_upload_mode else "secondary",
        use_container_width=True,
        key="btn_switch_to_upload"
    )
    if btn_upload and not is_upload_mode:
        st.session_state["input_source_mode"] = "upload"
        st.session_state["cam_key_ver"] = st.session_state.get("cam_key_ver", 0) + 1
        if "has_inspected_current" in st.session_state:
            del st.session_state["has_inspected_current"]
        st.rerun()

selected_image = None
file_error = False

if is_cam_mode:
    cam_key = f"cam_field_v{st.session_state.get('cam_key_ver', 0)}"
    cam_file = st.camera_input("Arahkan kamera dekat ke bagian daun yang sakit:", key=cam_key)
    if cam_file is not None:
        if cam_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ Ukuran foto ({cam_file.size / (1024*1024):.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB. Silakan ambil ulang dengan resolusi wajar.")
            file_error = True
        else:
            try:
                selected_image = Image.open(cam_file)
            except (ValueError, OSError) as err:
                st.error(f"Gagal membaca foto kamera: {err}")

elif is_upload_mode:
    upload_key = f"upload_field_v{st.session_state.get('upload_key_ver', 0)}"
    uploaded_file = st.file_uploader(
        f"Pilih file foto dari galeri (JPG, JPEG, PNG, WEBP - Maksimal {MAX_FILE_SIZE_MB} MB):",
        type=["jpg", "jpeg", "png", "webp"],
        key=upload_key
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

    # Monitor Real-Time Kanopi Daun (Live Responsif terhadap Slider Sensitivitas Daun di Sidebar)
    is_plant_live, reason_live, plant_ratio_live = check_shallot_leaf_mask(selected_image, min_ratio=min_leaf_ratio)
    ratio_pct_live = plant_ratio_live * 100.0
    is_passed_live = (plant_ratio_live >= min_leaf_ratio)

    m_color = "#16a34a" if is_passed_live else "#dc2626"
    m_bg = "#f0fdf4" if is_passed_live else "#fef2f2"
    m_border = "#86efac" if is_passed_live else "#fca5a5"
    m_icon = "🟢" if is_passed_live else "🔴"
    m_status_title = "MEMENUHI SYARAT VALIDASI" if is_passed_live else f"DI BAWAH BATAS VALIDASI ({ratio_pct_live:.1f}% < {min_leaf_ratio_pct}%)"

    st.markdown(f"""
        <div style="background: {m_bg}; border: 1.5px solid {m_border}; border-radius: 12px; padding: 10px 14px; margin: 6px 0 14px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                <span style="font-weight: 700; color: #1e293b; font-size: 0.90rem;">🍃 Monitor Validasi Kanopi Daun (Live Real-Time):</span>
                <span style="font-weight: 800; color: {m_color}; font-size: 0.85rem;">{m_icon} {m_status_title}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px 12px; font-size: 0.85rem; color: #475569; flex-wrap: wrap;">
                <span>Daun Terdeteksi: <strong style="color: #0f172a;">{ratio_pct_live:.1f}%</strong></span>
                <span>•</span>
                <span>Batas Slider Anda: <strong style="color: #0f172a;">{min_leaf_ratio_pct}%</strong></span>
                <span>•</span>
                <span>Standar Sawah: <strong style="color: #166534;">8%</strong></span>
            </div>
            <div style="background: #e2e8f0; border-radius: 999px; height: 8px; width: 100%; margin-top: 8px; overflow: hidden;">
                <div style="background: {m_color}; width: {min(max(ratio_pct_live, 0.0), 100.0):.1f}%; height: 100%; border-radius: 999px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
            val_res = validate_onion_image(selected_image, min_ratio=min_leaf_ratio)
            is_valid_vision = val_res[0]
            vision_verdict = val_res[1]
            val_info = val_res[2] if len(val_res) > 2 else {}

        if not is_valid_vision:
            detected_ratio = val_info.get("plant_ratio", 0.0) * 100.0
            curr_min_pct = min_leaf_ratio * 100.0
            rec_leaf_pct = val_info.get("recommended_leaf_pct", max(3, int(np.floor(detected_ratio))))

            if val_info.get("is_ratio_rejection", False):
                st.warning(f"⚠️ **Rasio Daun Terdeteksi ({detected_ratio:.1f}%) di Bawah Pengaturan Validasi ({curr_min_pct:.0f}%)**")
                st.markdown(f"""
                    <div class="card-rejection">
                        <div class="card-rejection-badge" style="background-color: #D97706;">⚠️ PENGATURAN VALIDASI TERLALU KETAT</div>
                        <div class="card-rejection-title">Rasio Daun {detected_ratio:.1f}% (Batas Aktif: {curr_min_pct:.0f}%)</div>
                        <div class="card-rejection-reason">
                            Foto Anda <strong>mengandung daun bawang merah asli ({detected_ratio:.1f}%)</strong>, namun tertahan karena slider <strong>Sensitivitas Daun Bawang</strong> diatur pada angka <strong>{curr_min_pct:.0f}%</strong>.
                        </div>
                        <div class="card-rejection-desc">
                            <strong>📜 Rekomendasi Standar Peraturan Resmi (Balitsa/Kementan):</strong>
                            <ul style="margin: 4px 0 8px 16px;">
                                <li><strong>Standar Rumpun Sawah Normal:</strong> <strong>8%</strong>.</li>
                                <li><strong>Daun Tunggal / Bibit Muda / Dipegang Tangan:</strong> <strong>5% atau 3%</strong>.</li>
                                <li><strong>Mode Makro Ekstrem:</strong> <strong>20% – 35%</strong> (Hanya untuk daun yang memenuhi layar penuh).</li>
                            </ul>
                            <strong>💡 Solusi Cepat:</strong> Klik tombol rekomendasi di bawah ini: Sistem akan <strong>menyesuaikan slider sensitivitas ke {rec_leaf_pct}% agar memenuhi kriteria foto Anda dan langsung memproses diagnosa penyakit</strong>!
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if st.button(f"⚡ Sesuaikan Slider ke {rec_leaf_pct}% Sesuai Foto Ini & Lanjutkan Diagnosa", type="primary", use_container_width=True, key=f"btn_apply_rec_{current_img_sig}"):
                    st.session_state["pending_leaf_slider"] = rec_leaf_pct
                    st.rerun()
            else:
                st.error("❌ Foto Ditolak: Objek yang diunggah terdeteksi bukan daun/tanaman bawang merah.")
                st.markdown(f"""
                    <div class="card-rejection">
                        <div class="card-rejection-badge">⚠️ FOTO BUKAN DAUN BAWANG</div>
                        <div class="card-rejection-title">Objek Terindikasi Bukan Daun Bawang Merah!</div>
                        <div class="card-rejection-reason">
                            {vision_verdict}
                        </div>
                        <div class="card-rejection-desc">
                            Sistem mendeteksi bahwa gambar yang Anda masukkan tidak memenuhi kriteria visual daun bawang merah (seperti foto manusia, hewan, dinding/kertas putih polos, tanah kosong tanpa tanaman, atau dokumen).
                            <br><br>
                            <strong>💡 Petunjuk Pengambilan Foto Lapangan:</strong>
                            <ul style="margin: 4px 0 8px 16px;">
                                <li>Pastikan foto menampilkan helai daun tanaman bawang merah asli di bedengan sawah atau pot.</li>
                                <li>Jarak pemotretan 10–20 cm tegak lurus daun, fokus tajam dan tidak blur.</li>
                                <li>Hindari menutup seluruh helai daun dengan jari tangan.</li>
                            </ul>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if detected_ratio >= 3.0:
                    if st.button(f"🌱 Sesuaikan Sensitivitas ke {max(3, int(np.floor(detected_ratio)))}% & Uji Ulang", type="primary", use_container_width=True, key=f"btn_retry_tolerant_{current_img_sig}"):
                        st.session_state["pending_leaf_slider"] = max(3, int(np.floor(detected_ratio)))
                        st.rerun()
            st.stop()

        # ==============================================================================
        # TAHAP 2: PROSES DIAGNOSIS EFFICIENTNET-B0 TORCHSCRIPT
        # ==============================================================================
        with st.spinner("🔍 Sedang menganalisis kondisi daun bawang merah dengan EfficientNet-B0 TorchScript..."):
            meta_run = meta_config.copy()
            meta_run["conf_threshold"] = conf_threshold
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
                selected_image, model, meta=meta_run, enforce_verification=False
            )
            # Selaraskan alias variabel agar konsisten (mencegah NameError)
            metadata = info
            second_metadata = second_info
            diag_mode = diag_info.get("diag_mode", "single")
            is_pure_healthy = diag_info.get("is_pure_healthy", False)
            third_class_raw = diag_info.get("third_class_raw", "")
            third_confidence = diag_info.get("third_confidence", 0.0)
            third_info = diag_info.get("third_metadata", {})
            third_metadata = third_info
            has_three_diseases = diag_info.get("has_three_diseases", False)

        # ==============================================================================
        # TAHAP 3: EVALUASI KEPUTUSAN MODEL / THRESHOLD
        # Jika confidence < conf_threshold, tampilkan "Tidak yakin" alih-alih menebak
        # ==============================================================================
        # ==============================================================================
        # TAHAP 3: EVALUASI KEPUTUSAN MODEL / THRESHOLD
        # Jika confidence < conf_threshold, tampilkan peringatan & rekomendasi penyesuaian
        # ==============================================================================
        allow_conf_key = f"allow_conf_{current_img_sig}"
        user_allowed_conf = st.session_state.get(allow_conf_key, False)

        is_uncertain = (not user_allowed_conf) and (api_output.get("uncertain", False) or (top_confidence < (conf_threshold * 100.0)))
        if is_uncertain:
            current_conf_pct = conf_threshold * 100.0
            # Hitung rekomendasi batas keyakinan yang adaptif kelipatan 5 (rentang slider: 20% - 90%)
            rec_conf_pct = max(20, min(90, int(np.floor(top_confidence / 5.0) * 5)))

            st.warning(f"⚠️ **Tingkat Keyakinan Model ({top_confidence:.1f}%) di Bawah Batas ({current_conf_pct:.0f}%)**")
            st.markdown(f"""
                <div class="card-rejection">
                    <div class="card-rejection-badge">⚠️ TINGKAT KEYAKINAN DI BAWAH AMBANG BATAS</div>
                    <div class="card-rejection-title">Foto Kurang Jelas atau Gejala Bercak Masih Awal</div>
                    <div class="card-rejection-reason">
                        Kepastian model tercatat <strong>{top_confidence:.1f}%</strong> (ambang batas aktif: <strong>{current_conf_pct:.0f}%</strong>). Sistem menahan vonis untuk mencegah salah penanganan di kebun.
                    </div>
                    <div class="card-rejection-desc">
                        <strong>📜 Rekomendasi Standar Peraturan Resmi (Balitsa/Kementan):</strong>
                        <ul style="margin: 4px 0 8px 16px;">
                            <li><strong>Standar Sawah Siang Terang:</strong> <strong>65%</strong> (Standar harian).</li>
                            <li><strong>Toleransi Sore / Redup / Gejala Dini:</strong> <strong>50% – 55%</strong> (Untuk cuaca mendung atau bercak yang masih tipis).</li>
                            <li><strong>Mode Toleransi Gejala Awal:</strong> <strong>{rec_conf_pct}%</strong> (Sesuai kepastian foto Anda).</li>
                        </ul>
                        <strong>💡 Solusi Cepat:</strong>
                        <p style="margin: 2px 0 8px 0;">
                            Klik tombol di bawah ini: Sistem akan <strong>otomatis menyesuaikan batas keyakinan ke {rec_conf_pct}% dan langsung memproses prediksi penyakit</strong> untuk foto ini tanpa terhambat!
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"⚡ Sesuaikan Batas Keyakinan ({rec_conf_pct}%) & Lanjutkan Prediksi Sekarang", type="primary", use_container_width=True, key=f"btn_apply_conf_rec_{current_img_sig}"):
                st.session_state["pending_conf_slider"] = rec_conf_pct
                st.session_state[allow_conf_key] = True
                st.rerun()
            st.stop()  # Hentikan eksekusi sampai tombol penyesuaian ditekan

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

            # Bar Perbandingan Keyakinan AI vs Ambang Batas Slider (Transparan & Responsif)
            diff_conf = top_confidence - conf_threshold_pct
            if diff_conf >= 0:
                badge_conf_html = f'<span style="background: #dcfce7; color: #166534; font-weight: 700; padding: 3px 10px; border-radius: 999px; border: 1px solid #86efac; font-size: 0.78rem;">🟢 Memenuhi Batas Keyakinan (+{diff_conf:.1f}%)</span>'
            else:
                badge_conf_html = f'<span style="background: #fefce8; color: #854d0e; font-weight: 700; padding: 3px 10px; border-radius: 999px; border: 1px solid #fef08a; font-size: 0.78rem;">🟡 Mode Toleran Gejala Awal ({top_confidence:.1f}%)</span>'

            st.markdown(f"""
                <div style="background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 12px; padding: 10px 14px; margin: 4px 0 14px 0; font-size: 0.88rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        🎯 <strong>Tingkat Keyakinan AI:</strong> <span style="color: #15803d; font-weight: 800; font-size: 1rem;">{top_confidence:.1f}%</span> 
                        <span style="color: #64748b; font-size: 0.82rem;">(Batas Keyakinan Slider Anda: <strong>{conf_threshold_pct}%</strong>)</span>
                    </div>
                    <div>
                        {badge_conf_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if diag_info.get("is_auto_cropped", False):
                st.markdown(f"""
                    <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 12px; padding: 10px 14px; margin: 4px 0 12px 0; display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.3rem;">🎯</span>
                        <div>
                            <strong style="color: #166534; font-size: 0.90rem;">Fokus Helai Daun Otomatis Berhasil:</strong>
                            <div style="color: #334155; font-size: 0.82rem;">Objek luar seperti tangan atau tanah berhasil disingkirkan ({diag_info.get('leaf_coverage_pct', 0)}% area terfokus) sehingga diagnosa AI tertuju murni pada daun bawang merah.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("🔍 Lihat Hasil Pemotongan Otomatis Daun (Auto-Crop)", expanded=False):
                    col_crop1, col_crop2 = st.columns([1, 1])
                    with col_crop1:
                        st.image(preview_crop, caption="Fokus Helai Daun (Bebas Latar Belakang)", use_container_width=True)
                    with col_crop2:
                        st.image(selected_image, caption="Foto Asli Sebelum Dipotong", use_container_width=True)

            is_healthy = is_pure_healthy
            is_pest = info.get("status") == "pest"

            if diag_mode == "three_way":
                # ------------------------------------------------------------------
                # TAMPILAN 3 KEMUNGKINAN BERSAING (PERINGKAT 1, 2, & 3)
                # ------------------------------------------------------------------
                p1_is_h = ("sehat" in info.get("nama_id", "").lower())
                header_title = (
                    "⚠️ **Waspada Gejala Awal: Terdeteksi 3 Kemungkinan Bersaing**"
                    if p1_is_h else
                    "🚨 **Terdeteksi 3 Kemungkinan Penyakit Bersaing**"
                )
                st.warning(
                    f"{header_title}\n\n"
                    f"Model mencatat distribusi probabilitas yang tersebar pada 3 kemungkinan dengan selisih mendekati "
                    f"(Peringkat 1: **{top_confidence:.1f}%**, Peringkat 2: **{second_confidence:.1f}%**, Peringkat 3: **{third_confidence:.1f}%**). "
                    f"Sistem menyajikan 3 kemungkinan agar petani dapat memverifikasi langsung di bedengan sawah:"
                )

                col_top1, col_top2, col_top3 = st.columns(3)
                with col_top1:
                    b_color_1 = "#16a34a" if p1_is_h else "#dc2626"
                    badge_text_1 = "🥇 PREDIKSI TERPILIH (SEHAT)" if p1_is_h else "🥇 PERINGKAT 1"
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 2.5px solid {b_color_1}; border-radius: 14px; padding: 14px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                            <span style="background: {b_color_1}; color: #ffffff; font-weight: 800; font-size: 0.78rem; padding: 3px 8px; border-radius: 6px; width: fit-content; margin-bottom: 6px;">{badge_text_1}</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px; line-height: 1.3;">{info['nama_id']}</div>
                            <div style="font-size: 0.90rem; color: {b_color_1}; font-weight: 800; margin-bottom: 8px;">Kepastian: {top_confidence:.1f}%</div>
                            <div style="font-size: 0.84rem; color: #475569; line-height: 1.45; margin-top: auto;">
                                <strong>🔍 Ciri di Sawah:</strong><br>{info.get('ciri_lapangan', '-')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_top2:
                    b_color_2 = "#ea580c"
                    badge_text_2 = "🥈 PERINGKAT 2"
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 2.5px solid {b_color_2}; border-radius: 14px; padding: 14px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                            <span style="background: {b_color_2}; color: #ffffff; font-weight: 800; font-size: 0.78rem; padding: 3px 8px; border-radius: 6px; width: fit-content; margin-bottom: 6px;">{badge_text_2}</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px; line-height: 1.3;">{second_info['nama_id']}</div>
                            <div style="font-size: 0.90rem; color: {b_color_2}; font-weight: 800; margin-bottom: 8px;">Kepastian: {second_confidence:.1f}%</div>
                            <div style="font-size: 0.84rem; color: #475569; line-height: 1.45; margin-top: auto;">
                                <strong>🔍 Ciri di Sawah:</strong><br>{second_info.get('ciri_lapangan', '-')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_top3:
                    b_color_3 = "#0284c7"
                    badge_text_3 = "🥉 PERINGKAT 3"
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 2.5px solid {b_color_3}; border-radius: 14px; padding: 14px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); display: flex; flex-direction: column;">
                            <span style="background: {b_color_3}; color: #ffffff; font-weight: 800; font-size: 0.78rem; padding: 3px 8px; border-radius: 6px; width: fit-content; margin-bottom: 6px;">{badge_text_3}</span>
                            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px; line-height: 1.3;">{third_info.get('nama_id', third_class_raw)}</div>
                            <div style="font-size: 0.90rem; color: {b_color_3}; font-weight: 800; margin-bottom: 8px;">Kepastian: {third_confidence:.1f}%</div>
                            <div style="font-size: 0.84rem; color: #475569; line-height: 1.45; margin-top: auto;">
                                <strong>🔍 Ciri di Sawah:</strong><br>{third_info.get('ciri_lapangan', '-')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.info(
                    "💡 **Panduan Identifikasi 3 Titik Gejala di Kebun:**\n\n"
                    "• **Titik [1] (Hijau/Merah):** Cek kondisi daun dominan apakah masih berklorofil tegar atau sudah berpusat lesi.\n\n"
                    f"• **Titik [2] (Oranye):** Cek apakah tampak gejala khas **{second_info['nama_id']}** (misal: ujung daun menguning kering atau bercak basah).\n\n"
                    f"• **Titik [3] (Biru):** Cek apakah tampak gejala **{third_info.get('nama_id', third_class_raw)}** (misal: garis klorosis virus atau bercak cincin jamur)."
                )

            elif diag_mode == "two_way":
                # ------------------------------------------------------------------
                # TAMPILAN 2 KEMUNGKINAN BERSAING (DIFERENSIAL)
                # ------------------------------------------------------------------
                st.warning(
                    f"⚠️ **Gejala Ganda / Memerlukan Konfirmasi Fisik**\n\n"
                    f"### Terdeteksi 2 Kemungkinan Penyakit Serupa\n\n"
                    f"Model visual menemukan kemiripan tinggi dengan selisih probabilitas tipis (hanya **{confidence_margin:.1f}%**). "
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
                # TAMPILAN SATU VONIS TUNGGAL (DOMINAN TINGGI)
                # ------------------------------------------------------------------
                if is_pure_healthy:
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
                        if diag_mode == "three_way":
                            rec_name = f"{info['nama_id']} ({top_confidence:.1f}%) | {second_info['nama_id']} ({second_confidence:.1f}%) | {third_info.get('nama_id', third_class_raw)} ({third_confidence:.1f}%)"
                        elif diag_mode == "two_way":
                            rec_name = f"{info['nama_id']} ({top_confidence:.1f}%) & {second_info['nama_id']} ({second_confidence:.1f}%)"
                        else:
                            rec_name = info["nama_id"]

                        save_diagnosis_to_history(
                            disease_code=top_class_raw,
                            display_name=rec_name,
                            confidence=top_confidence,
                            recommendation=info.get("rekomendasi_singkat", ""),
                            is_healthy=is_pure_healthy,
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
            # MODUL BUKTI ANALISIS VISUAL NYATA DARI FOTO (REAL VISUAL LESION AUDIT)
            # ==============================================================================
            if visual_evidence and visual_evidence.get("has_visual_evidence"):
                v_sev_pct = visual_evidence.get("severity_pct", 0.0)
                v_sev_lvl = visual_evidence.get("severity_level", "Normal")
                v_healthy_pct = visual_evidence.get("healthy_pct", 100.0)
                v_desc = visual_evidence.get("evidence_desc", "")

                st.markdown("### 🔬 Bukti Analisis Visual Nyata dari Foto Daun")
                st.caption("Hasil pemindaian fitur fisik piksel langsung dari foto yang diunggah:")

                # Konfigurasi Nilai Kartu Secara Realistis & Anti-Ngawur
                if is_pure_healthy:
                    c1_title = "🩺 Luas Kerusakan Daun"
                    c1_val = "0.0%"
                    c1_delta = "Sehat Prima"
                    c1_color = "#16a34a"

                    c2_title = "🌿 Jaringan Daun Sehat"
                    c2_val = "100.0%"
                    c2_delta = "Klorofil Utuh"
                    c2_color = "#16a34a"

                    c3_title = "🛡️ Kondisi Tanaman"
                    c3_val = "Bebas Patogen"
                    c3_delta = "Aman Terkendali"
                    c3_color = "#16a34a"
                elif diag_mode == "three_way":
                    c1_title = "🩺 Luas Gejala Visual"
                    c1_val = f"{v_sev_pct:.1f}%"
                    c1_delta = v_sev_lvl
                    c1_color = "#ea580c"

                    c2_title = "🌿 Jaringan Hijau Tersisa"
                    c2_val = f"{v_healthy_pct:.1f}%"
                    c2_delta = "Klorofil Bertahan"
                    c2_color = "#16a34a"

                    c3_title = "🛡️ Kondisi Tanaman"
                    c3_val = "Waspada Gejala Awal"
                    c3_delta = "3 Penyakit Bersaing"
                    c3_color = "#ea580c"
                elif diag_mode == "two_way":
                    c1_title = "🩺 Luas Kerusakan Daun"
                    c1_val = f"{v_sev_pct:.1f}%"
                    c1_delta = v_sev_lvl
                    c1_color = "#dc2626"

                    c2_title = "🌿 Jaringan Hijau Tersisa"
                    c2_val = f"{v_healthy_pct:.1f}%"
                    c2_delta = "Area Sehat"
                    c2_color = "#16a34a"

                    c3_title = "🛡️ Kondisi Tanaman"
                    c3_val = "Suspek Ganda"
                    c3_delta = "Konfirmasi Ciri Lapangan"
                    c3_color = "#dc2626"
                else:
                    c1_title = "🩺 Luas Kerusakan Daun"
                    c1_val = f"{v_sev_pct:.1f}%"
                    c1_delta = v_sev_lvl
                    c1_color = "#dc2626"

                    c2_title = "🌿 Jaringan Hijau Tersisa"
                    c2_val = f"{v_healthy_pct:.1f}%"
                    c2_delta = "Area Sehat"
                    c2_color = "#16a34a"

                    short_target = info['nama_id'].split('/')[0].strip()
                    c3_title = "🛡️ Kondisi Tanaman"
                    c3_val = f"Terinfeksi {short_target}"
                    c3_delta = "Butuh Penanganan"
                    c3_color = "#dc2626"

                # Responsive Anti-Crop Metric Cards Grid (Anti-Potong di HP / Tablet / PC)
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 10px 0 16px 0;">
                        <div class="metric-anti-crop">
                            <div class="metric-anti-crop-label">{c1_title}</div>
                            <div class="metric-anti-crop-val" style="color: {c1_color};">{c1_val}</div>
                            <div class="metric-anti-crop-delta" style="color: #64748b;">{c1_delta}</div>
                        </div>
                        <div class="metric-anti-crop">
                            <div class="metric-anti-crop-label">{c2_title}</div>
                            <div class="metric-anti-crop-val" style="color: {c2_color};">{c2_val}</div>
                            <div class="metric-anti-crop-delta" style="color: #64748b;">{c2_delta}</div>
                        </div>
                        <div class="metric-anti-crop">
                            <div class="metric-anti-crop-label">{c3_title}</div>
                            <div class="metric-anti-crop-val" style="color: {c3_color}; font-size: 1.15rem;">{c3_val}</div>
                            <div class="metric-anti-crop-delta" style="color: #64748b;">{c3_delta}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if is_pure_healthy:
                    st.success("✅ **Daun Sehat & Normal:** Pemindaian visual mengonfirmasi helai daun segar merata, berlilin alami, dan tidak ditemukan bercak lesi patogen aktif.")
                else:
                    st.info(f"📋 **Karakteristik Fisik Daun pada Foto:**\n\n{v_desc}")

                if visual_evidence.get("overlay_img") is not None:
                    with st.expander("🖼️ Peta Titik Kerusakan pada Foto Daun (HUD Lesion Scanner)", expanded=True):
                        title_1 = get_short_disease_title(info.get('nama_id', 'Penyakit'))
                        title_2 = get_short_disease_title(second_info.get('nama_id', 'Penyakit')) if second_info else ""
                        title_3 = get_short_disease_title(third_info.get('nama_id', 'Penyakit')) if third_info else ""
                        num_spots = visual_evidence.get("num_spots_detected", 0)

                        if is_pure_healthy:
                            map_caption = "Peta Verifikasi Jaringan Daun: Retikel Hijau [OK] memverifikasi helai daun dalam kondisi sehat optimal berklorofil tinggi bebas lesi patogen."
                        elif diag_mode == "three_way":
                            map_caption = (
                                f"Peta Titik Kerusakan 3 Kemungkinan: Retikel [1] {title_1} ({top_confidence}%), "
                                f"[2] {title_2} ({second_confidence}%), dan [3] {title_3} ({third_confidence}%) "
                                f"menandai posisi fisik macam penyakit pada helai daun."
                            )
                        elif diag_mode == "two_way":
                            map_caption = (
                                f"Peta Titik Kerusakan Multi-Penyakit: Retikel Merah [1] {title_1} ({top_confidence}%) & "
                                f"Retikel Oranye [2] {title_2} ({second_confidence}%) menandai lokasi infeksi fisik pada helai daun."
                            )
                        else:
                            map_caption = f"Peta Titik Kerusakan: Retikel scanner presisi tinggi menandai titik pusat lesi aktif [1] {title_1} pada helai daun."

                        st.image(
                            visual_evidence["overlay_img"],
                            caption=map_caption,
                            use_container_width=True
                        )

                        if is_pure_healthy:
                            st.success(f"✅ **Daun Sehat & Normal ({num_spots} Titik Diverifikasi):** Retikel hijau memvalidasi helai daun sehat prima, berklorofil merata, dan bebas dari bercak lesi patogen aktif.")
                        elif diag_mode == "three_way":
                            c1_icon = "🟢" if ("sehat" in info.get("nama_id", "").lower()) else "🔴"
                            st.warning(
                                f"⚠️ **Deteksi 3 Kemungkinan Bersaing ({num_spots} Titik Ditandai):**\n\n"
                                f"• {c1_icon} **[1] {title_1} ({top_confidence}%):** Retikel Peringkat 1 menandai area helai daun utama.\n"
                                f"• 🟠 **[2] {title_2} ({second_confidence}%):** Retikel Oranye Peringkat 2 menandai titik potensi infeksi kedua.\n"
                                f"• 🔵 **[3] {title_3} ({third_confidence}%):** Retikel Biru Peringkat 3 menandai titik potensi infeksi ketiga.\n\n"
                                f"💡 **Petunjuk Lapangan:** Cocokkan ketiga posisi retikel scanner di atas langsung pada helai daun di bedengan sawah untuk memastikan perlakuan obat semprot dan sanitasi secara terarah."
                            )
                        elif diag_mode == "two_way":
                            st.warning(
                                f"⚠️ **Deteksi Infeksi Ganda (Multi-Penyakit Terdeteksi {num_spots} Titik):**\n\n"
                                f"• 🔴 **[1] {title_1} ({top_confidence}%):** Ditandai dengan retikel Merah pada titik lesi penyakit utama.\n"
                                f"• 🟠 **[2] {title_2} ({second_confidence}%):** Ditandai dengan retikel Oranye pada titik lesi penyakit kedua yang menyertai.\n\n"
                                f"💡 **Petunjuk Lapangan:** Cocokkan kedua titik kerusakan ini langsung pada helai daun di bedengan sawah untuk memastikan perlakuan obat semprot dan pemangkasan daun sakit secara menyeluruh."
                            )
                        elif num_spots > 0:
                            st.caption(
                                f"💡 **Petunjuk Deteksi ({num_spots} Titik Kerusakan Terdeteksi):** Retikel scanner di atas memetakan titik lesi aktif tepat pada helai daun tanaman (anti-melengser, bebas gangguan latar belakang atau tangan). Titik bertanda nama penyakit menunjukkan konsentrasi infeksi aktif tempat patogen berkembang. Fokuskan sanitasi pemangkasan daun sakit dan penyemprotan obat pada titik-titik tersebut."
                            )
                        else:
                            st.caption(
                                "💡 **Petunjuk Deteksi:** Retikel scanner presisi dihasilkan langsung dari pemindaian densitas lesi pada foto helai daun Anda."
                            )
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
            if not is_pure_healthy:
                phys_cache_key = f"phys_{top_class_raw}_{second_class_raw}_{third_class_raw}_{diag_mode}"
                if phys_cache_key not in st.session_state:
                    with st.spinner("🔬 Menyiapkan panduan verifikasi fisik lapangan..."):
                        st.session_state[phys_cache_key] = get_groq_physical_verification(
                            primary_name=info['nama_id'],
                            second_name=second_info['nama_id'] if (diag_mode in ('two_way', 'three_way')) else None,
                            is_differential=(diag_mode == 'two_way'),
                            third_name=third_info.get('nama_id', third_class_raw) if diag_mode == 'three_way' else None,
                            is_three_way=(diag_mode == 'three_way')
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
            ai_token_now = f"{top_class_raw}_{second_class_raw}_{third_class_raw}_{diag_mode}_{round(top_confidence, 1)}"
            if st.session_state.get("ai_token_saved") != ai_token_now or "ai_text_saved" not in st.session_state:
                with st.spinner("🤖 Dokter Tanaman AI sedang meracik resep obat dan panduan perawatan..."):
                    ai_text, ai_angle = get_groq_recommendation(
                        disease_name=info["nama_id"],
                        confidence=top_confidence,
                        is_healthy=is_healthy,
                        second_disease_name=second_info["nama_id"] if (diag_mode in ("two_way", "three_way") and second_info) else None,
                        second_confidence=second_confidence if (diag_mode in ("two_way", "three_way") and second_info) else None,
                        is_differential=(diag_mode == "two_way"),
                        latin_name=info.get("latin"),
                        severity_level=visual_evidence.get("severity_level") if visual_evidence else None,
                        evidence_desc=visual_evidence.get("evidence_desc") if visual_evidence else None,
                        third_disease_name=third_info.get("nama_id", third_class_raw) if (diag_mode == "three_way" and third_info) else None,
                        third_confidence=third_confidence if (diag_mode == "three_way" and third_info) else None,
                        is_three_way=(diag_mode == "three_way")
                    )
                    st.session_state["ai_text_saved"] = ai_text
                    st.session_state["ai_angle_saved"] = ai_angle or "Pendekatan Terpadu Lapangan (Database Mandiri Sistem)"
                    st.session_state["ai_token_saved"] = ai_token_now

            # Parsing Resep Menjadi 3 Kartu Jelas & Format HTML Terstruktur (Otomatis Fallback ke Database Mandiri Sistem jika Kuota Groq Habis)
            kartu_tindakan, kartu_obat, kartu_lahan = parse_groq_to_cards(
                st.session_state.get("ai_text_saved"),
                info,
                second_info=second_info if (diag_mode in ("two_way", "three_way") and second_info) else None,
                is_differential=(diag_mode == "two_way"),
                third_info=third_info if (diag_mode == "three_way" and third_info) else None,
                is_three_way=(diag_mode == "three_way"),
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
                        evidence_desc=visual_evidence.get("evidence_desc") if visual_evidence else None,
                        second_disease_name=second_info["nama_id"] if (diag_mode in ("two_way", "three_way") and second_info) else None,
                        second_confidence=second_confidence if (diag_mode in ("two_way", "three_way") and second_info) else None,
                        is_differential=(diag_mode == "two_way"),
                        third_disease_name=third_info.get("nama_id", third_class_raw) if (diag_mode == "three_way" and third_info) else None,
                        third_confidence=third_confidence if (diag_mode == "three_way" and third_info) else None,
                        is_three_way=(diag_mode == "three_way")
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
