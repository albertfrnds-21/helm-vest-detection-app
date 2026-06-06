import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import tempfile
import os
import time
from datetime import datetime
import base64
from io import BytesIO

# =====================================================
# SUPABASE INTEGRATION
# =====================================================
try:
    from supabase import create_client, Client
    
    @st.cache_resource
    def init_supabase():
        """Initialize Supabase client from secrets"""
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            client = create_client(url, key)
            
            # Test connection dengan query sederhana
            try:
                client.table("ppe_detections").select("id").limit(1).execute()
                return client, True, None
            except Exception as test_err:
                return client, False, str(test_err)
                
        except Exception as e:
            return None, False, f"Config error: {str(e)}"
    
    supabase_client, supabase_connected, supabase_error = init_supabase()
    
except ImportError as import_err:
    supabase_client = None
    supabase_connected = False
    supabase_error = "Library 'supabase' tidak terinstall. Run: pip install supabase"

def save_log_to_supabase(media_type, area_status, counts, total):
    """Save detection log to Supabase database"""
    if supabase_client is None or not supabase_connected:
        return False
    
    try:
        # Insert sesuai dengan schema Supabase yang sudah ada
        # created_at akan otomatis terisi oleh DEFAULT now()
        response = supabase_client.table("ppe_detections").insert({
            "media_type": media_type,
            "area_status": area_status,
            "safe_count": counts["safe"],
            "warning_count": counts["warning"],
            "danger_count": counts["danger"],
            "total_count": total
            # created_at otomatis dari DEFAULT now()
            # session_id opsional (tidak dipakai untuk saat ini)
        }).execute()
        
        # Cek apakah insert berhasil
        if response.data:
            return True
        else:
            print(f"Supabase insert tidak mengembalikan data: {response}")
            return False
            
    except Exception as e:
        # Log error untuk debugging
        print(f"❌ Supabase save error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =====================================================
# PAGE CONFIG - must be first Streamlit call
# =====================================================
st.set_page_config(
    page_title="K3 Dermaga • PPE Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - Warm Industrial Light Theme
# Menggunakan Barlow + Source Code Pro, palette:
# Navy #0D2137 | Amber #F5A623 | Slate #64748B
# Safe=teal, Warning=amber, Danger=red
# =====================================================
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;900&family=Barlow+Condensed:wght@600;700&family=Source+Code+Pro:wght@400;600&display=swap');

    *, *::before, *::after {
        font-family: 'Barlow', sans-serif;
        box-sizing: border-box;
    }

    .stApp {
        background: #F4F6F9 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0D2137 !important;
        border-right: 3px solid #F5A623 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #E2EAF4 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #F5A623 !important;
    }

    /* DATABASE STATUS INDICATORS - Override sidebar text color */
    .db-status-connected {
        background: #D1FAE5 !important;
        border: 2px solid #10B981 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    .db-status-connected .status-title {
        color: #065F46 !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
        font-size: 0.85rem !important;
    }
    .db-status-connected .status-desc {
        color: #047857 !important;
        font-size: 0.75rem !important;
    }
    
    .db-status-disconnected {
        background: #FEE2E2 !important;
        border: 2px solid #EF4444 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    .db-status-disconnected .status-title {
        color: #991B1B !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
        font-size: 0.85rem !important;
    }
    .db-status-disconnected .status-desc {
        color: #7F1D1D !important;
        font-size: 0.75rem !important;
        line-height: 1.5 !important;
    }

    /* CONFIRMATION BOXES - Override sidebar text colors */
    section[data-testid="stSidebar"] div[style*="background:#DBEAFE"] * {
        color: inherit !important;
    }
    section[data-testid="stSidebar"] div[style*="background:#FEE2E2"] * {
        color: inherit !important;
    }
    section[data-testid="stSidebar"] div[style*="color:#1E3A8A"] {
        color: #1E3A8A !important;
    }
    section[data-testid="stSidebar"] div[style*="color:#1E40AF"] {
        color: #1E40AF !important;
    }
    section[data-testid="stSidebar"] div[style*="color:#7F1D1D"] {
        color: #7F1D1D !important;
    }
    section[data-testid="stSidebar"] div[style*="color:#991B1B"] {
        color: #991B1B !important;
    }

    /* Main container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1600px !important;
    }

    /* HERO HEADER */
    .hero-header {
        background: linear-gradient(110deg, #0D2137 60%, #1a3a5c 100%);
        border-radius: 16px;
        padding: 1.8rem 2.5rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-left: 6px solid #F5A623;
        position: relative;
        overflow: hidden;
    }
    .hero-header::after {
        content: "⚓";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.06;
    }
    .hero-title {
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.3px !important;
        margin: 0 !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        text-transform: uppercase !important;
    }
    .hero-sub {
        color: #94B8D4 !important;
        font-size: 0.9rem !important;
        margin: 0.3rem 0 0 0 !important;
    }
    .live-badge {
        background: #EF4444;
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        animation: pulse-badge 1.5s infinite;
        display: inline-block;
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.55; }
    }

    /* STATUS PILLS */
    .pill-safe {
        background: #ECFDF5; color: #065F46;
        border: 1.5px solid #6EE7B7;
        border-radius: 30px; padding: 6px 18px;
        font-weight: 700; font-size: 0.82rem;
        display: inline-block; text-transform: uppercase; letter-spacing: 0.8px;
    }
    .pill-warning {
        background: #FFFBEB; color: #92400E;
        border: 1.5px solid #FCD34D;
        border-radius: 30px; padding: 6px 18px;
        font-weight: 700; font-size: 0.82rem;
        display: inline-block; text-transform: uppercase; letter-spacing: 0.8px;
    }
    .pill-danger {
        background: #FEF2F2; color: #991B1B;
        border: 1.5px solid #FCA5A5;
        border-radius: 30px; padding: 6px 18px;
        font-weight: 700; font-size: 0.82rem;
        display: inline-block; text-transform: uppercase; letter-spacing: 0.8px;
    }

    /* METRIC CARDS */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: transform 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
    .metric-card-top {
        height: 4px;
        border-radius: 4px 4px 0 0;
        position: absolute;
        top: 0; left: 0; right: 0;
    }
    .metric-card .val {
        font-size: 2.4rem;
        font-weight: 900;
        font-family: 'Barlow Condensed', sans-serif;
        line-height: 1;
        margin: 0.5rem 0 0.3rem 0;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
    }
    .mc-safe .val { color: #059669; }
    .mc-safe .metric-card-top { background: #10B981; }
    .mc-warning .val { color: #D97706; }
    .mc-warning .metric-card-top { background: #F59E0B; }
    .mc-danger .val { color: #DC2626; }
    .mc-danger .metric-card-top { background: #EF4444; }
    .mc-total .val { color: #0D2137; }
    .mc-total .metric-card-top { background: #F5A623; }

    /* SECTION HEADING */
    .section-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0D2137;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-left: 4px solid #F5A623;
        padding-left: 12px;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* IMAGE FRAME */
    .img-frame {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 10px;
        overflow: hidden;
    }
    .img-frame-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        padding: 0 4px 6px 4px;
    }

    /* CHART PANEL */
    .chart-panel {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 1.25rem 1.5rem;
    }
    .chart-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #0D2137;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.75rem;
    }

    /* LOG TABLE */
    .log-row {
        display: flex;
        gap: 8px;
        align-items: center;
        padding: 7px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .log-row:nth-child(odd) { background: #F8FAFC; }
    .log-row .log-time { color: #94A3B8; font-family: 'Source Code Pro', monospace; font-size: 0.78rem; min-width: 80px; }
    .log-row .log-src { color: #475569; min-width: 70px; font-weight: 600; font-size: 0.78rem; }

    /* MODE TAB BUTTONS */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.3px;
        height: 3rem;
        text-transform: uppercase;
    }

    /* RESET BUTTON (in sidebar) */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%) !important;
        color: white !important;
        border: 2px solid #FCA5A5 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        border-color: #FEE2E2 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* RESET DATABASE BUTTON - Warning style (secondary button) */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: white !important;
        border: 2px solid #FCD34D !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%) !important;
        border-color: #FDE68A !important;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4) !important;
    }
    
    /* CONFIRM BUTTONS - Small inline buttons */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: #DC2626 !important;
        font-size: 0.75rem !important;
        padding: 0.4rem 0.6rem !important;
    }
    section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-secondary"] {
        background: #6B7280 !important;
        font-size: 0.75rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* Realtime info box */
    .rt-info {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        font-size: 0.85rem;
        color: #1E40AF;
        margin-bottom: 1rem;
    }

    /* Sidebar nav */
    .sb-section {
        background: rgba(245,166,35,0.1);
        border-left: 3px solid #F5A623;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #F5A623 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero-header">
    <div>
        <p class="hero-title">🦺 Sistem Monitoring Kepatuhan APD Dermaga</p>
        <p class="hero-sub">Real-time PPE Compliance Detection - YOLOv11 Multi-Anchor Engine v14.0</p>
    </div>
    <div style="text-align:right">
        <span class="live-badge">● LIVE</span>
        <p style="color:#94B8D4; font-size:0.78rem; margin-top:6px;">Helmet · Vest · Head · Person</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE INIT
# =====================================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "image"
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []
if "cumulative_safe" not in st.session_state:
    st.session_state.cumulative_safe = 0
if "cumulative_warning" not in st.session_state:
    st.session_state.cumulative_warning = 0
if "cumulative_danger" not in st.session_state:
    st.session_state.cumulative_danger = 0
if "db_stats_loaded" not in st.session_state:
    st.session_state.db_stats_loaded = False
    st.session_state.db_total_detections = 0
    st.session_state.db_total_safe = 0
    st.session_state.db_total_warning = 0
    st.session_state.db_total_danger = 0

# Load database statistics (one time per session)
if supabase_connected and not st.session_state.db_stats_loaded:
    try:
        # Get total stats from database
        response = supabase_client.table("ppe_detections").select("safe_count, warning_count, danger_count").execute()
        if response.data:
            st.session_state.db_total_detections = len(response.data)
            st.session_state.db_total_safe = sum(row['safe_count'] for row in response.data)
            st.session_state.db_total_warning = sum(row['warning_count'] for row in response.data)
            st.session_state.db_total_danger = sum(row['danger_count'] for row in response.data)
        st.session_state.db_stats_loaded = True
    except Exception as e:
        print(f"Failed to load DB stats: {e}")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 1rem 0;">
        <div style="font-size:2.8rem; margin-bottom:4px;">⚓</div>
        <div style="color:#F5A623; font-family:'Barlow Condensed'; font-size:1.3rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;">K3 Helm & Rompi</div>
        <div style="color:#64A0C8; font-size:0.78rem; margin-top:2px;">YOLOv11 PPE Detection Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # Klasifikasi APD
    st.markdown('<div class="sb-section">📋 Klasifikasi APD</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0.5rem 0.2rem; font-size: 0.84rem; line-height: 2;">
        <span style="background:#065F46; color:#A7F3D0; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.75rem;">SAFE</span>
        &nbsp; Person + Helm + Rompi<br>
        <span style="background:#92400E; color:#FDE68A; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.75rem;">WARNING</span>
        &nbsp; Person + Helm (tanpa rompi)<br>
        <span style="background:#991B1B; color:#FECACA; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.75rem;">DANGER</span>
        &nbsp; Person tanpa APD lengkap
    </div>
    """, unsafe_allow_html=True)

    # Default threshold values (fixed, no user adjustment)
    conf_helmet = 0.30
    conf_vest   = 0.10
    conf_head   = 0.35
    conf_person = 0.20
    iou_thresh  = 0.45

    st.markdown("---")

    # Statistik Sesi
    st.markdown('<div class="sb-section">📊 Statistik Sesi</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; line-height:2.1; padding:0.4rem 0.2rem;">
        <div>🟢 Safe: <b style="color:#10B981">{st.session_state.cumulative_safe}</b></div>
        <div>🟡 Warning: <b style="color:#F59E0B">{st.session_state.cumulative_warning}</b></div>
        <div>🔴 Danger: <b style="color:#EF4444">{st.session_state.cumulative_danger}</b></div>
        <div>📋 Log: <b style="color:#F5A623">{len(st.session_state.detection_log)}</b> entries</div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol Reset Statistik Sesi (dengan konfirmasi)
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    
    if 'confirm_session_reset' not in st.session_state:
        st.session_state.confirm_session_reset = False

    if not st.session_state.confirm_session_reset:
        if st.button("🗑 Reset Statistik Sesi", use_container_width=True, key="btn_reset_session"):
            st.session_state.confirm_session_reset = True
            st.rerun()
    else:
        st.markdown("""
        <div style="background:#DBEAFE !important; border:2px solid #3B82F6 !important; border-radius:6px; padding:0.5rem; margin:4px 0;">
            <div style="color:#1E3A8A !important; font-size:0.75rem; font-weight:700; margin-bottom:2px;">⚠️ Konfirmasi Reset Sesi</div>
            <div style="color:#1E40AF !important; font-size:0.7rem; line-height:1.3;">Log & statistik UI akan direset. Database tetap aman.</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            if st.button("✅ Reset", use_container_width=True, key="confirm_session_yes"):
                st.session_state.detection_log = []
                st.session_state.cumulative_safe = 0
                st.session_state.cumulative_warning = 0
                st.session_state.cumulative_danger = 0
                st.session_state.confirm_session_reset = False
                st.rerun()
        with col2:
            if st.button("❌ Batal", use_container_width=True, key="confirm_session_no"):
                st.session_state.confirm_session_reset = False
                st.rerun()

    # Statistik Database (jika terhubung)
    if supabase_connected and st.session_state.db_stats_loaded:
        st.markdown("---")
        st.markdown('<div class="sb-section">🗄️ Total Database</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(245,166,35,0.08); border:1px solid rgba(245,166,35,0.3); border-radius:6px; padding:0.6rem; font-size:0.8rem; line-height:2;">
            <div style="color:#10B981; font-size:0.75rem;">🟢 Safe: <b>{st.session_state.db_total_safe}</b></div>
            <div style="color:#F59E0B; font-size:0.75rem;">🟡 Warning: <b>{st.session_state.db_total_warning}</b></div>
            <div style="color:#EF4444; font-size:0.75rem;">🔴 Danger: <b>{st.session_state.db_total_danger}</b></div>
            <div style="color:#64A0C8; font-size:0.73rem; margin-top:4px; padding-top:4px; border-top:1px solid rgba(245,166,35,0.2);">📊 Total: <b>{st.session_state.db_total_detections}</b> records</div>
        </div>
        """, unsafe_allow_html=True)

        # Tombol Reset Database (dengan konfirmasi)
        st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
        
        if 'confirm_db_reset' not in st.session_state:
            st.session_state.confirm_db_reset = False
        
        if not st.session_state.confirm_db_reset:
            if st.button("⚠️ Reset Database", use_container_width=True, type="secondary", key="btn_reset_db"):
                st.session_state.confirm_db_reset = True
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#FEE2E2 !important; border:2px solid #DC2626 !important; border-radius:6px; padding:0.5rem; margin:4px 0;">
                <div style="color:#7F1D1D !important; font-size:0.75rem; font-weight:700; margin-bottom:2px;">🚨 Konfirmasi Hapus Database</div>
                <div style="color:#991B1B !important; font-size:0.7rem; line-height:1.3;">Semua data Supabase akan dihapus PERMANEN!</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                if st.button("✅ Hapus", use_container_width=True, type="primary", key="confirm_db_yes"):
                    try:
                        # Delete all records from database
                        response = supabase_client.table("ppe_detections").delete().neq("id", 0).execute()
                        
                        # Reset database stats
                        st.session_state.db_total_detections = 0
                        st.session_state.db_total_safe = 0
                        st.session_state.db_total_warning = 0
                        st.session_state.db_total_danger = 0
                        st.session_state.db_stats_loaded = False
                        st.session_state.confirm_db_reset = False
                        
                        st.success("✅ Database berhasil direset!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal reset database: {e}")
                        st.session_state.confirm_db_reset = False
            
            with col2:
                if st.button("❌ Batal", use_container_width=True, key="confirm_db_no"):
                    st.session_state.confirm_db_reset = False
                    st.rerun()

    # Status Koneksi Database
    st.markdown("---")
    st.markdown('<div class="sb-section">📡 Koneksi</div>', unsafe_allow_html=True)
    if supabase_connected:
        st.markdown("""
        <div class="db-status-connected">
            <div class="status-title">✅ Supabase Terhubung</div>
            <div class="status-desc">Data tersimpan otomatis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="db-status-disconnected">
            <div class="status-title">❌ Supabase Offline</div>
            <div class="status-desc">Data hanya di session</div>
        </div>
        """, unsafe_allow_html=True)
        if supabase_error:
            with st.expander("🔍 Detail Error", expanded=False):
                st.code(supabase_error, language="text")

    st.markdown("---")
    st.caption("© 2026 Proyek Akhir Informatika\nArea Dermaga — Keselamatan Kerja")

# =====================================================
# MODEL LOAD
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        upload_path = "/mnt/user-data/uploads/best.pt"
        if os.path.exists(upload_path):
            import shutil
            shutil.copy(upload_path, model_path)
        else:
            return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()
if model is None:
    st.error("❌ File `best.pt` tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
    st.stop()

# =====================================================
# DETECTION UTILITIES
# =====================================================
def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua

def box_inside(inner, outer, tol=60):
    return (inner[0] >= outer[0]-tol and inner[2] <= outer[2]+tol and
            inner[1] >= outer[1]-tol and inner[3] <= outer[3]+tol)

def center_dist(a, b):
    cx_a = (a[0]+a[2])/2; cy_a = (a[1]+a[3])/2
    cx_b = (b[0]+b[2])/2; cy_b = (b[1]+b[3])/2
    return ((cx_a-cx_b)**2 + (cy_a-cy_b)**2) ** 0.5

def run_detection(img_np, conf_h, conf_v, conf_hd, conf_p, iou_t):
    """
    Multi-anchor hybrid matching detection.
    Classes: 0=Helmet, 1=Vest, 2=Head, 3=Person
    """
    results = model(img_np, imgsz=640, conf=0.03, iou=iou_t,
                    augment=True, verbose=False)

    raw = {0: [], 1: [], 2: [], 3: []}
    thresholds = {0: conf_h, 1: conf_v, 2: conf_hd, 3: conf_p}

    if results and results[0].boxes is not None:
        for row in results[0].boxes.data.cpu().numpy():
            if len(row) < 6: continue
            x1,y1,x2,y2,conf,cls = row[:6]
            cls = int(cls)
            if cls in raw and conf >= thresholds.get(cls, 0.20):
                raw[cls].append({"box": [float(x1),float(y1),float(x2),float(y2)], "conf": float(conf)})

    helmets = raw[0]
    vests   = raw[1]
    heads   = raw[2]
    persons = raw[3]

    # Build candidate workers from multiple anchor sources
    candidates = []
    for p in persons:
        candidates.append({"box": p["box"], "conf": p["conf"], "src": "person"})

    for h in helmets:
        if not any(box_inside(h["box"], p["box"], 40) for p in persons):
            hb = h["box"]
            body = [hb[0]-30, hb[1], hb[2]+30, hb[3]+480]
            candidates.append({"box": body, "conf": h["conf"], "src": "helmet_anchor"})

    for hd in heads:
        if not any(box_inside(hd["box"], p["box"], 40) for p in persons):
            hdb = hd["box"]
            body = [hdb[0]-30, hdb[1], hdb[2]+30, hdb[3]+480]
            candidates.append({"box": body, "conf": hd["conf"], "src": "head_anchor"})

    # NMS-style deduplication
    workers = []
    for c in candidates:
        if not any(iou(c["box"], w["box"]) > 0.40 for w in workers):
            workers.append(c)

    results_out = []
    counts = {"safe": 0, "warning": 0, "danger": 0}

    for w in workers:
        wb = w["box"]
        has_helmet = any(box_inside(h["box"], wb, 65) for h in helmets)
        has_vest   = any(box_inside(v["box"], wb, 65) for v in vests)
        has_head   = any(box_inside(hd["box"], wb, 65) for hd in heads)

        if has_head and not has_helmet:
            status = "DANGER"
            color  = (68, 68, 239)   # red in BGR
        elif has_helmet and has_vest:
            status = "SAFE"
            color  = (16, 185, 10)   # green in BGR
        elif has_helmet and not has_vest:
            status = "WARNING"
            color  = (11, 158, 245)  # amber in BGR
        else:
            status = "DANGER"
            color  = (68, 68, 239)

        counts[status.lower()] += 1
        results_out.append({"box": wb, "status": status, "conf": w["conf"], "color": color})

    total = len(results_out)
    if total == 0:
        area_status = "TIDAK ADA AKTIVITAS"
    elif counts["danger"] > 0:
        area_status = "BAHAYA"
    elif counts["warning"] > 0:
        area_status = "WARNING"
    else:
        area_status = "AMAN"

    return results_out, area_status, counts, total

def render_boxes(img_arr, detections):
    out = img_arr.copy()
    H, W = out.shape[:2]
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["box"]]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(W,x2), min(H,y2)
        color = d["color"]
        status = d["status"]
        conf_pct = int(d["conf"]*100)

        # Thick outer box
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 3)

        # Corner accents
        clen = 18
        cv2.line(out, (x1,y1), (x1+clen,y1), color, 5)
        cv2.line(out, (x1,y1), (x1,y1+clen), color, 5)
        cv2.line(out, (x2,y1), (x2-clen,y1), color, 5)
        cv2.line(out, (x2,y1), (x2,y1+clen), color, 5)
        cv2.line(out, (x1,y2), (x1+clen,y2), color, 5)
        cv2.line(out, (x1,y2), (x1,y2-clen), color, 5)
        cv2.line(out, (x2,y2), (x2-clen,y2), color, 5)
        cv2.line(out, (x2,y2), (x2,y2-clen), color, 5)

        label = f"[{status}] {conf_pct}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        lx, ly = x1, max(y1-6, th+10)
        cv2.rectangle(out, (lx, ly-th-8), (lx+tw+10, ly+2), color, -1)
        cv2.putText(out, label, (lx+5, ly-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def log_detection(source, metrics, total, area_status=""):
    """Log detection locally and to Supabase"""
    st.session_state.cumulative_safe    += metrics["safe"]
    st.session_state.cumulative_warning += metrics["warning"]
    st.session_state.cumulative_danger  += metrics["danger"]
    
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "safe": metrics["safe"],
        "warning": metrics["warning"],
        "danger": metrics["danger"],
        "total": total,
    }
    st.session_state.detection_log.append(entry)
    if len(st.session_state.detection_log) > 50:
        st.session_state.detection_log = st.session_state.detection_log[-50:]
    
    # Save to Supabase
    if area_status and supabase_connected:
        success = save_log_to_supabase(source, area_status, metrics, total)
        if success:
            # Optional: tampilkan notifikasi sukses (bisa dikomentari jika terlalu banyak)
            # st.toast("💾 Data tersimpan ke Supabase", icon="✅")
            pass
        else:
            # Tampilkan error jika gagal
            st.toast("⚠️ Gagal menyimpan ke Supabase", icon="⚠️")

# =====================================================
# CHART COMPONENTS (Plotly-free, using Streamlit native + custom HTML)
# =====================================================
def render_analytics(metrics, total, title_prefix=""):
    """Renders pie chart + histogram side by side using Chart.js via st.components"""
    import streamlit.components.v1 as components

    safe_n    = metrics["safe"]
    warning_n = metrics["warning"]
    danger_n  = metrics["danger"]

    chart_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <style>
        body {{ margin:0; padding:0; background:transparent; font-family:'Barlow',sans-serif; }}
        .charts-wrap {{
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 12px 8px;
        }}
        .chart-card {{
            background: #FFFFFF;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            padding: 16px;
        }}
        .chart-label {{
            font-size:12px;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:1px;
            color:#0D2137;
            margin-bottom:10px;
            padding-left: 8px;
            border-left: 3px solid #F5A623;
        }}
        .legend-row {{
            display:flex;
            gap:18px;
            flex-wrap:wrap;
            margin-bottom:10px;
            font-size:11.5px;
            color:#475569;
        }}
        .legend-item {{
            display:flex;
            align-items:center;
            gap:5px;
            font-weight:600;
        }}
        .legend-dot {{
            width:11px;
            height:11px;
            border-radius:3px;
            flex-shrink:0;
        }}
        .no-data {{
            text-align:center;
            color:#94A3B8;
            font-size:13px;
            padding:40px 0;
        }}
    </style>
    </head>
    <body>
    <div class="charts-wrap">

        <!-- PIE CHART -->
        <div class="chart-card">
            <div class="chart-label">📊 Distribusi Status APD</div>
            <div class="legend-row">
                <span class="legend-item"><span class="legend-dot" style="background:#059669;"></span>Safe ({safe_n})</span>
                <span class="legend-item"><span class="legend-dot" style="background:#D97706;"></span>Warning ({warning_n})</span>
                <span class="legend-item"><span class="legend-dot" style="background:#DC2626;"></span>Danger ({danger_n})</span>
            </div>
            {"<div class='no-data'>Tidak ada pekerja terdeteksi</div>" if total == 0 else ""}
            <div style="position:relative; height:220px; {"display:none" if total == 0 else ""}">
                <canvas id="pieChart" role="img" aria-label="Pie chart distribusi APD pekerja"></canvas>
            </div>
        </div>

        <!-- BAR CHART -->
        <div class="chart-card">
            <div class="chart-label">📈 Jumlah Pekerja per Status</div>
            <div class="legend-row">
                <span class="legend-item"><span class="legend-dot" style="background:#059669;"></span>Safe</span>
                <span class="legend-item"><span class="legend-dot" style="background:#D97706;"></span>Warning</span>
                <span class="legend-item"><span class="legend-dot" style="background:#DC2626;"></span>Danger</span>
            </div>
            <div style="position:relative; height:220px;">
                <canvas id="barChart" role="img" aria-label="Histogram jumlah pekerja berdasarkan status APD"></canvas>
            </div>
        </div>

    </div>

    <script>
    (function() {{
        {"" if total == 0 else f"""
        new Chart(document.getElementById('pieChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Safe', 'Warning', 'Danger'],
                datasets: [{{
                    data: [{safe_n}, {warning_n}, {danger_n}],
                    backgroundColor: ['#059669', '#D97706', '#DC2626'],
                    borderColor: ['#047857', '#B45309', '#B91C1C'],
                    borderWidth: 2,
                    hoverOffset: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                const total = ctx.dataset.data.reduce((a,b)=>a+b, 0);
                                const pct = total > 0 ? Math.round(ctx.parsed / total * 100) : 0;
                                return ' ' + ctx.label + ': ' + ctx.parsed + ' orang (' + pct + '%)';
                            }}
                        }}
                    }}
                }}
            }}
        }});
        """}

        new Chart(document.getElementById('barChart'), {{
            type: 'bar',
            data: {{
                labels: ['Safe', 'Warning', 'Danger'],
                datasets: [{{
                    label: 'Jumlah Pekerja',
                    data: [{safe_n}, {warning_n}, {danger_n}],
                    backgroundColor: ['rgba(5,150,105,0.8)', 'rgba(217,119,6,0.8)', 'rgba(220,38,38,0.8)'],
                    borderColor: ['#047857', '#B45309', '#B91C1C'],
                    borderWidth: 2,
                    borderRadius: 6,
                    borderSkipped: false
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return ' ' + ctx.parsed.y + ' orang';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 1,
                            color: '#64748B',
                            font: {{ size: 11 }}
                        }},
                        grid: {{ color: 'rgba(100,116,139,0.12)' }}
                    }},
                    x: {{
                        ticks: {{
                            color: '#0D2137',
                            font: {{ size: 12, weight: '700' }}
                        }},
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
    }})();
    </script>
    </body>
    </html>
    """
    components.html(chart_html, height=320, scrolling=False)

def render_session_chart():
    """Renders cumulative session stats chart"""
    import streamlit.components.v1 as components
    log = st.session_state.detection_log
    if len(log) < 2:
        return

    labels  = [e["time"] for e in log[-15:]]
    safes   = [e["safe"] for e in log[-15:]]
    warns   = [e["warning"] for e in log[-15:]]
    dangs   = [e["danger"] for e in log[-15:]]

    labels_js  = str(labels).replace("'",'"')
    safes_js   = str(safes)
    warns_js   = str(warns)
    dangs_js   = str(dangs)

    html = f"""
    <!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <style>body{{margin:0;padding:8px;background:transparent;font-family:'Barlow',sans-serif;}}</style>
    </head><body>
    <div style="position:relative;height:180px;width:100%;">
        <canvas id="trendChart" role="img" aria-label="Tren deteksi APD per waktu"></canvas>
    </div>
    <script>
    new Chart(document.getElementById('trendChart'), {{
        type:'bar',
        data:{{
            labels:{labels_js},
            datasets:[
                {{label:'Safe', data:{safes_js}, backgroundColor:'rgba(5,150,105,0.75)', stack:'a', borderRadius:3}},
                {{label:'Warning', data:{warns_js}, backgroundColor:'rgba(217,119,6,0.75)', stack:'a', borderRadius:3}},
                {{label:'Danger', data:{dangs_js}, backgroundColor:'rgba(220,38,38,0.75)', stack:'a', borderRadius:3}}
            ]
        }},
        options:{{
            responsive:true,
            maintainAspectRatio:false,
            plugins:{{ legend:{{ display:true, position:'top', labels:{{ font:{{size:11}}, boxWidth:12, padding:12 }} }} }},
            scales:{{
                x:{{ stacked:true, ticks:{{color:'#64748B', font:{{size:9}}, maxRotation:45}}, grid:{{display:false}} }},
                y:{{ stacked:true, beginAtZero:true, ticks:{{stepSize:1, color:'#64748B', font:{{size:10}}}}, grid:{{color:'rgba(100,116,139,0.12)'}} }}
            }}
        }}
    }});
    </script></body></html>
    """
    components.html(html, height=220, scrolling=False)

# =====================================================
# STATUS BANNER
# =====================================================
def show_status_banner(area_status):
    if area_status == "AMAN":
        st.success("🟢 **STATUS AREA: AMAN** - Seluruh pekerja mematuhi regulasi K3 APD")
    elif area_status == "WARNING":
        st.warning("🟡 **STATUS AREA: WARNING** - Terdapat pekerja dengan APD tidak lengkap")
    elif area_status == "BAHAYA":
        st.error("🔴 **STATUS AREA: BAHAYA** - Terdeteksi pelanggaran fatal penggunaan APD!")
    else:
        st.info("⚪ **STATUS AREA: TIDAK ADA AKTIVITAS** - Tidak ada pekerja terdeteksi di area")

def show_metrics(counts, total):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card mc-safe">
            <div class="metric-card-top"></div>
            <div class="lbl">🟢 Safe Worker</div>
            <div class="val">{counts['safe']}</div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">Pekerja</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card mc-warning">
            <div class="metric-card-top"></div>
            <div class="lbl">🟡 Warning Worker</div>
            <div class="val">{counts['warning']}</div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">Pekerja</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card mc-danger">
            <div class="metric-card-top"></div>
            <div class="lbl">🔴 Danger Worker</div>
            <div class="val">{counts['danger']}</div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">Pekerja</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card mc-total">
            <div class="metric-card-top"></div>
            <div class="lbl">📊 Total Terdeteksi</div>
            <div class="val">{total}</div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:2px;">Pekerja</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

def show_log_table():
    log = st.session_state.detection_log
    if not log:
        st.caption("Belum ada log deteksi di sesi ini.")
        return
    rows_html = ""
    for e in reversed(log[-10:]):
        safe_badge    = f'<span style="background:#ECFDF5;color:#065F46;padding:2px 8px;border-radius:20px;font-weight:700;font-size:0.75rem;">{e["safe"]}</span>'
        warn_badge    = f'<span style="background:#FFFBEB;color:#92400E;padding:2px 8px;border-radius:20px;font-weight:700;font-size:0.75rem;">{e["warning"]}</span>'
        danger_badge  = f'<span style="background:#FEF2F2;color:#991B1B;padding:2px 8px;border-radius:20px;font-weight:700;font-size:0.75rem;">{e["danger"]}</span>'
        rows_html += f"""
        <div class="log-row">
            <span class="log-time">{e["time"]}</span>
            <span class="log-src">{e["source"]}</span>
            {safe_badge} {warn_badge} {danger_badge}
            <span style="color:#94A3B8;font-size:0.78rem;margin-left:auto;">Total: {e["total"]}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

# =====================================================
# MODE SELECTOR
# =====================================================
st.markdown('<div class="section-title">🎛️ Pilih Sumber Input Pemantauan</div>', unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3)
with mc1:
    if st.button("📸  Gambar Statis", use_container_width=True, type="primary" if st.session_state.app_mode=="image" else "secondary"):
        st.session_state.app_mode = "image"; st.rerun()
with mc2:
    if st.button("🎬  Stream Video", use_container_width=True, type="primary" if st.session_state.app_mode=="video" else "secondary"):
        st.session_state.app_mode = "video"; st.rerun()
with mc3:
    if st.button("📹  Kamera Realtime", use_container_width=True, type="primary" if st.session_state.app_mode=="realtime" else "secondary"):
        st.session_state.app_mode = "realtime"; st.rerun()

st.markdown("---")

# =====================================================
# MODE: IMAGE
# =====================================================
if st.session_state.app_mode == "image":
    uploaded = st.file_uploader("Pilih file gambar lapangan (JPG / PNG / JPEG)",
                                 type=["jpg","jpeg","png"],
                                 label_visibility="visible")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="img-frame"><div class="img-frame-label">📷 Gambar Asli</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("🕵️ Menganalisis gambar dengan YOLOv11..."):
            detections, area_status, counts, total = run_detection(
                img_np, conf_helmet, conf_vest, conf_head, conf_person, iou_thresh)
            output_img = render_boxes(img_np, detections)
            log_detection("image", counts, total, area_status)

        with col2:
            st.markdown('<div class="img-frame"><div class="img-frame-label">🤖 Hasil Deteksi APD</div>', unsafe_allow_html=True)
            st.image(output_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Metrik Deteksi</div>', unsafe_allow_html=True)
        show_metrics(counts, total)
        show_status_banner(area_status)

        st.markdown('<div class="section-title">📈 Visualisasi Analitik</div>', unsafe_allow_html=True)
        render_analytics(counts, total)

# =====================================================
# MODE: VIDEO
# =====================================================
elif st.session_state.app_mode == "video":
    uploaded_vid = st.file_uploader("Unggah file video CCTV (MP4 / AVI / MOV)",
                                     type=["mp4","avi","mov"],
                                     label_visibility="visible")
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Kolom untuk video dan stats
        col_vid, col_stats = st.columns([3, 2], gap="medium")
        
        with col_vid:
            st.markdown('<div class="section-title">🎬 Stream Deteksi</div>', unsafe_allow_html=True)
            frame_ph = st.empty()
        
        with col_stats:
            st.markdown('<div class="section-title">📊 Live Stats</div>', unsafe_allow_html=True)
            # Buat container yang akan diupdate setiap frame
            stats_container = st.empty()

        stop_btn  = st.button("⏹ Hentikan Pemrosesan Video", type="secondary")
        prog_bar  = st.progress(0)
        frame_count = 0
        SKIP = max(1, int(fps_src // 8))  # process ~8 fps for speed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn: break
            frame_count += 1
            if frame_count % SKIP != 0: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets, area_s, cnts, tot = run_detection(
                frame_rgb, conf_helmet, conf_vest, conf_head, conf_person, iou_thresh)
            ann = render_boxes(frame_rgb, dets)
            log_detection("video", cnts, tot, area_s)

            # Update frame video
            frame_ph.image(ann, channels="RGB", use_container_width=True)

            # Update stats dalam 1 container saja (lebih simple & smooth)
            with stats_container.container():
                show_metrics(cnts, tot)
                show_status_banner(area_s)

            prog = min(frame_count / max(total_frames, 1), 1.0)
            prog_bar.progress(prog)

        cap.release()
        # Close the temporary file properly before deleting
        try:
            tfile.close()
        except:
            pass
        
        # Wait a bit and retry deletion if needed
        import time as time_module
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                os.unlink(tfile.name)
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time_module.sleep(0.5)
                else:
                    # If still can't delete, just ignore
                    pass
        
        st.success("✅ Pemrosesan video selesai.")
        
        # Tampilkan analitik sesi setelah video selesai
        st.markdown('<div class="section-title">📈 Analitik Sesi</div>', unsafe_allow_html=True)
        render_session_chart()

# =====================================================
# MODE: REALTIME WEBCAM (continuous frame prediction)
# =====================================================
elif st.session_state.app_mode == "realtime":
    st.markdown("""
    <div class="rt-info">
        📹 <b>Mode Kamera Realtime</b> - Deteksi berjalan otomatis setiap frame dari webcam.
        Kamera langsung memproses tanpa perlu menekan tombol. Klik <b>Stop</b> untuk menghentikan.
    </div>
    """, unsafe_allow_html=True)

    run_cam = st.toggle("▶ Aktifkan Kamera Realtime", value=True, key="cam_toggle")

    # Layout untuk kamera dan stats
    col_cam, col_cam_stats = st.columns([3, 2], gap="medium")
    
    with col_cam:
        st.markdown('<div class="section-title">📹 Live Feed Deteksi</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        fps_placeholder   = st.empty()
    
    with col_cam_stats:
        st.markdown('<div class="section-title">📊 Live Metrik</div>', unsafe_allow_html=True)
        # 1 container saja untuk semua stats
        rt_stats_container = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Kamera tidak ditemukan. Pastikan webcam tersambung dan izin diberikan.")
        else:
            prev_time = time.time()

            while st.session_state.get("cam_toggle", True):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Gagal membaca frame dari kamera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets, area_s, cnts, tot = run_detection(
                    frame_rgb, conf_helmet, conf_vest, conf_head, conf_person, iou_thresh)
                ann = render_boxes(frame_rgb, dets)
                log_detection("webcam", cnts, tot, area_s)

                # Hitung FPS
                now = time.time()
                fps_val = 1.0 / max(now - prev_time, 0.001)
                prev_time = now

                # Update frame
                frame_placeholder.image(ann, channels="RGB", use_container_width=True)
                fps_placeholder.caption(f"⚡ FPS: {fps_val:.1f}  |  Frame terakhir: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                # Update stats dalam 1 container (lebih simple)
                with rt_stats_container.container():
                    show_metrics(cnts, tot)
                    show_status_banner(area_s)

            cap.release()
            st.info("Kamera dihentikan.")

    # Log & chart below
    st.markdown('<div class="section-title">📋 Log Deteksi Sesi</div>', unsafe_allow_html=True)
    col_log, col_trend = st.columns([1,1], gap="medium")
    with col_log:
        show_log_table()
    with col_trend:
        st.markdown('<div class="section-title">📈 Tren Deteksi</div>', unsafe_allow_html=True)
        render_session_chart()

# =====================================================
# BOTTOM LOG (for image/video modes)
# =====================================================
if st.session_state.app_mode in ("image", "video"):
    st.markdown("---")
    col_log2, col_trend2 = st.columns([1,1], gap="medium")
    with col_log2:
        st.markdown('<div class="section-title">📋 Log Deteksi Sesi (10 Terakhir)</div>', unsafe_allow_html=True)
        show_log_table()
    with col_trend2:
        st.markdown('<div class="section-title">📈 Tren Deteksi Sesi</div>', unsafe_allow_html=True)
        render_session_chart()
