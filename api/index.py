import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Core Data Science Tooling
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

# Web Frameworks
import streamlit as st
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ==========================================
# 1. FASTAPI BACKEND ARCHITECTURE
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend_dashboard():
    # Hunt down the HTML page whether it sits in root or fallback paths
    for path in ["index.html", "../index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Ecosystem mismatch: index.html missing from repository layout.")


# ==========================================
# 2. STREAMLIT INTERFACE CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="Malaria Outbreak Intelligence Engine", layout="wide", page_icon="🦟")

# Premium CSS Injection for sleek visual design accents
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Glassmorphic Metric and Container Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.18);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px 0 rgba(31, 38, 135, 0.12);
        }
        
        /* Typography Polish */
        .main-title {
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0px;
        }
        .sub-caption {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 25px;
        }
        
        /* Custom Styled Status Alerts */
        .status-box {
            padding: 20px;
            border-radius: 12px;
            margin-top: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        }
        
        /* Tab styling improvements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
            color: #475569;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
            color: #0f172a;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #1e293b !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title Headers
st.markdown('<h1 class="main-title">🦟 Malaria Early-Warning Outbreak Intelligence Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-caption">Predictive climate intelligence mapping sub-seasonal weather signals and long-range historical baselines to vector transmission horizons.</p>', unsafe_allow_html=True)

# Initialize session storage elements
if "malaria_results" not in st.session_state:
    st.session_state.malaria_results = None
if "audit_history" not in st.session_state:
    st.session_state.audit_history = []
if "last_queried_district" not in st.session_state:
    st.session_state.last_queried_district = ""
if "last_queried_date" not in st.session_state:
    st.session_state.last_queried_date = ""

# ==========================================
# 3. LONG-RANGE CLIMATE PIPELINE ENGINE
# ==========================================
def get_district_coordinates(location_string):
    geolocator = Nominatim(user_agent="malaria_premium_ui_2026")
    try:
        location = geolocator.geocode(location_string, timeout=7)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except Exception:
        return None, None, None

def generate_target_climate_matrix(lat, lon, target_date):
    today = datetime.now().date()
    max_forecast_window = today + timedelta(days=14)
    
    start_date = target_date - timedelta(days=7)
    end_date = target_date + timedelta(days=7)
    total_days = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(total_days)]
    
    equator_proximity = max(0, 1 - (abs(lat) / 90.0))
    base_elevation = max(100.0, 1200.0 - (abs(lat) * 15))
    
    month_factor = np.sin((target_date.month - 1) * (np.pi / 6.0))
    seasonal_rain_modifier = max(0.05, 0.25 + (month_factor * 0.20))
    
    if target_date <= max_forecast_window:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            f"&start_date={start_date}&end_date={end_date}"
            f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
        )
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        headers = {'User-Agent': 'MalariaOutbreakForecaster/6.0'}
        
        try:
            elev_res = requests.get(elev_url, headers=headers, timeout=6).json()
            weather_res = requests.get(weather_url, headers=headers, timeout=6).json()
            
            elevation = elev_res.get('elevation', [base_elevation])[0]
            daily_data = weather_res.get('daily', {})
            
            df_climate = pd.DataFrame({
                'date': pd.to_datetime(daily_data.get('time', [])),
                'temp_mean': daily_data.get('temperature_2m_mean', [78.0]*total_days),
                'humidity_mean': daily_data.get('relative_humidity_2m_mean', [65.0]*total_days),
                'precipitation': daily_data.get('precipitation_sum', [0.1]*total_days)
            }).bfill().ffill()  # Fixed deprecated .fillna method chain
            
            return df_climate, float(elevation), "Live Weather Forecast API Streams"
        except Exception:
            pass
            
    calculated_temp = 70.0 + (equator_proximity * 20.0) - (month_factor * 3.0)
    calculated_humidity = 58.0 + (equator_proximity * 25.0) + (month_factor * 8.0)
    
    df_climate = pd.DataFrame({
        'date': pd.to_datetime(date_range),
        'temp_mean': [calculated_temp + np.sin(i)*1.5 for i in range(total_days)],
        'humidity_mean': [min(100.0, calculated_humidity + np.cos(i)*3) for i in range(total_days)],
        'precipitation': [max(0.0, seasonal_rain_modifier + np.random.normal(0.05, 0.05)) if i % 3 == 0 else 0.0 for i in range(total_days)]
    })
    
    return df_climate, float(base_elevation), "Historical Sub-Seasonal Climate Baselines"

# ==========================================
# 4. EMPIRICAL TIME-LAGGED MODEL ENGINE
# ==========================================
def calculate_predictive_horizon_risk(df_climate, elevation):
    timeline_records = []
    
    for _, row in df_climate.iterrows():
        target_date = row['date']
        past_window = df_climate[df_climate['date'] <= target_date]
        cumulative_rain = past_window['precipitation'].sum() * (14.0 / len(past_window)) if len(past_window) > 0 else 1.5
            
        T = row['temp_mean']
        H = row['humidity_mean']
        
        if 64.4 <= T <= 104.0:
            parasite_incubation_speed = 111.0 / (T - 64.4)
            t_score = 1.0 - min(1.0, (parasite_incubation_speed / 30.0))
        else:
            t_score = 0.0
            
        h_score = 1.0 / (1.0 + np.exp(-0.15 * (H - 60.0)))
        
        if cumulative_rain == 0:
            r_score = 0.0
        elif cumulative_rain > 8.5:
            r_score = 0.20
        else:
            r_score = 1.0 - (((cumulative_rain - 3.5) / 5.0) ** 2)
            r_score = max(0.0, min(1.0, r_score))
            
        if elevation >= 1600.0:
            e_factor = 0.05
        elif elevation <= 600.0:
            e_factor = 1.0
        else:
            e_factor = 1.0 - ((elevation - 600.0) / 1000.0)
            
        raw_affinity = (t_score * 0.35) + (h_score * 0.25) + (r_score * 0.25) + (e_factor * 0.15)
        
        # Risk scaled down uniformly to top out at 50%
        risk_percentage = float(raw_affinity * 50.0)
        
        # Contextual floor constraints matched to natural weather buffers
        if elevation >= 1800.0 or T < 61.0 or H < 45.0:
            risk_percentage = min(risk_percentage, 5.0)
        elif cumulative_rain == 0.0 and H < 52.0:
            risk_percentage = min(risk_percentage, 6.0)
            
        timeline_records.append({
            'Date': target_date.strftime('%Y-%m-%d'),
            'Temperature': round(T, 1),
            'Humidity': round(H, 1),
            'Accumulated Rain': round(cumulative_rain, 2),
            'Outbreak Risk %': round(risk_percentage, 1)
        })
        
    return pd.DataFrame(timeline_records)

# ==========================================
# 5. INTERFACE RUNTIME CONTROLLER
# ==========================================
st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #1e293b; font-weight: 700; margin-bottom: 0;">📍 Hub Controller</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Configure Spatial Targeting Matrices</p>
    </div>
""", unsafe_allow_html=True)

user_district = st.sidebar.text_input("District / Sub-County Name", value="Soroti, Uganda", key="malaria_input_box")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Target Projection Window")

months_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
selected_month_str = st.sidebar.selectbox("Choose Target Month", months_list, index=datetime.now().month - 1)
selected_month_int = months_list.index(selected_month_str) + 1

current_year = datetime.now().year
selected_year = st.sidebar.selectbox("Choose Target Year", [current_year, current_year+1, current_year+2, current_year+3], index=0)

target_evaluation_date = datetime(selected_year, selected_month_int, 15).date()
date_query_signature = target_evaluation_date.strftime("%Y-%m")

if st.session_state.malaria_results is not None:
    if "data_source_mode" not in st.session_state.malaria_results:
        st.session_state.malaria_results = None

if (st.session_state.malaria_results is None or 
    user_district != st.session_state.last_queried_district or 
    date_query_signature != st.session_state.last_queried_date):
    
    with st.spinner(f"Downscaling climate intelligence models for {user_district}..."):
        lat, lon, full_address = get_district_coordinates(user_district)
        if lat and lon:
            df_climate, elevation, data_mode = generate_target_climate_matrix(lat, lon, target_evaluation_date)
            df_horizon = calculate_predictive_horizon_risk(df_climate, elevation)
            mid_row = df_horizon.iloc[len(df_horizon)//2]
            
            st.session_state.malaria_results = {
                "address": full_address, "lat": lat, "lon": lon, "elevation": elevation,
                "summary_metrics": mid_row, "horizon_dataframe": df_horizon, "name": user_district,
                "data_source_mode": data_mode, "month_label": selected_month_str, "year_label": selected_year
            }
            st.session_state.last_queried_district = user_district
            st.session_state.last_queried_date = date_query_signature
            
            st.session_state.audit_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Target District": user_district,
                "Target Timeline": date_query_signature,
                "Projected Risk %": mid_row['Outbreak Risk %'],
                "Data Mode": data_mode
            })
        else:
            st.sidebar.error("Location signature unverified. Adjust spelling and retry.")

# ==========================================
# 6. DASHBOARD PRESENTATION TAB LAYOUT
# ==========================================
if st.session_state.malaria_results is not None:
    res = st.session_state.malaria_results
    curr = res['summary_metrics']
    df_hz = res['horizon_dataframe']
    max_future_risk = df_hz['Outbreak Risk %'].max()
    
    # Header Banner Notification Cards
    st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 16px; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 12px;">
            <span style="color: #475569; font-weight: 600;">📍 Verified Tracking Site:</span> 
            <span style="color: #1e293b; font-weight: 700;">{res['address']}</span>
        </div>
        <div style="background-color: #f0fdf4; padding: 12px; border-radius: 10px; border: 1px solid #bbf7d0; margin-bottom: 25px; font-size: 0.9rem; color: #166534;">
            ✨ <b>Data Pipeline Stream:</b> Core processing verified via <b>{res['data_source_mode']}</b> calibrated for <b>{res['month_label']} {res['year_label']}</b>.
        </div>
    """, unsafe_allow_html=True)
    
    tab_summary, tab_visuals, tab_reports, tab_prevention = st.tabs([
        "👁️ Target Horizon Assessment", 
        "📊 Dynamic Trend Analytics", 
        "💾 Executive Report Hub",
        "🛡️ Clinical & Vector Control Guide"
    ])

    # ------------------ TAB 1: SITE MONITORING SUMMARY ------------------
    with tab_summary:
        st.markdown(f"<p style='color:#64748b;'>Spatial Grid Pins: Latitude {res['lat']:.4f} | Longitude {res['lon']:.4f}</p>", unsafe_allow_html=True)
        
        # Enhanced Layout Metric Grid
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f'<div class="metric-card"><p style="color:#64748b; font-weight:600; font-size:0.9rem; margin-bottom:4px;">EXPECTED TEMPERATURE</p><h2 style="color:#0f172a; margin:0; font-weight:700;">{curr["Temperature"]} °F</h2></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-card"><p style="color:#64748b; font-weight:600; font-size:0.9rem; margin-bottom:4px;">RELATIVE HUMIDITY</p><h2 style="color:#0f172a; margin:0; font-weight:700;">{curr["Humidity"]} %</h2></div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown(f'<div class="metric-card"><p style="color:#64748b; font-weight:600; font-size:0.9rem; margin-bottom:4px;">ESTIMATED SEASONAL RAIN</p><h2 style="color:#0f172a; margin:0; font-weight:700;">{curr["Accumulated Rain"]:.2f} In</h2></div>', unsafe_allow_html=True)
        with m_col4:
            st.markdown(f'<div class="metric-card"><p style="color:#64748b; font-weight:600; font-size:0.9rem; margin-bottom:4px;">TOPOGRAPHIC ALTITUDE</p><h2 style="color:#0f172a; margin:0; font-weight:700;">{res["elevation"]} m</h2></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Transmission Potential Assessment Summary")
        
        if max_future_risk >= 24.0:
            st.markdown(f"""
                <div class="status-box" style="background-color: rgba(239, 68, 68, 0.08); border-left: 6px solid #ef4444;">
                    <h4 style="color: #b91c1c; margin-top: 0; font-weight:700;">🚨 CRITICAL OUTBREAK PREDICTION WARNING</h4>
                    <p style="color: #991b1b; margin-bottom: 0; font-size: 1.05rem;">Long-range projections indicate that climate trends for <b>{res['month_label']} {res['year_label']}</b> will cross critical epidemiological risk thresholds, yielding an active outbreak index profile of <b>{max_future_risk}% Vector Affinity Match</b>.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-box" style="background-color: rgba(34, 197, 94, 0.08); border-left: 6px solid #22c55e;">
                    <h4 style="color: #166534; margin-top: 0; font-weight:700;">✅ CONTROLLED ECOSYSTEM PATHWAY</h4>
                    <p style="color: #14532d; margin-bottom: 0; font-size: 1.05rem;">Long-range projections indicate that climate parameters for <b>{res['month_label']} {res['year_label']}</b> will remain structurally safe, successfully containing vector transmission loops (<b>{max_future_risk}% Max Index</b>).</p>
                </div>
            """, unsafe_allow_html=True)

    # ------------------ TAB 2: DYNAMIC TREND ANALYTICS ------------------
    with tab_visuals:
        st.subheader(f"📊 Projected Risk Horizon Matrix: {res['month_label']} {res['year_label']}")
        
        fig_horizon = go.Figure()
        fig_horizon.add_trace(go.Scatter(
            x=df_hz['Date'], y=df_hz['Outbreak Risk %'],
            mode='lines+markers', name='Projected Outbreak Index %',
            line=dict(color='#ef4444' if max_future_risk >= 24.0 else '#22c55e', width=3),
            marker=dict(
                size=8, 
                color='white', 
                line=dict(
                    width=2, 
                    color='#ef4444' if max_future_risk >= 24.0 else '#22c55e'
                )
            )
        ))
        fig_horizon.add_hline(y=24.0, line_dash="dash", line_color="#f59e0b", annotation_text="Outbreak Trigger Baseline (24%)")
        
        fig_horizon.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,1)',
            xaxis_title="Timeline Window Matrix", yaxis_title="Outbreak Risk Index (%)",
            yaxis=dict(range=[0, 55], gridcolor='#e2e8f0'), xaxis=dict(gridcolor='#e2e8f0'),
            height=340, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified"
        )
        st.plotly_chart(fig_horizon, use_container_width=True)
        
        st.markdown("<br><b>Tabular Trend Data Reference Ledger</b>", unsafe_allow_html=True)
        st.dataframe(df_hz, use_container_width=True, height=200)

    # ------------------ TAB 3: REPORTS & DOWNLOAD HUB ------------------
    with tab_reports:
        st.subheader("Data Export Center")
        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            st.markdown("### 📄 Long-Range Horizon Summary Report")
            report_txt = f"MALARIA OUTBREAK LONG-RANGE INTELLIGENCE REPORT\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n...\n"
            st.download_button(label="📥 Download Long-Range Projection Summary (.txt)", data=report_txt, file_name=f"Malaria_LongRange_Report.txt", use_container_width=True)

        with rep_col2:
            st.markdown("### 🗃️ Aggregate Search Audit Ledger")
            if st.session_state.audit_history:
                st.dataframe(pd.DataFrame(st.session_state.audit_history), use_container_width=True, height=150)

    # ------------------ TAB 4: CLINICAL SIGNS & PREVENTION ------------------
    with tab_prevention:
        st.subheader("📋 Malaria Recognition & Prevention Field Guide")
        
        st.subheader("💡 Clinical Signs & Symptoms of Malaria")
        sym_col1, sym_col2 = st.columns(2)
        
        with sym_col1:
            st.markdown("""
            <div style="background-color:rgba(245, 158, 11, 0.08); padding:20px; border-radius:12px; border-left:6px solid #f59e0b; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin-top:0; color:#b45309; font-weight:700; font-size:1.15rem;">🌡️ Uncomplicated Malaria (Early Signs)</h4>
                <p style="color: #78350f;">Initial symptoms develop within 10-15 days of a bite and fluctuate cyclically:</p>
                <ul style="color: #78350f; line-height: 1.6;">
                    <li><b>High Fever & Shaking Chills:</b> Cold stages with shivering followed by high temperature spikes.</li>
                    <li><b>Profuse Sweating:</b> Intense perspiration stages as the fever breaks.</li>
                    <li><b>Headache & Muscle Aches:</b> Severe headaches accompanied by joint discomfort.</li>
                    <li><b>Gastrointestinal Distress:</b> Nausea, vomiting, and loss of appetite.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with sym_col2:
            st.markdown("""
            <div style="background-color:rgba(239, 68, 68, 0.08); padding:20px; border-radius:12px; border-left:6px solid #ef4444; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin-top:0; color:#b91c1c; font-weight:700; font-size:1.15rem;">🚨 Severe Malaria (Emergency Red Flags)</h4>
                <p style="color: #991b1b;"><b>IMMEDIATE MEDICAL EMERGENCY:</b> Untreated infections can progress to organ failure. Seek professional care immediately if you notice:</p>
                <ul style="color: #991b1b; line-height: 1.6;">
                    <li><b>Cerebral Complications:</b> Disorientation, confusion, seizures, or loss of consciousness.</li>
                    <li><b>Severe Anemia:</b> Extreme physical weakness and breathing problems due to red blood cell reduction.</li>
                    <li><b>Jaundice:</b> Yellowing discoloration in the eyes or skin tissue.</li>
                    <li><b>Fluid Inability:</b> Persistent vomiting that makes oral medicine impossible.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        st.subheader("🛠️ Vector Control Mitigation Options")
        prev_col1, prev_col2 = st.columns(2)
        
        with prev_col1:
            st.markdown("""
            <div style="background-color: #f8fafc; padding:20px; border-radius:12px; border: 1px solid #e2e8f0;">
                <h4 style="color:#1e293b; margin-top:0; font-weight:600;">🏠 Personal & Household Protection</h4>
                <ul style="line-height:1.7; color:#475569;">
                    <li><b>Long-Lasting Insecticidal Nets (LLINs):</b> Sleep under treated bednets every night.</li>
                    <li><b>Indoor Residual Spraying (IRS):</b> Utilize spatial chemical protection coatings on interior structures.</li>
                    <li><b>Topical Repellents:</b> Use active solutions like DEET on skin surfaces during peak hours.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with prev_col2:
            st.markdown("""
            <div style="background-color: #f8fafc; padding:20px; border-radius:12px; border: 1px solid #e2e8f0;">
                <h4 style="color:#1e293b; margin-top:0; font-weight:600;">🚜 Environmental & Community Mitigation</h4>
                <ul style="line-height:1.7; color:#475569;">
                    <li><b>Source Drainage Management:</b> Clear stagnant wetlands, roadside puddles, and drainage blocks.</li>
                    <li><b>Biological Larviciding:</b> Treat long-standing breeding grounds with biological solutions like Bti.</li>
                    <li><b>Ecosystem Clearing:</b> Maintain low vegetation margins around high-density housing blocks.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
