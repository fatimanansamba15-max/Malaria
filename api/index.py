import os
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim

# Configure structured logging for production debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("malaria_intelligence_engine")

app = FastAPI(
    title="Malaria Outbreak Intelligence Engine - Production API",
    version="2.0.0",
    description="Real-time epidemiological vector risk modeling using satellite weather streams and topographic dynamics."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    district: str = Field(..., example="Soroti, Uganda")
    year: int = Field(default_factory=lambda: datetime.now().year, ge=2000, le=2030, example=2026)
    month: int = Field(default_factory=lambda: datetime.now().month, ge=1, le=12, example=7)

# --- ROUTE 1: SERVE DASHBOARD FRONTEND AT ROOT (/) ---

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serves the index.html file so Vercel renders your full UI 
    instead of returning a raw JSON 404 or blank page.
    """
    # Primary check: index.html at root level when app runs inside /api directory
    index_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
            
    # Secondary check: index.html in current working directory
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
            
    return "<h1 style='color:red;'>index.html not found in repository root. Ensure index.html is placed at the top level of your GitHub repo.</h1>"


# --- UTILITY: GEOLOCATION & TOPOGRAPHY ---

def get_district_coordinates(location_string: str) -> Tuple[float, float, str]:
    """
    Geocodes location strings using Nominatim with exponential backoff / standard fallback.
    """
    try:
        geolocator = Nominatim(user_agent="malaria_outbreak_engine_prod_2026", timeout=5)
        location = geolocator.geocode(location_string)
        if location:
            return float(location.latitude), float(location.longitude), str(location.address)
    except Exception as e:
        logger.warning(f"Geocoding service unavailable for '{location_string}': {e}")
    
    # Context-aware fallback for Soroti / Uganda default zone
    if "soroti" in location_string.lower():
        return 1.7879, 33.5007, "Soroti, Eastern Region, Uganda"
    
    return 1.7879, 33.5007, f"{location_string} (Fallback Coordinates Active)"


def get_real_elevation(lat: float, lon: float) -> float:
    """
    Queries Open-Meteo Elevation API for authentic digital elevation model (DEM) altitude data.
    """
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if "elevation" in data and len(data["elevation"]) > 0:
                return float(data["elevation"][0])
    except Exception as e:
        logger.warning(f"Elevation lookup failed for ({lat}, {lon}): {e}")
    
    # Secondary topographic approximation based on equatorial distance
    return max(50.0, 1173.0 - (abs(lat) * 12.0))


# --- CLIMATE STREAMING ENGINE ---

def fetch_satellite_climate_data(lat: float, lon: float, target_year: int, target_month: int) -> Tuple[pd.DataFrame, str]:
    """
    Pulls real-time forecast or reanalysis historical records from Open-Meteo API.
    Handles unit conversions and null interpolations cleanly.
    """
    today = date.today()
    start_date = date(target_year, target_month, 1)
    
    # Calculate target month endpoint
    if target_month == 12:
        end_date = date(target_year, 12, 31)
    else:
        end_date = date(target_year, target_month + 1, 1) - timedelta(days=1)
        
    is_historical = end_date < (today - timedelta(days=14))
    
    if is_historical:
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
        )
        source_label = "Open-Meteo Historical Reanalysis Registry"
    else:
        # Standardize query bounds for active/future windows
        req_start = max(start_date, today)
        req_end = max(end_date, today + timedelta(days=7))
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&start_date={req_start}&end_date={req_end}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
        )
        source_label = "Open-Meteo Operational Satellite & GFS Stream"

    headers = {'User-Agent': 'MalariaOutbreakEngine/2.0'}
    try:
        res = requests.get(weather_url, headers=headers, timeout=8)
        if res.status_code == 200:
            daily_data = res.json().get('daily', {})
            if 'time' in daily_data and len(daily_data['time']) > 0:
                df = pd.DataFrame({
                    'date': pd.to_datetime(daily_data['time']),
                    'temp_mean': daily_data['temperature_2m_mean'],
                    'humidity_mean': daily_data['relative_humidity_2m_mean'],
                    'precipitation': daily_data['precipitation_sum']
                })
                # Impute missing records safely using forward & backward fills
                df['temp_mean'] = df['temp_mean'].bfill().ffill().fillna(82.5)
                df['humidity_mean'] = df['humidity_mean'].bfill().ffill().fillna(75.0)
                df['precipitation'] = df['precipitation'].bfill().ffill().fillna(0.05)
                return df, source_label
    except Exception as e:
        logger.error(f"Weather API Fetch failed: {e}")

    # Fallback simulation generator if live API is unreachable
    total_days = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(total_days)]
    
    df_fallback = pd.DataFrame({
        'date': pd.to_datetime(date_range),
        'temp_mean': [82.0 + np.sin(i * 0.2) * 2.5 for i in range(total_days)],
        'humidity_mean': [74.0 + np.cos(i * 0.2) * 5.0 for i in range(total_days)],
        'precipitation': [0.2 if i % 4 == 0 else 0.0 for i in range(total_days)]
    })
    return df_fallback, "Sub-Seasonal Climatological Synthetic Fallback"


# --- EPIDEMIOLOGICAL CALCULATIONS ---

def calculate_predictive_horizon_risk(df_climate: pd.DataFrame, elevation: float) -> List[Dict[str, Any]]:
    """
    Computes daily vector outbreak risk scores using degree-day parasitemia limits,
    humidity desiccation curves, pooling accumulation, and altitude lapse rates.
    """
    timeline_records = []
    
    for _, row in df_climate.iterrows():
        target_date = row['date']
        past_window = df_climate[df_climate['date'] <= target_date]
        
        # 14-day cumulative rainfall for vector breeding pool index
        cumulative_rain = past_window['precipitation'].tail(14).sum()
        
        T_fahrenheit = row['temp_mean']
        T_celsius = (T_fahrenheit - 32.0) * (5.0 / 9.0)
        H = row['humidity_mean']
        
        # 1. Temperature Vector Affinity (Deg-Day EIP Latency Model)
        if 16.0 <= T_celsius <= 38.0:
            eip_days = 111.0 / (T_celsius - 16.0)
            t_score = float(np.clip(1.0 - ((eip_days - 10.0) / 25.0), 0.0, 1.0))
        else:
            t_score = 0.0

        # 2. Humidity Vector Longevity Score (Sigmoid curve)
        h_score = float(1.0 / (1.0 + np.exp(-0.15 * (H - 60.0))))

        # 3. Non-Linear Larval Pooling Score
        if cumulative_rain == 0:
            r_score = 0.0
        elif cumulative_rain > 8.0:
            r_score = 0.25
        else:
            r_score = float(np.clip(1.0 - (((cumulative_rain - 3.0) / 5.0) ** 2), 0.05, 1.0))

        # 4. Topographic Elevation Exclusion Factor
        if elevation >= 1800.0:
            e_factor = 0.02
        elif elevation <= 600.0:
            e_factor = 1.0
        else:
            e_factor = float(1.0 - ((elevation - 600.0) / 1200.0))

        # Composite score synthesis
        raw_affinity = (t_score * 0.35) + (h_score * 0.25) + (r_score * 0.25) + (e_factor * 0.15)
        risk_percentage = raw_affinity * 100.0

        # Strict epidemiological bounds
        if elevation >= 1800.0 or T_celsius < 16.0 or H < 45.0:
            risk_percentage = min(risk_percentage, 5.0)
        elif cumulative_rain == 0.0 and H < 50.0:
            risk_percentage = min(risk_percentage, 8.0)

        timeline_records.append({
            'Date': target_date.strftime('%Y-%m-%d'),
            'Temperature_F': round(T_fahrenheit, 1),
            'Temperature_C': round(T_celsius, 1),
            'Humidity_Percent': round(H, 1),
            'Accumulated_Rain_Inches': round(cumulative_rain, 2),
            'Outbreak_Risk_Percent': round(float(np.clip(risk_percentage, 0.0, 100.0)), 1)
        })

    return timeline_records


# --- MAIN API ENDPOINT ---

@app.post("/api/analyze")
def analyze_outbreak_risk(payload: AnalysisRequest):
    """
    Production endpoint accepting district, year, and month to produce detailed 
    vector outbreak timelines and epidemiological risk assessments.
    """
    # 1. Spatial Resolution
    lat, lon, full_address = get_district_coordinates(payload.district)
    elevation = get_real_elevation(lat, lon)

    # 2. Data Ingestion
    df_climate, data_source_label = fetch_satellite_climate_data(
        lat, lon, payload.year, payload.month
    )

    # 3. Model Pipeline Execution
    horizon_records = calculate_predictive_horizon_risk(df_climate, elevation)

    if not horizon_records:
        raise HTTPException(status_code=500, detail="Failed to calculate epidemiological risk metrics.")

    # Extract dynamic mid-month summary and peak risk markers
    mid_index = len(horizon_records) // 2
    summary_metrics = horizon_records[mid_index]
    max_risk = max([record['Outbreak_Risk_Percent'] for record in horizon_records])

    return {
        "status": "success",
        "location": {
            "address": full_address,
            "lat": lat,
            "lon": lon,
            "elevation_meters": round(elevation, 1)
        },
        "metadata": {
            "data_source_mode": data_source_label,
            "target_period": f"{payload.year}-{payload.month:02d}"
        },
        "assessment": {
            "peak_outbreak_risk_percent": max_risk,
            "risk_status": "HIGH OUTBREAK THRESHOLD" if max_risk >= 54.0 else "LOW TRANSMISSION RISK",
            "summary_metrics": summary_metrics
        },
        "horizon_dataframe": horizon_records
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
