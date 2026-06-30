import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from geopy.geocoders import Nominatim

app = FastAPI(title="Malaria Outbreak Intelligence Engine Backend")

# Enable robust cross-origin resource sharing for smooth frontend synchronization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    district: str
    year: int
    month: int

def get_district_coordinates(location_string: str):
    """
    Translates free-text inputs into accurate geographic spatial parameters
    using the Nominatim geocoding engine.
    """
    try:
        # Initializing clean user agent matching global registration standards
        geolocator = Nominatim(user_agent="malaria_early_warning_system_2026")
        location = geolocator.geocode(location_string, timeout=6)
        if location:
            return location.latitude, location.longitude, location.address
    except Exception:
        pass
    
    # Context-aware structural defaults for Soroti if the geolocator times out
    if "soroti" in location_string.lower():
        return 1.7879, 33.5007, "Soroti, Eastern Region, Uganda"
    
    # Universal fallback coordinates
    return 1.7879, 33.5007, f"{location_string} (Fallback Coordinates Active)"

def generate_target_climate_matrix(lat: float, lon: float, target_date: datetime.date):
    """
    Queries real-world historical archives and sub-seasonal weather engines.
    Switches dynamically to reanalysis engines if requested target windows extend into past years.
    """
    today = datetime.now().date()
    start_date = datetime(target_date.year, target_date.month, 1).date()
    # Calculate target month endpoint boundary
    if target_date.month == 12:
        end_date = datetime(target_date.year, 12, 31).date()
    else:
        end_date = (datetime(target_date.year, target_date.month + 1, 1) - timedelta(days=1)).date()
        
    total_days = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(total_days)]
    
    # Base topographic evaluations derived from geographic coordinates
    equator_proximity = max(0, 1 - (abs(lat) / 90.0))
    base_elevation = max(100.0, 1173.18 - (abs(lat) * 15))
    
    # Check if target date requires historical archive API or standard forecast API
    is_historical = start_date < (today - timedelta(days=14))
    
    if is_historical:
        # Pull authentic historical records from the open-meteo archive engine
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
        )
    else:
        # Pull live operational forecasting windows
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
            f"&start_date={start_date if start_date >= today else today}"
            f"&end_date={end_date if end_date >= today else today + timedelta(days=7)}"
            f"&temperature_unit=fahrenheit&precipitation_unit=inch&timezone=auto"
        )

    headers = {'User-Agent': 'MalariaOutbreakForecaster/6.0'}
    try:
        response = requests.get(weather_url, headers=headers, timeout=7)
        if response.status_code == 200:
            daily_data = response.json().get('daily', {})
            if 'time' in daily_data and len(daily_data['time']) > 0:
                df_climate = pd.DataFrame({
                    'date': pd.to_datetime(daily_data.get('time')),
                    'temp_mean': daily_data.get('temperature_2m_mean'),
                    'humidity_mean': daily_data.get('relative_humidity_2m_mean'),
                    'precipitation': daily_data.get('precipitation_sum')
                })
                # Interpolate missing atmospheric signals cleanly via forward/backward fills
                df_climate['temp_mean'] = df_climate['temp_mean'].bfill().ffill().fillna(82.5)
                df_climate['humidity_mean'] = df_climate['humidity_mean'].bfill().ffill().fillna(78.0)
                df_climate['precipitation'] = df_climate['precipitation'].bfill().ffill().fillna(0.1)
                
                source_label = "Live Open-Meteo Climate Engine Connection" if not is_historical else "Historical Weather Reanalysis Registry"
                return df_climate, float(base_elevation), source_label
    except Exception:
        pass
            
    # Premium statistical fallback matrix generation used only if web API streams completely fail
    month_factor = np.sin((target_date.month - 1) * (np.pi / 6.0))
    seasonal_rain_modifier = max(0.05, 0.25 + (month_factor * 0.20))
    calculated_temp = 72.0 + (equator_proximity * 18.0) - (month_factor * 2.0)
    calculated_humidity = 60.0 + (equator_proximity * 25.0) + (month_factor * 5.0)
    
    df_climate = pd.DataFrame({
        'date': pd.to_datetime(date_range),
        'temp_mean': [calculated_temp + np.sin(i)*1.2 for i in range(total_days)],
        'humidity_mean': [min(100.0, calculated_humidity + np.cos(i)*2) for i in range(total_days)],
        'precipitation': [max(0.0, seasonal_rain_modifier + 0.1) if i % 3 == 0 else 0.0 for i in range(total_days)]
    })
    
    return df_climate, float(base_elevation), "Local Sub-Seasonal Structural Models (API Offline Fallback)"

def calculate_predictive_horizon_risk(df_climate: pd.DataFrame, elevation: float) -> List[Dict[str, Any]]:
    """
    Evaluates ecological indicators to construct vector affinity metrics matching 0-100% boundaries.
    """
    timeline_records = []
    
    for i, row in df_climate.iterrows():
        target_date = row['date']
        past_window = df_climate[df_climate['date'] <= target_date]
        
        # Track 14-day trailing rainfall to calculate breeding ground potential
        cumulative_rain = past_window['precipitation'].tail(14).sum()
            
        T = row['temp_mean']
        H = row['humidity_mean']
        
        # 1. Temperature Vector Score Calculation
        if 64.4 <= T <= 104.0:
            parasite_incubation_speed = 111.0 / (T - 64.4)
            t_score = 1.0 - min(1.0, (parasite_incubation_speed / 30.0))
        else:
            t_score = 0.0
            
        # 2. Humidity Vector Score Evaluation
        h_score = 1.0 / (1.0 + np.exp(-0.15 * (H - 60.0)))
        
        # 3. Non-linear Precipitation Vector Score Matching
        if cumulative_rain == 0:
            r_score = 0.0
        elif cumulative_rain > 8.5:
            r_score = 0.20  # Heavy downpours wash away larvae fields
        else:
            r_score = 1.0 - (((cumulative_rain - 3.5) / 5.0) ** 2)
            r_score = max(0.0, min(1.0, r_score))
            
        # 4. Topographic Altitude Influence Factor
        if elevation >= 1800.0:
            e_factor = 0.02
        elif elevation <= 600.0:
            e_factor = 1.0
        else:
            e_factor = 1.0 - ((elevation - 600.0) / 1200.0)
            
        # Weighted affinity formula synthesis
        raw_affinity = (t_score * 0.35) + (h_score * 0.25) + (r_score * 0.25) + (e_factor * 0.15)
        
        # FIXED: Multiplied by 100 to scale across a full 0-100% risk axis to respect the 60% UI threshold
        risk_percentage = float(raw_affinity * 100.0)
        
        # Hard limits based on biological containment realities
        if elevation >= 1800.0 or T < 61.0 or H < 45.0:
            risk_percentage = min(risk_percentage, 5.0)
        elif cumulative_rain == 0.0 and H < 52.0:
            risk_percentage = min(risk_percentage, 6.0)
            
        timeline_records.append({
            'Date': target_date.strftime('%Y-%m-%d'),
            'Temperature': round(T, 1),
            'Humidity': round(H, 1),
            'Accumulated_Rain': round(cumulative_rain, 2),
            'Outbreak_Risk_Percent': round(max(0.0, min(100.0, risk_percentage)), 1)
        })
        
    return timeline_records

@app.post("/api/analyze")
def analyze_outbreak_risk(payload: AnalysisRequest):
    # Geocode free-text strings to retrieve authentic spatial metadata
    lat, lon, full_address = get_district_coordinates(payload.district)
    
    try:
        target_date = datetime(payload.year, payload.month, 15).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid calendar structure selection parameters.")
        
    # Extract live climate data streams from web servers
    df_climate, elevation, data_mode = generate_target_climate_matrix(lat, lon, target_date)
    
    # Parse dataframes through epidemiology tracking layers
    horizon_records = calculate_predictive_horizon_risk(df_climate, elevation)
    
    # Calculate mid-month index summary points and monthly peak markers
    mid_row = horizon_records[len(horizon_records) // 2]
    max_future_risk = max([day['Outbreak_Risk_Percent'] for day in horizon_records])
    
    # Return structured JSON payload to the user dashboard interface
    return {
        "status": "success",
        "location": { 
            "address": full_address, 
            "lat": lat, 
            "lon": lon, 
            "elevation": round(elevation, 3) 
        },
        "metadata": { 
            "data_source_mode": data_mode, 
            "target_period": target_date.strftime("%Y-%m") 
        },
        "assessment": {
            "max_future_risk": max_future_risk,
            "summary_metrics": mid_row
        },
        "horizon_dataframe": horizon_records
    }
