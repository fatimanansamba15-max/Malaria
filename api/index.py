from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # <-- ADD THIS IMPORT LAYER

from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from typing import List, Dict, Any

app = FastAPI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 COPY AND PASTE THIS EXACT NEW ROUTE BLOCK 👇
@app.get("/")
def serve_frontend_dashboard():
    import os
    # Hunt down the HTML page whether it sits in root or fallback paths
    for path in ["index.html", "../index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Ecosystem mismatch: index.html missing from repository layout.")


audit_history: List[Dict[str, Any]] = []



