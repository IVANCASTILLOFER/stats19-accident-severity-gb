"""Configuración central del proyecto: rutas, semilla y columnas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
MAPS = ROOT / "reports" / "maps"

ACCIDENTS_CSV = RAW / "Accident_Information.csv"
VEHICLES_CSV = RAW / "Vehicle_Information.csv"
ACCIDENTS_PARQUET = PROCESSED / "accidents_dft.parquet"          # fuente oficial DfT 2005–2025 (src/dft.py)
ACCIDENTS_KAGGLE_PARQUET = PROCESSED / "accidents.parquet"        # conjunto de Kaggle 2005–2017 (archivado)

SEED = 42
TARGET = "Accident_Severity"
CLASSES = ["Slight", "Serious", "Fatal"]

# Columnas de accidentes que se conservan (el resto no se usa).
ACCIDENT_COLUMNS = [
    "Accident_Index", "Accident_Severity", "Date", "Time", "Year", "Day_of_Week",
    "Latitude", "Longitude", "Police_Force", "Local_Authority_(District)",
    "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit",
    "Road_Type", "1st_Road_Class", "Junction_Detail", "Junction_Control",
    "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions",
    "Urban_or_Rural_Area", "Pedestrian_Crossing-Physical_Facilities",
    "Special_Conditions_at_Site", "Carriageway_Hazards",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]

# Variables que se conocen DESPUÉS del accidente: no usar como predictoras.
LEAKAGE_COLUMNS = ["Number_of_Casualties", "Did_Police_Officer_Attend_Scene_of_Accident"]
