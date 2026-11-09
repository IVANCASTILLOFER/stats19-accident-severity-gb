"""Datos oficiales del DfT (STATS19): lectura, decodificación con la guía oficial y agregación de vehículos.

Entradas (data/oficial_dft/): colisiones y vehículos 1979–último año (CSV) y la guía de datos (xlsx).
Salidas (data/processed/): accidents_dft.parquet y vehicles_dft_agg.parquet.

El resultado conserva los nombres de columna del pipeline anterior (Accident_Index, Year, Accident_Severity,
Latitude, …) para reutilizar features.py, train.py, evaluate.py y la app.
Las etiquetas salen de la hoja «2024_code_list» de la guía oficial; no se escribe ninguna a mano.
"""
import numpy as np
import pandas as pd

from config import PROCESSED, ROOT
from xlsx_min import read_xlsx

OFICIAL = ROOT / "data" / "oficial_dft"
COLLISIONS_CSV = OFICIAL / "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
VEHICLES_CSV = OFICIAL / "dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv"
GUIDE_XLSX = OFICIAL / "data-guide-2025.xlsx"
ACCIDENTS_DFT = PROCESSED / "accidents_dft.parquet"
VEHICLES_DFT = PROCESSED / "vehicles_dft_agg.parquet"

YEAR_MIN, YEAR_MAX = 2005, 2025          # el DfT recomienda usar el ajuste de severidad desde 2005
MISSING_LABELS = {"Data missing or out of range"}
VALID_SPEEDS = {20, 30, 40, 50, 60, 70}

# Códigos de tipo de vehículo (guía oficial, hoja 2024_code_list, tabla vehicle / vehicle_type)
MOTORCYCLE = {2, 3, 4, 5, 23, 97, 103, 104, 105, 106}
PEDAL_CYCLE = {1}
BUS = {10, 11, 110}
HEAVY_GOODS = {20, 21, 98, 113}
VAN = {19}
AGE_MID = {1: 3.0, 2: 8.0, 3: 13.0, 4: 18.0, 5: 23.0, 6: 30.5, 7: 40.5, 8: 50.5, 9: 60.5, 10: 70.5, 11: 80.0}

VEHICLE_COLUMNS = [
    "has_vehicle_info", "veh_motorcycle_any", "veh_pedalcycle_any", "veh_bus_any", "veh_heavy_goods_any",
    "veh_van_any", "driver_age_min", "driver_age_max", "young_driver_any", "elderly_driver_any",
    "male_share", "veh_age_mean", "engine_cc_max",
]


def load_lookups() -> dict:
    """{(tabla, campo): {código: etiqueta}} desde la guía oficial."""
    rows = read_xlsx(GUIDE_XLSX)["2024_code_list"][1:]
    out = {}
    for r in rows:
        r = r + [""] * (5 - len(r))
        table_, field, code, label = r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip()
        if not field or code == "":
            continue
        try:
            code = int(float(code))
        except ValueError:
            continue
        out.setdefault((table_, field), {})[code] = label
    return out


def decode(series: pd.Series, mapping: dict) -> pd.Series:
    labels = series.map(mapping)
    labels = labels.where(~labels.isin(MISSING_LABELS))
    return labels.astype("category")


def build_collisions(save: bool = True) -> pd.DataFrame:
    lk = load_lookups()
    cols = ["collision_index", "collision_year", "longitude", "latitude", "police_force", "collision_severity",
            "number_of_vehicles", "number_of_casualties", "date", "time", "first_road_class", "road_type",
            "speed_limit", "junction_detail", "junction_control", "pedestrian_crossing", "light_conditions",
            "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
            "did_police_officer_attend_scene_of_accident", "collision_injury_based",
            "collision_adjusted_severity_serious", "collision_adjusted_severity_slight"]
    d = pd.read_csv(COLLISIONS_CSV, usecols=cols, low_memory=False)
    d = d[d["collision_year"].between(YEAR_MIN, YEAR_MAX)].copy()
    c = lambda f: lk[("collision", f)]
    out = pd.DataFrame({
        "Accident_Index": d["collision_index"].astype(str),
        "Year": d["collision_year"].astype("int16"),
        "Accident_Severity": decode(d["collision_severity"], c("collision_severity")),
        "Date": pd.to_datetime(d["date"], format="%d/%m/%Y", errors="coerce"),
        "Time": d["time"].astype(str),
        "Latitude": d["latitude"], "Longitude": d["longitude"],
        "Police_Force": d["police_force"].astype("int16"),
        "Number_of_Vehicles": d["number_of_vehicles"], "Number_of_Casualties": d["number_of_casualties"],
        "Speed_limit": d["speed_limit"].where(d["speed_limit"].isin(VALID_SPEEDS)).astype("float32"),
        "Road_Type": decode(d["road_type"], c("road_type")),
        "1st_Road_Class": decode(d["first_road_class"], c("first_road_class")),
        "Junction_Detail": decode(d["junction_detail"], c("junction_detail")),
        "Junction_Control": decode(d["junction_control"], c("junction_control")),
        "Pedestrian_Crossing": decode(d["pedestrian_crossing"], c("pedestrian_crossing")),
        "Light_Conditions": decode(d["light_conditions"], c("light_conditions")),
        "Weather_Conditions": decode(d["weather_conditions"], c("weather_conditions")),
        "Road_Surface_Conditions": decode(d["road_surface_conditions"], c("road_surface_conditions")),
        "Urban_or_Rural_Area": decode(d["urban_or_rural_area"], c("urban_or_rural_area")),
        "Did_Police_Officer_Attend_Scene_of_Accident": d["did_police_officer_attend_scene_of_accident"],
        "Injury_Based": (d["collision_injury_based"] == 1).astype("int8"),
        "Adj_Serious": d["collision_adjusted_severity_serious"].astype("float32"),
        "Adj_Slight": d["collision_adjusted_severity_slight"].astype("float32"),
    })
    out["Accident_Severity"] = out["Accident_Severity"].astype(str).astype("category")
    out = out.reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(ACCIDENTS_DFT, index=False)
    return out


def build_vehicles(save: bool = True, chunksize: int = 2_000_000) -> pd.DataFrame:
    cols = ["collision_index", "collision_year", "vehicle_type", "age_band_of_driver", "sex_of_driver",
            "engine_capacity_cc", "age_of_vehicle"]
    parts = []
    for ch in pd.read_csv(VEHICLES_CSV, usecols=cols, chunksize=chunksize, low_memory=False):
        ch = ch[ch["collision_year"].between(YEAR_MIN, YEAR_MAX)]
        if len(ch):
            parts.append(ch)
    v = pd.concat(parts, ignore_index=True)
    v["collision_index"] = v["collision_index"].astype(str)
    vt = v["vehicle_type"]
    v["motorcycle"] = vt.isin(MOTORCYCLE)
    v["pedalcycle"] = vt.isin(PEDAL_CYCLE)
    v["bus"] = vt.isin(BUS)
    v["heavy"] = vt.isin(HEAVY_GOODS)
    v["van"] = vt.isin(VAN)
    v["age_mid"] = v["age_band_of_driver"].map(AGE_MID)
    v["male"] = v["sex_of_driver"].map({1: 1.0, 2: 0.0})
    v["eng"] = v["engine_capacity_cc"].where(v["engine_capacity_cc"].between(50, 10_000))
    v["vage"] = v["age_of_vehicle"].where(v["age_of_vehicle"].between(0, 60))
    g = v.groupby("collision_index")
    agg = pd.DataFrame({
        "veh_motorcycle_any": g["motorcycle"].max().astype("int8"),
        "veh_pedalcycle_any": g["pedalcycle"].max().astype("int8"),
        "veh_bus_any": g["bus"].max().astype("int8"),
        "veh_heavy_goods_any": g["heavy"].max().astype("int8"),
        "veh_van_any": g["van"].max().astype("int8"),
        "driver_age_min": g["age_mid"].min(), "driver_age_max": g["age_mid"].max(),
        "male_share": g["male"].mean(), "veh_age_mean": g["vage"].mean(), "engine_cc_max": g["eng"].max(),
    })
    agg["young_driver_any"] = (agg["driver_age_min"] <= 25).astype("int8")
    agg["elderly_driver_any"] = (agg["driver_age_max"] > 75).astype("int8")
    agg["has_vehicle_info"] = np.int8(1)
    agg = agg.reset_index().rename(columns={"collision_index": "Accident_Index"})
    if save:
        agg.to_parquet(VEHICLES_DFT, index=False)
    return agg


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "collisions"):
        a = build_collisions()
        print("colisiones:", a.shape, "| años", int(a.Year.min()), "-", int(a.Year.max()))
        print("gravedad:", a.Accident_Severity.value_counts(normalize=True).round(4).to_dict())
        print("nulos (%):", (a.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict())
        print("Injury_Based por año (%):", (a.groupby("Year")["Injury_Based"].mean() * 100).round(1).to_dict())
    if what in ("all", "vehicles"):
        v = build_vehicles()
        print("vehículos agregados:", v.shape)
