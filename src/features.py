"""Ingeniería de variables sin fuga: todo lo aprendido se ajusta SOLO con entrenamiento."""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from config import MODELS, PROCESSED, SEED
from data import load_accidents
from vehicles import VEHICLE_COLUMNS, load_vehicle_agg

TRAIN_YEARS = range(2005, 2015)   # 2005-2014
VAL_YEARS = [2015]
TEST_YEARS = [2016, 2017]
# Conjuntos de evaluación posteriores: los modelos se congelan antes de verlos.
EXTRA_SETS = {
    "holdout": [2018, 2019],          # prueba independiente, antes de la pandemia
    "estres": [2020, 2021],           # pandemia: prueba de estrés (no se entrena ni se ajusta con ella)
    "post": [2022, 2023, 2024, 2025], # recuperación posterior
}

CELL = 0.05                       # rejilla ~5 km
N_CLUSTERS = 60
LONDON = (51.5074, -0.1278)
SMOOTH_M = 20                     # suavizado de la tasa por celda

CATEGORICAL = [
    "Road_Type", "1st_Road_Class", "Junction_Detail", "Junction_Control",
    "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions",
    "Urban_or_Rural_Area", "Pedestrian_Crossing",
]   # Police_Force se excluye: los códigos de Escocia cambiaron en 2019
NUMERIC_BASE = ["Latitude", "Longitude", "Speed_limit", "Number_of_Vehicles", "Injury_Based"]


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def temporal_split(df: pd.DataFrame):
    return (df[df["Year"].isin(TRAIN_YEARS)].copy(),
            df[df["Year"].isin(VAL_YEARS)].copy(),
            df[df["Year"].isin(TEST_YEARS)].copy())


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables que no aprenden nada del conjunto (seguras en cualquier partición)."""
    out = df.copy()
    hour = pd.to_numeric(out["Time"].astype(str).str[:2], errors="coerce")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out["is_dark"] = out["Light_Conditions"].astype(str).str.startswith("Darkness").astype("int8")
    lat, lon = np.radians(out["Latitude"]), np.radians(out["Longitude"])
    la0, lo0 = np.radians(LONDON[0]), np.radians(LONDON[1])
    a = np.sin((lat - la0) / 2) ** 2 + np.cos(lat) * np.cos(la0) * np.sin((lon - lo0) / 2) ** 2
    out["dist_london_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    out["ksi"] = (out["Accident_Severity"] != "Slight").astype("int8")
    out["severity"] = out["Accident_Severity"].map({"Slight": 0, "Serious": 1, "Fatal": 2}).astype("int8")
    return out


def _cell_id(df: pd.DataFrame) -> pd.Series:
    la = np.floor(df["Latitude"] / CELL).astype("int64")
    lo = np.floor(df["Longitude"] / CELL).astype("int64") + 1000
    return la * 10_000 + lo


class GeoFeatures:
    """Clusters, densidad y tasa de gravedad por celda; se ajusta con entrenamiento."""

    def fit(self, train: pd.DataFrame):
        self.kmeans = MiniBatchKMeans(N_CLUSTERS, random_state=SEED, n_init=3, batch_size=20_000)
        self.kmeans.fit(self._xy(train))
        cid = _cell_id(train)
        years = train["Year"].nunique()
        self.global_rate = train["ksi"].mean()
        agg = train.assign(cell=cid).groupby("cell").agg(n=("ksi", "size"), k=("ksi", "sum"))
        self.cell_count = (agg["n"] / years).to_dict()
        self.cell_n, self.cell_k = agg["n"].to_dict(), agg["k"].to_dict()
        # tasa por celda y año, para la versión leave-one-year-out
        by_year = train.assign(cell=cid).groupby(["cell", "Year"]).agg(n=("ksi", "size"), k=("ksi", "sum"))
        self._by_year = by_year
        return self

    @staticmethod
    def _xy(df):
        return np.c_[df["Latitude"].values, df["Longitude"].values * np.cos(np.radians(54.0))]

    def _rate(self, n, k):
        return (k + SMOOTH_M * self.global_rate) / (n + SMOOTH_M)

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        out = df.copy()
        cid = _cell_id(out)
        out["geo_cluster"] = pd.Categorical(self.kmeans.predict(self._xy(out)), categories=range(N_CLUSTERS))
        out["cell_density"] = cid.map(self.cell_count).fillna(0.0)
        n = cid.map(self.cell_n).fillna(0).astype(float)
        k = cid.map(self.cell_k).fillna(0).astype(float)
        if is_train:  # quitar el año de la propia fila: sin ver su propia etiqueta
            idx = pd.MultiIndex.from_arrays([cid, out["Year"]])
            n_y = self._by_year["n"].reindex(idx).fillna(0).values
            k_y = self._by_year["k"].reindex(idx).fillna(0).values
            n, k = n - n_y, k - k_y
        out["cell_ksi_rate"] = self._rate(n, k)
        return out


def feature_columns(df: pd.DataFrame):
    num = NUMERIC_BASE + ["hour_sin", "hour_cos", "dayofweek", "month", "is_weekend",
                          "is_dark", "dist_london_km", "cell_density", "cell_ksi_rate"]
    cat = CATEGORICAL + ["geo_cluster"]
    return num, cat


def build_features(save: bool = True):
    df = load_accidents().dropna(subset=["Latitude", "Longitude"])
    df = df.merge(load_vehicle_agg(), on="Accident_Index", how="left")
    df["has_vehicle_info"] = df["has_vehicle_info"].fillna(0).astype("int8")
    df = basic_features(df)
    train, val, test = temporal_split(df)
    extras = {k: df[df["Year"].isin(v)].copy() for k, v in EXTRA_SETS.items()}
    geo = GeoFeatures().fit(train)
    train, val, test = geo.transform(train, is_train=True), geo.transform(val), geo.transform(test)
    extras = {k: geo.transform(v) for k, v in extras.items()}
    num, cat = feature_columns(train)
    keep = num + cat + VEHICLE_COLUMNS + ["ksi", "severity", "Year", "Adj_Serious", "Adj_Slight"]
    parts = {}
    for name, d in [("train", train), ("val", val), ("test", test)] + list(extras.items()):
        x = d[keep].copy()
        x.columns = [_safe(c) for c in x.columns]
        parts[name] = x.reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        MODELS.mkdir(parents=True, exist_ok=True)
        for name, x in parts.items():
            x.to_parquet(PROCESSED / f"features_{name}.parquet", index=False)
        joblib.dump(geo, MODELS / "geo_features.joblib")
    return parts, [_safe(c) for c in num], [_safe(c) for c in cat]


def run():
    """Ejecutar con:  python -c "import features; features.run()"  (así GeoFeatures se guarda bajo el módulo features)."""
    parts, num, cat = build_features()
    for n, x in parts.items():
        print(f"{n:5s} {x.shape}  años {x.Year.min()}-{x.Year.max()}  "
              f"KSI={x.ksi.mean():.4f}  Fatal={(x.severity == 2).mean():.4f}")
    print("numéricas:", num)
    print("categóricas:", cat)
    t = parts["train"]
    print(t[["cell_density", "cell_ksi_rate", "dist_london_km"]].describe().round(3))
    print("nulos train (>0):", t.isna().mean()[t.isna().mean() > 0].round(4).to_dict())
