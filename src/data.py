"""Carga de datos de accidentes.

Fuente actual: archivo oficial del DfT (2005–2025), procesado por src/dft.py.
La función `build_accidents_parquet` conserva la construcción histórica con el conjunto de Kaggle
(solo para reproducir los resultados archivados en archivo_kaggle_2005_2017/).
"""
import pandas as pd

from config import (ACCIDENT_COLUMNS, ACCIDENTS_CSV, ACCIDENTS_KAGGLE_PARQUET, ACCIDENTS_PARQUET, PROCESSED)


def build_accidents_parquet(chunksize: int = 500_000) -> pd.DataFrame:
    """[Histórico] Lee el CSV de Kaggle por bloques, conserva columnas útiles y guarda Parquet."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    parts = []
    for chunk in pd.read_csv(ACCIDENTS_CSV, usecols=ACCIDENT_COLUMNS, chunksize=chunksize, low_memory=False):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in df.select_dtypes(["object", "string"]).columns:
        if col not in ("Accident_Index", "Time"):
            df[col] = df[col].astype("category")
    df.to_parquet(ACCIDENTS_KAGGLE_PARQUET, index=False)
    return df


def load_accidents() -> pd.DataFrame:
    """Carga los accidentes oficiales 2005–2025; si el Parquet no existe, lo construye desde el CSV del DfT."""
    if not ACCIDENTS_PARQUET.exists():
        from dft import build_collisions
        return build_collisions()
    return pd.read_parquet(ACCIDENTS_PARQUET)


if __name__ == "__main__":
    d = load_accidents()
    print(d.shape)
    print("Años:", int(d["Year"].min()), "-", int(d["Year"].max()))
    print(d["Accident_Severity"].value_counts(normalize=True).round(4))
    print("Nulos lat/lon:", int(d["Latitude"].isna().sum()))
