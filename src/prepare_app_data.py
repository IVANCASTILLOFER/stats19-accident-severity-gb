"""Precalcula los agregados geográficos que usa la app (evita cargar millones de filas al abrirla).

Genera data/processed/grid_cells.parquet con una fila por celda de 0,05° (≈5,5 km de lado en latitud),
año y franja horaria: nº de accidentes, de graves o mortales y de mortales. Así la app puede filtrar por
periodo y por hora sin recalcular. Se mantiene también grid_stats.parquet (todos los años juntos).
"""
import numpy as np
import pandas as pd

from config import PROCESSED
from data import load_accidents

CELL = 0.05
BAND_LABELS = {0: "00–05 h", 1: "06–11 h", 2: "12–17 h", 3: "18–23 h"}


def build_cells(df: pd.DataFrame, cell: float = CELL) -> pd.DataFrame:
    g = df.dropna(subset=["Latitude", "Longitude"]).copy()
    g["lat_c"] = (g["Latitude"] // cell) * cell + cell / 2
    g["lon_c"] = (g["Longitude"] // cell) * cell + cell / 2
    hour = pd.to_numeric(g["Time"].astype(str).str[:2], errors="coerce")
    g["band"] = (hour // 6).clip(0, 3).fillna(-1).astype("int8")
    g = g[g["band"] >= 0]
    g["ksi"] = (g["Accident_Severity"] != "Slight").astype("int32")
    g["fatal"] = (g["Accident_Severity"] == "Fatal").astype("int32")
    out = (g.groupby(["lat_c", "lon_c", "Year", "band"], observed=True)
             .agg(n=("ksi", "size"), ksi=("ksi", "sum"), fatal=("fatal", "sum")).reset_index()
             .rename(columns={"Year": "year"}))
    out["year"] = out["year"].astype("int16")
    for c in ("n", "ksi", "fatal"):
        out[c] = out[c].astype("int32")
    return out


if __name__ == "__main__":
    df = load_accidents()
    cells = build_cells(df)
    cells.to_parquet(PROCESSED / "grid_cells.parquet", index=False)
    tot = cells.groupby(["lat_c", "lon_c"], as_index=False)[["n", "ksi", "fatal"]].sum()
    tot["ksi_rate"] = tot["ksi"] / tot["n"]
    tot[tot["n"] >= 30].reset_index(drop=True).to_parquet(PROCESSED / "grid_stats.parquet", index=False)
    print("grid_cells:", cells.shape, "| años:", int(cells.year.min()), "-", int(cells.year.max()),
          "| accidentes:", int(cells.n.sum()))
