"""Análisis geoespacial: mapas de calor y puntos negros (hotspots) con Folium."""
import branca.colormap as cm
import folium
import pandas as pd
from folium.plugins import HeatMap

from config import MAPS, SEED

UK_CENTER = [54.0, -2.5]


def _base_map(zoom: int = 6) -> folium.Map:
    return folium.Map(location=UK_CENTER, zoom_start=zoom, tiles="OpenStreetMap")


def heatmap(df: pd.DataFrame, filename: str, n: int = 100_000) -> str:
    """Mapa de calor de una muestra de accidentes."""
    MAPS.mkdir(parents=True, exist_ok=True)
    pts = df.dropna(subset=["Latitude", "Longitude"])
    if len(pts) > n:
        pts = pts.sample(n, random_state=SEED)
    m = _base_map()
    HeatMap(pts[["Latitude", "Longitude"]].values.tolist(),
            radius=6, blur=8, min_opacity=0.3).add_to(m)
    path = MAPS / filename
    m.save(str(path))
    return str(path)


def grid_stats(df: pd.DataFrame, cell: float = 0.05) -> pd.DataFrame:
    """Agrega accidentes en una rejilla de `cell` grados (~5 km)."""
    g = df.dropna(subset=["Latitude", "Longitude"]).copy()
    g["lat_c"] = (g["Latitude"] // cell) * cell + cell / 2
    g["lon_c"] = (g["Longitude"] // cell) * cell + cell / 2
    g["ksi"] = (g["Accident_Severity"] != "Slight").astype(int)
    g["fatal"] = (g["Accident_Severity"] == "Fatal").astype(int)
    out = (g.groupby(["lat_c", "lon_c"], observed=True)
             .agg(n=("ksi", "size"), ksi=("ksi", "sum"), fatal=("fatal", "sum"))
             .reset_index())
    out["ksi_rate"] = out["ksi"] / out["n"]
    return out


def _wilson_lower(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    return ((p + z ** 2 / (2 * n)) - z * (p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / den


def hotspots_map(df: pd.DataFrame, filename: str, cell: float = 0.05,
                 min_n: int = 300, top: int = 400) -> str:
    """Zonas con mayor PROPORCIÓN de graves+mortales (límite inferior del IC 95 % de Wilson, mínimo de accidentes
    por celda). No ordena por volumen: las celdas con más accidentes suelen ser las de más tráfico y no son más graves."""
    MAPS.mkdir(parents=True, exist_ok=True)
    cells = grid_stats(df, cell)
    cells = cells[cells["n"] >= min_n].copy()
    cells["lo"] = _wilson_lower(cells["ksi"], cells["n"])
    cells = cells.nlargest(top, "lo")
    cmap = cm.LinearColormap(["#fee08b", "#f46d43", "#a50026"],
                             vmin=float(cells["ksi_rate"].min()),
                             vmax=float(cells["ksi_rate"].max()),
                             caption="Tasa de accidentes graves o mortales")
    m = _base_map()
    for r in cells.itertuples():
        folium.CircleMarker(
            [r.lat_c, r.lon_c], radius=3 + (r.ksi ** 0.5) / 3,
            color=cmap(r.ksi_rate), fill=True, fill_opacity=0.7, weight=1,
            tooltip=f"{r.n} accidentes · {r.ksi} graves/mortales ({r.ksi_rate:.1%}) · {r.fatal} mortales",
        ).add_to(m)
    cmap.add_to(m)
    path = MAPS / filename
    m.save(str(path))
    return str(path)
