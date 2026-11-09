"""App Streamlit: estimador de severidad de accidentes de tráfico en Reino Unido.

Ejecutar desde la carpeta del proyecto:  streamlit run app/streamlit_app.py
Proyecto académico: no apto para decisiones reales de seguridad vial.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import branca.colormap as cm
import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

BASE_RATE_KSI = 0.179        # tasa graves+mortales observada en 2016-17
BASE_RATE_FATAL = 0.0126     # tasa de mortales observada en 2016-17
DEFAULT_POINT = (53.4808, -2.2426)   # Manchester
UK_BOUNDS = (49.8, 60.9, -8.7, 1.8)  # lat_min, lat_max, lon_min, lon_max
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MONTHS = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
          "Septiembre", "Octubre", "Noviembre", "Diciembre"]
CLASS_ES = ["Leve", "Grave", "Mortal"]

st.set_page_config(page_title="Severidad de accidentes en UK", page_icon="🚦", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    a = joblib.load(ROOT / "models" / "modelo_con_vehiculos.joblib")
    b = joblib.load(ROOT / "models" / "modelo_sin_vehiculos.joblib")
    geo = joblib.load(ROOT / "models" / "geo_features.joblib")
    return a, b, geo


@st.cache_data
def load_cells():
    """Accidentes por celda de ≈5 km, año y franja horaria (ver src/prepare_app_data.py)."""
    return pd.read_parquet(ROOT / "data" / "processed" / "grid_cells.parquet")


@st.cache_data
def load_validation():
    """Métricas congeladas con IC 95 % (reports/holdout_ic95.csv y holdout_metricas.csv)."""
    try:
        return (pd.read_csv(ROOT / "reports" / "holdout_ic95.csv"), pd.read_csv(ROOT / "reports" / "holdout_metricas.csv"))
    except Exception:
        return None


BAND_LABELS = {0: "00–05 h", 1: "06–11 h", 2: "12–17 h", 3: "18–23 h"}
# Edad exacta -> punto medio de la banda STATS19 (así se codificó en el entrenamiento)
BAND_EDGES = [(5, 3.0), (10, 8.0), (15, 13.0), (20, 18.0), (25, 23.0), (35, 30.5), (45, 40.5),
              (55, 50.5), (65, 60.5), (75, 70.5)]
VALIDATION_NOTE = ("Entrenado con 2005–2014 y umbrales ajustados con 2015. Los modelos se congelaron antes de evaluarlos "
                   "en 2018–2019, 2020–2021 (pandemia) y 2022–2025, años que no intervinieron en ninguna decisión de "
                   "diseño. Los intervalos son bootstrap del 95 % sobre las filas evaluadas; no incluyen la variabilidad "
                   "del entrenamiento. Las probabilidades mostradas se recalibraron con 2022–2025 por sistema de registro.")
SET_ORDER = ["test 2016-17", "holdout 2018-19", "estres 2020-21", "post 2022-25"]


def band_mid(age):
    for hi, mid in BAND_EDGES:
        if age <= hi:
            return mid
    return 80.0


def wilson_ci(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción k/n (n > 0)."""
    n = np.asarray(n, float)
    p = np.asarray(k, float) / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def aggregate_cells(cells, years, bands):
    sub = cells[cells["year"].between(years[0], years[1]) & cells["band"].isin(bands)]
    return sub.groupby(["lat_c", "lon_c"], as_index=False)[["n", "ksi", "fatal"]].sum()


def haversine_london(lat, lon):
    la, lo, la0, lo0 = map(np.radians, (lat, lon, 51.5074, -0.1278))
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la) * np.cos(la0) * np.sin((lo - lo0) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def build_row(bundle, geo, inp, veh):
    """Construye la fila de variables igual que en el entrenamiento."""
    lat, lon = inp["lat"], inp["lon"]
    hour = inp["hour"]
    row = {
        "Latitude": lat, "Longitude": lon, "Speed_limit": float(inp["speed"]),
        "Number_of_Vehicles": inp["n_veh"], "Injury_Based": inp["injury_based"],
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dayofweek": inp["dow"], "month": inp["month"], "is_weekend": int(inp["dow"] >= 5),
        "is_dark": int(inp["Light_Conditions"].startswith("Darkness")),
        "dist_london_km": haversine_london(lat, lon),
    }
    g = geo.transform(pd.DataFrame({"Latitude": [lat], "Longitude": [lon]}))
    row["geo_cluster"] = g["geo_cluster"].iloc[0]
    row["cell_density"] = float(g["cell_density"].iloc[0])
    row["cell_ksi_rate"] = float(g["cell_ksi_rate"].iloc[0])
    for c in ("Road_Type", "1st_Road_Class", "Junction_Detail", "Junction_Control", "Light_Conditions",
              "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area",
              "Pedestrian_Crossing"):
        row[c] = inp[c]
    if veh is not None:
        row.update(veh)
    X = pd.DataFrame([row])
    for c, cats in bundle["categories"].items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return X[bundle["columns"]], row["cell_density"]


def vehicle_features(sel, age_min, age_max, vage, cc):
    return {
        "has_vehicle_info": 1,
        "veh_motorcycle_any": int("Moto" in sel), "veh_pedalcycle_any": int("Bicicleta" in sel),
        "veh_bus_any": int("Autobús / minibús" in sel), "veh_heavy_goods_any": int("Camión pesado" in sel),
        "veh_van_any": int("Furgoneta" in sel),
        "driver_age_min": band_mid(age_min), "driver_age_max": band_mid(age_max),
        "young_driver_any": int(age_min <= 25), "elderly_driver_any": int(age_max > 75),
        "male_share": np.nan, "veh_age_mean": float(vage), "engine_cc_max": float(cc),
    }


def predict(bundle, X):
    p = bundle["model"].predict_proba(X)[0]
    w = np.asarray(bundle["class_weights"])
    return p, int((p * w).argmax())


MODE_RATE = "Proporción de graves o mortales"
MODE_VOLUME = "Número de accidentes (volumen)"


def zones_table(agg, mode, min_n, top=300):
    """Celdas a mostrar. Proporción: se ordena por el límite inferior del IC 95 % (prudencia con pocas muestras)."""
    a = agg[agg["n"] >= min_n].copy()
    if a.empty:
        return a
    a["rate"] = a["ksi"] / a["n"]
    a["lo"], a["hi"] = wilson_ci(a["ksi"], a["n"])
    return a.nlargest(top, "lo" if mode == MODE_RATE else "n")


def zones_layer(m, shown, mode):
    if shown.empty:
        return
    if mode == MODE_RATE:
        lo, hi = float(shown["rate"].quantile(0.05)), float(shown["rate"].quantile(0.95))
        cmap = cm.LinearColormap(["#fee08b", "#f46d43", "#a50026"], vmin=lo, vmax=max(hi, lo + 1e-6),
                                 caption="Proporción de graves o mortales")
    else:
        cmap = cm.LinearColormap(["#deebf7", "#6baed6", "#08306b"], vmin=float(shown["n"].min()),
                                 vmax=float(shown["n"].max()), caption="Número de accidentes en la celda")
    for r in shown.itertuples():
        color = cmap(r.rate if mode == MODE_RATE else r.n)
        folium.CircleMarker(
            [r.lat_c, r.lon_c], radius=3 + (r.n ** 0.5) / 8, color=color, fill=True, fill_opacity=0.7, weight=1,
            tooltip=(f"{r.n} accidentes · {r.ksi} graves/mortales · proporción {r.rate:.1%} "
                     f"(IC 95 %: {r.lo:.1%}–{r.hi:.1%})")).add_to(m)
    cmap.add_to(m)


def compare_layer(m, a1, a2, min_n):
    """Diferencia de proporción entre dos periodos; solo celdas con diferencia estadísticamente clara (|z| > 2)."""
    j = (a1[a1["n"] >= min_n].merge(a2[a2["n"] >= min_n], on=["lat_c", "lon_c"], suffixes=("1", "2")))
    if len(j) < 3:
        j["z"] = np.nan
        return j, np.nan
    j["r1"], j["r2"] = j["ksi1"] / j["n1"], j["ksi2"] / j["n2"]
    se = np.sqrt(j["r1"] * (1 - j["r1"]) / j["n1"] + j["r2"] * (1 - j["r2"]) / j["n2"])
    j["z"] = (j["r2"] - j["r1"]) / se.replace(0, np.nan)
    sig = j[j["z"].abs() > 2]
    for r in sig.itertuples():
        folium.CircleMarker(
            [r.lat_c, r.lon_c], radius=3 + ((r.n1 + r.n2) ** 0.5) / 12, fill=True, fill_opacity=0.65, weight=1,
            color="#c0392b" if r.r2 > r.r1 else "#2471a3",
            tooltip=f"periodo 1: {r.r1:.1%} ({r.n1} acc.) → periodo 2: {r.r2:.1%} ({r.n2} acc.)").add_to(m)
    return j, float(j["r1"].corr(j["r2"]))


# ----------------------------------------------------------------- UI
st.title("🚦 Gravedad de los accidentes de tráfico — Gran Bretaña")
st.caption("Proyecto académico. Datos: STATS19 oficial del Department for Transport (2005–2025, Open Government Licence "
           "v3.0); modelos LightGBM entrenados con 2005–2014. **No apto para decisiones reales de seguridad vial.**")

modelA, modelB, geo = load_models()
cells = load_cells()
cats = modelB["categories"]

if "point" not in st.session_state:
    st.session_state.point = DEFAULT_POINT
    st.session_state.view = {"center": list(DEFAULT_POINT), "zoom": 7}

tab1, tab2 = st.tabs(["📍 Estimador de gravedad", "🗺️ Zonas y periodos"])

with tab1:
    st.info("**Qué estima este modelo:** *si ocurre* un accidente en este lugar y con estas condiciones, la probabilidad "
            "de que sea leve, grave o mortal. **No** estima la probabilidad de que ocurra un accidente ni el riesgo por "
            "unidad de tráfico (no hay datos de exposición).")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("1. Elige el lugar (clic en el mapa)")
        m = folium.Map(location=st.session_state.view["center"], zoom_start=st.session_state.view["zoom"],
                       tiles="OpenStreetMap")
        folium.Marker(st.session_state.point, tooltip="Punto elegido",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        out = st_folium(m, height=430, width="stretch", key="map_pick",
                        returned_objects=["last_clicked", "center", "zoom"])
        if out and out.get("center"):
            st.session_state.view = {"center": [out["center"]["lat"], out["center"]["lng"]], "zoom": out["zoom"]}
        if out and out.get("last_clicked"):
            new = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
            if new != st.session_state.point:
                st.session_state.point = new
                st.rerun()
        lat, lon = st.session_state.point
        st.write(f"**Punto:** {lat:.4f}, {lon:.4f}")
        in_bounds = UK_BOUNDS[0] <= lat <= UK_BOUNDS[1] and UK_BOUNDS[2] <= lon <= UK_BOUNDS[3]
        dens0 = float(geo.transform(pd.DataFrame({"Latitude": [lat], "Longitude": [lon]}))["cell_density"].iloc[0])
        valid = in_bounds and dens0 > 0
        if not valid:
            st.error("**Sin estimación para este punto.** " + (
                "Está fuera de Gran Bretaña, que es lo que cubre STATS19." if not in_bounds else
                "No hay accidentes registrados en esta celda de ≈5 km (puede ser mar, Irlanda del Norte u otra zona "
                "sin datos): el modelo no se aplica sin historial local."))

    with right:
        st.subheader("2. Condiciones del accidente")
        c1, c2 = st.columns(2)
        hour = c1.slider("Hora", 0, 23, 17)
        speed = c2.select_slider("Límite de velocidad (mph)", [20, 30, 40, 50, 60, 70], value=30)
        c3, c4 = st.columns(2)
        dow = c3.selectbox("Día", range(7), format_func=lambda i: DAYS[i], index=4)
        month = c4.selectbox("Mes", range(1, 13), format_func=lambda i: MONTHS[i - 1], index=10)
        n_veh = st.slider("Vehículos implicados", 1, 6, 2)
        fields = {
            "Light_Conditions": "Luz", "Weather_Conditions": "Clima", "Road_Surface_Conditions": "Superficie",
            "Road_Type": "Tipo de vía", "1st_Road_Class": "Clase de vía", "Urban_or_Rural_Area": "Zona",
            "Junction_Detail": "Cruce", "Junction_Control": "Control del cruce",
            "Pedestrian_Crossing": "Paso de peatones",
        }
        reg = st.radio("Sistema de registro de la gravedad", ["Por lesiones (el más extendido hoy)", "Por juicio del agente"],
                       help="Desde 2012–2016 muchas fuerzas policiales derivan la gravedad de la lesión más severa. "
                            "El modelo estima la gravedad *registrada*, que depende de este sistema.")
        inp = {"lat": lat, "lon": lon, "hour": hour, "speed": speed, "dow": dow, "month": month, "n_veh": n_veh,
               "injury_based": int(reg.startswith("Por lesiones"))}
        defaults = {"Light_Conditions": "Daylight", "Weather_Conditions": "Fine no high winds",
                    "Road_Surface_Conditions": "Dry", "Road_Type": "Single carriageway",
                    "Urban_or_Rural_Area": "Urban", "1st_Road_Class": "A", "Junction_Detail": "Not at junction or within 20 metres",
                    "Junction_Control": "Give way or uncontrolled"}
        for key, label in fields.items():
            opts = [o for o in cats[key] if str(o) != "nan"]
            d = defaults.get(key)
            inp[key] = st.selectbox(label, opts, index=opts.index(d) if d in opts else 0, key=key)

        use_veh = st.checkbox("Conozco los vehículos y conductores (usa el modelo A, más preciso)")
        veh = None
        if use_veh:
            sel = st.multiselect("Tipos de vehículo", ["Coche / otro", "Moto", "Bicicleta", "Autobús / minibús",
                                                      "Camión pesado", "Furgoneta"], default=["Coche / otro"])
            a1, a2 = st.columns(2)
            age_min = a1.slider("Edad del conductor más joven", 16, 90, 30)
            age_max = a2.slider("Edad del conductor mayor", age_min, 90, max(age_min, 40))
            v1, v2 = st.columns(2)
            vage = v1.slider("Antigüedad media del vehículo (años)", 0, 25, 7)
            cc = v2.number_input("Cilindrada máxima (cc)", 50, 8000, 1600, step=100)
            veh = vehicle_features(sel, age_min, age_max, vage, cc)

    if valid:
        bundle = modelA if veh is not None else modelB
        X, dens = build_row(bundle, geo, inp, veh)
        p_raw, pred = predict(bundle, X)       # decisiones y alerta: probabilidades originales
        p = p_raw
        cal = bundle.get("calibration")
        base_ksi, base_fatal = BASE_RATE_KSI, BASE_RATE_FATAL
        if cal:                                # probabilidades mostradas: recalibradas por sistema de registro
            ibk = str(inp["injury_based"])
            t = np.asarray(cal["params_by_ib"][ibk])          # softmax(a·log p + [0, b1, b2])
            z = t[0] * np.log(np.clip(p_raw, 1e-9, 1)) + np.array([0.0, t[1], t[2]])
            p = np.exp(z - z.max())
            p = p / p.sum()
            base_ksi, base_fatal = cal["obs_ksi_by_ib"][ibk], cal["obs_fatal"]
        p_ksi = 1 - p[0]
        st.divider()
        st.subheader(f"Resultado — modelo {'A (con vehículos)' if veh is not None else 'B (sin vehículos)'}")
        st.caption("Probabilidades **si ocurre** un accidente con estas características (no es la probabilidad de que ocurra).")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Leve", f"{p[0]:.1%}")
        r2.metric("Grave", f"{p[1]:.1%}")
        r3.metric("Mortal", f"{p[2]:.1%}", f"{p[2] / base_fatal:.1f}× la media 2022–25", delta_color="off")
        r4.metric("Grave o mortal", f"{p_ksi:.1%}", f"{p_ksi / base_ksi:.1f}× la media 2022–25", delta_color="off")
        st.progress(float(min(p_ksi / 0.6, 1.0)),
                    text="Probabilidad de que el accidente sea grave o mortal, si ocurre (escala 0–60 %)")
        alert = p_raw[2] >= bundle["alert_threshold"]
        if alert:
            st.error(f"⚠️ **Modo alerta:** probabilidad de accidente mortal (si ocurre) por encima del umbral "
                     f"({bundle['alert_threshold']:.1%}). En validación, los casos marcados así fueron mortales "
                     f"≈{bundle['min_precision_fatal']:.0%} de las veces.")
        else:
            st.success("Sin alerta de accidente mortal.")
        st.info(f"Clase más probable según el umbral ajustado (equilibra clases): **{CLASS_ES[pred]}**.")
        st.caption(f"Probabilidades recalibradas con datos de {cal['fit_on'] if cal else '—'} para el sistema de registro "
                   "elegido (la proporción de graves o mortales subió mucho desde 2016 y el modelo se entrenó con 2005–2014). "
                   "Úsalo sobre todo como comparación relativa. El efecto del lugar proviene de la tasa histórica de "
                   "gravedad de la celda y no es causal.")
        val = load_validation()
        with st.expander("Validación del modelo y limitaciones"):
            if val is not None:
                ci_df, met_df = val
                mk = "A" if veh is not None else "B"
                rows = []
                for s in SET_ORDER:
                    c = ci_df[(ci_df["modelo"] == mk) & (ci_df["conjunto"] == s)].set_index("metrica")
                    mm = met_df[(met_df["modelo"] == mk) & (met_df["conjunto"] == s)]
                    if c.empty or mm.empty:
                        continue
                    f3 = lambda m: f"{c.loc[m, 'media']:.3f} [{c.loc[m, 'ic95_inf']:.3f}; {c.loc[m, 'ic95_sup']:.3f}]"
                    rows.append({"Periodo evaluado": s.split(" ", 1)[1], "Macro-F1 [IC 95 %]": f3("macro_f1"),
                                 "Precisión promedio grave/mortal": f3("ap_ksi"),
                                 "Alerta: mortales detectados": f"{mm.iloc[0]['alert_recall']:.1%}",
                                 "Alerta: precisión": f"{mm.iloc[0]['alert_precision']:.1%}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("La precisión promedio depende de la proporción de casos graves (que sube con el tiempo): "
                           "no es comparable entre periodos sin dividirla por esa proporción.")
            st.markdown(VALIDATION_NOTE)
            st.markdown("- La etiqueta «grave» cambió de criterio de registro desde 2012–2016 (sistemas por lesiones): "
                        "las comparaciones entre años deben leerse con cautela.\n"
                        "- 2020–2021 (pandemia) tuvo un patrón de tráfico atípico.\n"
                        "- No hay datos de exposición al tráfico: no se estima riesgo por vehículo-kilómetro.\n"
                        "- Proyecto académico: no apto para decisiones reales de seguridad vial.")

with tab2:
    st.subheader("Dónde se concentran los accidentes y cómo varía la proporción de graves")
    st.warning("⚠️ **Más accidentes registrados no significa más peligro.** Las zonas con muchos accidentes suelen ser las "
               "de más tráfico y no hay datos de exposición (vehículos-kilómetro). La proporción de graves tampoco prueba "
               "causas: mezcla tipo de vía, velocidad y prácticas de registro.")
    y_min, y_max = int(cells["year"].min()), int(cells["year"].max())
    c1, c2, c3 = st.columns([1.4, 1, 1])
    years = c1.slider("Periodo (años)", y_min, y_max, (y_min, y_max))
    bands = c2.multiselect("Franja horaria", list(BAND_LABELS), default=list(BAND_LABELS),
                           format_func=lambda b: BAND_LABELS[b])
    min_n = c3.slider("Mínimo de accidentes por celda", 50, 1000, 300, step=50,
                      help="Las celdas con pocos accidentes dan proporciones muy inestables.")
    mode = st.radio("Qué mostrar", [MODE_RATE, MODE_VOLUME], horizontal=True)
    compare = st.checkbox("Comparar con otro periodo (cambios en la proporción de graves o mortales)")
    years2 = None
    if compare:
        years2 = st.slider("Periodo de comparación (años)", y_min, y_max, (y_max, y_max))

    def period_notes(yr):
        if yr[1] >= 2016:
            st.info(f"Periodo {yr[0]}–{yr[1]}: desde 2016 muchas fuerzas policiales registran la gravedad según la lesión "
                    "(en lugar del juicio del agente), lo que eleva la proporción de «graves» sin que cambie necesariamente "
                    "el riesgo real (fuente: Department for Transport).")
        if yr[0] <= 2021 and yr[1] >= 2020:
            st.warning(f"Periodo {yr[0]}–{yr[1]} incluye 2020–2021 (pandemia): el tráfico y la mezcla de accidentes "
                       "fueron atípicos.")

    period_notes(years)
    if compare:
        period_notes(years2)
    if not bands:
        st.info("Selecciona al menos una franja horaria.")
    else:
        agg = aggregate_cells(cells, years, bands)
        m2 = folium.Map(location=[54.0, -2.5], zoom_start=6, tiles="OpenStreetMap")
        if compare:
            agg2 = aggregate_cells(cells, years2, bands)
            j, corr = compare_layer(m2, agg, agg2, min_n)
            st.caption("**Rojo:** la proporción de graves o mortales **sube** en el periodo de comparación; **azul:** baja. "
                       "Solo se dibujan las celdas con diferencia estadísticamente clara (|z| > 2).")
            k = int((j["z"].abs() > 2).sum()) if len(j) else 0
            st.markdown(f"Celdas comparables (≥ {min_n} accidentes en ambos periodos): **{len(j)}** · con cambio claro: "
                        f"**{k}** · correlación entre periodos de la proporción por celda: "
                        f"**{'—' if np.isnan(corr) else f'{corr:.2f}'}**")
        else:
            shown = zones_table(agg, mode, min_n)
            if shown.empty:
                st.info("Ninguna celda cumple el mínimo de accidentes con estos filtros.")
            else:
                zones_layer(m2, shown, mode)
                base = agg[agg["n"] >= min_n]
                st.caption(
                    "Se muestran las 300 celdas "
                    + ("con **mayor límite inferior del IC 95 %** de la proporción de graves o mortales (prudencia con "
                       "pocas muestras)." if mode == MODE_RATE else "con **más accidentes** (volumen).")
                    + f" Proporción de graves o mortales: celdas mostradas **{shown['ksi'].sum() / shown['n'].sum():.1%}**"
                      f" · todas las celdas con ≥ {min_n} accidentes **{base['ksi'].sum() / base['n'].sum():.1%}**.")
        st_folium(m2, height=620, width="stretch", key="map_zones", returned_objects=[])
        if not compare and not shown.empty:
            t = shown.head(10).copy()
            t["proporción"] = (t["rate"] * 100).round(1).astype(str) + " %"
            t["IC 95 %"] = (t["lo"] * 100).round(1).astype(str) + "–" + (t["hi"] * 100).round(1).astype(str) + " %"
            st.dataframe(t[["lat_c", "lon_c", "n", "ksi", "fatal", "proporción", "IC 95 %"]].rename(
                columns={"lat_c": "lat", "lon_c": "lon", "n": "accidentes", "ksi": "graves+mortales", "fatal": "mortales"}),
                hide_index=True, width="stretch")
