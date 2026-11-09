"""Figuras claras para lectores no técnicos (se usan en el informe y la presentación).

Criterios: todo en español; porcentajes (no proporciones); títulos que dicen el mensaje; sin jerga (los términos técnicos
van en la leyenda); referencias visibles (ciudades, línea de la media, modelo trivial); colores con significado fijo:
gris oscuro = lo real/observado · azul = modelo A · naranja = modelo B · rojo = gravedad/mortales.
Cada figura se guarda como reports/figures/F*.png y su pregunta y mensaje en reports/figures/LEYENDAS_FIGURAS.md.
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

from config import ACCIDENTS_PARQUET, FIGURES, MODELS, PROCESSED
from train import REPORTS, load_splits

plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True})
REAL, BLUE, ORANGE, RED, GRAY, GREEN = "#333333", "#1f6f8b", "#e07b39", "#b22222", "#9aa0a6", "#4d8f4d"
PCT = FuncFormatter(lambda v, _: (f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}") + " %")
LEGENDS = {}


def dec(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def coma_decimal(fig):
    """Convierte en coma el separador decimal de las marcas de todos los ejes."""
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            axis.set_major_formatter(FuncFormatter(lambda v, pos, f=fmt: f(v, pos).replace(".", ",")))


def save(fig, name, question, message):
    coma_decimal(fig)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LEGENDS[name] = (question, message)


def covid(ax, label=True):
    ax.axvspan(2019.5, 2021.5, color=GRAY, alpha=0.2, lw=0)
    if label:
        ax.text(2020.5, ax.get_ylim()[1], "pandemia", ha="center", va="top", fontsize=9, color="#555555")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


# ----------------------------------------------------------------------------------------------------------- F1
LUZ = {"Daylight": "De día", "Darkness - lights lit": "De noche, con alumbrado", "Darkness - no lighting": "De noche, sin alumbrado",
       "Darkness - lights unlit": "De noche, alumbrado apagado", "Darkness - lighting unknown": "De noche, alumbrado desconocido"}


def f1_condiciones(a):
    a = a.copy()
    a["ksi"] = (a["Accident_Severity"] != "Slight").astype(int)
    a["hora"] = pd.to_numeric(a["Time"].astype(str).str[:2], errors="coerce")
    a["franja"] = pd.cut(a["hora"], [-1, 5, 9, 15, 19, 23], labels=["00–05 h\n(madrugada)", "06–09 h", "10–15 h", "16–19 h", "20–23 h"])
    base = a["ksi"].mean() * 100
    panels = [
        ("Franja horaria", a.groupby("franja", observed=True)["ksi"].agg(["mean", "size"])),
        ("Límite de velocidad", a.assign(v=a["Speed_limit"].map(lambda x: f"{x:.0f} mph" if pd.notna(x) else None))
         .groupby("v")["ksi"].agg(["mean", "size"]).sort_index(key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))),
        ("Iluminación", a.assign(l=a["Light_Conditions"].map(LUZ)).groupby("l")["ksi"].agg(["mean", "size"])
         .sort_values("mean")),
        ("Zona", a[a["Urban_or_Rural_Area"].isin(["Urban", "Rural"])].assign(z=lambda d: d["Urban_or_Rural_Area"].map(
            {"Urban": "Urbana", "Rural": "Rural"})).groupby("z")["ksi"].agg(["mean", "size"])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax, (title, t) in zip(axes.ravel(), panels):
        t = t[t["size"] >= 5000]
        vals = t["mean"] * 100
        cols = [RED if v > base + 1.5 else (GREEN if v < base - 1.5 else GRAY) for v in vals]
        ax.barh(range(len(t)), vals, color=cols)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=10); ax.invert_yaxis()
        ax.axvline(base, color=REAL, ls="--", lw=1.2)
        ax.set_title(title); ax.xaxis.set_major_formatter(PCT); ax.set_xlim(0, max(vals) * 1.18)
        for i, v in enumerate(vals):
            ax.text(v + 0.3, i, f"{dec(v)} %", va="center", fontsize=9)
    fig.suptitle("Porcentaje de accidentes graves o mortales según las condiciones del accidente",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01, f"Línea discontinua: media de todos los accidentes ({dec(base)} %). Rojo: por encima de la media; verde: por debajo. Periodo 2005–2025. "
             "Las proporciones corresponden a accidentes registrados, no al riesgo de sufrirlos, y no establecen causalidad.", ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout()
    hr = panels[0][1]["mean"] * 100; sp = panels[1][1]["mean"] * 100; lz = panels[2][1]["mean"] * 100
    save(fig, "F1_gravedad_por_condicion.png",
         "¿En qué condiciones es mayor la proporción de accidentes graves o mortales?",
         f"La proporción es mayor de madrugada ({hr.iloc[0]:.1f} %), a 60 mph ({sp.get('60 mph', np.nan):.1f} %) y de noche sin "
         f"alumbrado ({lz.max():.1f} %) que la media ({base:.1f} %). Son asociaciones dentro de los accidentes registrados.")


# ----------------------------------------------------------------------------------------------------------- F2
def f2_evolucion(a):
    g = a.groupby("Year")
    res = pd.DataFrame({"rep": g["Accident_Severity"].apply(lambda s: (s == "Serious").mean() * 100),
                        "aj": g["Adj_Serious"].apply(lambda s: s.dropna().sum()) / g.size() * 100,
                        "ib": g["Injury_Based"].mean() * 100,
                        "mort": g["Accident_Severity"].apply(lambda s: (s == "Fatal").sum())})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(res.index, res["rep"], "o-", color=BLUE, lw=2.5, label="Graves según el registro original")
    ax.plot(res.index, res["aj"], "s--", color=ORANGE, lw=2.5, label="Graves ajustados por el DfT\n(equivalentes al registro por lesiones)")
    ax.set_ylim(10, 27); ax.yaxis.set_major_formatter(PCT)
    covid(ax); ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("Proporción de accidentes graves,\nregistrada y ajustada")
    ax.set_ylabel("Porcentaje de accidentes graves"); ax.set_xlabel("Año")
    ax.annotate("2016: numerosas fuerzas\nadoptan el registro\npor lesiones", xy=(2016, res.loc[2016, "rep"]),
                xytext=(2008.3, 22.5), fontsize=9.5, arrowprops=dict(arrowstyle="->", color=REAL))
    ax = axes[1]
    ax.bar(res.index, res["ib"], color=GREEN, alpha=0.75)
    ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PCT); covid(ax)
    ax.set_ylabel("Accidentes registrados por lesiones (%)", color=GREEN); ax.set_xlabel("Año")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax2.plot(res.index, res["mort"], "o-", color=RED, lw=2.5)
    ax2.set_ylim(0, 3400); ax2.set_ylabel("Accidentes mortales (número anual)", color=RED)
    ax.set_title("Sistema de registro y\naccidentes mortales")
    fig.suptitle("Evolución de la proporción de accidentes graves y del sistema de registro, 2005–2025", fontsize=14,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "F2_evolucion_graves.png",
         "¿Por qué sube la proporción de accidentes graves desde 2016 si los mortales casi no cambian?",
         f"Los graves registrados pasan de {res.loc[2015, 'rep']:.1f} % (2015) a {res.loc[2017, 'rep']:.1f} % (2017), mientras los "
         f"ajustados se mantienen en ≈{res.loc[2015, 'aj']:.1f}–{res.loc[2017, 'aj']:.1f} %; el registro por lesiones pasa de "
         f"{res.loc[2015, 'ib']:.1f} % a {res.loc[2016, 'ib']:.1f} % entre 2015 y 2016. Los mortales rondan {int(res.loc[2010:2025, 'mort'].min())}–"
         f"{int(res.loc[2010:2025, 'mort'].max())} al año desde 2010. Desde 2018 también suben los ajustados: no todo es registro.")
    return res


# ----------------------------------------------------------------------------------------------------------- F3
def captura(y, score):
    o = np.argsort(-score)
    c = np.cumsum(y[o]) / y.sum()
    return (np.arange(1, len(y) + 1) / len(y)) * 100, c * 100


def f3_capturas(models, ev):
    y = ev["severity"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    out = {}
    for ax, (target, name, col) in zip(axes, (("fatal", "mortales", 2), ("ksi", "graves o mortales", None))):
        yy = (y == 2) if target == "fatal" else (y > 0)
        ax.plot([0, 100], [0, 100], ":", color=GRAY, lw=2, label="Selección aleatoria")
        for mname, (p, b, color) in models.items():
            s = p[:, 2] if target == "fatal" else 1 - p[:, 0]
            x, c = captura(yy.astype(int), s)
            ax.plot(x, c, color=color, lw=2.6, label=f"Modelo {mname}")
            c5 = float(np.interp(5, x, c)); out[(target, mname)] = c5
            if target == "fatal":
                flag = (p[:, 2] >= b["alert_threshold"])
                xa, ya = flag.mean() * 100, (flag & yy).sum() / yy.sum() * 100
                ax.scatter([xa], [ya], s=90, color=color, edgecolor="black", zorder=6)
                ax.annotate(f"Alerta del modelo {mname}: marca el {dec(xa)} %\ny capta el {ya:.0f} % de los mortales", xy=(xa, ya),
                            xytext=(20, 32 if mname == "A" else 12), fontsize=9, color=color,
                            bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.25"), arrowprops=dict(arrowstyle="->", color=color))
        ax.set_xlim(0, 60 if target == "fatal" else 100); ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Porcentaje de accidentes marcados\n(ordenados de mayor a menor probabilidad estimada)")
        ax.set_ylabel(f"Porcentaje de los accidentes {name} captados")
        ax.set_title(f"Accidentes {name}"); ax.legend(fontsize=9.5, loc="center right" if target == "fatal" else "lower right")
    fig.suptitle("Captación de accidentes mortales y graves según el porcentaje de accidentes marcados",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "F3_capturas.png",
         "Si revisamos primero los accidentes que el modelo considera de mayor riesgo, ¿qué parte de los realmente mortales se captura?",
         f"Marcando el 5 % de los accidentes con mayor probabilidad, el modelo A captura el {out[('fatal', 'A')]:.0f} % de los mortales y "
         f"el B el {out[('fatal', 'B')]:.0f} % (al azar sería el 5 %). Es mejor que el azar pero lejos de perfecto. Años 2018–2019, "
         "modelos entrenados con 2005–2014.")
    return out


# ----------------------------------------------------------------------------------------------------------- F4
def f4_anual(a):
    y = pd.read_csv(REPORTS / "holdout_por_anio.csv")
    s = a.groupby("Year")["Accident_Severity"].apply(lambda x: (x == "Slight").mean())
    triv = (2 * s / (1 + s)) / 3
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for m, c in (("A", BLUE), ("B", ORANGE)):
        d = y[y.modelo == m]
        ax.plot(d["año"], d["macro_f1"], "o-", color=c, lw=2.5, label=f"Modelo {m}" + (" (con vehículos)" if m == "A" else " (sin vehículos)"))
    t = triv.loc[2015:2025]
    ax.plot(t.index, t.values, "--", color=GRAY, lw=2, label="Modelo trivial (siempre «leve»)")
    ax.set_ylim(0.26, 0.48); ax.axvline(2015.5, color=REAL, ls=":", lw=1)
    ax.text(2015.6, 0.27, "Años posteriores al\najuste de los modelos", fontsize=9)
    covid(ax); ax.legend(fontsize=9, loc="center right")
    ax.set_title("Índice de acierto por año de evaluación")
    ax.set_xlabel("Año"); ax.set_ylabel("Índice de acierto (macro-F1)")
    ax = axes[1]
    d = y[y.modelo == "A"]
    ax.plot(d["año"], d["obs_ksi"] * 100, "o-", color=REAL, lw=2.5, label="Observado")
    ax.plot(d["año"], d["pred_ksi_raw"] * 100, "s-", color=BLUE, lw=2.5, label="Estimado por el modelo A\n(entrenado con 2005–2014)")
    ax.yaxis.set_major_formatter(PCT); covid(ax); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Proporción observada y estimada de\naccidentes graves o mortales")
    ax.set_xlabel("Año"); ax.set_ylabel("Porcentaje de accidentes graves o mortales")
    fig.suptitle("Desempeño de los modelos entrenados con 2005–2014 en los años posteriores", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    a15, a25 = y[(y.modelo == "A") & (y["año"] == 2015)].iloc[0], y[(y.modelo == "A") & (y["año"] == 2025)].iloc[0]
    save(fig, "F4_rendimiento_anual.png",
         "¿Siguen funcionando los modelos años después de entrenarlos?",
         f"El índice de acierto del modelo A baja de {a15.macro_f1:.3f} (2015) a {a25.macro_f1:.3f} (2025), siempre por encima del modelo "
         f"trivial (≈{triv.loc[2025]:.2f}), sin ningún salto en 2020–2021. La proporción real de graves o mortales sube de "
         f"{a15.obs_ksi*100:.1f} % a {a25.obs_ksi*100:.1f} % y el modelo (sin recalibrar) predice {a25.pred_ksi_raw*100:.1f} % en 2025.")


# ----------------------------------------------------------------------------------------------------------- F5
def f5_calibracion(models, ev_all):
    from calibration import apply_slope_bias
    y = ev_all["severity"].values
    yr = ev_all["Year"].values
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    e = (yr >= 2024)
    res = {}
    for ax, (mname, (p, b, color)) in zip(axes, models.items()):
        ib = ev_all["Injury_Based"].values.astype(int)
        t = b["calibration"]["params_by_ib"]
        pc = p.copy()
        for k in (0, 1):
            m = ib == k
            pc[m] = apply_slope_bias(p[m], t[str(k)])
        for nm, pp, c, ls in (("Sin recalibrar", p, GRAY, "o-"), ("Recalibrado", pc, color, "o-")):
            s, tt = 1 - pp[e, 0], (y[e] > 0).astype(float)
            ed = np.quantile(s, np.linspace(0, 1, 11)); ix = np.clip(np.searchsorted(ed, s, side="right") - 1, 0, 9)
            xs = np.array([s[ix == k].mean() for k in range(10)]) * 100
            ys = np.array([tt[ix == k].mean() for k in range(10)]) * 100
            ax.plot(xs, ys, ls, color=c, lw=2.5, label=nm)
            res[(mname, nm)] = (xs, ys)
        ax.plot([0, 60], [0, 60], ":", color=REAL, lw=1.5, label="Calibración perfecta")
        ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Probabilidad estimada de accidente grave o mortal")
        ax.set_ylabel("Porcentaje observado de accidentes graves o mortales")
        ax.set_title(f"Modelo {mname}"); ax.legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Calibración de las probabilidades estimadas (accidentes de 2024–2025)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "F5_calibracion.png",
         "Cuando el modelo dice que un accidente tiene un X % de probabilidad de ser grave o mortal, ¿ocurre realmente en un X % de los casos?",
         "Sin recalibrar, el modelo se queda corto en todos los niveles (los puntos grises quedan por encima de la línea de predicción "
         "perfecta); tras recalibrar con 2022–2023 los puntos caen casi sobre la línea en 2024–2025.")


# ----------------------------------------------------------------------------------------------------------- F6
NOMBRES = {
    "Number_of_Vehicles": "Número de vehículos implicados", "Speed_limit": "Límite de velocidad",
    "veh_motorcycle_any": "Motocicleta implicada", "veh_pedalcycle_any": "Ciclista implicado",
    "Junction_Control": "Control del cruce", "male_share": "Proporción de conductores hombres",
    "driver_age_max": "Edad del conductor de más edad", "driver_age_min": "Edad del conductor más joven",
    "engine_cc_max": "Cilindrada del vehículo", "veh_age_mean": "Antigüedad del vehículo",
    "Urban_or_Rural_Area": "Zona urbana o rural", "1st_Road_Class": "Clase de vía", "Road_Type": "Tipo de vía",
    "Junction_Detail": "Tipo de cruce", "Weather_Conditions": "Clima", "Light_Conditions": "Iluminación",
    "Road_Surface_Conditions": "Superficie de la vía", "Pedestrian_Crossing": "Paso de peatones",
    "Injury_Based": "Sistema de registro (por lesiones)", "dayofweek": "Día de la semana", "month": "Mes del año",
    "is_weekend": "Fin de semana", "is_dark": "Oscuridad", "veh_bus_any": "Autobús implicado",
    "veh_heavy_goods_any": "Camión pesado implicado", "veh_van_any": "Furgoneta implicada",
    "young_driver_any": "Conductor de 25 años o menos", "elderly_driver_any": "Conductor de más de 75 años",
    "has_vehicle_info": "Datos de vehículos disponibles", "geo_cluster": "Zona del país (agrupación)",
    "cell_density": "Accidentes previos en la zona", "cell_ksi_rate": "Proporción previa de graves en la zona",
    "dist_london_km": "Distancia a Londres", "hour": "Hora del día", "coords": "Coordenadas (latitud y longitud)"}
GRUPO_DE = {}
for g, vs in {"Vía y cruce": ["Speed_limit", "Junction_Control", "Urban_or_Rural_Area", "1st_Road_Class", "Road_Type", "Junction_Detail",
                              "Pedestrian_Crossing"],
              "Vehículos y conductores": ["veh_motorcycle_any", "veh_pedalcycle_any", "male_share", "driver_age_max", "driver_age_min",
                                          "engine_cc_max", "veh_age_mean", "veh_bus_any", "veh_heavy_goods_any", "veh_van_any",
                                          "young_driver_any", "elderly_driver_any", "has_vehicle_info"],
              "Número de vehículos": ["Number_of_Vehicles"],
              "Lugar": ["geo_cluster", "cell_density", "cell_ksi_rate", "dist_london_km", "coords"],
              "Momento": ["hour", "dayofweek", "month", "is_weekend"],
              "Entorno": ["Weather_Conditions", "Light_Conditions", "Road_Surface_Conditions", "is_dark"],
              "Sistema de registro": ["Injury_Based"]}.items():
    for v in vs:
        GRUPO_DE[v] = g
COLOR_GRUPO = {"Vía y cruce": "#c0392b", "Vehículos y conductores": "#1f6f8b", "Número de vehículos": "#8e44ad", "Lugar": "#e07b39",
               "Momento": "#4d8f4d", "Entorno": "#7f8c8d", "Sistema de registro": "#b7950b"}


def shap_plain(bundle, X, n=6000, seed=42):
    Xs = X[bundle["columns"]].sample(min(n, len(X)), random_state=seed)
    F = len(bundle["columns"])
    c = bundle["model"].booster_.predict(Xs, pred_contrib=True).reshape(len(Xs), 3, F + 1)[:, :, :F]
    cols = bundle["columns"]
    imp = {}
    for i, v in enumerate(cols):
        imp[v] = np.abs(c[:, :, i]).mean()
    # variables que son partes de una misma magnitud: se suman por fila antes de promediar
    def merged(names, label):
        idx = [cols.index(v) for v in names if v in cols]
        if idx:
            imp[label] = np.abs(c[:, :, idx].sum(2)).mean()
            for v in names:
                imp.pop(v, None)
    merged(["hour_sin", "hour_cos"], "hour"); merged(["Latitude", "Longitude"], "coords")
    s = pd.Series(imp); return (s / s.sum() * 100).sort_values(ascending=False)


def f6_factores(models, ev):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True)
    tops = {}
    for ax, (mname, (p, b, color)) in zip(axes, models.items()):
        s = shap_plain(b, ev).head(10)
        tops[mname] = s
        ax.barh(range(len(s)), s.values, color=[COLOR_GRUPO[GRUPO_DE[v]] for v in s.index])
        ax.set_yticks(range(len(s))); ax.set_yticklabels([NOMBRES.get(v, v) for v in s.index], fontsize=10.5); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PCT); ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_title("Modelo A (con datos de vehículos)" if mname == "A" else "Modelo B (sin datos de vehículos)")
        ax.set_xlabel("Contribución a las predicciones (% del total)")
        for i, v in enumerate(s.values):
            ax.text(v + 0.2, i, f"{dec(v)} %", va="center", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR_GRUPO.values()]
    fig.legend(handles, COLOR_GRUPO.keys(), loc="lower center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Variables de mayor peso en las predicciones de cada modelo", fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.5, -0.09, "Indica cuánto utiliza el modelo cada variable, no si la causa ni en qué sentido. Cada porcentaje es la parte del "
             "total de la contribución (valores SHAP). Accidentes de 2018–2019.", ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout()
    save(fig, "F6_factores_principales.png",
         "¿Qué factores pesan más en las predicciones de los modelos?",
         "En el modelo A pesan sobre todo el número de vehículos, el límite de velocidad, si hay una moto o un ciclista y las "
         "características de los conductores; en el B, el número de vehículos, el límite de velocidad, la hora, el control del cruce y el lugar. "
         "Son pesos en el modelo, no causas.")
    return tops


# ----------------------------------------------------------------------------------------------------------- F7
def f7_subgrupos():
    t = pd.read_csv(REPORTS / "subgrupos_modelo_B.csv")
    t["grupo"] = t["grupo"].astype(str)
    rename = {"Darkness - lights lit": "De noche, con alumbrado", "Darkness - no lighting": "De noche, sin alumbrado",
              "Darkness - lighting unknown": "De noche, alumbrado desconocido", "Daylight": "De día", "Rural": "Zona rural",
              "Urban": "Zona urbana", "juicio del agente": "Registro por juicio del agente", "por lesiones": "Registro por lesiones"}
    t["grupo"] = t["grupo"].map(lambda g: rename.get(g, g))
    t.loc[t.variable == "Speed_limit", "grupo"] = t.loc[t.variable == "Speed_limit", "grupo"].map(lambda g: f"{float(g):.0f} mph")
    t.loc[t.variable == "franja", "grupo"] = t.loc[t.variable == "franja", "grupo"].map(lambda g: f"{g.replace('-', '–')} h")
    titles = {"franja": "Hora del día", "Speed_limit": "Límite de velocidad", "luz": "Iluminación", "zona": "Zona", "registro": "Sistema de registro"}
    order = ["franja", "Speed_limit", "luz", "zona", "registro"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for ax, v in zip(axes.ravel(), order):
        s = t[t.variable == v].reset_index(drop=True)
        for i, r in s.iterrows():
            ax.plot([r.predicho * 100, r.observado * 100], [i, i], color=GRAY, lw=2)
        ax.scatter(s["observado"] * 100, range(len(s)), color=REAL, s=70, zorder=5, label="Observado")
        ax.scatter(s["predicho"] * 100, range(len(s)), color=ORANGE, s=70, zorder=5, label="Estimado por el modelo B")
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s["grupo"], fontsize=10); ax.invert_yaxis()
        ax.set_title(titles[v]); ax.xaxis.set_major_formatter(PCT)
        ax.set_xlim(10, 36); ax.set_ylim(len(s) - 0.5, -0.5)
    ax = axes.ravel()[-1]; ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    ax.legend(h, l, loc="center", fontsize=11, frameon=False)
    ax.text(0.5, 0.25, "El modelo subestima de forma\nsimilar en todos los grupos:\nel sesgo es de nivel general.",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    fig.suptitle("Porcentaje observado y estimado de accidentes graves o mortales por subgrupos (modelo B sin recalibrar, 2018–2019)", fontsize=14,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "F7_subgrupos.png",
         "¿Falla el modelo más en algún tipo de accidente?",
         "El modelo (sin recalibrar) predice una proporción menor que la real en todos los grupos, con una razón real/predicha de 1,1 a 1,4: "
         "no hay un grupo anómalo, el desfase es general.")


# ----------------------------------------------------------------------------------------------------------- F8
CIUDADES = {"Londres": (51.51, -0.13), "Birmingham": (52.49, -1.89), "Manchester": (53.48, -2.24), "Leeds": (53.80, -1.55),
            "Newcastle": (54.98, -1.62), "Glasgow": (55.86, -4.25), "Edimburgo": (55.95, -3.19), "Cardiff": (51.48, -3.18),
            "Bristol": (51.45, -2.59), "Aberdeen": (57.15, -2.09), "Plymouth": (50.37, -4.14), "Norwich": (52.63, 1.30)}


def f8_mapa():
    g = pd.read_parquet(PROCESSED / "grid_stats.parquet")
    g = g[g["n"] >= 150].copy()
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    sc = ax.scatter(g["lon_c"], g["lat_c"], s=5 + np.sqrt(g["n"]) * 0.55, c=g["ksi_rate"] * 100, cmap="YlOrRd", vmin=10, vmax=32,
                    alpha=0.9, linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(54))); ax.grid(False)
    for nm, (la, lo) in CIUDADES.items():
        ax.scatter([lo], [la], marker="*", s=130, color="black", zorder=6)
        izq = nm in ("Cardiff", "Norwich", "Glasgow")
        ax.text(lo + (-0.12 if izq else 0.12), la + 0.05, nm, fontsize=9.5, fontweight="bold", zorder=7, ha="right" if izq else "left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Porcentaje de accidentes graves o mortales"); cb.ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Longitud\n\nCada punto es una celda; su tamaño es proporcional al número de accidentes (más accidentes no implica mayor peligrosidad).", fontsize=9.5)
    ax.set_ylabel("Latitud")
    ax.set_title("Proporción de accidentes graves o mortales por zonas\n(celdas de ≈5 km con al menos 150 accidentes, 2005–2025)", fontsize=12.5)
    save(fig, "F8_mapa_proporcion.png",
         "¿Dónde es mayor la proporción de accidentes graves o mortales?",
         f"La proporción varía entre zonas (de ≈10 % a ≈32 %): suele ser menor en las grandes ciudades y mayor en zonas rurales y vías rápidas. "
         f"Se muestran {len(g):,} zonas.".replace(",", "."))


# ----------------------------------------------------------------------------------------------------------- F9
def f9_estabilidad():
    e = pd.read_csv(REPORTS / "estabilidad_espacial.csv")
    labels = [f"{a.replace('-', '–')}\nfrente a\n{b.replace('-', '–')}" for a, b in zip(e.periodo_1, e.periodo_2)]
    x = np.arange(len(e)); w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w / 2, e["corr_numero_accidentes"], w, color=BLUE, label="Número de accidentes por zona")
    b2 = ax.bar(x + w / 2, e["corr_proporcion_pearson"], w, color=ORANGE, label="Proporción de accidentes graves por zona")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}".replace(".", ","), ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10); ax.set_ylim(0, 1.3); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Correlación entre periodos\n(0 = sin relación, 1 = idéntica)")
    ax.legend(fontsize=10, loc="upper center", ncol=2)
    ax.set_title("Estabilidad entre periodos del número y de la proporción de accidentes graves por zona")
    fig.text(0.5, -0.02, "Correlación entre periodos para zonas de ≈5 km con al menos 150 accidentes en ambos. El número de accidentes por zona es muy estable; "
             "la proporción de graves lo es bastante menos.", ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout()
    save(fig, "F9_estabilidad_zonas.png",
         "Si una zona destaca por tener muchos accidentes, o una proporción alta de graves, ¿lo sigue haciendo en otro periodo?",
         f"El volumen de accidentes de cada zona es muy estable ({e['corr_numero_accidentes'].min():.2f}–{e['corr_numero_accidentes'].max():.2f}); la "
         f"proporción de graves lo es mucho menos ({e['corr_proporcion_pearson'].iloc[0]:.2f} entre periodos próximos y "
         f"{e['corr_proporcion_pearson'].iloc[1]:.2f} entre 2005–2011 y 2022–2025).")


# ----------------------------------------------------------------------------------------------------------- F10
def f10_geografia():
    a = pd.read_csv(REPORTS / "holdout_contrastes.csv")
    a = a[a["contraste"].str.startswith("con geografía")]
    b = pd.read_csv(REPORTS / "separacion_geografica_contraste.csv").rename(columns={"dif_macro_f1 (con - sin geografía)": "dif"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    ax = axes[0]
    lab = {"test 2016-17": "2016–2017", "holdout 2018-19": "2018–2019", "estres 2020-21": "2020–2021 (pandemia)", "post 2022-25": "2022–2025"}
    for i, r in enumerate(a.itertuples()):
        ax.errorbar(r.dif_macro_f1 * 100, i, xerr=[[(r.dif_macro_f1 - r.ic95_inf) * 100], [(r.ic95_sup - r.dif_macro_f1) * 100]], fmt="o",
                    color=BLUE, capsize=5, ms=9, lw=2)
    ax.set_yticks(range(len(a))); ax.set_yticklabels([lab[c] for c in a["conjunto"]]); ax.invert_yaxis()
    ax.axvline(0, color=REAL, lw=1.5); ax.set_xlim(-1.2, 2.4)
    ax.set_xlabel("Diferencia del índice de acierto al añadir la ubicación\n(centésimas; barras: IC del 95 %)\n← empeora  |  mejora →")
    ax.set_title("Efecto de la ubicación por periodo")
    ax = axes[1]
    for i, r in enumerate(b.itertuples()):
        col = BLUE if "NO" not in r.grupo else ORANGE
        ax.errorbar(r.dif * 100, i, xerr=[[(r.dif - r.ic95_inf) * 100], [(r.ic95_sup - r.dif) * 100]], fmt="o", color=col, capsize=5, ms=9, lw=2)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels([f"{r.conjunto.split(' ')[1].replace('-', '–')}\n" + ("zonas observadas" if "NO" not in r.grupo else "zonas no observadas")
                        for r in b.itertuples()], fontsize=10)
    ax.invert_yaxis(); ax.axvline(0, color=REAL, lw=1.5); ax.set_xlim(-1.2, 2.4)
    ax.set_xlabel("Diferencia del índice de acierto al añadir la ubicación\n(centésimas)")
    ax.set_title("Efecto según zonas observadas o no\nobservadas en el entrenamiento")
    fig.suptitle("Aportación de la ubicación al índice de acierto", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "F10_utilidad_ubicacion.png",
         "¿Mejora la predicción saber dónde ocurrió el accidente, también en lugares nuevos?",
         "Sí mejora, un poco y de forma creciente (de +0,9 a +1,7 centésimas del índice), pero solo en zonas con historial: en zonas "
         "reservadas que el modelo nunca vio, la mejora no se distingue de cero.")


# ----------------------------------------------------------------------------------------------------------- F11
def f11_algoritmos():
    c = pd.read_csv(REPORTS / "comparacion_modelos.csv")
    names = {"LightGBM": ("LightGBM", BLUE), "XGBoost": ("XGBoost", GREEN), "Regresión logística": ("Regresión logística\n(modelo simple)", ORANGE)}
    sets = ["test 2016-17", "holdout 2018-19", "estres 2020-21", "post 2022-25"]
    lab = {"test 2016-17": "2016–2017", "holdout 2018-19": "2018–2019", "estres 2020-21": "2020–2021", "post 2022-25": "2022–2025"}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    off = {"LightGBM": -0.18, "XGBoost": 0.0, "Regresión logística": 0.18}
    for m, (nm, col) in names.items():
        d = c[c.modelo == m].set_index("conjunto").loc[sets]
        y = np.arange(len(sets)) + off[m]
        ax.errorbar(d["macro_f1"], y, xerr=[d["macro_f1"] - d["macro_f1_ic95_inf"], d["macro_f1_ic95_sup"] - d["macro_f1"]], fmt="o", color=col,
                    capsize=4, ms=9, lw=2, label=nm.replace("\n", " "))
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([lab[s] for s in sets]); ax.invert_yaxis()
    ax.set_xlabel("Índice de acierto (macro-F1)\n(el eje no comienza en cero)")
    ax.legend(fontsize=10, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0)); ax.set_title("Comparación de algoritmos según el periodo de evaluación", pad=38)
    fig.tight_layout()
    d1 = c[(c.modelo == "LightGBM") & (c.conjunto == "holdout 2018-19")].iloc[0]; d3 = c[(c.modelo == "Regresión logística") & (c.conjunto == "holdout 2018-19")].iloc[0]
    save(fig, "F11_comparacion_algoritmos.png",
         "¿Importa qué algoritmo se use? ¿Compensa un modelo complejo frente a uno simple?",
         f"LightGBM y XGBoost dan prácticamente el mismo resultado; la regresión logística (modelo simple) queda por debajo "
         f"({d3.macro_f1:.3f} frente a {d1.macro_f1:.3f} en 2018–2019) y la diferencia se mantiene en todos los periodos (de 1,8 a 2,8 centésimas del índice). Mismas variables, mismos años.")


# ----------------------------------------------------------------------------------------------------------- F12
def f12_confusion(models, ev):
    from sklearn.metrics import confusion_matrix
    y = ev["severity"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    names = ["Leve", "Grave", "Mortal"]
    res = {}
    for ax, (mname, (p, b, color)) in zip(axes, models.items()):
        pred = (p * np.asarray(b["class_weights"])).argmax(1)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2]); nm = cm / cm.sum(1, keepdims=True)
        ax.imshow(nm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{nm[i, j]*100:.0f} %", ha="center", va="center", fontsize=15, fontweight="bold" if i == j else None,
                        color="white" if nm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_yticks(range(3))
        ax.set_yticklabels([f"{n}\n(real)" for n in names]); ax.grid(False)
        ax.set_xlabel("Clase estimada por el modelo")
        ax.set_title("Modelo A (con vehículos)" if mname == "A" else "Modelo B (sin vehículos)")
        res[mname] = nm
    fig.suptitle("Clasificación de los accidentes por el modelo según su gravedad real (2018–2019)", fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Cada fila suma 100 %. La diagonal (en negrita) corresponde a los aciertos.",
             ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout()
    save(fig, "F12_aciertos_por_tipo.png",
         "¿Cuántos accidentes de cada tipo acierta el modelo y con qué los confunde?",
         f"El modelo A acierta el {res['A'][0, 0]*100:.0f} % de los leves, el {res['A'][1, 1]*100:.0f} % de los graves y el {res['A'][2, 2]*100:.0f} % de los "
         f"mortales; un {res['A'][2, 0]*100:.0f} % de los mortales se clasifica como leve. Para esos casos es mejor usar el modo alerta que la clase más probable.")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(ACCIDENTS_PARQUET, columns=["Year", "Time", "Accident_Severity", "Speed_limit", "Light_Conditions",
                                                    "Urban_or_Rural_Area", "Injury_Based", "Adj_Serious"])
    f1_condiciones(a); f2_evolucion(a); f4_anual(a); f7_subgrupos(); f8_mapa(); f9_estabilidad(); f10_geografia(); f11_algoritmos()
    d, _, _ = load_splits()
    A = joblib.load(MODELS / "modelo_con_vehiculos.joblib"); B = joblib.load(MODELS / "modelo_sin_vehiculos.joblib")
    ev = d["holdout"].reset_index(drop=True)
    models = {"A": (A["model"].predict_proba(ev[A["columns"]]), A, BLUE), "B": (B["model"].predict_proba(ev[B["columns"]]), B, ORANGE)}
    f3_capturas(models, ev); f6_factores(models, ev); f12_confusion(models, ev)
    allv = pd.concat([d["test"], d["holdout"], d["estres"], d["post"]], ignore_index=True)
    models_all = {"A": (A["model"].predict_proba(allv[A["columns"]]), A, BLUE), "B": (B["model"].predict_proba(allv[B["columns"]]), B, ORANGE)}
    f5_calibracion(models_all, allv)
    lines = ["# Leyendas de las figuras claras\n", "Cada figura: qué pregunta responde y cuál es el mensaje (cifras calculadas de los datos).\n"]
    for name, (q, m) in sorted(LEGENDS.items()):
        lines += [f"## {name}", f"- **Pregunta:** {q}", f"- **Mensaje:** {m}", ""]
    (FIGURES / "LEYENDAS_FIGURAS.md").write_text("\n".join(lines), encoding="utf-8")
    print("figuras:", sorted(LEGENDS))


if __name__ == "__main__":
    main()

