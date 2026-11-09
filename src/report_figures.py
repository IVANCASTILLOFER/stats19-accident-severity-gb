"""Figuras del informe con datos oficiales del DfT (2005–2025). Solo presenta resultados ya calculados."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from config import ACCIDENTS_PARQUET, FIGURES, PROCESSED
from train import REPORTS

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY, RED, GREEN = "#1f6f8b", "#e07b39", "#8a8f98", "#b22222", "#4d8f4d"


def covid(ax):
    ax.axvspan(2019.5, 2021.5, color=GRAY, alpha=0.18, lw=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


def fig_severidad_anual():
    """Gravedad reportada frente a ajustada y registro por lesiones, por año."""
    d = pd.read_parquet(ACCIDENTS_PARQUET, columns=["Year", "Accident_Severity", "Injury_Based", "Adj_Serious"])
    g = d.groupby("Year")
    res = pd.DataFrame({
        "n": g.size(),
        "graves_rep": g["Accident_Severity"].apply(lambda s: (s == "Serious").mean() * 100),
        "graves_aj": g["Adj_Serious"].apply(lambda s: s.dropna().sum()) / g.size() * 100,
        "ibrs": g["Injury_Based"].mean() * 100,
        "mortales": g["Accident_Severity"].apply(lambda s: (s == "Fatal").sum()),
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    ax.plot(res.index, res["graves_rep"], "o-", color=BLUE, lw=2, label="Graves reportados")
    ax.plot(res.index, res["graves_aj"], "s--", color=ORANGE, lw=2, label="Graves ajustados (DfT)")
    covid(ax)
    ax.set_title("Accidentes graves (% de las colisiones)"); ax.set_xlabel("Año"); ax.set_ylabel("%"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(res.index, res["ibrs"], color=GREEN, alpha=0.75, label="% con registro por lesiones")
    ax.set_ylabel("% registrado por lesiones"); ax.set_xlabel("Año"); ax.set_ylim(0, 100)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(res.index, res["mortales"], "o-", color=RED, lw=2, label="Accidentes mortales (n)")
    ax2.set_ylabel("Accidentes mortales"); ax2.set_ylim(0, res["mortales"].max() * 1.15)
    covid(ax)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Cambio de registro y mortales")
    fig.tight_layout(); fig.savefig(FIGURES / "20_severidad_anual.png", dpi=200); plt.close(fig)
    res.round(3).to_csv(REPORTS / "serie_severidad_anual.csv")
    return res.round(2)


def fig_mapa_celdas():
    g = pd.read_parquet(PROCESSED / "grid_stats.parquet")
    g = g[g["n"] >= 150].copy()
    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    ax.scatter(g["lon_c"], g["lat_c"], s=4 + np.sqrt(g["n"]) * 0.5, c=g["ksi_rate"], cmap="YlOrRd",
               vmin=0.10, vmax=0.32, alpha=0.85, linewidths=0)
    cb = fig.colorbar(ax.collections[0], ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Proporción de accidentes graves o mortales")
    ax.set_aspect(1 / np.cos(np.radians(54)))
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    ax.set_title("Celdas de ≈5 km con 150 o más accidentes, 2005–2025\n(tamaño: nº de accidentes)")
    fig.tight_layout(); fig.savefig(FIGURES / "21_mapa_celdas.png", dpi=200); plt.close(fig)
    return len(g)


def fig_validacion_anual():
    y = pd.read_csv(REPORTS / "holdout_por_anio.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    for m, c in (("A", BLUE), ("B", ORANGE)):
        s = y[y.modelo == m]
        ax.plot(s["año"], s["macro_f1"], "o-", color=c, lw=2, label=f"Modelo {m}")
    covid(ax); ax.axvline(2015.5, color="black", ls=":", lw=1)
    ax.text(2015.6, ax.get_ylim()[0] + 0.002, "modelos congelados", fontsize=8)
    ax.set_title("Macro-F1 por año (modelos entrenados con 2005–2014)"); ax.set_xlabel("Año"); ax.set_ylabel("Macro-F1")
    ax.legend(fontsize=8)
    ax = axes[1]
    s = y[y.modelo == "A"]
    ax.plot(s["año"], s["obs_ksi"] * 100, "o-", color=RED, lw=2, label="Observado")
    ax.plot(s["año"], s["pred_ksi_raw"] * 100, "s-", color=BLUE, lw=2, label="Predicho por el modelo A (sin recalibrar)")
    covid(ax)
    ax.set_title("Proporción de graves o mortales"); ax.set_xlabel("Año"); ax.set_ylabel("%"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "12_calibracion_anual.png", dpi=200); plt.close(fig)


def fig_estabilidad_espacial():
    e = pd.read_csv(REPORTS / "estabilidad_espacial.csv")
    lab = [f"{a} vs\n{b}" for a, b in zip(e.periodo_1, e.periodo_2)]
    x = np.arange(len(e)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    ax.bar(x - w / 2, e["corr_proporcion_pearson"], w, color=ORANGE, label="Proporción de graves o mortales")
    ax.bar(x + w / 2, e["corr_numero_accidentes"], w, color=BLUE, label="Número de accidentes (volumen)")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("Correlación entre periodos")
    ax.set_title("¿Se mantiene el patrón por celda?"); ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    chance = 300 * 300 / e["celdas_comparables"]
    ax.bar(x - w / 2, e["solape_top300_por_proporcion"], w, color=ORANGE, label="Top-300 por proporción (IC inferior)")
    ax.bar(x + w / 2, e["solape_top300_por_volumen"], w, color=BLUE, label="Top-300 por volumen")
    ax.scatter(x, chance, marker="_", s=900, color="black", zorder=5, label="Solape esperado por azar")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("Celdas que siguen en el top-300")
    ax.set_title("Persistencia de las zonas destacadas"); ax.set_ylim(0, 400)
    ax.legend(fontsize=7, loc="upper center", ncol=1, frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "24_estabilidad_espacial.png", dpi=200); plt.close(fig)


def fig_separacion_geografica():
    a = pd.read_csv(REPORTS / "holdout_contrastes.csv")
    a = a[a["contraste"].str.startswith("con geografía")]
    b = pd.read_csv(REPORTS / "separacion_geografica_contraste.csv").rename(
        columns={"dif_macro_f1 (con - sin geografía)": "dif"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    y = np.arange(len(a))
    ax.errorbar(a["dif_macro_f1"], y, xerr=[a["dif_macro_f1"] - a["ic95_inf"], a["ic95_sup"] - a["dif_macro_f1"]], fmt="o",
                color=BLUE, capsize=4)
    ax.set_yticks(y); ax.set_yticklabels(a["conjunto"]); ax.invert_yaxis(); ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Diferencia de macro-F1 (con − sin variables geográficas)")
    ax.set_title("Aporte de la geografía por periodo (modelo A)")
    ax = axes[1]
    labels = [f"{r.conjunto.split(' ')[1]}\n{r.grupo}" for r in b.itertuples()]
    y = np.arange(len(b))
    colors = [BLUE if "NO" not in g else ORANGE for g in b["grupo"]]
    for i, r in enumerate(b.itertuples()):
        ax.errorbar(r.dif, i, xerr=[[r.dif - r.ic95_inf], [r.ic95_sup - r.dif]], fmt="o", color=colors[i], capsize=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Diferencia de macro-F1 (con − sin variables geográficas)")
    ax.set_title("Separación geográfica: bloques vistos y no vistos")
    for a_ in axes:
        a_.xaxis.set_major_locator(MaxNLocator(5))
        a_.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.tight_layout(); fig.savefig(FIGURES / "25_separacion_geografica.png", dpi=200); plt.close(fig)


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=color, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def fig_proceso():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.25, 3.35); ax.axis("off")
    steps = [("Datos oficiales\nSTATS19 (DfT)\n2005–2025,\n2,89 M de colisiones", 0.1, "#dbe9f1"),
             ("Preparación y\nvalidación de datos\n(decodificación con la\nguía oficial)", 2.1, "#dbe9f1"),
             ("Variables\ntemporales,\ngeográficas y\nde vehículos", 4.1, "#dbe9f1"),
             ("Modelos\nLightGBM (A y B)\ncongelados", 6.1, "#f8e3d2"),
             ("Evaluación temporal,\nSHAP, calibración\ny separación\ngeográfica", 8.1, "#f8e3d2")]
    for t, x, c in steps:
        box(ax, x, 1.2, 1.75, 1.3, t, c)
    for x in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", xy=(x + 0.2, 1.85), xytext=(x - 0.02, 1.85), arrowprops=dict(arrowstyle="-|>", color="#333333"))
    box(ax, 5.2, -0.1, 4.65, 0.85, "Prototipo: estimador de gravedad (si ocurre el accidente)\ny mapa de zonas y periodos", "#e3f0dc")
    ax.annotate("", xy=(8.0, 0.77), xytext=(8.0, 1.18), arrowprops=dict(arrowstyle="-|>", color="#333333"))
    ax.text(0.1, 2.85, "Entrenamiento 2005–2014 · validación 2015 · prueba 2016–17 · holdout 2018–19 · estrés 2020–21 · post 2022–25",
            fontsize=9, color="#444444")
    fig.tight_layout(); fig.savefig(FIGURES / "22_proceso.png", dpi=200); plt.close(fig)


def fig_causa_efecto():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.plot([0.4, 9.2], [3, 3], color="#333333", lw=2.5)
    box(ax, 9.3, 2.2, 2.5, 1.6, "Dificultad para anticipar\nla gravedad de un accidente\ny localizar dónde se\nconcentran los graves", "#f8e3d2")
    top = [("Vía", 1.8, ["velocidad máxima", "tipo de vía y cruce", "zona rural o urbana"]),
           ("Entorno", 4.6, ["iluminación", "clima", "superficie de la vía"]),
           ("Tiempo", 7.4, ["hora del día", "día y mes", "evolución anual y pandemia"])]
    bottom = [("Territorio", 1.8, ["concentración geográfica", "heterogeneidad regional", "zonas sin historial"]),
              ("Vehículos y\nconductores", 4.6, ["tipo de vehículo", "edad del conductor", "ciclistas y motos"]),
              ("Registro de datos", 7.4, ["mortales: 1,3 % del total", "registro por lesiones\n(desde 2012)", "cambios de especificación"])]
    for (name, x, causes), up in [(c, True) for c in top] + [(c, False) for c in bottom]:
        y1 = 5.0 if up else 1.0
        ax.plot([x + 0.9, x], [3, y1], color="#333333", lw=1.5)
        ax.text(x - 0.1, y1 + (0.35 if up else -0.35), name, ha="center", va="center", fontsize=10, fontweight="bold")
        for i, c in enumerate(causes):
            yy = y1 + (-0.55 - 0.5 * i if up else 0.55 + 0.5 * i)
            xx = x + 0.9 - 0.9 * (yy - 3) / (y1 - 3)
            ax.plot([xx - 0.5, xx], [yy, yy], color=GRAY, lw=1)
            ax.text(xx - 0.55, yy, c, ha="right", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "23_causa_efecto.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    print(fig_severidad_anual().to_string())
    print("celdas dibujadas:", fig_mapa_celdas())
    fig_validacion_anual(); fig_estabilidad_espacial(); fig_separacion_geografica(); fig_proceso(); fig_causa_efecto()
    print("ok")
