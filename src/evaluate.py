"""Evaluación e interpretación de los modelos A (con vehículos) y B (sin vehículos) sobre datos oficiales.

Conjunto de evaluación principal: holdout 2018–2019 (independiente: no intervino en ninguna decisión).
Produce: matrices de confusión, curvas precisión-recall, SHAP (mortal) y aporte por grupos de variables,
análisis por subgrupos, mapa de residuos y perfil de los mortales detectados y perdidos.
Las métricas con intervalos y la calibración están en holdout_eval.py y recalibrate.py.
"""
import time

import folium
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES, MAPS, MODELS, SEED
from train import REPORTS, load_splits
from vehicles import VEHICLE_COLUMNS

CLASS_ES = ["Leve", "Grave", "Mortal"]
C_A, C_B, C_GRAY = "#1f6f8b", "#e07b39", "#8a8f98"
SHAP_N = 6000
SET_KEY, SET_LABEL = "holdout", "2018–19"
GROUPS = {
    "Geografía": ["Latitude", "Longitude", "dist_london_km", "geo_cluster", "cell_density", "cell_ksi_rate"],
    "Tiempo": ["hour_sin", "hour_cos", "dayofweek", "month", "is_weekend"],
    "Entorno": ["Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "is_dark"],
    "Vía": ["Road_Type", "1st_Road_Class", "Junction_Detail", "Junction_Control", "Speed_limit",
            "Urban_or_Rural_Area", "Pedestrian_Crossing"],
    "Nº de vehículos": ["Number_of_Vehicles"],
    "Sistema de registro": ["Injury_Based"],
    "Vehículos y conductores": list(VEHICLE_COLUMNS),
}
sns.set_theme(style="whitegrid", font_scale=0.95)


def predict(bundle, X):
    return bundle["model"].predict_proba(X[bundle["columns"]])


def decide(p, bundle):
    return (p * np.asarray(bundle["class_weights"])).argmax(1)


def hour_of(df):
    ang = np.arctan2(df["hour_sin"].values, df["hour_cos"].values) % (2 * np.pi)
    return (np.round(ang * 24 / (2 * np.pi)) % 24).astype(int)


def fig_confusion(sets):
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.4))
    for ax, (title, y, pred) in zip(np.atleast_1d(axes), sets):
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        norm = cm / cm.sum(1, keepdims=True)
        sns.heatmap(norm, annot=False, cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=CLASS_ES, yticklabels=CLASS_ES)
        for i in range(3):
            for j in range(3):
                ax.text(j + 0.5, i + 0.5, f"{norm[i, j]:.0%}\n({cm[i, j]:,})".replace(",", "."),
                        ha="center", va="center", fontsize=9, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real (cada fila suma 100 %)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIGURES / "10_matriz_confusion.png", dpi=160); plt.close(fig)


def fig_pr(items):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (target, name) in zip(axes, [("fatal", "Mortal"), ("ksi", "Grave o mortal")]):
        for label, y, p, color, thr in items:
            yy = (y == 2) if target == "fatal" else (y > 0)
            s = p[:, 2] if target == "fatal" else 1 - p[:, 0]
            prec, rec, _ = precision_recall_curve(yy, s)
            ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AP {average_precision_score(yy, s):.3f})")
            if target == "fatal":
                f = p[:, 2] >= thr
                ax.scatter((f & yy).sum() / yy.sum(), (f & yy).sum() / max(f.sum(), 1), color=color, s=70, zorder=5,
                           edgecolor="black")
        base = ((items[0][1] == 2) if target == "fatal" else (items[0][1] > 0)).mean()
        ax.axhline(base, color=C_GRAY, ls="--", lw=1, label=f"azar ({base:.1%})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.set_title(f"Curva precisión-recall: {name}")
        ax.legend(fontsize=8, loc="upper right")
        if target == "fatal":
            ax.set_xlim(0, 1); ax.set_ylim(0, 0.35)
            ax.text(0.98, 0.52, "● modo alerta", transform=ax.transAxes, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "11_curvas_pr.png", dpi=160); plt.close(fig)


def shap_contrib(bundle, X, n=SHAP_N):
    Xs = X[bundle["columns"]].sample(min(n, len(X)), random_state=SEED)
    contrib = bundle["model"].booster_.predict(Xs, pred_contrib=True)
    F = len(bundle["columns"])
    return Xs, contrib.reshape(len(Xs), 3, F + 1)[:, :, :F]


def fig_shap(tag, bundle, X):
    Xs, c = shap_contrib(bundle, X)
    cols = bundle["columns"]
    Xn = Xs.copy()
    for col in cols:
        if str(Xn[col].dtype) == "category":
            Xn[col] = Xn[col].cat.codes
    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(c[:, 2, :], Xn, feature_names=cols, show=False, max_display=14, plot_size=None)
    plt.title(f"SHAP · clase Mortal · modelo {tag}"); plt.tight_layout()
    plt.savefig(FIGURES / f"13_shap_mortal_{tag}.png", dpi=160, bbox_inches="tight"); plt.close()
    glob = np.abs(c).mean(0).mean(0)
    grp = {g: float(sum(glob[cols.index(v)] for v in vs if v in cols)) for g, vs in GROUPS.items()}
    return pd.Series(glob, index=cols).sort_values(ascending=False), {g: v for g, v in grp.items() if v > 0}


def fig_groups(g_a, g_b):
    df = pd.DataFrame({"A (con vehículos)": pd.Series(g_a), "B (sin vehículos)": pd.Series(g_b)}).reindex(list(GROUPS)).fillna(0)
    df = df / df.sum() * 100
    ax = df.plot.barh(figsize=(8, 4.4), color=[C_A, C_B], width=0.75)
    ax.invert_yaxis(); ax.set_xlabel("% del aporte total (media |SHAP|)"); ax.set_title("Qué grupos de variables explican el modelo")
    plt.tight_layout(); plt.savefig(FIGURES / "14_aporte_por_grupo.png", dpi=160); plt.close()
    return df


def subgroup_table(df, p):
    d = pd.DataFrame({"hora": hour_of(df), "ksi": (df["severity"] > 0).astype(int), "pred": 1 - p[:, 0],
                      "Speed_limit": df["Speed_limit"].values, "zona": df["Urban_or_Rural_Area"].astype(str).values,
                      "luz": df["Light_Conditions"].astype(str).values,
                      "registro": np.where(df["Injury_Based"].values == 1, "por lesiones", "juicio del agente")})
    d["franja"] = pd.cut(d["hora"], [-1, 5, 9, 15, 19, 23], labels=["00-05", "06-09", "10-15", "16-19", "20-23"])
    rows = []
    for var in ["franja", "Speed_limit", "zona", "luz", "registro"]:
        for k, g in d.groupby(var, observed=True):
            if len(g) < 2000 or g["ksi"].nunique() < 2:
                continue
            rows.append({"variable": var, "grupo": str(k), "n": len(g), "observado": g["ksi"].mean(),
                         "predicho": g["pred"].mean(), "AUC": roc_auc_score(g["ksi"], g["pred"])})
    t = pd.DataFrame(rows)
    t["razon_obs_pred"] = t["observado"] / t["predicho"]
    return t


def fig_subgroups(t):
    titles = {"franja": "Franja horaria", "Speed_limit": "Límite de velocidad (mph)", "zona": "Zona", "luz": "Luz",
              "registro": "Sistema de registro"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, v in zip(axes.ravel(), list(titles)):
        s = t[t.variable == v]
        y = np.arange(len(s))
        ax.barh(y + 0.2, s["observado"], 0.4, color=C_GRAY, label="Observado")
        ax.barh(y - 0.2, s["predicho"], 0.4, color=C_B, label="Predicho (sin recalibrar)")
        ax.set_yticks(y); ax.set_yticklabels(s["grupo"], fontsize=8); ax.invert_yaxis()
        ax.set_title(titles[v]); ax.set_xlabel("Tasa grave o mortal")
        for yi, a in zip(y, s["AUC"]):
            ax.text(ax.get_xlim()[1], yi, f"AUC {a:.2f}", va="center", ha="right", fontsize=7)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Modelo B en {SET_LABEL}: observado frente a predicho por subgrupo", y=1.0)
    fig.tight_layout(); fig.savefig(FIGURES / "15_subgrupos.png", dpi=160); plt.close(fig)


def residual_map(df, p, cell=0.1, min_n=150):
    d = pd.DataFrame({"lat": (df["Latitude"].values // cell) * cell + cell / 2,
                      "lon": (df["Longitude"].values // cell) * cell + cell / 2,
                      "obs": (df["severity"].values > 0).astype(int), "pred": 1 - p[:, 0]})
    g = d.groupby(["lat", "lon"]).agg(n=("obs", "size"), obs=("obs", "mean"), pred=("pred", "mean")).reset_index()
    g = g[g["n"] >= min_n].copy()
    g["resid"] = g["obs"] - g["pred"]
    g["z"] = g["resid"] / np.sqrt(np.maximum(g["pred"] * (1 - g["pred"]), 1e-6) / g["n"])
    sig = g[np.abs(g["z"]) > 2.5]
    m = folium.Map(location=[54.0, -2.5], zoom_start=6, tiles="OpenStreetMap")
    for r in sig.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=3 + min(abs(r.resid) * 40, 12), color="#c0392b" if r.resid > 0 else "#2471a3",
                            fill=True, fill_opacity=0.6, weight=1,
                            tooltip=f"{r.n} accidentes · observado {r.obs:.1%} · predicho {r.pred:.1%}").add_to(m)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #888;font-size:12px'><b>Residuo (2018–19, modelo B, sin recalibrar)</b><br>"
              "<span style='color:#c0392b'>●</span> el modelo SUBESTIMA<br><span style='color:#2471a3'>●</span> SOBRESTIMA</div>")
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(MAPS / "mapa_residuos_modelo_B.html"))
    return g, sig


def missed_fatals(name, df, p, thr):
    y = df["severity"].values == 2
    flag = p[:, 2] >= thr

    def prof(mask):
        s = df[mask]
        return {"n": int(mask.sum()), "% urbano": float((s["Urban_or_Rural_Area"].astype(str) == "Urban").mean() * 100),
                "% oscuridad": float(s["is_dark"].mean() * 100), "límite medio (mph)": float(s["Speed_limit"].mean()),
                "% fin de semana": float(s["is_weekend"].mean() * 100),
                "% madrugada (0-5 h)": float((hour_of(s) <= 5).mean() * 100),
                "vehículos medios": float(s["Number_of_Vehicles"].mean())}
    return pd.DataFrame({f"{name} · mortales detectados": prof(y & flag), f"{name} · mortales NO detectados": prof(y & ~flag),
                         f"{name} · todos los accidentes": prof(np.ones(len(df), bool))})


def main():
    t0 = time.time()
    FIGURES.mkdir(parents=True, exist_ok=True); MAPS.mkdir(parents=True, exist_ok=True)
    d, _, _ = load_splits()
    A = joblib.load(MODELS / "modelo_con_vehiculos.joblib")
    B = joblib.load(MODELS / "modelo_sin_vehiculos.joblib")
    ev = d[SET_KEY].reset_index(drop=True)
    y = ev["severity"].values
    pA, pB = predict(A, ev), predict(B, ev)
    print(f"evaluando en {SET_LABEL}: {len(ev):,} accidentes", flush=True)

    fig_confusion([(f"Modelo A · {SET_LABEL}", y, decide(pA, A)), (f"Modelo B · {SET_LABEL}", y, decide(pB, B))])
    fig_pr([(f"A · {SET_LABEL}", y, pA, C_A, A["alert_threshold"]), (f"B · {SET_LABEL}", y, pB, C_B, B["alert_threshold"])])
    print("matrices de confusión y curvas PR", flush=True)

    imp_a, g_a = fig_shap("A", A, ev)
    imp_b, g_b = fig_shap("B", B, ev)
    imp_a.head(15).rename("mean_abs_shap").to_csv(REPORTS / "shap_importancia_A.csv")
    imp_b.head(15).rename("mean_abs_shap").to_csv(REPORTS / "shap_importancia_B.csv")
    grp = fig_groups(g_a, g_b); grp.to_csv(REPORTS / "shap_por_grupo.csv")
    print("SHAP top A:", imp_a.head(8).round(4).to_dict(), flush=True)
    print("SHAP top B:", imp_b.head(8).round(4).to_dict(), flush=True)
    print("grupos (%):\n", grp.round(1).to_string(), flush=True)

    sub = subgroup_table(ev, pB); sub.to_csv(REPORTS / "subgrupos_modelo_B.csv", index=False)
    fig_subgroups(sub)
    print(sub.round(3).to_string(index=False), flush=True)
    g, sig = residual_map(ev, pB)
    sig.sort_values("z").to_csv(REPORTS / "residuos_celdas_significativas.csv", index=False)
    print(f"mapa de residuos: {len(g)} celdas, {len(sig)} significativas ({(sig.resid > 0).sum()} subestimadas / "
          f"{(sig.resid < 0).sum()} sobrestimadas); residuo medio global={g['resid'].mean():+.4f}", flush=True)

    miss = pd.concat([missed_fatals(f"A {SET_LABEL}", ev, pA, A["alert_threshold"]),
                      missed_fatals(f"B {SET_LABEL}", ev, pB, B["alert_threshold"])], axis=1)
    miss.to_csv(REPORTS / "mortales_detectados_vs_perdidos.csv")
    print(miss.round(1).to_string(), flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
