"""Recalibración de probabilidades por sistema de registro de la gravedad (Injury_Based = 0 o 1).

Validación pasado → futuro: factores ajustados con 2022–2023 y evaluados en 2024–2025, frente a
(a) sin recalibrar y (b) factores globales «viejos» ajustados con 2016–2017.
Los factores finales (2022–2025, por sistema de registro) se guardan en los modelos para la app.
Las decisiones y la alerta siguen usando las probabilidades ORIGINALES (sus umbrales se ajustaron con 2015).
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration import apply_factors, apply_slope_bias, ece_ksi, fit_factors, fit_slope_bias, metrics
from config import FIGURES, MODELS
from train import REPORTS, load_splits

C_RAW, C_OLD, C_NEW, C_GRAY = "#8a8f98", "#e07b39", "#1f6f8b", "#444444"


def predict(b, X):
    return b["model"].predict_proba(X[b["columns"]])


def reliability(y, p, bins=10):
    s, t = 1 - p[:, 0], (y > 0).astype(float)
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
    return (np.array([s[idx == b].mean() for b in range(bins)]),
            np.array([t[idx == b].mean() for b in range(bins)]))


def by_stratum_factors(p, y, ib, fit=fit_factors):
    return {int(k): fit(p[ib == k], y[ib == k]) for k in (0, 1)}


def apply_by_stratum(p, ib, fac, apply=apply_factors):
    out = p.copy()
    for k, f in fac.items():
        m = ib == k
        out[m] = apply(p[m], f)
    return out


def main():
    d, _, _ = load_splits()
    allv = pd.concat([d["test"], d["holdout"], d["estres"], d["post"]], ignore_index=True)
    y, yr, ib = allv["severity"].values, allv["Year"].values, allv["Injury_Based"].values.astype(int)
    rows, plots = [], []
    for mname, path in (("A", "modelo_con_vehiculos"), ("B", "modelo_sin_vehiculos")):
        b = joblib.load(MODELS / f"{path}.joblib")
        p = predict(b, allv)
        old = fit_factors(p[(yr >= 2016) & (yr <= 2017)], y[(yr >= 2016) & (yr <= 2017)])      # global, viejo
        fit_m = (yr >= 2022) & (yr <= 2023)
        eva_m = (yr >= 2024) & (yr <= 2025)
        glob_new = fit_factors(p[fit_m], y[fit_m])
        strat_new = by_stratum_factors(p[fit_m], y[fit_m], ib[fit_m])
        sb_new = by_stratum_factors(p[fit_m], y[fit_m], ib[fit_m], fit_slope_bias)
        variants = {"Sin recalibrar": p[eva_m],
                    "Factores globales de 2016–17 (desfasados)": apply_factors(p[eva_m], old),
                    "Factores globales de 2022–23": apply_factors(p[eva_m], glob_new),
                    "Factores por clase y por sistema de registro, 2022–23": apply_by_stratum(p[eva_m], ib[eva_m], strat_new),
                    "Pendiente y sesgo por sistema de registro, 2022–23":
                        apply_by_stratum(p[eva_m], ib[eva_m], sb_new, apply_slope_bias)}
        for vname, pv in variants.items():
            m = metrics(y[eva_m], pv)
            rows.append({"modelo": mname, "esquema": vname, "evaluado_en": "2024–2025", **m})
            # error dentro de cada sistema de registro
            for k, nm in ((0, "juicio del agente"), (1, "por lesiones")):
                mk = ib[eva_m] == k
                rows[-1][f"ece_{nm}"] = ece_ksi(y[eva_m][mk], pv[mk])
        plots.append((mname, y[eva_m], variants))
        # factores finales para la app: 2022–2025 por sistema de registro
        fin_m = (yr >= 2022) & (yr <= 2025)
        fin = by_stratum_factors(p[fin_m], y[fin_m], ib[fin_m], fit_slope_bias)
        b["calibration"] = {
            "method": "pendiente y sesgo por sistema de registro",
            "fit_on": "2022-2025",
            "params_by_ib": {str(k): v.tolist() for k, v in fin.items()},
            "obs_ksi_by_ib": {str(k): float((y[fin_m & (ib == k)] > 0).mean()) for k in (0, 1)},
            "obs_ksi": float((y[fin_m] > 0).mean()), "obs_fatal": float((y[fin_m] == 2).mean())}
        joblib.dump(b, MODELS / f"{path}.joblib")
        print(f"modelo {mname}: parámetros finales (a, b1, b2) por sistema de registro (2022–25):",
              {k: np.round(v, 3).tolist() for k, v in fin.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "recalibracion.csv", index=False)
    print(res[["modelo", "esquema", "pred_ksi", "obs_ksi", "ece_ksi", "ece_juicio del agente", "ece_por lesiones",
               "brier", "logloss"]].round(4).to_string(index=False), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, (mname, ye, variants) in zip(axes, plots):
        for (vname, pv), color in zip(variants.items(), (C_RAW, C_OLD, "#9ecae1", "#6baed6", C_NEW)):
            x, o = reliability(ye, pv)
            ax.plot(x, o, "o-", color=color, lw=2.2 if vname.startswith("Pendiente") else 1.2, label=vname)
        ax.plot([0, 0.6], [0, 0.6], "--", color=C_GRAY, lw=1)
        ax.set_title(f"Modelo {mname} · evaluado en 2024–2025", fontsize=10)
        ax.set_xlabel("P(grave o mortal) predicha (deciles)")
    axes[0].set_ylabel("Tasa observada"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "16_recalibracion.png", dpi=160); plt.close(fig)
    print("== Terminado", flush=True)


if __name__ == "__main__":
    main()
