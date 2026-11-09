"""Fase 5c: dos modelos finales (con y sin datos de vehículos) y modo alerta para mortales.

A) con vehículos: entrenado solo con accidentes que tienen datos de vehículos.
B) sin vehículos: entrenado con todos los accidentes, sin esas variables.
Ambos sin Police_Force. Umbrales por clase y umbral de alerta ajustados en validación (2015).
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import MODELS
from train import REPORTS, evaluate, fit_lgb, lgb_params, load_splits, tune_weights
from vehicles import VEHICLE_COLUMNS

MIN_PRECISION_FATAL = 0.08   # precisión mínima exigida al modo alerta


def alert_threshold(y_fatal, p_fatal, min_precision):
    """Umbral de p(Fatal) con la mayor recall posible y precisión >= min_precision (en validación)."""
    prec, rec, thr = precision_recall_curve(y_fatal, p_fatal)
    ok = np.where(prec[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return None
    i = ok[np.argmax(rec[:-1][ok])]
    return float(thr[i])


def alert_metrics(y_fatal, p_fatal, thr):
    if thr is None:
        return {"alert_precision": np.nan, "alert_recall": np.nan, "alert_flagged_pct": np.nan}
    flag = p_fatal >= thr
    tp = int((flag & (y_fatal == 1)).sum())
    return {"alert_precision": tp / max(flag.sum(), 1), "alert_recall": tp / max(int(y_fatal.sum()), 1),
            "alert_flagged_pct": float(flag.mean())}


def run_model(name, d, cols, train_mask, rows):
    ytr = d["train"]["severity"].values[train_mask]
    yva, yte = d["val"]["severity"].values, d["test"]["severity"].values
    Xtr = d["train"][cols][train_mask]
    t0 = time.time()
    m = fit_lgb(Xtr, ytr, d["val"][cols], yva, lgb_params())
    pv, pt = m.predict_proba(d["val"][cols]), m.predict_proba(d["test"][cols])
    w = tune_weights(yva, pv)
    thr = alert_threshold((yva == 2).astype(int), pv[:, 2], MIN_PRECISION_FATAL)
    year_te = d["test"]["Year"].values
    print(f"  {name}: {m.best_iteration_} árboles, {len(ytr):,} filas, {time.time()-t0:.0f}s, "
          f"umbral alerta={thr}", flush=True)
    subsets = {"val 2015": (np.ones(len(yva), bool), yva, pv), "test 2016": (year_te == 2016, yte, pt),
               "test 2017": (year_te == 2017, yte, pt), "test 2016-17": (np.ones(len(yte), bool), yte, pt)}
    for tag, (mask, y, p) in subsets.items():
        r = evaluate(y[mask], p[mask], w)
        rows.append({"modelo": name, "conjunto": tag, **r,
                     **alert_metrics((y[mask] == 2).astype(int), p[mask][:, 2], thr)})
    cats = {c: list(d["train"][c].cat.categories) for c in cols if str(d["train"][c].dtype) == "category"}
    return {"model": m, "columns": cols, "class_weights": w, "alert_threshold": thr,
            "min_precision_fatal": MIN_PRECISION_FATAL, "categories": cats}, pt


def main():
    t0 = time.time()
    d, num, cat = load_splits()
    base = [c for c in num + cat if c != "Police_Force"]
    veh = list(VEHICLE_COLUMNS)
    rows = []
    has_veh = d["train"]["has_vehicle_info"].values == 1
    print(f"filas de entrenamiento con vehículos: {has_veh.sum():,} de {len(has_veh):,}", flush=True)
    A, ptA = run_model("A con vehículos", d, base + veh, has_veh, rows)
    B, ptB = run_model("B sin vehículos", d, base, np.ones(len(has_veh), bool), rows)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(A, MODELS / "modelo_con_vehiculos.joblib")
    joblib.dump(B, MODELS / "modelo_sin_vehiculos.joblib")
    np.save(MODELS / "proba_test_A.npy", ptA)
    np.save(MODELS / "proba_test_B.npy", ptB)
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "modelos_finales.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False), flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
