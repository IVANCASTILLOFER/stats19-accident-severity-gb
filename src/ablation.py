"""Fase 5b: ablaciones y mejoras con LightGBM (parámetros por defecto, rápido).

V0 base · V1 sin Number_of_Vehicles · V2 sin Police_Force · V3 sin ambas
V4 = V3 + vehículos · V5 = base + vehículos · V6 = V4 con peso por año
V7 = V4 sin variables geográficas (para H1, con bootstrap)
Los umbrales por clase se ajustan en validación (2015) y se aplican a prueba.
Se informa prueba completa (2016-2017), solo 2016 (con datos de vehículos) y solo 2017 (sin ellos).
"""
import time

import joblib
import numpy as np
import pandas as pd

from config import MODELS, SEED
from train import REPORTS, GEO_COLS, evaluate, fit_lgb, lgb_params, load_splits, tune_weights
from vehicles import VEHICLE_COLUMNS

N_BOOT = 300


def macro_f1_from_pred(y, pred):
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def main():
    t0 = time.time()
    d, num, cat = load_splits()
    base = num + cat
    veh = [c for c in VEHICLE_COLUMNS]
    minus = lambda cols, drop: [c for c in cols if c not in drop]
    variants = {
        "V0 base (con geografía)": (base, None),
        "V1 sin Number_of_Vehicles": (minus(base, ["Number_of_Vehicles"]), None),
        "V2 sin Police_Force": (minus(base, ["Police_Force"]), None),
        "V3 sin ambas": (minus(base, ["Number_of_Vehicles", "Police_Force"]), None),
        "V4 V3 + vehículos": (minus(base, ["Number_of_Vehicles", "Police_Force"]) + veh, None),
        "V5 base + vehículos": (base + veh, None),
        "V6 V4 + peso por año": (minus(base, ["Number_of_Vehicles", "Police_Force"]) + veh, "year"),
        "V7 V4 sin geografía": (minus(minus(base, ["Number_of_Vehicles"]) + veh, GEO_COLS), None),
    }
    ytr, yva, yte = (d[n]["severity"].values for n in ("train", "val", "test"))
    year_te = d["test"]["Year"].values
    rows, store = [], {}
    for name, (cols, wmode) in variants.items():
        t1 = time.time()
        p = lgb_params()
        m = lgb_fit(d, cols, ytr, yva, p, wmode)
        pv, pt = m.predict_proba(d["val"][cols]), m.predict_proba(d["test"][cols])
        w = tune_weights(yva, pv)
        store[name] = (m, cols, w, pv, pt)
        val_f1 = evaluate(yva, pv, w)["macro_f1"]
        for tag, mask in (("test 2016-17", np.ones(len(yte), bool)),
                          ("test 2016", year_te == 2016), ("test 2017", year_te == 2017)):
            r = evaluate(yte[mask], pt[mask], w)
            rows.append({"variante": name, "conjunto": tag, "n_vars": len(cols),
                         "macro_f1_val": val_f1, **r})
        a = [r for r in rows if r["variante"] == name and r["conjunto"] == "test 2016-17"][0]
        b = [r for r in rows if r["variante"] == name and r["conjunto"] == "test 2016"][0]
        print(f"  {name:28s} val={val_f1:.4f} test={a['macro_f1']:.4f} test2016={b['macro_f1']:.4f} "
              f"APksi={a['ap_ksi']:.3f} recF={a['recall_fatal']:.3f} ({time.time()-t1:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "ablaciones.csv", index=False)

    # H1: bootstrap pareado V4 (con geografía) vs V7 (sin geografía) en prueba 2016
    print("== Bootstrap H1 (V4 vs V7), prueba 2016", flush=True)
    mask = year_te == 2016
    y = yte[mask]
    pa = (store["V4 V3 + vehículos"][4][mask] * store["V4 V3 + vehículos"][2]).argmax(1)
    pb = (store["V7 V4 sin geografía"][4][mask] * store["V7 V4 sin geografía"][2]).argmax(1)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        diffs.append(macro_f1_from_pred(y[i], pa[i]) - macro_f1_from_pred(y[i], pb[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  diferencia macro-F1 (con geo - sin geo): {np.mean(diffs):+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]", flush=True)
    pd.DataFrame({"diff_macro_f1": diffs}).to_csv(REPORTS / "bootstrap_h1.csv", index=False)

    # mejor variante elegida por validación (no por prueba)
    best = res[res.conjunto == "test 2016-17"].sort_values("macro_f1_val", ascending=False).iloc[0]["variante"]
    m, cols, w, pv, pt = store[best]
    joblib.dump({"model": m, "columns": cols, "class_weights": w, "variante": best},
                MODELS / "modelo_mejorado.joblib")
    np.save(MODELS / "proba_test_mejorado.npy", pt)
    print(f"== Mejor por validación: {best}", flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


def lgb_fit(d, cols, ytr, yva, params, wmode):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**params)
    sw = None
    if wmode == "year":
        sw = 1.0 + 2.0 * (d["train"]["Year"].values - 2005) / 9.0
    m.fit(d["train"][cols], ytr, sample_weight=sw, eval_set=[(d["val"][cols], yva)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    return m


if __name__ == "__main__":
    main()
