"""Validación con separación geográfica.

Se reservan al azar el 20 % de los bloques de 0,5° (≈55 × 33 km). El modelo A se entrena con 2005–2014 SOLO en los
bloques restantes (también las variables geográficas aprendidas: clústeres, densidad y tasa por celda) y se evalúa en
2018–2019 y 2022–2025 dentro de los bloques vistos y de los bloques no vistos. Se compara con y sin variables geográficas.
Pregunta: ¿la ubicación ayuda también en lugares que el modelo no vio durante el entrenamiento?
"""
import time

import numpy as np
import pandas as pd

from ablation import macro_f1_from_pred
from config import SEED
from data import load_accidents
from features import EXTRA_SETS, TRAIN_YEARS, VAL_YEARS, GeoFeatures, _safe, basic_features, feature_columns
from train import GEO_COLS, REPORTS, evaluate, fit_lgb, lgb_params, tune_weights
from vehicles import VEHICLE_COLUMNS, load_vehicle_agg

BLOCK = 0.5
HELD_FRACTION = 0.20
N_BOOT = 150


def main():
    t0 = time.time()
    df = load_accidents().dropna(subset=["Latitude", "Longitude"])
    df = df.merge(load_vehicle_agg(), on="Accident_Index", how="left")
    df["has_vehicle_info"] = df["has_vehicle_info"].fillna(0).astype("int8")
    df = basic_features(df)
    block = (np.floor(df["Latitude"] / BLOCK).astype("int64") * 1000 + np.floor(df["Longitude"] / BLOCK).astype("int64") + 100)
    ub = np.unique(block)
    rng = np.random.RandomState(SEED)
    held = set(rng.choice(ub, int(len(ub) * HELD_FRACTION), replace=False).tolist())
    df["held"] = block.isin(held).values
    print(f"bloques totales: {len(ub)} | reservados: {len(held)} | accidentes en bloques reservados: {df.held.mean():.1%}", flush=True)

    tr = df[df["Year"].isin(TRAIN_YEARS) & ~df["held"]].copy()
    va = df[df["Year"].isin(VAL_YEARS) & ~df["held"]].copy()
    geo = GeoFeatures().fit(tr)                       # SOLO con bloques vistos
    tr, va = geo.transform(tr, is_train=True), geo.transform(va)
    num, cat = feature_columns(tr)
    cols_raw = num + cat + VEHICLE_COLUMNS
    cols = [_safe(c) for c in cols_raw]
    sets = {"holdout 2018-19": geo.transform(df[df["Year"].isin(EXTRA_SETS["holdout"])].copy()),
            "post 2022-25": geo.transform(df[df["Year"].isin(EXTRA_SETS["post"])].copy())}

    cat_safe = [_safe(c) for c in cat]
    ref_cats = {c: tr[raw].astype("category").cat.categories for c, raw in zip(cat_safe, cat)}

    def prep2(d):
        x = d[cols_raw].copy()
        x.columns = cols
        for c in cat_safe:
            x[c] = pd.Categorical(x[c].astype(object).where(x[c].notna(), None), categories=ref_cats[c])
        return x

    Xtr, Xva = prep2(tr), prep2(va)
    ytr, yva = tr["severity"].values, va["severity"].values
    variants = {"con geografía": cols, "sin geografía": [c for c in cols if c not in GEO_COLS]}
    models, weights = {}, {}
    for name, vc in variants.items():
        t1 = time.time()
        m = fit_lgb(Xtr[vc], ytr, Xva[vc], yva, lgb_params())
        pv = m.predict_proba(Xva[vc])
        models[name], weights[name] = m, tune_weights(yva, pv)
        print(f"  {name}: {m.best_iteration_} árboles, {time.time()-t1:.0f}s", flush=True)

    rows, diffs = [], []
    for label, d in sets.items():
        X = prep2(d)
        y = d["severity"].values
        hm = d["held"].values
        preds, probs = {}, {}
        for name, vc in variants.items():
            p = models[name].predict_proba(X[vc])
            probs[name] = p
            preds[name] = (p * weights[name]).argmax(1)
        for gname, mask in (("bloques vistos", ~hm), ("bloques NO vistos", hm)):
            for name in variants:
                e = evaluate(y[mask], probs[name][mask], weights[name])
                rows.append({"conjunto": label, "grupo": gname, "variante": name, "n": int(mask.sum()),
                             "macro_f1": e["macro_f1"], "ap_ksi": e["ap_ksi"], "ap_fatal": e["ap_fatal"]})
            idx = np.where(mask)[0]
            rb = np.random.RandomState(SEED)
            dd = []
            for _ in range(N_BOOT):
                i = idx[rb.randint(0, len(idx), len(idx))]
                dd.append(macro_f1_from_pred(y[i], preds["con geografía"][i]) - macro_f1_from_pred(y[i], preds["sin geografía"][i]))
            lo, hi = np.percentile(dd, [2.5, 97.5])
            diffs.append({"conjunto": label, "grupo": gname, "dif_macro_f1 (con - sin geografía)": float(np.mean(dd)),
                          "ic95_inf": float(lo), "ic95_sup": float(hi), "n": int(mask.sum())})
    pd.DataFrame(rows).to_csv(REPORTS / "separacion_geografica_metricas.csv", index=False)
    out = pd.DataFrame(diffs)
    out.to_csv(REPORTS / "separacion_geografica_contraste.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
