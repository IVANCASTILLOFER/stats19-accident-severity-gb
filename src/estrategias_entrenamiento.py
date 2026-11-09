"""Experimento complementario: ¿conviene reentrenar con datos recientes o basta con recalibrar el modelo antiguo?

Se comparan tres estrategias de entrenamiento, evaluadas en los MISMOS años finales (2024–2025):

  S0 · Actual        modelo congelado, entrenado con 2005–2014.
  S1 · Ventana       entrenado con 2016–2022 (años con registro por lesiones ya extendido).
  S2 · Histórico     entrenado con 2005–2022 (todos los años anteriores a la validación).

Validación común: 2023 (parada temprana, umbrales de clase y recalibración). Prueba: 2024–2025.
Para cada estrategia se informan las probabilidades sin recalibrar y recalibradas (pendiente y sesgo por sistema de registro,
ajustadas SOLO con 2023). El macro-F1 usa multiplicadores de clase ajustados con 2023 para todas las estrategias.
Las variables geográficas aprendidas (clusters, densidad y tasa por celda) se ajustan solo con los años de entrenamiento de cada estrategia.

Ejecutar desde src:   python -c "import features, estrategias_entrenamiento as e; e.main()"
"""
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

from ablation import macro_f1_from_pred
from calibration import fit_slope_bias, apply_slope_bias, metrics
from config import MODELS, SEED
from data import load_accidents
from features import GeoFeatures, _safe, basic_features, feature_columns
from recalibrate import apply_by_stratum, by_stratum_factors
from train import REPORTS, evaluate, fit_lgb, lgb_params, load_splits, tune_weights
from vehicles import VEHICLE_COLUMNS, load_vehicle_agg

N_BOOT = 100
VAL_YEAR = 2023
TEST_YEARS = [2024, 2025]
ESTRATEGIAS = {
    "S1 · Ventana reciente (2016–2022)": range(2016, 2023),
    "S2 · Histórico completo (2005–2022)": range(2005, 2023),
}


def evalua(y, p, w):
    e = evaluate(y, p, w)
    m = metrics(y, p)
    return {"macro_f1": e["macro_f1"], "recall_mortal": e["recall_fatal"], "ap_ksi": e["ap_ksi"], "ap_mortal": e["ap_fatal"],
            "pred_ksi": m["pred_ksi"], "obs_ksi": m["obs_ksi"], "brier": m["brier"], "logloss": m["logloss"], "ece": m["ece_ksi"]}


def filas(version, estrategia, n_train, arboles, y_va, p_va, ib_va, y_te, p_te, ib_te, w, fit_va=None):
    """Dos filas: sin recalibrar y recalibrada con 2023 (pendiente y sesgo por sistema de registro)."""
    out = [{"version": version, "estrategia": estrategia, "esquema": "Sin recalibrar", "n_entrenamiento": n_train, "arboles": arboles,
            **evalua(y_te, p_te, w)}]
    par = by_stratum_factors(p_va, y_va, ib_va, fit_slope_bias)
    pr = apply_by_stratum(p_te, ib_te, par, apply_slope_bias)
    out.append({"version": version, "estrategia": estrategia, "esquema": "Recalibrada con 2023", "n_entrenamiento": n_train, "arboles": arboles,
                **evalua(y_te, pr, w)})
    return out


def marco():
    df = load_accidents().dropna(subset=["Latitude", "Longitude"])
    df = df.merge(load_vehicle_agg(), on="Accident_Index", how="left")
    df["has_vehicle_info"] = df["has_vehicle_info"].fillna(0).astype("int8")
    return basic_features(df)


def particiones(df, anios_train):
    """Variables con ajuste geográfico solo con los años de entrenamiento de la estrategia."""
    tr = df[df["Year"].isin(list(anios_train))].copy()
    va = df[df["Year"] == VAL_YEAR].copy()
    te = df[df["Year"].isin(TEST_YEARS)].copy()
    geo = GeoFeatures().fit(tr)
    tr, va, te = geo.transform(tr, is_train=True), geo.transform(va), geo.transform(te)
    num, cat = feature_columns(tr)
    keep = num + cat + VEHICLE_COLUMNS + ["ksi", "severity", "Year"]
    parts = {}
    for nombre, x in (("train", tr), ("val", va), ("test", te)):
        x = x[keep].copy()
        x.columns = [_safe(c) for c in x.columns]
        parts[nombre] = x.reset_index(drop=True)
    num_s, cat_s = [_safe(c) for c in num], [_safe(c) for c in cat]
    for c in cat_s:   # mismas categorías en todas las particiones
        cats = parts["train"][c].astype("category").cat.categories
        for x in parts.values():
            x[c] = pd.Categorical(x[c], categories=cats)
    return parts, num_s, cat_s


def main():
    t0 = time.time()
    resultados = []
    guardado = {}   # (versión, estrategia) -> (y, probabilidades sin recalibrar, multiplicadores)

    # ---------------------------------------------------------------- S0: modelo congelado (2005–2014)
    d, num, cat = load_splits()
    post = d["post"]
    va, te = post[post["Year"] == VAL_YEAR].reset_index(drop=True), post[post["Year"].isin(TEST_YEARS)].reset_index(drop=True)
    for version, ruta in (("A", "modelo_con_vehiculos"), ("B", "modelo_sin_vehiculos")):
        b = joblib.load(MODELS / f"{ruta}.joblib")
        pv, pt = b["model"].predict_proba(va[b["columns"]]), b["model"].predict_proba(te[b["columns"]])
        w = tune_weights(va["severity"].values, pv)
        guardado[(version, "S0")] = (te["severity"].values, pt, w)
        resultados += filas(version, "S0 · Actual (2005–2014)", int(len(d["train"])), int(b["model"].best_iteration_), va["severity"].values, pv,
                            va["Injury_Based"].values.astype(int), te["severity"].values, pt, te["Injury_Based"].values.astype(int), w)
        print(f"[S0 {version}] listo · {(time.time()-t0)/60:.1f} min", flush=True)
    del d, post, va, te

    # ---------------------------------------------------------------- S1 y S2: reentrenamiento
    df = marco()
    for nombre, anios in ESTRATEGIAS.items():
        parts, num, cat = particiones(df, anios)
        tr, va, te = parts["train"], parts["val"], parts["test"]
        base = num + cat
        for version, cols in (("A", base + list(VEHICLE_COLUMNS)), ("B", base)):
            t1 = time.time()
            m = fit_lgb(tr[cols], tr["severity"].values, va[cols], va["severity"].values, lgb_params())
            pv, pt = m.predict_proba(va[cols]), m.predict_proba(te[cols])
            w = tune_weights(va["severity"].values, pv)
            guardado[(version, nombre[:2])] = (te["severity"].values, pt, w)
            resultados += filas(version, nombre, int(len(tr)), int(m.best_iteration_), va["severity"].values, pv,
                                va["Injury_Based"].values.astype(int), te["severity"].values, pt, te["Injury_Based"].values.astype(int), w)
            print(f"[{nombre} {version}] {len(tr):,} filas · {m.best_iteration_} árboles · {time.time()-t1:.0f} s · total {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- contrastes pareados (remuestreo con los mismos accidentes para las tres estrategias)
    cont = []
    for version in ("A", "B"):
        y0, p0, w0 = guardado[(version, "S0")]
        for s in ("S1", "S2"):
            assert np.array_equal(y0, guardado[(version, s)][0]), "los accidentes de prueba no coinciden entre estrategias"
        rng = np.random.RandomState(SEED)
        dif = {s: {"macro_f1": [], "ap_ksi": [], "ap_mortal": []} for s in ("S1", "S2")}
        for _ in range(N_BOOT):
            i = rng.randint(0, len(y0), len(y0))
            def met(p, w):
                return (macro_f1_from_pred(y0[i], (p[i] * w).argmax(1)), average_precision_score(y0[i] > 0, 1 - p[i, 0]),
                        average_precision_score(y0[i] == 2, p[i, 2]))
            b0 = met(p0, w0)
            for s in ("S1", "S2"):
                _, ps, ws = guardado[(version, s)]
                bs = met(ps, ws)
                for k, nombre_m in enumerate(("macro_f1", "ap_ksi", "ap_mortal")):
                    dif[s][nombre_m].append(bs[k] - b0[k])
        for s in ("S1", "S2"):
            for nombre_m, v in dif[s].items():
                lo, hi = np.percentile(v, [2.5, 97.5])
                cont.append({"version": version, "contraste": f"{s} − S0", "metrica": nombre_m, "dif_media": float(np.mean(v)), "ic95_inf": float(lo), "ic95_sup": float(hi)})
        print(f"[contrastes {version}] listos · {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(cont).to_csv(REPORTS / "estrategias_contrastes.csv", index=False)
    print(pd.DataFrame(cont).round(4).to_string(index=False), flush=True)

    res = pd.DataFrame(resultados)
    res.to_csv(REPORTS / "estrategias_entrenamiento.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["version", "estrategia", "esquema", "macro_f1", "ap_mortal", "ap_ksi", "pred_ksi", "obs_ksi", "ece", "brier"]].round(4).to_string(index=False))
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
