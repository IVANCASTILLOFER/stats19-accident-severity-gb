"""Comparación de algoritmos con el mismo conjunto de variables (modelo A, 37 variables) y los mismos años.

Regresión logística multinomial (interpretable; entrenada con TODOS los datos de entrenamiento), XGBoost y LightGBM.
Entrenamiento 2005–2014; umbrales por clase ajustados con 2015; evaluación en 2016–17, 2018–19, 2020–21 y 2022–25.
Sin recalibrar y con parámetros por defecto razonables (no se ajustaron hiperparámetros a ningún modelo).
"""
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ablation import macro_f1_from_pred
from calibration import ece_ksi
from config import MODELS, SEED
from train import REPORTS, evaluate, load_splits, tune_weights

N_BOOT = 100
SETS = [("test 2016-17", "test"), ("holdout 2018-19", "holdout"), ("estres 2020-21", "estres"), ("post 2022-25", "post")]


def main():
    t0 = time.time()
    d, _, _ = load_splits()
    A = joblib.load(MODELS / "modelo_con_vehiculos.joblib")
    cols = A["columns"]
    cat_cols = [c for c in cols if str(d["train"][c].dtype) == "category"]
    num_cols = [c for c in cols if c not in cat_cols]
    ytr, yva = d["train"]["severity"].values, d["val"]["severity"].values
    probs, weights, info = {}, {}, {}

    # ---- LightGBM (el modelo A ya entrenado y congelado)
    probs["LightGBM"] = {k: A["model"].predict_proba(d[k][cols]) for k in ["val"] + [s[1] for s in SETS]}
    weights["LightGBM"] = np.asarray(A["class_weights"])
    info["LightGBM"] = f"{A['model'].best_iteration_} árboles"

    # ---- XGBoost
    t1 = time.time()
    xg = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", enable_categorical=True, n_estimators=1000,
                           learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           early_stopping_rounds=40, eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    xg.fit(d["train"][cols], ytr, eval_set=[(d["val"][cols], yva)], verbose=False)
    probs["XGBoost"] = {k: xg.predict_proba(d[k][cols]) for k in ["val"] + [s[1] for s in SETS]}
    weights["XGBoost"] = tune_weights(yva, probs["XGBoost"]["val"])
    info["XGBoost"] = f"{xg.best_iteration} árboles, {time.time()-t1:.0f}s"
    print("  XGBoost:", info["XGBoost"], flush=True)

    # ---- Regresión logística (todos los datos)
    t1 = time.time()
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_cols),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=500), cat_cols)])
    lr = Pipeline([("p", pre), ("m", LogisticRegression(max_iter=150, tol=1e-3))])

    def lr_x(df):
        x = df[cols].copy()
        for c in cat_cols:
            x[c] = x[c].astype(str)
        return x

    lr.fit(lr_x(d["train"]), ytr)
    probs["Regresión logística"] = {k: lr.predict_proba(lr_x(d[k])) for k in ["val"] + [s[1] for s in SETS]}
    weights["Regresión logística"] = tune_weights(yva, probs["Regresión logística"]["val"])
    info["Regresión logística"] = f"{time.time()-t1:.0f}s (iteraciones: {lr.named_steps['m'].n_iter_})"
    print("  Regresión logística:", info["Regresión logística"], flush=True)

    rows, ci_rows = [], []
    for mname in probs:
        for label, key in SETS:
            y = d[key]["severity"].values
            p = probs[mname][key]
            e = evaluate(y, p, weights[mname])
            rows.append({"modelo": mname, "conjunto": label, **e, "ece_ksi": ece_ksi(y, p),
                         "prevalencia_ksi": float((y > 0).mean())})
            pred = (p * weights[mname]).argmax(1)
            rng = np.random.RandomState(SEED)
            v = [macro_f1_from_pred(y[i], pred[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(N_BOOT))]
            lo, hi = np.percentile(v, [2.5, 97.5])
            ci_rows.append({"modelo": mname, "conjunto": label, "macro_f1_ic95_inf": lo, "macro_f1_ic95_sup": hi})
    res = pd.DataFrame(rows).merge(pd.DataFrame(ci_rows), on=["modelo", "conjunto"])
    res.to_csv(REPORTS / "comparacion_modelos.csv", index=False)

    diffs = []
    for label, key in SETS:
        y = d[key]["severity"].values
        pr = {m: (probs[m][key] * weights[m]).argmax(1) for m in probs}
        for other in ("XGBoost", "Regresión logística"):
            rng = np.random.RandomState(SEED)
            dd = []
            for _ in range(N_BOOT):
                i = rng.randint(0, len(y), len(y))
                dd.append(macro_f1_from_pred(y[i], pr["LightGBM"][i]) - macro_f1_from_pred(y[i], pr[other][i]))
            lo, hi = np.percentile(dd, [2.5, 97.5])
            diffs.append({"conjunto": label, "contraste": f"LightGBM − {other}", "dif_macro_f1": float(np.mean(dd)),
                          "ic95_inf": float(lo), "ic95_sup": float(hi)})
    out = pd.DataFrame(diffs)
    out.to_csv(REPORTS / "comparacion_contrastes.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["modelo", "conjunto", "macro_f1", "macro_f1_ic95_inf", "macro_f1_ic95_sup", "ap_ksi", "ap_fatal", "recall_fatal",
               "ece_ksi"]].round(4).to_string(index=False), flush=True)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
