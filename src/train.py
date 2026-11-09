"""Fase 5: modelado. Baselines, LightGBM (ablación geo), ajuste de umbrales y Optuna.

Uso:  python -u train.py            (todo)      |  python -u train.py quick   (sin Optuna)
"""
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS, PROCESSED, SEED
from features import _safe, feature_columns

REPORTS = PROCESSED.parent.parent / "reports"
GEO_COLS = ["Latitude", "Longitude", "dist_london_km", "geo_cluster", "cell_density",
            "cell_ksi_rate", "Police_Force"]
N_OPTUNA = 12
OPTUNA_SUBSAMPLE = 600_000


# ---------- datos ----------
def load_splits():
    raw_num, raw_cat = feature_columns(None)
    num, cat = [_safe(c) for c in raw_num], [_safe(c) for c in raw_cat]
    names = ["train", "val", "test"] + [n for n in ("holdout", "estres", "post")
                                        if (PROCESSED / f"features_{n}.parquet").exists()]
    d = {n: pd.read_parquet(PROCESSED / f"features_{n}.parquet") for n in names}
    for c in cat:  # mismas categorías en todas las particiones
        cats = d["train"][c].astype("category").cat.categories
        for n in d:
            d[n][c] = pd.Categorical(d[n][c], categories=cats)
    return d, num, cat


# ---------- métricas ----------
def evaluate(y, proba, w=None):
    w = np.ones(3) if w is None else np.asarray(w)
    pred = (proba * w).argmax(1)
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y, pred),
        "recall_slight": rec[0], "recall_serious": rec[1], "recall_fatal": rec[2],
        "ap_fatal": average_precision_score(y == 2, proba[:, 2]),
        "ap_ksi": average_precision_score(y > 0, 1 - proba[:, 0]),
    }


def tune_weights(y, proba):
    """Multiplicadores por clase (Serious, Fatal) que maximizan macro-F1 en validación."""
    grid = np.logspace(-0.3, 1.7, 21)
    best, best_w = -1, np.ones(3)
    for w1 in grid:
        for w2 in grid:
            w = np.array([1.0, w1, w2])
            s = f1_score(y, (proba * w).argmax(1), average="macro")
            if s > best:
                best, best_w = s, w
    return best_w


RESULTS = []


def log(model, w, y_val, p_val, y_test, p_test, note=""):
    for split, y, p in (("val", y_val, p_val), ("test", y_test, p_test)):
        r = {"modelo": model, "ajuste_umbral": note, "split": split, **evaluate(y, p, w)}
        RESULTS.append(r)
    t = RESULTS[-1]
    print(f"  [{model} {note}] test macroF1={t['macro_f1']:.4f} recallFatal={t['recall_fatal']:.3f} "
          f"recallSerious={t['recall_serious']:.3f} AP_ksi={t['ap_ksi']:.3f}", flush=True)


# ---------- modelos ----------
def run_baselines(d, num, cat):
    Xtr, ytr = d["train"][num + cat], d["train"]["severity"].values
    Xva, yva = d["val"][num + cat], d["val"]["severity"].values
    Xte, yte = d["test"][num + cat], d["test"]["severity"].values

    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    log("Dummy (clase mayoritaria)", None, yva, dummy.predict_proba(Xva), yte, dummy.predict_proba(Xte))

    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=200), cat),
    ])
    lr = Pipeline([("p", pre), ("m", LogisticRegression(max_iter=300, class_weight="balanced"))])
    sub = Xtr.assign(_y=ytr).sample(400_000, random_state=SEED)
    sub_y = sub.pop("_y")
    for c in cat:
        sub[c] = sub[c].astype(str)
    for X in (Xva, Xte):
        for c in cat:
            X[c] = X[c].astype(str)
    t0 = time.time()
    lr.fit(sub, sub_y)
    print(f"  logreg entrenada en {time.time()-t0:.0f}s", flush=True)
    pv, pt = lr.predict_proba(Xva), lr.predict_proba(Xte)
    log("Regresión logística (balanced)", None, yva, pv, yte, pt, "argmax")
    w = tune_weights(yva, pv)
    log("Regresión logística (balanced)", w, yva, pv, yte, pt, "umbral ajustado")
    for X in (Xva, Xte):
        for c in cat:
            X[c] = pd.Categorical(X[c], categories=d["train"][c].cat.categories)


def lgb_params(**kw):
    p = dict(objective="multiclass", num_class=3, learning_rate=0.1, num_leaves=63,
             min_child_samples=100, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
             reg_lambda=1.0, n_estimators=1000, random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(kw)
    return p


def fit_lgb(Xtr, ytr, Xva, yva, params):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    return m


def run_lgbm(d, num, cat):
    ytr, yva, yte = (d[n]["severity"].values for n in ("train", "val", "test"))
    out = {}
    for name, drop in (("LightGBM sin variables geográficas", GEO_COLS), ("LightGBM con geografía", [])):
        cols = [c for c in num + cat if c not in drop]
        t0 = time.time()
        m = fit_lgb(d["train"][cols], ytr, d["val"][cols], yva, lgb_params())
        pv, pt = m.predict_proba(d["val"][cols]), m.predict_proba(d["test"][cols])
        print(f"  {name}: {m.best_iteration_} árboles, {time.time()-t0:.0f}s", flush=True)
        log(name, None, yva, pv, yte, pt, "argmax")
        w = tune_weights(yva, pv)
        log(name, w, yva, pv, yte, pt, "umbral ajustado")
        out[name] = (m, cols, w, pv, pt)
    return out


def run_optuna(d, num, cat):
    cols = num + cat
    ytr, yva, yte = (d[n]["severity"].values for n in ("train", "val", "test"))
    idx = np.random.RandomState(SEED).choice(len(ytr), OPTUNA_SUBSAMPLE, replace=False)
    Xs, ys = d["train"][cols].iloc[idx], ytr[idx]

    def objective(trial):
        p = lgb_params(
            learning_rate=0.1,
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            cat_smooth=trial.suggest_float("cat_smooth", 5, 100),
            n_estimators=500)
        m = fit_lgb(Xs, ys, d["val"][cols], yva, p)
        pv = m.predict_proba(d["val"][cols])
        w = tune_weights(yva, pv)
        return f1_score(yva, (pv * w).argmax(1), average="macro")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=N_OPTUNA, callbacks=[
        lambda s, t: print(f"  trial {t.number}: macroF1(val)={t.value:.4f} ({time.time()-t0:.0f}s)", flush=True)])
    best = lgb_params(**study.best_params)
    m = fit_lgb(d["train"][cols], ytr, d["val"][cols], yva, best)
    pv, pt = m.predict_proba(d["val"][cols]), m.predict_proba(d["test"][cols])
    w = tune_weights(yva, pv)
    name = "LightGBM + Optuna (final)"
    log(name, None, yva, pv, yte, pt, "argmax")
    log(name, w, yva, pv, yte, pt, "umbral ajustado")
    return m, cols, w, pv, pt, study.best_params


def main(quick=False):
    t0 = time.time()
    d, num, cat = load_splits()
    print(f"datos: {({n: len(x) for n, x in d.items()})}", flush=True)
    print("== Baselines", flush=True); run_baselines(d, num, cat)
    print("== LightGBM (ablación geo)", flush=True); lg = run_lgbm(d, num, cat)
    final = lg["LightGBM con geografía"]
    best_params = {}
    if not quick:
        print("== Optuna", flush=True)
        m, cols, w, pv, pt, best_params = run_optuna(d, num, cat)
        final = (m, cols, w, pv, pt)
    m, cols, w, pv, pt = final
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": m, "columns": cols, "class_weights": w, "best_params": best_params},
                MODELS / "modelo_final.joblib")
    np.save(MODELS / "proba_test.npy", pt)
    pd.DataFrame({"feature": cols, "importance_gain": m.booster_.feature_importance("gain")}) \
        .sort_values("importance_gain", ascending=False).to_csv(REPORTS / "importancia_variables.csv", index=False)
    pd.DataFrame(RESULTS).to_csv(REPORTS / "experimentos.csv", index=False)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")
