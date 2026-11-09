"""Evaluación de los modelos CONGELADOS en años nunca usados para decidir nada.

Conjuntos: validación 2015 · prueba 2016–17 (ya vista en el análisis anterior) ·
           holdout 2018–19 (independiente, previo a la pandemia) · estrés 2020–21 (pandemia) · post 2022–25.
Además: sensibilidad al registro por lesiones (Injury_Based), calibración con y sin recalibración, y
ablaciones (sin geografía, sin Injury_Based) evaluadas en los conjuntos nuevos con bootstrap pareado.
Los umbrales y factores de clase se fijan con 2015; los factores de calibración con 2016–17.
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ablation import macro_f1_from_pred
from calibration import apply_factors, ece_ksi, fit_factors
from config import MODELS, SEED
from train import GEO_COLS, REPORTS, evaluate, fit_lgb, lgb_params, load_splits, tune_weights

N_BOOT = 150
SETS = [("val 2015", "val"), ("test 2016-17", "test"), ("holdout 2018-19", "holdout"),
        ("estres 2020-21", "estres"), ("post 2022-25", "post")]


def predict(b, X):
    return b["model"].predict_proba(X[b["columns"]])


def set_metrics(y, p, bundle, factors):
    w = np.asarray(bundle["class_weights"])
    thr = bundle["alert_threshold"]
    e = evaluate(y, p, w)
    flag = p[:, 2] >= thr
    fa = y == 2
    pc = apply_factors(p, factors)
    return {**e,
            "alert_recall": float((flag & fa).sum() / max(fa.sum(), 1)),
            "alert_precision": float((flag & fa).sum() / max(flag.sum(), 1)),
            "alert_flagged_pct": float(flag.mean()),
            "obs_ksi": float((y > 0).mean()), "pred_ksi_raw": float((1 - p[:, 0]).mean()),
            "pred_ksi_recal": float((1 - pc[:, 0]).mean()), "obs_fatal": float(fa.mean()),
            "ece_raw": ece_ksi(y, p), "ece_recal": ece_ksi(y, pc),
            "ap_fatal_vs_nonfatal": float(average_precision_score(fa, p[:, 2])),
            "n": int(len(y))}


def boot_ci(y, p, bundle, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    w = np.asarray(bundle["class_weights"])
    pred = (p * w).argmax(1)
    out = {"macro_f1": [], "ap_ksi": [], "ap_fatal": []}
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        out["macro_f1"].append(macro_f1_from_pred(y[i], pred[i]))
        out["ap_ksi"].append(average_precision_score(y[i] > 0, 1 - p[i, 0]))
        out["ap_fatal"].append(average_precision_score(y[i] == 2, p[i, 2]))
    return {k: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for k, v in out.items()}


def main():
    t0 = time.time()
    d, num, cat = load_splits()
    bundles = {"A": joblib.load(MODELS / "modelo_con_vehiculos.joblib"),
               "B": joblib.load(MODELS / "modelo_sin_vehiculos.joblib")}
    rows, ci_rows, strata, years_rows = [], [], [], []
    probs = {}
    for mname, b in bundles.items():
        ptest = predict(b, d["test"])
        factors = fit_factors(ptest, d["test"]["severity"].values)       # calibrado con 2016–17
        b["calibration_eval"] = {"factors": factors.tolist(), "fit_on": "2016-2017"}
        print(f"modelo {mname}: factores de calibración (2016–17) = {np.round(factors, 3)}", flush=True)
        for label, key in SETS:
            df = d[key]
            y = df["severity"].values
            p = predict(b, df)
            probs[(mname, key)] = p
            rows.append({"modelo": mname, "conjunto": label, **set_metrics(y, p, b, factors)})
            c = boot_ci(y, p, b)
            for k, (m, lo, hi) in c.items():
                ci_rows.append({"modelo": mname, "conjunto": label, "metrica": k, "media": m, "ic95_inf": lo, "ic95_sup": hi})
            for yr in sorted(df["Year"].unique()):
                msk = (df["Year"] == yr).values
                years_rows.append({"modelo": mname, "año": int(yr), **{k: v for k, v in set_metrics(y[msk], p[msk], b, factors).items()
                                   if k in ("macro_f1", "ap_ksi", "ap_fatal", "alert_recall", "alert_precision", "obs_ksi",
                                            "pred_ksi_raw", "pred_ksi_recal", "n")}})
            # sensibilidad al sistema de registro (por lesiones o por juicio del agente)
            ib = df["Injury_Based"].values
            for flag, name in ((0, "severidad reportada"), (1, "por lesiones (IBRS)")):
                msk = ib == flag
                if msk.sum() > 2000 and len(np.unique(y[msk] > 0)) == 2:
                    strata.append({"modelo": mname, "conjunto": label, "sistema_registro": name, "n": int(msk.sum()),
                                   "obs_ksi": float((y[msk] > 0).mean()), "pred_ksi_raw": float((1 - p[msk, 0]).mean()),
                                   "ap_ksi": float(average_precision_score(y[msk] > 0, 1 - p[msk, 0])),
                                   "obs_fatal": float((y[msk] == 2).mean()),
                                   "ap_fatal": float(average_precision_score(y[msk] == 2, p[msk, 2]))})
    pd.DataFrame(rows).to_csv(REPORTS / "holdout_metricas.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(REPORTS / "holdout_ic95.csv", index=False)
    pd.DataFrame(years_rows).to_csv(REPORTS / "holdout_por_anio.csv", index=False)
    pd.DataFrame(strata).to_csv(REPORTS / "holdout_sistema_registro.csv", index=False)
    print("métricas congeladas listas", flush=True)

    # ---- ablaciones sobre el modelo A (mismos parámetros por defecto), evaluadas en datos frescos
    bA = bundles["A"]
    cols = bA["columns"]
    ytr, yva = d["train"]["severity"].values, d["val"]["severity"].values
    variants = {"A completo": cols,
                "A sin geografía": [c for c in cols if c not in GEO_COLS],
                "A sin Injury_Based": [c for c in cols if c != "Injury_Based"]}
    abl, ablp = [], {}
    for vname, vcols in variants.items():
        if vname == "A completo":
            m, w = bA["model"], np.asarray(bA["class_weights"])
            pv = m.predict_proba(d["val"][vcols])
        else:
            t1 = time.time()
            m = fit_lgb(d["train"][vcols], ytr, d["val"][vcols], yva, lgb_params())
            pv = m.predict_proba(d["val"][vcols])
            w = tune_weights(yva, pv)
            print(f"  {vname}: {m.best_iteration_} árboles, {time.time()-t1:.0f}s", flush=True)
        for label, key in SETS[1:]:
            df = d[key]
            y = df["severity"].values
            p = m.predict_proba(df[vcols])
            ablp[(vname, key)] = (y, (p * w).argmax(1), p)
            e = evaluate(y, p, w)
            abl.append({"variante": vname, "conjunto": label, "n_vars": len(vcols), "macro_f1": e["macro_f1"],
                        "ap_ksi": e["ap_ksi"], "ap_fatal": e["ap_fatal"], "recall_fatal": e["recall_fatal"]})
    pd.DataFrame(abl).to_csv(REPORTS / "holdout_ablacion.csv", index=False)

    diffs = []
    for label, key in SETS[1:]:
        y, pa, _ = ablp[("A completo", key)]
        _, pb, _ = ablp[("A sin geografía", key)]
        _, pc_, _ = ablp[("A sin Injury_Based", key)]
        rng = np.random.RandomState(SEED)
        dg, di = [], []
        for _ in range(N_BOOT):
            i = rng.randint(0, len(y), len(y))
            fa = macro_f1_from_pred(y[i], pa[i])
            dg.append(fa - macro_f1_from_pred(y[i], pb[i]))
            di.append(fa - macro_f1_from_pred(y[i], pc_[i]))
        for nm, v in (("con geografía − sin geografía", dg), ("con Injury_Based − sin Injury_Based", di)):
            lo, hi = np.percentile(v, [2.5, 97.5])
            diffs.append({"conjunto": label, "contraste": nm, "dif_macro_f1": float(np.mean(v)), "ic95_inf": float(lo),
                          "ic95_sup": float(hi)})
    pd.DataFrame(diffs).to_csv(REPORTS / "holdout_contrastes.csv", index=False)
    print(pd.DataFrame(diffs).round(4).to_string(index=False), flush=True)
    print(f"== Terminado en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
