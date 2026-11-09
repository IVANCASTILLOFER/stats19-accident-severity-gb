"""Comprobación del cambio de sistema de registro de la gravedad usando las fuerzas policiales como grupos de comparación.

Cada fuerza adoptó el registro basado en lesiones (campo oficial `collision_injury_based`) en un año distinto, y varias no lo hicieron hasta 2019.
Se comparan:
  · fuerzas que CAMBIAN: ≤ 25 % de accidentes registrados por lesiones en 2013 y 2015 y ≥ 75 % en 2017 (casi todas, en 2016);
  · fuerzas de CONTROL: 0 % por lesiones en todos los años 2013–2018.
Se excluyen las fuerzas escocesas (sus códigos cambiaron en 2019) y la fuerza que adoptó el sistema antes (código 45).
Estimación: diferencia de diferencias del porcentaje de accidentes graves o mortales (antes 2013–2015; después 2017–2018), con intervalo
de confianza por remuestreo de las fuerzas (no de los accidentes), pues los accidentes de una misma fuerza no son independientes.
"""
import numpy as np
import pandas as pd

from config import ACCIDENTS_PARQUET, ROOT

PRE, POST = [2013, 2014, 2015], [2017, 2018]
SEED = 42


def _grupos(a):
    ib = a.pivot_table(index="Police_Force", columns="Year", values="Injury_Based", aggfunc="mean")
    cambian = list(ib.index[(ib[2013] <= 0.25) & (ib[2015] <= 0.25) & (ib[2017] >= 0.75)])
    control = list(ib.index[ib.max(axis=1) == 0])
    return cambian, control


def _did(df, col):
    m = df.groupby(["grupo", df["Year"].isin(POST)])[col].mean().unstack()
    return (m.loc["cambian", True] - m.loc["cambian", False]) - (m.loc["control", True] - m.loc["control", False]), m


def analizar(n_boot: int = 500) -> dict:
    a = pd.read_parquet(ACCIDENTS_PARQUET, columns=["Year", "Police_Force", "Accident_Severity", "Injury_Based", "Adj_Serious"])
    a = a[(a["Police_Force"] < 90) & a["Year"].between(2013, 2018) & (a["Police_Force"] != 45)].copy()
    a["ksi"] = (a["Accident_Severity"] != "Slight").astype(int) * 100.0
    a["mortal"] = (a["Accident_Severity"] == "Fatal").astype(int)
    a["adj_ksi"] = (a["Adj_Serious"].fillna(0) + a["mortal"]) * 100.0
    cambian, control = _grupos(a)
    g = a[a["Police_Force"].isin(cambian + control)].copy()
    g["grupo"] = np.where(g["Police_Force"].isin(cambian), "cambian", "control")

    tabla = (g.groupby(["grupo", "Year"]).agg(accidentes=("ksi", "size"), ksi_pct=("ksi", "mean"), ajustado_pct=("adj_ksi", "mean"),
                                              mortales=("mortal", "sum"), lesiones_pct=("Injury_Based", "mean")).reset_index())
    tabla["lesiones_pct"] *= 100
    base = tabla[tabla["Year"] == 2015].set_index("grupo")["mortales"]
    tabla["mortales_indice_2015"] = tabla.apply(lambda r: 100 * r["mortales"] / base[r["grupo"]], axis=1)

    gg = g[g["Year"].isin(PRE + POST)]
    d_reg, m_reg = _did(gg, "ksi")
    _, m_adj = _did(gg, "adj_ksi")
    rng = np.random.RandomState(SEED)
    por_fuerza = {f: gg[gg["Police_Force"] == f] for f in cambian + control}
    boots = []
    for _ in range(n_boot):
        sel = list(rng.choice(cambian, len(cambian))) + list(rng.choice(control, len(control)))
        boots.append(_did(pd.concat([por_fuerza[f].assign(_i=i) for i, f in enumerate(sel)]), "ksi")[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    cam_reg = m_reg.loc["cambian", True] - m_reg.loc["cambian", False]
    con_reg = m_reg.loc["control", True] - m_reg.loc["control", False]
    cam_adj = m_adj.loc["cambian", True] - m_adj.loc["cambian", False]
    did = pd.DataFrame([
        {"medida": "Aumento en las fuerzas que cambiaron (puntos)", "valor": cam_reg, "ic95_inf": np.nan, "ic95_sup": np.nan},
        {"medida": "Aumento en las fuerzas de control (puntos)", "valor": con_reg, "ic95_inf": np.nan, "ic95_sup": np.nan},
        {"medida": "Diferencia entre ambos: efecto estimado del cambio de registro (puntos)", "valor": d_reg, "ic95_inf": lo, "ic95_sup": hi},
        {"medida": "Efecto según la serie ajustada del DfT (puntos)", "valor": cam_reg - cam_adj, "ic95_inf": np.nan, "ic95_sup": np.nan},
    ])
    tabla.to_csv(ROOT / "reports" / "cambio_registro_fuerzas.csv", index=False)
    did.to_csv(ROOT / "reports" / "cambio_registro_efecto.csv", index=False)
    return {"tabla": tabla, "did": did, "n_cambian": len(cambian), "n_control": len(control), "cambian": cambian, "control": control}


if __name__ == "__main__":
    r = analizar()
    pd.set_option("display.width", 250)
    print(r["n_cambian"], "fuerzas que cambian ·", r["n_control"], "de control")
    print(r["tabla"].round(1).to_string(index=False))
    print(r["did"].round(2).to_string(index=False))
