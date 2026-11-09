"""Estabilidad espacial entre periodos (datos oficiales 2005–2025, celdas de ≈5 km).

Pregunta: ¿las zonas con mayor proporción de graves o mortales en un periodo lo siguen siendo en otro?
¿Y las zonas con más accidentes (volumen)? Los periodos evitan la pandemia (2020–2021).
Salida: reports/estabilidad_espacial.csv
"""
import numpy as np
import pandas as pd

from config import PROCESSED
from train import REPORTS

PERIODS = {"2005-2011": (2005, 2011), "2012-2019": (2012, 2019), "2022-2025": (2022, 2025)}
MIN_N, TOP = 150, 300


def wilson_lo(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    return ((p + z ** 2 / (2 * n)) - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / den


def agg(cells, lo, hi):
    s = cells[cells.year.between(lo, hi)]
    return s.groupby(["lat_c", "lon_c"], as_index=False)[["n", "ksi", "fatal"]].sum()


def main():
    cells = pd.read_parquet(PROCESSED / "grid_cells.parquet")
    A = {k: agg(cells, *v) for k, v in PERIODS.items()}
    rows = []
    names = list(PERIODS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            m = A[p1].merge(A[p2], on=["lat_c", "lon_c"], suffixes=("1", "2"))
            m = m[(m.n1 >= MIN_N) & (m.n2 >= MIN_N)].copy()
            m["r1"], m["r2"] = m.ksi1 / m.n1, m.ksi2 / m.n2
            m["lo1"], m["lo2"] = wilson_lo(m.ksi1, m.n1), wilson_lo(m.ksi2, m.n2)
            top_rate1 = set(m.nlargest(TOP, "lo1").index)
            top_rate2 = set(m.nlargest(TOP, "lo2").index)
            top_vol1 = set(m.nlargest(TOP, "n1").index)
            top_vol2 = set(m.nlargest(TOP, "n2").index)
            top_ksi1 = set(m.nlargest(TOP, "ksi1").index)
            top_ksi2 = set(m.nlargest(TOP, "ksi2").index)
            g1 = m.loc[list(top_rate1)]
            rows.append({
                "periodo_1": p1, "periodo_2": p2, "celdas_comparables": len(m),
                "corr_proporcion_pearson": m.r1.corr(m.r2), "corr_proporcion_spearman": m.r1.corr(m.r2, method="spearman"),
                "corr_numero_accidentes": m.n1.corr(m.n2), "corr_numero_graves": m.ksi1.corr(m.ksi2),
                "solape_top300_por_proporcion": len(top_rate1 & top_rate2),
                "solape_top300_por_volumen": len(top_vol1 & top_vol2),
                "solape_top300_por_numero_graves": len(top_ksi1 & top_ksi2),
                "solape_proporcion_vs_volumen_periodo1": len(top_rate1 & top_vol1),
                "prop_media_top_proporcion_p1_en_p1": g1.ksi1.sum() / g1.n1.sum(),
                "prop_media_top_proporcion_p1_en_p2": g1.ksi2.sum() / g1.n2.sum(),
                "prop_media_todas_en_p2": m.ksi2.sum() / m.n2.sum(),
                "prop_media_top_volumen_p1_en_p2": m.loc[list(top_vol1)].ksi2.sum() / m.loc[list(top_vol1)].n2.sum(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "estabilidad_espacial.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string(header=False))


if __name__ == "__main__":
    main()
