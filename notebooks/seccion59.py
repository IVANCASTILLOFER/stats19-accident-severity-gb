"""Celdas del apartado 5.9 (estrategias de actualización: recalibrar o reentrenar).

Se usa tanto en build_notebook.py como al insertar el apartado en un cuaderno ya editado.
Resultados: reports/estrategias_entrenamiento.csv y reports/estrategias_contrastes.csv (src/estrategias_entrenamiento.py).
"""


def agregar(md, code):
    md("""
### 5.9 Estrategias de actualización: recalibrar o reentrenar
El desajuste descrito en los apartados 5.3 y 5.6 plantea una cuestión práctica: cuando el modelo envejece, ¿basta con recalibrarlo o conviene reentrenarlo con datos más recientes?
Para responderla se comparan tres estrategias, todas validadas con 2023 y evaluadas con los mismos años finales (2024–2025):

- **S0 (actual):** modelo entrenado con 2005–2014, sin reentrenar.
- **S1 (ventana reciente):** modelo entrenado con 2016–2022, periodo en el que el registro por lesiones ya está extendido.
- **S2 (histórico completo):** modelo entrenado con 2005–2022.

Cada estrategia se evalúa sin recalibrar y recalibrada (pendiente y sesgo por sistema de registro, ajustados únicamente con 2023). Las variables geográficas aprendidas se recalculan con los años de
entrenamiento de cada estrategia. El procedimiento se encuentra en `src/estrategias_entrenamiento.py`.
""")
    md("*Tabla 12. Resultados de las tres estrategias en 2024–2025, por versión del modelo (AP: precisión promedio; el macro-F1 se informa solo para las probabilidades sin recalibrar).*")
    code("""
estr = pd.read_csv(REPORTS / "estrategias_entrenamiento.csv")
estr["macro-F1"] = np.where(estr["esquema"] == "Sin recalibrar", estr["macro_f1"], np.nan)
cols_t = {"estrategia": "estrategia", "esquema": "probabilidades", "n_entrenamiento": "accidentes de entrenamiento", "macro-F1": "macro-F1",
          "ap_ksi": "AP graves+mortales", "ap_mortal": "AP mortales", "pred_ksi": "% estimado", "obs_ksi": "% observado", "ece": "error de calibración", "brier": "Brier"}
for v in ("A", "B"):
    t_v = estr[estr["version"] == v][list(cols_t)].rename(columns=cols_t).copy()
    t_v["% estimado"] = (t_v["% estimado"] * 100).round(1); t_v["% observado"] = (t_v["% observado"] * 100).round(1)
    print(f"Versión {v} ({'con' if v == 'A' else 'sin'} datos de vehículos)")
    display(t_v.round(4).fillna("—").reset_index(drop=True))
""")
    code("""
cols_s = {"S0": "#8a8f98", "S1": "#1f6f8b", "S2": "#e07b39"}
nombres_s = {"S0": "S0 · 2005–2014", "S1": "S1 · 2016–2022", "S2": "S2 · 2005–2022"}
va_ = estr[estr["version"] == "A"].copy(); va_["s"] = va_["estrategia"].str[:2]
raw_ = va_[va_["esquema"] == "Sin recalibrar"].set_index("s"); rec_ = va_[va_["esquema"] != "Sin recalibrar"].set_index("s")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
ax = axes[0]
ax.bar([nombres_s[k] for k in raw_.index], raw_["macro_f1"], color=[cols_s[k] for k in raw_.index]); ax.set_ylim(0.43, 0.46)
for i, v in enumerate(raw_["macro_f1"]): ax.text(i, v + 0.0004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9)
ax.set_title("Macro-F1 (sin recalibrar)"); ax.tick_params(axis="x", labelsize=8)
ax = axes[1]; w_ = 0.38; x_ = np.arange(3)
ax.bar(x_ - w_/2, raw_["ece"], w_, color="#b3b8bf", label="Sin recalibrar"); ax.bar(x_ + w_/2, rec_["ece"], w_, color="#1f6f8b", label="Recalibrada con 2023")
ax.set_xticks(x_); ax.set_xticklabels([nombres_s[k] for k in raw_.index], fontsize=8); ax.set_title("Error de calibración"); ax.legend(fontsize=8)
for i, (a1, a2) in enumerate(zip(raw_["ece"], rec_["ece"])):
    ax.text(i - w_/2, a1 + 0.001, f"{a1:.3f}".replace(".", ","), ha="center", fontsize=8); ax.text(i + w_/2, a2 + 0.001, f"{a2:.3f}".replace(".", ","), ha="center", fontsize=8)
ax = axes[2]
ax.bar([nombres_s[k] for k in raw_.index], raw_["pred_ksi"] * 100, color=[cols_s[k] for k in raw_.index])
ax.axhline(raw_["obs_ksi"].iloc[0] * 100, color="k", ls="--", lw=1.2); ax.text(2.45, raw_["obs_ksi"].iloc[0] * 100 + .3, "observado", ha="right", fontsize=9)
for i, v in enumerate(raw_["pred_ksi"] * 100): ax.text(i, v + 0.3, f"{v:.1f} %".replace(".", ","), ha="center", fontsize=9)
ax.set_ylim(0, 30); ax.set_title("% estimado de graves o mortales (sin recalibrar)"); ax.tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.show()
""")
    md("""
*Figura 16. Comparación de las estrategias S0, S1 y S2 en 2024–2025 (versión A): macro-F1, error de calibración y proporción estimada de accidentes graves o mortales frente a la observada. El eje del macro-F1 no comienza en cero. Fuente: elaboración propia con datos del DfT.*

*Tabla 13. Diferencia de cada estrategia respecto de S0, con intervalo de confianza del 95 % (remuestreo pareado de los accidentes de prueba).*
""")
    code("""
contr = pd.read_csv(REPORTS / "estrategias_contrastes.csv")
contr["contraste"] = contr["contraste"].str.replace("\\u2212", "-", regex=False)
contr["metrica"] = contr["metrica"].map({"macro_f1": "macro-F1", "ap_ksi": "AP graves+mortales", "ap_mortal": "AP mortales"})
display(contr.rename(columns={"dif_media": "diferencia", "ic95_inf": "IC95 inf", "ic95_sup": "IC95 sup"}).round(4))
""")
    code("""
def es(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")
cA = contr[contr["version"] == "A"].set_index(["contraste", "metrica"])
estr["s"] = estr["estrategia"].str[:2]
a_raw = estr[(estr["version"] == "A") & (estr["esquema"] == "Sin recalibrar")].set_index("s")
a_rec = estr[(estr["version"] == "A") & (estr["esquema"] != "Sin recalibrar")].set_index("s")
dif59 = lambda c, m, col: cA.loc[(c, m), col]
display(Markdown(f'''
**Capacidad de ordenar los casos**
- Reentrenar mejora el orden de los accidentes graves o mortales: la precisión promedio sube {es(dif59('S1 - S0', 'AP graves+mortales', 'dif_media'), 3)} con la ventana reciente y {es(dif59('S2 - S0', 'AP graves+mortales', 'dif_media'), 3)} con el histórico. Los intervalos excluyen el cero.
- El índice de acierto sube {es(dif59('S1 - S0', 'macro-F1', 'dif_media'), 3)} y {es(dif59('S2 - S0', 'macro-F1', 'dif_media'), 3)}, respectivamente.
- Para los mortales, solo mejora el histórico completo ({es(dif59('S2 - S0', 'AP mortales', 'dif_media'), 3)}; IC95 {es(dif59('S2 - S0', 'AP mortales', 'ic95_inf'), 3)} a {es(dif59('S2 - S0', 'AP mortales', 'ic95_sup'), 3)}). Con la ventana reciente no se distingue de cero.

**Nivel de los porcentajes (2024–2025)**
- Real: {es(a_raw.loc['S0', 'obs_ksi'] * 100)} %. Estimado sin ajustar: {es(a_raw.loc['S0', 'pred_ksi'] * 100)} % (S0), {es(a_raw.loc['S1', 'pred_ksi'] * 100)} % (S1) y {es(a_raw.loc['S2', 'pred_ksi'] * 100)} % (S2).
- Reentrenar reduce la diferencia de {es((a_raw.loc['S0', 'obs_ksi'] - a_raw.loc['S0', 'pred_ksi']) * 100)} a {es((a_raw.loc['S1', 'obs_ksi'] - a_raw.loc['S1', 'pred_ksi']) * 100)} y {es((a_raw.loc['S2', 'obs_ksi'] - a_raw.loc['S2', 'pred_ksi']) * 100)} puntos, pero no la elimina.

**Ajustar los porcentajes (recalibrar)**
- El error queda en {es(a_rec.loc['S0', 'ece'], 3)} (S0), {es(a_rec.loc['S1', 'ece'], 3)} (S1) y {es(a_rec.loc['S2', 'ece'], 3)} (S2). El índice de Brier pasa de {es(a_rec.loc['S0', 'brier'], 4)} a {es(a_rec.loc['S1', 'brier'], 4)} y {es(a_rec.loc['S2', 'brier'], 4)}.

**Lectura**
- Ajustar los porcentajes es lo que más corrige el nivel (el error de S0 baja de {es(a_raw.loc['S0', 'ece'], 3)} a {es(a_rec.loc['S0', 'ece'], 3)}). Reentrenar mejora sobre todo el orden de los casos.
- Hacer las dos cosas da los mejores resultados.
- S1 y S2 se parecen. S2 es la única que mejora los mortales, pero entrena con 2,6 millones de registros frente a 0,8.
- La diferencia que queda es coherente con que la proporción real siguió subiendo en 2024–2025; este experimento no lo prueba.
- Los intervalos solo reflejan el muestreo de los accidentes de prueba, no la variación propia del entrenamiento.
'''))
""")
