"""Construye el cuaderno del proceso completo (datos oficiales DfT 2005–2025) y lo deja listo para ejecutar.

Uso (desde la carpeta notebooks):  python build_notebook.py   ->   Proyecto1_Proceso_Completo.ipynb
Después se ejecuta con run_notebook.py (los resultados quedan guardados dentro del archivo).
"""
import nbformat as nbf

from conclusiones_texto import CONC, NOTA_TABLA

NB = "Proyecto1_Proceso_Completo.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
# Análisis multivariante de la gravedad de los accidentes de tráfico en Gran Bretaña (2005–2025)

Cuaderno de trabajo del Proyecto de Posgrado · Diplomado en Análisis de Datos · Universidad Católica Boliviana «San Pablo»
Autor: Iván Ricardo Castillo Fernández

## Presentación
El presente cuaderno documenta el proceso completo del estudio: obtención y verificación de los datos, análisis exploratorio,
construcción de variables, estimación de modelos, evaluación con datos no utilizados en el entrenamiento e interpretación de los
resultados. Las salidas de cada celda se conservan en el documento, por lo que puede revisarse sin necesidad de ejecutarlo.

**Fuente de datos.** Registro oficial de accidentes con víctimas (STATS19) del *Department for Transport* del Reino Unido, tablas de
colisiones y de vehículos, periodo 2005–2025, publicado bajo la licencia *Open Government Licence v3.0*.

**Pregunta de investigación.** ¿Qué variables (vía, entorno, hora, lugar, vehículos y conductores) explican en mayor medida la gravedad
de un accidente de tráfico y en qué medida se modifica el análisis al incorporar la información de los vehículos?

**Alcance.** Se analiza la gravedad de accidentes que ya ocurrieron. El estudio no pronostica accidentes futuros ni estima el riesgo de
circulación, pues no se dispone de datos de exposición (volumen de tráfico, kilómetros recorridos). Asimismo, la importancia estadística
de una variable no implica una relación causal.

| Apartado | Contenido |
|---|---|
| 1 | Datos: origen, verificación, distribución de la gravedad y cambios en el registro |
| 2 | Análisis exploratorio: patrones temporales, condiciones de la vía y distribución espacial |
| 3 | Variables y partición temporal de los datos |
| 4 | Modelos: versión A (con datos de vehículos), versión B (sin ellos) y comparación de algoritmos |
| 5 | Evaluación con años y zonas no utilizados en el entrenamiento, calibración y análisis por subgrupos |
| 6 | Interpretación de los modelos mediante valores SHAP |
| 7 | Pruebas automáticas del prototipo de aplicación |
| 8 | Resultados, conclusiones y limitaciones |
""")

md("## 0. Configuración del entorno")
code("""
import sys, warnings, time, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from IPython.display import Image, display, Markdown

import features                      # se importa primero para que los objetos serializados se carguen correctamente
import config, dft, train, calibration
import figuras_claras as fc
from config import ACCIDENTS_PARQUET, MODELS, FIGURES
get_ipython().run_line_magic("matplotlib", "inline")   # el módulo de figuras fija un modo sin pantalla; se restablece el del cuaderno

REPORTS = ROOT / "reports"
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})

_mostrar_figura = plt.show
def _mostrar_con_coma(*args, **kwargs):          # separador decimal en coma en los ejes de todas las figuras del cuaderno
    for n in plt.get_fignums():
        fc.coma_decimal(plt.figure(n))
    _mostrar_figura(*args, **kwargs)
plt.show = _mostrar_con_coma

def mostrar(nombre, ancho=950):
    display(Image(filename=str(FIGURES / nombre), width=ancho))

print("Directorio del proyecto:", ROOT)
print("Directorio de datos oficiales:", dft.OFICIAL)
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 1. Datos

### 1.1 Origen de los datos y preparación
Los datos son publicados por el *Department for Transport* (DfT) del Reino Unido, que los recopila a partir de los informes de la policía de cada fuerza (registro STATS19). Se utilizan dos tablas y una guía de códigos:

| Archivo oficial | Contenido | Registros 2005–2025 | Uso |
|---|---|---|---|
| Colisiones | Un registro por accidente: ubicación, hora, vía y gravedad | 2.892.427 | Base principal del análisis |
| Vehículos | Un registro por vehículo implicado: tipo y características del conductor | 5.301.126 | Se resume en un registro por accidente (versión A) |
| Guía de datos (`data-guide-2025.xlsx`) | Tabla oficial de códigos y significados | — | Decodificación de las categorías |

El DfT publica además una tabla de víctimas que no se utiliza en este estudio. La preparación de los datos (lectura, selección del periodo, decodificación con la guía oficial, agregación de los vehículos por accidente y construcción de variables) fue realizada por el autor mediante el módulo `src/dft.py`; los archivos se encuentran en `data/oficial_dft/`.

Como control de integridad, se comprueba que el número de accidentes del periodo 2005–2025 en el archivo oficial coincide con el de la base procesada.
""")
code("""
raw = pd.read_csv(dft.COLLISIONS_CSV, usecols=["collision_year"])
n_oficial = int(raw["collision_year"].between(dft.YEAR_MIN, dft.YEAR_MAX).sum())
a = pd.read_parquet(ACCIDENTS_PARQUET)
print(f"Accidentes 2005–2025 en el archivo oficial : {format(n_oficial, ',').replace(',', '.')}")
print(f"Accidentes en la base procesada            : {format(len(a), ',').replace(',', '.')}")
assert n_oficial == len(a), "El número de registros no coincide con el archivo oficial."
print("Verificación superada. Periodo:", int(a.Year.min()), "–", int(a.Year.max()))
""")

md("### 1.2 Estructura de la base y datos ausentes")
md("*Tabla 1. Variables de la base procesada, tipo de dato y porcentaje de valores ausentes.*")
code("""
info = pd.DataFrame({"tipo": a.dtypes.astype(str), "% ausentes": (a.isna().mean() * 100).round(2), "ejemplo": a.iloc[0].astype(str)})
display(info)
""")
md("""
Las coordenadas geográficas (`Latitude`, `Longitude`) presentan una proporción reducida de datos ausentes (Tabla 1), lo que permite el análisis espacial.
Las categorías declaradas como desconocidas en la guía oficial se tratan como valores ausentes y no se imputan.
""")

md("""
### 1.3 Distribución de la gravedad
Cada accidente se clasifica en tres niveles: leve, grave o mortal. La proporción de accidentes mortales es reducida, situación conocida como
desbalance de clases. En este contexto la exactitud (*accuracy*) es una medida inadecuada, dado que un clasificador que asignara siempre la
categoría «leve» obtendría un valor elevado sin aportar información. Por ello se emplean métricas específicas para clases desbalanceadas
(macro-F1, precisión promedio y medidas de calibración).
""")
code("""
orden = ["Slight", "Serious", "Fatal"]; nombres = ["Leve", "Grave", "Mortal"]
pct = a["Accident_Severity"].value_counts(normalize=True).reindex(orden) * 100
fig, ax = plt.subplots(figsize=(6.5, 3.8))
b = ax.bar(nombres, pct.values, color=["#8fbc8f", "#f4a460", "#b22222"])
for r, v in zip(b, pct.values): ax.text(r.get_x() + r.get_width()/2, v + 1, f"{v:.1f} %".replace(".", ","), ha="center", fontweight="bold")
ax.set_ylabel("Porcentaje de accidentes"); ax.set_ylim(0, 100); ax.set_title("Distribución de la gravedad de los accidentes")
plt.show()
print("Proporción de accidentes mortales:", f"{pct['Fatal']:.2f} %".replace(".", ","))
""")
md("*Figura 1. Distribución porcentual de los accidentes según su gravedad, 2005–2025. Fuente: elaboración propia con datos del DfT.*")

md("""
### 1.4 Cambio en el sistema de registro de la gravedad
A partir de 2016, numerosas fuerzas policiales sustituyeron la clasificación de la gravedad basada en la valoración del agente por un sistema
basado en las lesiones (*injury-based reporting*). Este cambio incrementa la proporción de accidentes clasificados como graves sin que
necesariamente varíe la situación real en las vías. Para corregirlo, el DfT publica una gravedad ajustada, aplicable a los accidentes no mortales.

La Figura 2 contrasta, en el panel izquierdo, la proporción de accidentes graves según el registro original y según la serie ajustada; el panel
derecho presenta la proporción de accidentes registrados por lesiones y el número anual de accidentes mortales.
""")
code("""
res = fc.f2_evolucion(a)
mostrar("F2_evolucion_graves.png", 1000)
display(res.round(2).tail(12))
""")
md("""
*Figura 2. Evolución de la proporción de accidentes graves (registrada y ajustada) y del sistema de registro, 2005–2025. Fuente: elaboración propia con datos del DfT.*

Hasta 2017, buena parte del incremento de los accidentes graves registrados coincide con el cambio de sistema de registro, pues la serie ajustada por el DfT permanece prácticamente estable. A partir de 2018 esa serie también aumenta, por lo que el cambio de registro no es la única explicación. En consecuencia, el sistema de registro se incorpora como variable del modelo y la calibración se realiza de forma diferenciada
según dicho sistema (apartado 5.6). A continuación se comprueba y se mide el efecto del cambio comparando fuerzas policiales.
""")

import seccion14
seccion14.agregar(md, code)

md("""
### 1.5 Variables excluidas por riesgo de fuga de información
Se denomina fuga de información a la utilización, durante el entrenamiento, de variables que no estarían disponibles en el momento de la
estimación o que contienen directamente la respuesta. Su inclusión produciría resultados artificialmente favorables. La Tabla 3 resume las
variables excluidas y el criterio aplicado.
""")
md("*Tabla 3. Variables excluidas del análisis y justificación.*")
code("""
fuga = pd.DataFrame([
    ("Number_of_Casualties", "Se conoce una vez ocurrido el accidente y está directamente relacionada con su gravedad."),
    ("Did_Police_Officer_Attend_Scene_of_Accident", "La asistencia policial depende de la gravedad del accidente."),
    ("Year", "Se excluye para que el modelo no dependa del año de ocurrencia."),
    ("Police_Force", "Los códigos de las fuerzas escocesas se modificaron en 2019, por lo que no son comparables entre periodos."),
    ("Variables de dinámica del choque (maniobra, punto de impacto, etc.)", "Describen el desarrollo del choque y no las condiciones previas."),
], columns=["Variable excluida", "Justificación"])
display(fuga)
print("Variables definidas como fuga en config.py:", config.LEAKAGE_COLUMNS)
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 2. Análisis exploratorio de datos
Este apartado es descriptivo y no incluye modelos. Se examina cuándo y dónde ocurren los accidentes y cuándo y dónde son más graves, distinguiendo
dos magnitudes diferentes: el número de accidentes y la proporción de accidentes graves o mortales.

### 2.1 Evolución anual
""")
code("""
yr = a.groupby("Year").agg(accidentes=("Accident_Severity", "size"), mortales=("Accident_Severity", lambda s: (s == "Fatal").sum()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, color, tit in ((axes[0], "accidentes", "#1f6f8b", "Accidentes con víctimas por año"), (axes[1], "mortales", "#b22222", "Accidentes mortales por año")):
    ax.plot(yr.index, yr[col], "o-", color=color, lw=2.4)
    ax.axvspan(2019.5, 2021.5, color="gray", alpha=.2); ax.text(2020.5, ax.get_ylim()[1], "pandemia", ha="center", va="top", fontsize=9)
    ax.set_title(tit); ax.set_xlabel("Año"); ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=7)); ax.set_ylim(0, None)
plt.tight_layout(); plt.show()
""")
md("""
*Figura 4. Número anual de accidentes con víctimas y de accidentes mortales, 2005–2025. La banda gris señala el periodo 2020–2021. Fuente: elaboración propia con datos del DfT.*

El número de accidentes disminuye de forma sostenida a lo largo del periodo y presenta un descenso pronunciado en 2020, coincidente con las restricciones de
movilidad de la pandemia de COVID-19. Por este motivo, los años 2020 y 2021 se analizan como un periodo separado.
""")

md("### 2.2 Distribución horaria")
code("""
a["hora"] = pd.to_numeric(a["Time"].astype(str).str[:2], errors="coerce")
a["grave_o_mortal"] = (a["Accident_Severity"] != "Slight").astype(int)
h = a.groupby("hora").agg(n=("grave_o_mortal", "size"), r=("grave_o_mortal", "mean"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(h.index, h["n"] / 1000, color="#1f6f8b"); axes[0].set_title("Número de accidentes (miles) según la hora"); axes[0].set_xlabel("Hora del día")
axes[1].plot(h.index, h["r"] * 100, "o-", color="#b22222"); axes[1].axhline(a["grave_o_mortal"].mean() * 100, color="k", ls="--", lw=1)
axes[1].set_title("Porcentaje de accidentes graves o mortales según la hora"); axes[1].set_xlabel("Hora del día"); axes[1].set_ylabel("Porcentaje")
axes[1].text(0.5, a["grave_o_mortal"].mean() * 100 + .4, "media", fontsize=9)
plt.tight_layout(); plt.show()
""")
md("""
*Figura 5. Número de accidentes y porcentaje de accidentes graves o mortales según la hora del día, 2005–2025. La línea discontinua indica la media general. Fuente: elaboración propia con datos del DfT.*

La mayor concentración de accidentes se produce en las horas de mayor circulación (mañana y tarde), mientras que la proporción más elevada de accidentes graves
o mortales corresponde a la madrugada. Ambas magnitudes no evolucionan de forma paralela, lo que justifica analizarlas por separado.
""")

md("### 2.3 Gravedad según las condiciones de la vía y del entorno")
code("""
fc.f1_condiciones(a)
mostrar("F1_gravedad_por_condicion.png", 1000)
""")
md("""
*Figura 6. Porcentaje de accidentes graves o mortales según franja horaria, límite de velocidad, iluminación y tipo de zona. Fuente: elaboración propia con datos del DfT.*

Las proporciones corresponden al conjunto de accidentes registrados y no al riesgo de sufrir un accidente; tampoco permiten establecer relaciones causales. Variables como
el límite de velocidad, el tipo de zona y la iluminación están asociadas entre sí, de modo que su efecto no puede aislarse mediante comparaciones univariantes. Esta
limitación motiva el empleo de técnicas multivariantes, que consideran simultáneamente el conjunto de variables.
""")

md("""
### 2.4 Distribución espacial
El territorio se divide en celdas de aproximadamente 5 km y se calcula, para aquellas con un número suficiente de accidentes, la proporción de accidentes
graves o mortales. El tamaño de cada punto es proporcional al número de accidentes de la celda.
""")
code("""
fc.f8_mapa()
mostrar("F8_mapa_proporcion.png", 700)
""")
md("""
*Figura 7. Proporción de accidentes graves o mortales por celdas de aproximadamente 5 km (celdas con al menos 150 accidentes), 2005–2025. Fuente: elaboración propia con datos del DfT.*

Para valorar si las zonas que destacan en un periodo mantienen su posición en otros, se calculó la correlación entre periodos de dos magnitudes: el número de accidentes
y la proporción de accidentes graves de cada celda.
""")
code("""
fc.f9_estabilidad()
mostrar("F9_estabilidad_zonas.png", 850)
""")
md("""
*Figura 8. Correlación entre periodos del número de accidentes y de la proporción de accidentes graves por celda. Fuente: elaboración propia con datos del DfT.*

La distribución del número de accidentes por zona es muy estable entre periodos, mientras que la proporción de accidentes graves lo es considerablemente menos. En consecuencia, una
clasificación de zonas según la proporción de accidentes graves debe interpretarse con cautela, ya que es más sensible al periodo y al tamaño de la muestra.
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 3. Variables y partición temporal

### 3.1 Partición de los datos por años
La evaluación se diseña de manera que reproduzca un uso real: los modelos se entrenan con los años más antiguos y se evalúan con años posteriores, que no intervienen
en el entrenamiento ni en el ajuste. Una partición aleatoria, o el entrenamiento con todos los años, impediría comprobar el comportamiento del modelo ante datos nuevos y
conduciría a resultados excesivamente favorables. Los modelos se fijan antes de utilizar los conjuntos de evaluación.
""")
md("*Tabla 4. Partición temporal de los datos y composición de cada conjunto.*")
code("""
d, num, cat = train.load_splits()
etiquetas = {"train": "Entrenamiento", "val": "Validación (ajustes)", "test": "Prueba", "holdout": "Evaluación principal (reserva)",
             "estres": "Pandemia", "post": "Años posteriores"}
reparto = pd.DataFrame({etiquetas[k]: {"años": (f"{int(v.Year.min())}" if v.Year.min() == v.Year.max() else f"{int(v.Year.min())}–{int(v.Year.max())}"), "accidentes": f"{len(v):,}".replace(",", "."),
                                       "% graves o mortales": round(v.ksi.mean() * 100, 1), "% mortales": round((v.severity == 2).mean() * 100, 2)}
                        for k, v in d.items()}).T
display(reparto)
""")

md("""
### 3.2 Variables empleadas
Las variables se construyen evitando el uso de información posterior. Todo lo que se estima a partir de los datos (agrupación geográfica, densidad y proporción histórica de accidentes
graves por celda) se calcula exclusivamente con los años de entrenamiento (`src/features.py`). Se definen dos versiones del análisis:

- **Versión A (con vehículos):** variables de la vía, el entorno, la hora y el lugar, más las características de los vehículos y conductores implicados.
- **Versión B (sin vehículos):** únicamente variables de la vía, el entorno, la hora y el lugar.
""")
md("*Tabla 5. Variables de los modelos, agrupadas por categoría.*")
code("""
A = joblib.load(MODELS / "modelo_con_vehiculos.joblib"); B = joblib.load(MODELS / "modelo_sin_vehiculos.joblib")
filas = []
extra_grupo = {"hour_sin": "Momento", "hour_cos": "Momento", "Latitude": "Lugar", "Longitude": "Lugar"}
extra_nombre = {"hour_sin": "Hora del día (componente seno)", "hour_cos": "Hora del día (componente coseno)", "Latitude": "Latitud", "Longitude": "Longitud"}
for v in A["columns"]:
    grupo = fc.GRUPO_DE.get(v, extra_grupo.get(v, "Otros"))
    filas.append((v, fc.NOMBRES.get(v, extra_nombre.get(v, v)), grupo, "A y B" if v in B["columns"] else "solo A"))
tabla_vars = pd.DataFrame(filas, columns=["variable", "significado", "grupo", "en modelo"])
print(f"Versión A: {len(A['columns'])} variables · Versión B: {len(B['columns'])} variables")
display(tabla_vars.groupby("grupo").agg(variables=("variable", "count")).T)
display(tabla_vars.sort_values(["grupo", "variable"]).reset_index(drop=True))
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 4. Modelos

### 4.1 Diseño de la comparación
Se establecen dos comparaciones. La primera contrasta las versiones A y B, de modo que la diferencia de desempeño entre ambas permita estimar la aportación de los datos
de vehículos y conductores. La segunda compara tres algoritmos con las mismas variables y los mismos años: la regresión logística, técnica multivariante clásica que sirve
de referencia y es de fácil interpretación, y dos algoritmos de *boosting* (XGBoost y LightGBM), capaces de representar relaciones más complejas entre las variables.

Los modelos finales A y B emplean LightGBM y se entrenaron con los años 2005–2014 (`src/final_models.py`). En este cuaderno se cargan ya entrenados y no se modifican.
""")
md("*Tabla 6. Configuración de los modelos finales.*")
code("""
REENTRENAR = False      # si se establece True, se vuelven a entrenar A y B (proceso de varios minutos)
if REENTRENAR:
    import final_models; final_models.main()
resumen = pd.DataFrame({
    "Versión A (con vehículos)": {"variables": len(A["columns"]), "árboles": A["model"].best_iteration_, "umbral de alerta (mortales)": round(A["alert_threshold"], 4),
                                  "pesos de clase (leve, grave, mortal)": np.round(A["class_weights"], 2).tolist()},
    "Versión B (sin vehículos)": {"variables": len(B["columns"]), "árboles": B["model"].best_iteration_, "umbral de alerta (mortales)": round(B["alert_threshold"], 4),
                                  "pesos de clase (leve, grave, mortal)": np.round(B["class_weights"], 2).tolist()},
})
display(resumen)
""")

md("### 4.2 Comparación de algoritmos")
md("*Tabla 7. Macro-F1 de cada algoritmo según el periodo de evaluación.*")
code("""
cmp = pd.read_csv(REPORTS / "comparacion_modelos.csv")
display(cmp.pivot(index="conjunto", columns="modelo", values="macro_f1").round(4))
fc.f11_algoritmos()
mostrar("F11_comparacion_algoritmos.png", 850)
""")
md("""
*Figura 9. Macro-F1 de LightGBM, XGBoost y regresión logística con su intervalo de confianza del 95 %, según el periodo de evaluación. Fuente: elaboración propia con datos del DfT.*

*Tabla 8. Diferencias de macro-F1 de LightGBM respecto de los otros algoritmos, con intervalo de confianza del 95 % (remuestreo).*
""")
code("""
display(pd.read_csv(REPORTS / "comparacion_contrastes.csv").round(4))
""")
md("""
LightGBM y XGBoost presentan un desempeño equivalente: los intervalos de confianza de sus diferencias incluyen el valor cero en todos los periodos. La regresión logística
obtiene valores inferiores en todos los casos, por lo que la mayor complejidad de los algoritmos de *boosting* se asocia con una mejora moderada.
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 5. Evaluación con datos no utilizados en el entrenamiento
La evaluación emplea accidentes reales posteriores al periodo de entrenamiento. Los resultados se acompañan de intervalos de confianza obtenidos por remuestreo (*bootstrap*) y
se interpretan en comparación con un modelo trivial que asignara siempre la categoría «leve».

### 5.1 Verificación con el periodo de evaluación principal (2018–2019)
Se calculan las predicciones directamente en este cuaderno con los modelos entrenados y se verifica que coinciden con los resultados almacenados.
""")
code("""
ev = d["holdout"].reset_index(drop=True)
y = ev["severity"].values
models = {}
for nombre, b, color in (("A", A, fc.BLUE), ("B", B, fc.ORANGE)):
    p = b["model"].predict_proba(ev[b["columns"]])
    models[nombre] = (p, b, color)
    e = train.evaluate(y, p, b["class_weights"])
    guardado = pd.read_csv(REPORTS / "holdout_metricas.csv").query("modelo == @nombre and conjunto == 'holdout 2018-19'").iloc[0]
    assert np.isclose(e["macro_f1"], guardado["macro_f1"], atol=1e-6), "El resultado no coincide con el almacenado."
    print((f"Versión {nombre}: macro-F1 = {e['macro_f1']:.4f} (coincide con el resultado almacenado) · "
          f"AP mortales = {e['ap_fatal']:.4f} · sensibilidad en mortales (clase más probable) = {e['recall_fatal']:.3f}").replace(".", ","))
print(f"Accidentes evaluados: {format(len(y), ',').replace(',', '.')}")
""")

md("### 5.2 Resultados por periodo")
md("*Tabla 9. Métricas de desempeño de las versiones A y B en cada periodo de evaluación (AP: precisión promedio).*")
code("""
m = pd.read_csv(REPORTS / "holdout_metricas.csv")
ci = pd.read_csv(REPORTS / "holdout_ic95.csv").query("metrica == 'macro_f1'")
t = m[["modelo", "conjunto", "macro_f1", "ap_ksi", "ap_fatal", "alert_recall", "alert_precision", "ece_raw", "ece_recal", "n"]].merge(
    ci[["modelo", "conjunto", "ic95_inf", "ic95_sup"]], on=["modelo", "conjunto"]).rename(columns={
    "macro_f1": "macro-F1", "ap_ksi": "AP graves+mortales", "ap_fatal": "AP mortales", "alert_recall": "alerta: sensibilidad",
    "alert_precision": "alerta: precisión", "ece_raw": "error de calibración", "ece_recal": "error de calibración (recalibrado)",
    "ic95_inf": "IC95 inf", "ic95_sup": "IC95 sup"})
display(t.round(4))
""")
md("""
El macro-F1 promedia el desempeño en las tres clases y toma el valor 1 ante una clasificación perfecta; el modelo trivial alcanza aproximadamente 0,3. La precisión promedio (AP) mide la capacidad
de ordenar los accidentes de mayor a menor gravedad. El modo de alerta marca un porcentaje reducido de accidentes como de mayor probabilidad de resultar mortales, con un umbral fijado en el año
de validación (2015).
""")

md("### 5.3 Estabilidad del desempeño en el tiempo")
code("""
fc.f4_anual(a)
mostrar("F4_rendimiento_anual.png", 1000)
""")
md("""
*Figura 10. Evolución del macro-F1 por año de evaluación y comparación entre la proporción observada y la estimada de accidentes graves o mortales. Fuente: elaboración propia con datos del DfT.*

El desempeño de los modelos disminuye gradualmente a lo largo de los años y no se observa una discontinuidad en 2020–2021; en todos los casos permanece por encima del modelo trivial. En cambio, la
proporción observada de accidentes graves o mortales crece más de lo que estiman los modelos, lo que se atribuye al cambio de sistema de registro y a otros factores. Esta desviación del nivel
se corrige mediante la recalibración (apartado 5.6).
""")

md("### 5.4 Clasificación por tipo de accidente")
code("""
fc.f12_confusion(models, ev)
mostrar("F12_aciertos_por_tipo.png", 950)
""")
md("""
*Figura 11. Matriz de confusión normalizada por fila de las versiones A y B, 2018–2019. Fuente: elaboración propia con datos del DfT.*

Los accidentes leves se clasifican correctamente con mayor frecuencia. Los mortales se confunden con frecuencia con las categorías leve y grave, lo que limita el uso de la clase más probable como
criterio para identificarlos y motiva el empleo del modo de alerta.
""")

md("### 5.5 Modo de alerta para accidentes mortales")
code("""
fc.f3_capturas(models, ev)
mostrar("F3_capturas.png", 1000)
""")
md("""
*Figura 12. Porcentaje acumulado de accidentes mortales (izquierda) y graves o mortales (derecha) captados según el porcentaje de accidentes marcados, ordenados por la probabilidad estimada. Fuente: elaboración propia con datos del DfT.*

La ordenación de los accidentes por probabilidad estimada supera claramente a la selección aleatoria (línea punteada), aunque dista de ser perfecta. El modelo se plantea, por tanto, como un
apoyo al análisis exploratorio y no como un mecanismo de decisión automática.
""")

md("### 5.6 Calibración de las probabilidades")
code("""
allv = pd.concat([d["test"], d["holdout"], d["estres"], d["post"]], ignore_index=True)
models_all = {n: (b["model"].predict_proba(allv[b["columns"]]), b, c) for n, b, c in (("A", A, fc.BLUE), ("B", B, fc.ORANGE))}
fc.f5_calibracion(models_all, allv)
mostrar("F5_calibracion.png", 950)
""")
md("""
*Figura 13. Curvas de calibración de la probabilidad estimada de accidente grave o mortal, con y sin recalibración, 2024–2025. Fuente: elaboración propia con datos del DfT.*

*Tabla 10. Métricas de calibración según el esquema de recalibración aplicado.*
""")
code("""
rc = pd.read_csv(REPORTS / "recalibracion.csv")
display(rc[["modelo", "esquema", "evaluado_en", "pred_ksi", "obs_ksi", "ece_ksi", "brier", "logloss"]].round(4))
""")
md("""
Sin recalibración, los modelos subestiman la probabilidad de accidente grave o mortal en todos los niveles. La recalibración mediante pendiente y sesgo, diferenciada por sistema de registro, se ajusta
con los años 2022–2023 y se verifica con 2024–2025, periodo en el que las probabilidades estimadas se aproximan a las frecuencias observadas.
""")

md("### 5.7 Análisis por subgrupos")
code("""
fc.f7_subgrupos()
mostrar("F7_subgrupos.png", 1000)
""")
md("""
*Figura 14. Porcentaje observado y estimado por la versión B (sin recalibrar) de accidentes graves o mortales, según subgrupos, 2018–2019. Fuente: elaboración propia con datos del DfT.*

El modelo sin recalibrar subestima la proporción de accidentes graves o mortales de forma similar en todos los subgrupos. El sesgo es, por lo tanto, de nivel general y no se concentra en una categoría concreta.
""")

md("""
### 5.8 Aportación de la ubicación y evaluación en zonas no vistas
Se compara el desempeño del modelo con y sin variables geográficas, tanto en periodos posteriores como en bloques geográficos reservados (20 % de las zonas, excluidas del entrenamiento).
""")
code("""
fc.f10_geografia()
mostrar("F10_utilidad_ubicacion.png", 1000)
""")
md("""
*Figura 15. Diferencia de macro-F1 al incorporar la ubicación, por periodo (izquierda) y según se trate de zonas ya observadas o no observadas en el entrenamiento (derecha), con intervalo de confianza del 95 %. Fuente: elaboración propia con datos del DfT.*

*Tabla 11. Resultados de la comparación con y sin variables geográficas (muestra de filas).*
""")
code("""
display(pd.read_csv(REPORTS / "holdout_ablacion.csv").round(4).head(12))
""")
md("""
La incorporación de la ubicación mejora ligeramente el desempeño, pero solo en zonas con historial previo; en zonas no observadas la diferencia no se distingue de cero. La información geográfica resulta útil,
por tanto, para interpolar entre zonas conocidas y no para extrapolar a zonas nuevas.
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
import seccion59
seccion59.agregar(md, code)

md("""
## 6. Interpretación de los modelos mediante valores SHAP
Los valores SHAP (*SHapley Additive exPlanations*) cuantifican la contribución de cada variable a las predicciones del modelo. Permiten ordenar las variables según su importancia y comprobar
que el modelo no se apoya en variables que filtran la respuesta. No informan sobre relaciones causales ni sobre el sentido del efecto de cada variable.
""")
code("""
tops = fc.f6_factores(models, ev)
mostrar("F6_factores_principales.png", 1050)
""")
md("""
*Figura 17. Diez variables de mayor peso en las predicciones de las versiones A y B (porcentaje del total de contribución SHAP), 2018–2019. Fuente: elaboración propia con datos del DfT.*

*Tabla 14. Peso de cada grupo de variables en las predicciones (porcentaje del total).*
""")
code("""
g = pd.read_csv(REPORTS / "shap_por_grupo.csv", index_col=0)
display(g.round(1))
""")
md("""
En ambas versiones, el número de vehículos implicados y el límite de velocidad ocupan los dos primeros lugares. En la versión A, las variables relativas a los vehículos y
conductores (presencia de una motocicleta o de un ciclista, edad y sexo de los conductores) ocupan posiciones destacadas, lo que indica su aportación al análisis y es coherente con la mejora de la
versión A respecto de la B.
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 7. Pruebas automáticas del prototipo de aplicación
El prototipo (Streamlit) cuenta con pruebas automáticas que verifican su carga sin errores, la respuesta del estimador, el rechazo de ubicaciones fuera de Gran Bretaña, el funcionamiento del mapa
y de la comparación entre periodos, y la ausencia de etiquetas que sugieran una estimación de «riesgo de accidente».
""")
code("""
t0 = time.time()
r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_app.py")], cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-3500:])
if r.returncode != 0 and r.stderr.strip(): print("Mensajes de error:", r.stderr[-800:])
print(f"Código de salida: {r.returncode} · duración: {time.time()-t0:.0f} s")
""")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
md("""
## 8. Resultados y conclusiones
Este apartado integra los resultados de los apartados anteriores en relación con la pregunta de investigación. Las cifras provienen de los resultados
almacenados; el periodo de referencia es la evaluación principal (2018–2019), salvo indicación contraria.

### 8.1 Aportación de los datos de vehículos y conductores
""")
code("""
def es(x, nd=1):                       # formato numérico con coma decimal
    return f"{x:.{nd}f}".replace(".", ",")

def mil(n):                            # separador de miles con punto
    return f"{int(n):,}".replace(",", ".")

sg = pd.read_csv(REPORTS / "shap_por_grupo.csv", index_col=0)
mt = pd.read_csv(REPORTS / "holdout_metricas.csv").set_index(["modelo", "conjunto"])
hA, hB = mt.loc[("A", "holdout 2018-19")], mt.loc[("B", "holdout 2018-19")]
yf = (y == 2).astype(int)
cap = {}
for k in ("A", "B"):
    xx, cc = fc.captura(yf, models[k][0][:, 2]); cap[k] = float(np.interp(5, xx, cc))
pA, pB = tops["A"], tops["B"]
gv = sg.loc["Vehículos y conductores", "A (con vehículos)"]; gvia_A, gvia_B = sg.loc["Vía", "A (con vehículos)"], sg.loc["Vía", "B (sin vehículos)"]
ggeo_A, ggeo_B = sg.loc["Geografía", "A (con vehículos)"], sg.loc["Geografía", "B (sin vehículos)"]
gtie_A, gtie_B = sg.loc["Tiempo", "A (con vehículos)"], sg.loc["Tiempo", "B (sin vehículos)"]
assert bool((mt.loc["A"]["macro_f1"].values > mt.loc["B"]["macro_f1"].values).all()), "A no supera a B en todos los periodos"

display(Markdown(f'''
**Cuánto mejoran los datos de vehículos y conductores**
- La versión A (con vehículos) supera a la B (sin ellos) en todos los años evaluados.
- En 2018–2019, el índice de acierto (macro-F1) es {es(hA.macro_f1, 3)} frente a {es(hB.macro_f1, 3)}.
- La capacidad de ordenar los accidentes mortales (precisión promedio) sube de {es(hB.ap_fatal, 3)} a {es(hA.ap_fatal, 3)}.
- Si se revisa primero el 5 % de accidentes con más riesgo, A capta el {es(cap['A'])} % de los mortales y B el {es(cap['B'])} %.

**Qué peso tienen dentro del modelo A**
- Los vehículos y conductores son el grupo que más pesa: {es(gv)} % del total, por delante de la vía ({es(gvia_A)} %).
- La moto es la tercera variable más importante ({es(pA['veh_motorcycle_any'])} %), después del número de vehículos ({es(pA['Number_of_Vehicles'])} %) y del límite de velocidad ({es(pA['Speed_limit'])} %).
- Siguen la proporción de conductores hombres ({es(pA['male_share'])} %), la edad del conductor mayor ({es(pA['driver_age_max'])} %) y la presencia de un ciclista ({es(pA['veh_pedalcycle_any'])} %).

**Qué pasa en el modelo B**
- No tiene datos de vehículos, así que no puede usar la moto ni al ciclista.
- Su peso se traslada a la vía (de {es(gvia_A)} % a {es(gvia_B)} %), al lugar (de {es(ggeo_A)} % a {es(ggeo_B)} %) y a la hora (de {es(gtie_A)} % a {es(gtie_B)} %).
- Aun así acierta menos: esas variables solo reemplazan en parte lo que aportan los vehículos.
'''))
""")

md("""
### 8.2 Factores de mayor peso y su sentido descriptivo
Los valores SHAP indican cuánto utiliza el modelo cada variable, pero no el sentido de su asociación con la gravedad. Para complementarlo, la Tabla 15 presenta la proporción observada de accidentes graves o
mortales según la presencia de motocicletas o ciclistas y según el número de vehículos. Se utilizan únicamente los accidentes con información de vehículos.

*Tabla 15. Porcentaje de accidentes graves o mortales y de accidentes mortales según características de los vehículos implicados, 2018–2019.*
""")
code("""
ev_v = ev[ev["has_vehicle_info"] == 1]
def fila(nombre, mask):
    s = ev_v[mask]
    return {"grupo": nombre, "accidentes": len(s), "% graves o mortales": round(s["ksi"].mean() * 100, 1), "% mortales": round((s["severity"] == 2).mean() * 100, 2)}
nv = ev_v["Number_of_Vehicles"]
tabla12 = pd.DataFrame([
    fila("Con motocicleta", ev_v["veh_motorcycle_any"] == 1), fila("Sin motocicleta", ev_v["veh_motorcycle_any"] == 0),
    fila("Con ciclista", ev_v["veh_pedalcycle_any"] == 1), fila("Sin ciclista", ev_v["veh_pedalcycle_any"] == 0),
    fila("1 vehículo", nv == 1), fila("2 vehículos", nv == 2), fila("3 o más vehículos", nv >= 3),
]).set_index("grupo")
display(tabla12)
""")
md(NOTA_TABLA)
code("""
t12 = tabla12
display(Markdown(f'''
- **Moto.** Con moto, el {es(t12.loc['Con motocicleta', '% graves o mortales'])} % de los accidentes es grave o mortal; sin moto, el {es(t12.loc['Sin motocicleta', '% graves o mortales'])} %. Mortales: {es(t12.loc['Con motocicleta', '% mortales'], 2)} % frente a {es(t12.loc['Sin motocicleta', '% mortales'], 2)} %.
- **Ciclista.** Con ciclista hay más accidentes graves o mortales ({es(t12.loc['Con ciclista', '% graves o mortales'])} % frente a {es(t12.loc['Sin ciclista', '% graves o mortales'])} %), pero menos mortales ({es(t12.loc['Con ciclista', '% mortales'], 2)} % frente a {es(t12.loc['Sin ciclista', '% mortales'], 2)} %).
- **Número de vehículos.** Con 1 vehículo, el {es(t12.loc['1 vehículo', '% graves o mortales'])} % es grave o mortal; con 2, el {es(t12.loc['2 vehículos', '% graves o mortales'])} %; con 3 o más, el {es(t12.loc['3 o más vehículos', '% graves o mortales'])} %. No sube de forma pareja.
- **Importancia en el modelo.** El número de vehículos y el límite de velocidad son los dos factores más importantes en ambas versiones ({es(pA['Number_of_Vehicles'])} % y {es(pA['Speed_limit'])} % en A; {es(pB['Number_of_Vehicles'])} % y {es(pB['Speed_limit'])} % en B).

Son comparaciones simples: no tienen en cuenta el resto de variables y no prueban que una cosa cause la otra.
'''))
""")

md("### 8.3 Estabilidad temporal y calibración")
code("""
py = pd.read_csv(REPORTS / "holdout_por_anio.csv")
g = lambda mod, yr, col: float(py[(py["modelo"] == mod) & (py["año"] == yr)][col].iloc[0])
rc = pd.read_csv(REPORTS / "recalibracion.csv")
rc_A = rc[(rc["modelo"] == "A") & (rc["evaluado_en"] == rc["evaluado_en"].iloc[0])]
sin = rc_A[rc_A["esquema"] == "Sin recalibrar"].iloc[0]; mejor = rc_A.sort_values("ece_ksi").iloc[0]
display(Markdown(f'''
- **Pierde acierto poco a poco.** El índice de acierto de A baja de {es(g('A', 2015, 'macro_f1'), 3)} (2015) a {es(g('A', 2025, 'macro_f1'), 3)} (2025); el de B, de {es(g('B', 2015, 'macro_f1'), 3)} a {es(g('B', 2025, 'macro_f1'), 3)}.
- **La pandemia no lo afecta.** En A, el acierto es {es(g('A', 2019, 'macro_f1'), 3)} en 2019 y {es(g('A', 2020, 'macro_f1'), 3)} en 2020: no hay salto.
- **Lo que sí cambia es el nivel.** La proporción real de accidentes graves o mortales pasa de {es(g('A', 2015, 'obs_ksi') * 100)} % (2015) a {es(g('A', 2025, 'obs_ksi') * 100)} % (2025); el modelo A estima {es(g('A', 2015, 'pred_ksi_raw') * 100)} % y {es(g('A', 2025, 'pred_ksi_raw') * 100)} %.
  En 2025 se queda corto por {es((g('A', 2025, 'obs_ksi') - g('A', 2025, 'pred_ksi_raw')) * 100)} puntos.
- **Ajustar los porcentajes lo corrige.** Con la recalibración, el error baja de {es(sin.ece_ksi, 3)} a {es(mejor.ece_ksi, 3)} en {sin.evaluado_en}.
'''))
""")

md("### 8.4 Qué accidentes mortales detecta el modelo y cuáles no")
code("""
mdv = pd.read_csv(REPORTS / "mortales_detectados_vs_perdidos.csv", index_col=0)
cd, cn, ct = "A 2018–19 · mortales detectados", "A 2018–19 · mortales NO detectados", "A 2018–19 · todos los accidentes"
bd = "B 2018–19 · mortales detectados"
display(Markdown(f'''
- El modo de alerta de A detecta {mil(mdv.loc['n', cd])} de los {mil(mdv.loc['n', cd] + mdv.loc['n', cn])} accidentes mortales de 2018–2019.
- **Los que detecta** ocurren sobre todo fuera de la ciudad: solo el {es(mdv.loc['% urbano', cd])} % es urbano, frente al {es(mdv.loc['% urbano', cn])} % de los que se le escapan.
- Tienen más velocidad (límite medio de {es(mdv.loc['límite medio (mph)', cd])} mph frente a {es(mdv.loc['límite medio (mph)', cn])} mph) y ocurren más de madrugada ({es(mdv.loc['% madrugada (0-5 h)', cd])} % frente a {es(mdv.loc['% madrugada (0-5 h)', cn])} %).
- **En resumen:** el modelo reconoce el perfil de vía rápida, fuera de la ciudad y de noche, y falla más con los mortales urbanos, que son casi la mitad de los que no detecta.
- La precisión de la alerta es de {es(hA.alert_precision * 100, 0)} %: de cada 10 accidentes marcados, solo {es(hA.alert_precision * 10, 0)} es mortal. Sirve para priorizar la revisión, no para decidir caso por caso.
'''))
""")

md("### 8.5 Información geográfica, sistema de registro y algoritmos")
code("""
hc = pd.read_csv(REPORTS / "holdout_contrastes.csv")
geo_c = hc[hc["contraste"].str.startswith("con geografía")].set_index("conjunto")
inj_c = hc[hc["contraste"].str.startswith("con Injury")].set_index("conjunto")
sc = pd.read_csv(REPORTS / "separacion_geografica_contraste.csv"); sc.columns = ["conjunto", "grupo", "dif", "inf", "sup", "n"]
nuevo = sc[sc["grupo"].str.contains("NO")].set_index("conjunto"); visto = sc[~sc["grupo"].str.contains("NO")].set_index("conjunto")
rs = pd.read_csv(REPORTS / "holdout_sistema_registro.csv")
r_ib = rs[(rs["modelo"] == "A") & (rs["conjunto"] == "post 2022-25") & rs["sistema_registro"].str.contains("lesiones")].iloc[0]
r_rp = rs[(rs["modelo"] == "A") & (rs["conjunto"] == "post 2022-25") & rs["sistema_registro"].str.contains("reportada")].iloc[0]
cc_ = pd.read_csv(REPORTS / "comparacion_contrastes.csv")
lr = cc_[cc_["contraste"].str.contains("logística")].set_index("conjunto"); xg = cc_[cc_["contraste"].str.contains("XGBoost")].set_index("conjunto")
est = pd.read_csv(REPORTS / "estabilidad_espacial.csv")
display(Markdown(f'''
**Ubicación**
- Añadir el lugar mejora el índice de acierto en {es(geo_c.loc['holdout 2018-19', 'dif_macro_f1'], 3)} (2018–2019) y en {es(geo_c.loc['post 2022-25', 'dif_macro_f1'], 3)} (2022–2025).
- Esa mejora está en zonas que el modelo ya conocía ({es(visto.loc['holdout 2018-19', 'dif'], 3)} y {es(visto.loc['post 2022-25', 'dif'], 3)}). En zonas nuevas no se distingue de cero ({es(nuevo.loc['holdout 2018-19', 'dif'], 3)} y {es(nuevo.loc['post 2022-25', 'dif'], 3)}).
- El lugar pesa entre {es(ggeo_A)} % y {es(ggeo_B)} % en el modelo, pero quitarlo cuesta solo entre una y dos centésimas: parte de su información ya está en la vía, la velocidad o la zona.

**Forma de registrar la gravedad**
- Quitar esta variable casi no cambia el acierto ({es(inj_c.loc['holdout 2018-19', 'dif_macro_f1'], 4)} en 2018–2019 y {es(inj_c.loc['post 2022-25', 'dif_macro_f1'], 4)} en 2022–2025).
- Pero el nivel sí cambia: en 2022–2025, el {es(r_ib.obs_ksi * 100)} % de los accidentes registrados por lesiones es grave o mortal, frente al {es(r_rp.obs_ksi * 100)} % de los registrados por valoración del agente.
- El modelo se queda corto en ambos ({es(r_ib.pred_ksi_raw * 100)} % y {es(r_rp.pred_ksi_raw * 100)} %), y más en el primero. Por eso se ajusta por separado.

**Algoritmos**
- LightGBM supera a la regresión logística en {es(lr.loc['holdout 2018-19', 'dif_macro_f1'], 3)} (2018–2019) y en {es(lr.loc['post 2022-25', 'dif_macro_f1'], 3)} (2022–2025). Con XGBoost no hay diferencia ({es(xg.loc['holdout 2018-19', 'dif_macro_f1'], 3)}).

**Zonas**
- Cuántos accidentes tiene cada zona se mantiene entre periodos (correlación de {es(est['corr_numero_accidentes'].min(), 2)} a {es(est['corr_numero_accidentes'].max(), 2)}).
- La proporción de accidentes graves cambia mucho: {es(est['corr_proporcion_pearson'].max(), 2)} entre periodos cercanos y {es(est['corr_proporcion_pearson'].min(), 2)} entre 2005–2011 y 2022–2025.
'''))
""")

md("### 8.6 Resumen de las pruebas")
code("""
display(Markdown(f'''
- Base: {mil(len(a))} accidentes oficiales (STATS19, 2005–2025), verificados frente al archivo original.
- Evaluación principal (2018–2019): macro-F1 de {es(hA.macro_f1, 3)} (versión A) y {es(hB.macro_f1, 3)} (versión B); el modelo trivial se sitúa en torno a 0,3.
- Modo de alerta (versión A): marca el {es(hA.alert_flagged_pct * 100)} % de los accidentes y capta el {es(hA.alert_recall * 100, 0)} % de los mortales, con una precisión de {es(hA.alert_precision * 100, 0)} %.
- Pruebas automáticas del prototipo: 13 de 13 superadas (apartado 7).
'''))
""")

md(CONC)

md("""
### 8.8 Limitaciones
- El modelo estima la gravedad de accidentes ocurridos; no pronostica accidentes futuros ni mide el riesgo de circulación, por la ausencia de datos de exposición (tráfico, kilómetros recorridos).
- Los valores SHAP y las proporciones descriptivas son asociaciones y no demuestran causalidad; no controlan factores no registrados (comportamiento del conductor, estado del vehículo, entre otros).
- La gravedad es la registrada oficialmente y su criterio cambió durante el periodo; la gravedad ajustada del DfT solo está disponible para accidentes no mortales.
- Los resultados corresponden al Reino Unido; lo transferible a otros contextos es el método de análisis y de validación, no los valores estimados.
- La aportación de los vehículos se mide de forma agregada (versión A frente a B). No se ha descompuesto la aportación individual de cada tipo de vehículo, lo que constituye una línea natural de continuación.
- La comprobación del cambio de registro con fuerzas policiales es aproximada, pues las fuerzas no se eligieron al azar; y los intervalos del experimento de reentrenamiento solo reflejan el muestreo de los accidentes de prueba, no la variabilidad del entrenamiento.

### Referencias
- Department for Transport (Reino Unido). *Road Safety Data* (registros STATS19, 1979–2025). Open Government Licence v3.0.
  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- Department for Transport. *Guide to severity adjustments for reported road casualties, Great Britain*.
  https://www.gov.uk/government/publications/guide-to-severity-adjustments-for-reported-road-casualty-statistics/guide-to-severity-adjustments-for-reported-road-casualties-great-britain


""")

for _c in cells:   # el código que solo genera texto se oculta en el HTML
    if _c.cell_type == "code" and "display(Markdown(f'''" in _c.source:
        _c.metadata["tags"] = ["ocultar_codigo"]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "ml-kaggle", "language": "python", "name": "ml-kaggle"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, NB)
print("escrito:", NB, "·", len(cells), "celdas")
