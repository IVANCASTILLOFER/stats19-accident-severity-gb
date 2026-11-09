"""Pruebas de humo y de comportamiento de la app Streamlit (sin navegador).

Ejecutar desde la carpeta del proyecto:   python tests/test_app.py
Comprueba: que la app carga, que el estimador responde, que bloquea puntos fuera de Gran Bretaña,
que el mapa de zonas y la comparación entre periodos no fallan y que las etiquetas no hablan de «riesgo de accidente».
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("  OK   " if cond else "  FALLA"), name, detail)


def metrics(at):
    return {m.label: m.value for m in at.metric}


at = AppTest.from_file(APP, default_timeout=180).run()
check("la app carga sin excepciones", not at.exception, str([e.value for e in at.exception]))
check("el estimador muestra 4 métricas", len(at.metric) == 4)
check("no aparece la etiqueta ambigua «Riesgo de accidente»",
      not any("Riesgo de accidente" in str(x.value) for x in at.markdown) and
      "Riesgo de accidente" not in str([getattr(p.proto, "text", "") for p in at.get("progress")]))

base = float(metrics(at)["Grave o mortal"].replace("%", ""))
at.select_slider[0].set_value(60)
at.run()
check("subir el límite a 60 mph aumenta la probabilidad de grave o mortal",
      float(metrics(at)["Grave o mortal"].replace("%", "")) > base)
at.select_slider[0].set_value(30)

at.session_state["point"] = (40.4168, -3.7038)   # Madrid
at.run()
check("fuera de Gran Bretaña: sin métricas y con error visible", len(at.metric) == 0 and len(at.error) >= 1)
at.session_state["point"] = (52.0, -15.0)         # Atlántico
at.run()
check("océano: sin métricas", len(at.metric) == 0)
at.session_state["point"] = (53.4808, -2.2426)    # Manchester
at.run()
check("punto válido: vuelve a calcular", len(at.metric) == 4)

at.checkbox[0].check()
at.run()
check("modelo A (con vehículos) calcula", len(at.metric) == 4 and not at.exception)

check("la pestaña de zonas tiene el selector de periodo", any("Periodo" in s.label for s in at.slider))
reg = [r for r in at.radio if "registro" in r.label][0]
ksi_les = float(metrics(at)["Grave o mortal"].replace("%", ""))
reg.set_value("Por juicio del agente")
at.run()
ksi_juicio = float(metrics(at)["Grave o mortal"].replace("%", ""))
check("el sistema de registro cambia la estimación (por lesiones > por juicio del agente)", ksi_les > ksi_juicio,
      f"{ksi_les:.1f} % frente a {ksi_juicio:.1f} %")
reg.set_value("Por lesiones (el más extendido hoy)")
at.run()

radio = [r for r in at.radio if "Qué mostrar" in r.label][0]
radio.set_value("Número de accidentes (volumen)")
at.run()
check("modo volumen sin excepciones", not at.exception)
radio.set_value("Proporción de graves o mortales")
at.run()
check("modo proporción sin excepciones", not at.exception)

comp = [c for c in at.checkbox if "Comparar" in c.label][0]
comp.check()
at.run()
check("comparación entre periodos sin excepciones", not at.exception, str([e.value for e in at.exception]))

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)} de {len(results)} comprobaciones correctas")
sys.exit(1 if fails else 0)
