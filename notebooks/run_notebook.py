"""Ejecuta el notebook (kernel ml-kaggle), guarda los resultados dentro del .ipynb y lo exporta a HTML."""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from traitlets.config import Config

AQUI = Path(__file__).parent
NB = AQUI / "Proyecto1_Proceso_Completo.ipynb"

t0 = time.time()
nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="ml-kaggle", resources={"metadata": {"path": str(AQUI)}}, allow_errors=False)
try:
    client.execute()
    ok = True
except Exception as e:  # se guarda lo que haya salido para poder revisar
    ok = False
    print("ERROR al ejecutar:", type(e).__name__, str(e)[:1500])
nbformat.write(nb, NB)
cfg = Config(); cfg.TagRemovePreprocessor.enabled = True; cfg.TagRemovePreprocessor.remove_input_tags = {"ocultar_codigo"}
html, _ = HTMLExporter(config=cfg).from_notebook_node(nb)
(AQUI / "Proyecto1_Proceso_Completo.html").write_text(html, encoding="utf-8")
print(f"{'OK' if ok else 'CON ERRORES'} · {(time.time()-t0)/60:.1f} min · {NB.name} y .html guardados")
sys.exit(0 if ok else 1)
