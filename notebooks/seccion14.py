"""Celdas de la comprobación del cambio de registro con fuerzas policiales (dentro del apartado 1.4).

Se usa tanto en build_notebook.py como al insertar el apartado en un cuaderno ya editado.
Análisis: src/cambio_registro_fuerzas.py. Los números del texto corresponden a reports/cambio_registro_fuerzas.csv y cambio_registro_efecto.csv.
"""


def agregar(md, code):
    md("""
#### Comprobación con las fuerzas policiales
Para verificar el cambio de registro y medir cuánto influye, se aprovecha que cada fuerza lo adoptó en un año distinto. Se comparan dos grupos con los datos del propio DfT:

- **Fuerzas que cambiaron (18):** pasaron de registrar casi todo según la valoración del agente (hasta 25 % por lesiones en 2013 y 2015) a registrar casi todo por lesiones (75 % o más en 2017). La mayoría lo hizo en 2016.
- **Fuerzas de control (16):** no cambiaron en ningún año hasta 2018 (0 % por lesiones).

Se excluyen las fuerzas escocesas, cuyos códigos cambiaron en 2019, y la fuerza que adoptó el sistema antes (código 45).
""")
    code("""
import cambio_registro_fuerzas as cr
fz = cr.analizar()
tabla_fz, efecto_fz = fz["tabla"], fz["did"]
fig, ax = plt.subplots(figsize=(8, 4.2))
for g_, col_, lab in (("cambian", "#b22222", f"Fuerzas que cambiaron de sistema ({fz['n_cambian']})"), ("control", "#1f6f8b", f"Fuerzas que no cambiaron ({fz['n_control']})")):
    s_ = tabla_fz[tabla_fz["grupo"] == g_]
    ax.plot(s_["Year"], s_["ksi_pct"], "o-", color=col_, lw=2.4, label=lab)
ax.axvline(2016, color="gray", ls="--", lw=1); ax.text(2016.05, ax.get_ylim()[0] + 0.15, "2016", fontsize=9, color="#555555")
ax.set_xlabel("Año"); ax.set_ylabel("Porcentaje de accidentes graves o mortales"); ax.legend(fontsize=9)
ax.set_title("Accidentes graves o mortales según si la fuerza cambió de sistema de registro")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); plt.show()
""")
    md("*Figura 3. Porcentaje de accidentes graves o mortales en las fuerzas que cambiaron de sistema de registro y en las que no, 2013–2018. Fuente: elaboración propia con datos del DfT.*")
    md("*Tabla 2. Efecto estimado del cambio de registro: diferencia entre antes (2013–2015) y después (2017–2018), con intervalo de confianza del 95 % calculado remuestreando fuerzas.*")
    code("""
display(efecto_fz.rename(columns={"medida": "medida", "valor": "valor", "ic95_inf": "IC95 inf", "ic95_sup": "IC95 sup"}).round(2).fillna("—"))
""")
    md("""
**Qué muestra**
- En las fuerzas que cambiaron, el porcentaje de accidentes graves o mortales era estable (15,7 %, 15,1 % y 15,6 % entre 2013 y 2015) y salta a 18,7 % en 2016, 20,2 % en 2017 y 21,8 % en 2018.
- En las fuerzas de control sube poco a poco (de 16,7 % en 2013 a 20,6 % en 2018), sin ningún salto en 2016.
- Los accidentes mortales, que no dependen de cómo se clasifique, no muestran un salto comparable: en 2018 están un 5 % (fuerzas que cambiaron) y un 7 % (control) por encima de 2015.

**Qué significa**
- El cambio de registro es real y tiene efecto: las fuerzas que cambiaron subieron 2,9 puntos más que las de control (IC95 de 0,8 a 5,0), de un aumento total de 5,6 puntos.
- Pero las fuerzas de control también suben (2,7 puntos), así que una parte del aumento no depende del registro.
- La serie ajustada del DfT atribuye al registro algo más (5,1 puntos). El efecto del cambio de registro está, por tanto, entre unos 3 y 5 puntos.

Esta comparación es aproximada: las fuerzas no se eligieron al azar y sus tendencias previas no son idénticas.
""")
