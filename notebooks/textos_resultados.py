"""Textos cortos de los apartados 5.9 y 8.1 a 8.5. Cada bloque reemplaza el `display(Markdown(f'''...'''))` de su celda.
Clave: frase que identifica la celda (en la versión anterior del texto) -> bloque nuevo."""

BLOQUES = {}

BLOQUES["**Desempeño.**"] = """display(Markdown(f'''
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
'''))"""

BLOQUES["Cuando interviene una motocicleta"] = """display(Markdown(f'''
- **Moto.** Con moto, el {es(t12.loc['Con motocicleta', '% graves o mortales'])} % de los accidentes es grave o mortal; sin moto, el {es(t12.loc['Sin motocicleta', '% graves o mortales'])} %. Mortales: {es(t12.loc['Con motocicleta', '% mortales'], 2)} % frente a {es(t12.loc['Sin motocicleta', '% mortales'], 2)} %.
- **Ciclista.** Con ciclista hay más accidentes graves o mortales ({es(t12.loc['Con ciclista', '% graves o mortales'])} % frente a {es(t12.loc['Sin ciclista', '% graves o mortales'])} %), pero menos mortales ({es(t12.loc['Con ciclista', '% mortales'], 2)} % frente a {es(t12.loc['Sin ciclista', '% mortales'], 2)} %).
- **Número de vehículos.** Con 1 vehículo, el {es(t12.loc['1 vehículo', '% graves o mortales'])} % es grave o mortal; con 2, el {es(t12.loc['2 vehículos', '% graves o mortales'])} %; con 3 o más, el {es(t12.loc['3 o más vehículos', '% graves o mortales'])} %. No sube de forma pareja.
- **Importancia en el modelo.** El número de vehículos y el límite de velocidad son los dos factores más importantes en ambas versiones ({es(pA['Number_of_Vehicles'])} % y {es(pA['Speed_limit'])} % en A; {es(pB['Number_of_Vehicles'])} % y {es(pB['Speed_limit'])} % en B).

Son comparaciones simples: no tienen en cuenta el resto de variables y no prueban que una cosa cause la otra.
'''))"""

BLOQUES["**Pérdida gradual de desempeño.**"] = """display(Markdown(f'''
- **Pierde acierto poco a poco.** El índice de acierto de A baja de {es(g('A', 2015, 'macro_f1'), 3)} (2015) a {es(g('A', 2025, 'macro_f1'), 3)} (2025); el de B, de {es(g('B', 2015, 'macro_f1'), 3)} a {es(g('B', 2025, 'macro_f1'), 3)}.
- **La pandemia no lo afecta.** En A, el acierto es {es(g('A', 2019, 'macro_f1'), 3)} en 2019 y {es(g('A', 2020, 'macro_f1'), 3)} en 2020: no hay salto.
- **Lo que sí cambia es el nivel.** La proporción real de accidentes graves o mortales pasa de {es(g('A', 2015, 'obs_ksi') * 100)} % (2015) a {es(g('A', 2025, 'obs_ksi') * 100)} % (2025); el modelo A estima {es(g('A', 2015, 'pred_ksi_raw') * 100)} % y {es(g('A', 2025, 'pred_ksi_raw') * 100)} %.
  En 2025 se queda corto por {es((g('A', 2025, 'obs_ksi') - g('A', 2025, 'pred_ksi_raw')) * 100)} puntos.
- **Ajustar los porcentajes lo corrige.** Con la recalibración, el error baja de {es(sin.ece_ksi, 3)} a {es(mejor.ece_ksi, 3)} en {sin.evaluado_en}.
'''))"""

BLOQUES["El modo de alerta de la versión A identifica"] = """display(Markdown(f'''
- El modo de alerta de A detecta {mil(mdv.loc['n', cd])} de los {mil(mdv.loc['n', cd] + mdv.loc['n', cn])} accidentes mortales de 2018–2019.
- **Los que detecta** ocurren sobre todo fuera de la ciudad: solo el {es(mdv.loc['% urbano', cd])} % es urbano, frente al {es(mdv.loc['% urbano', cn])} % de los que se le escapan.
- Tienen más velocidad (límite medio de {es(mdv.loc['límite medio (mph)', cd])} mph frente a {es(mdv.loc['límite medio (mph)', cn])} mph) y ocurren más de madrugada ({es(mdv.loc['% madrugada (0-5 h)', cd])} % frente a {es(mdv.loc['% madrugada (0-5 h)', cn])} %).
- **En resumen:** el modelo reconoce el perfil de vía rápida, fuera de la ciudad y de noche, y falla más con los mortales urbanos, que son casi la mitad de los que no detecta.
- La precisión de la alerta es de {es(hA.alert_precision * 100, 0)} %: de cada 10 accidentes marcados, solo {es(hA.alert_precision * 10, 0)} es mortal. Sirve para priorizar la revisión, no para decidir caso por caso.
'''))"""

BLOQUES["**Ubicación.**"] = """display(Markdown(f'''
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
'''))"""

BLOQUES["**Capacidad de ordenar.**"] = """display(Markdown(f'''
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
'''))"""
