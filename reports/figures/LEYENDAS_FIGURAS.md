# Leyendas de las figuras claras

Cada figura: qué pregunta responde y cuál es el mensaje (cifras calculadas de los datos).

## F10_utilidad_ubicacion.png
- **Pregunta:** ¿Mejora la predicción saber dónde ocurrió el accidente, también en lugares nuevos?
- **Mensaje:** Sí mejora, un poco y de forma creciente (de +0,9 a +1,7 centésimas del índice), pero solo en zonas con historial: en zonas reservadas que el modelo nunca vio, la mejora no se distingue de cero.

## F11_comparacion_algoritmos.png
- **Pregunta:** ¿Importa qué algoritmo se use? ¿Compensa un modelo complejo frente a uno simple?
- **Mensaje:** LightGBM y XGBoost dan prácticamente el mismo resultado; la regresión logística (modelo simple) queda por debajo (0.425 frente a 0.443 en 2018–2019) y la diferencia se mantiene en todos los periodos (de 1,8 a 2,8 centésimas del índice). Mismas variables, mismos años.

## F12_aciertos_por_tipo.png
- **Pregunta:** ¿Cuántos accidentes de cada tipo acierta el modelo y con qué los confunde?
- **Mensaje:** El modelo A acierta el 86 % de los leves, el 34 % de los graves y el 12 % de los mortales; un 45 % de los mortales se clasifica como leve. Para esos casos es mejor usar el modo alerta que la clase más probable.

## F1_gravedad_por_condicion.png
- **Pregunta:** ¿En qué condiciones es mayor la proporción de accidentes graves o mortales?
- **Mensaje:** La proporción es mayor de madrugada (24.8 %), a 60 mph (25.1 %) y de noche sin alumbrado (26.9 %) que la media (17.5 %). Son asociaciones dentro de los accidentes registrados.

## F2_evolucion_graves.png
- **Pregunta:** ¿Por qué sube la proporción de accidentes graves desde 2016 si los mortales casi no cambian?
- **Mensaje:** Los graves registrados pasan de 14.3 % (2015) a 17.2 % (2017), mientras los ajustados se mantienen en ≈19.8–19.9 %; el registro por lesiones pasa de 5.9 % a 31.0 % entre 2015 y 2016. Los mortales rondan 1391–1797 al año desde 2010. Desde 2018 también suben los ajustados: no todo es registro.

## F3_capturas.png
- **Pregunta:** Si revisamos primero los accidentes que el modelo considera de mayor riesgo, ¿qué parte de los realmente mortales se captura?
- **Mensaje:** Marcando el 5 % de los accidentes con mayor probabilidad, el modelo A captura el 34 % de los mortales y el B el 27 % (al azar sería el 5 %). Es mejor que el azar pero lejos de perfecto. Años 2018–2019, modelos entrenados con 2005–2014.

## F4_rendimiento_anual.png
- **Pregunta:** ¿Siguen funcionando los modelos años después de entrenarlos?
- **Mensaje:** El índice de acierto del modelo A baja de 0.451 (2015) a 0.433 (2025), siempre por encima del modelo trivial (≈0.28), sin ningún salto en 2020–2021. La proporción real de graves o mortales sube de 15.4 % a 26.2 % y el modelo (sin recalibrar) predice 18.8 % en 2025.

## F5_calibracion.png
- **Pregunta:** Cuando el modelo dice que un accidente tiene un X % de probabilidad de ser grave o mortal, ¿ocurre realmente en un X % de los casos?
- **Mensaje:** Sin recalibrar, el modelo se queda corto en todos los niveles (los puntos grises quedan por encima de la línea de predicción perfecta); tras recalibrar con 2022–2023 los puntos caen casi sobre la línea en 2024–2025.

## F6_factores_principales.png
- **Pregunta:** ¿Qué factores pesan más en las predicciones de los modelos?
- **Mensaje:** En el modelo A pesan sobre todo el número de vehículos, el límite de velocidad, si hay una moto o un ciclista y las características de los conductores; en el B, el número de vehículos, el límite de velocidad, la hora, el control del cruce y el lugar. Son pesos en el modelo, no causas.

## F7_subgrupos.png
- **Pregunta:** ¿Falla el modelo más en algún tipo de accidente?
- **Mensaje:** El modelo (sin recalibrar) predice una proporción menor que la real en todos los grupos, con una razón real/predicha de 1,1 a 1,4: no hay un grupo anómalo, el desfase es general.

## F8_mapa_proporcion.png
- **Pregunta:** ¿Dónde es mayor la proporción de accidentes graves o mortales?
- **Mensaje:** La proporción varía entre zonas (de ≈10 % a ≈32 %): suele ser menor en las grandes ciudades y mayor en zonas rurales y vías rápidas. Se muestran 2.988 zonas.

## F9_estabilidad_zonas.png
- **Pregunta:** Si una zona destaca por tener muchos accidentes, o una proporción alta de graves, ¿lo sigue haciendo en otro periodo?
- **Mensaje:** El volumen de accidentes de cada zona es muy estable (0.94–0.98); la proporción de graves lo es mucho menos (0.53 entre periodos próximos y 0.20 entre 2005–2011 y 2022–2025).
