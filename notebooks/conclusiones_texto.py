"""Texto de las conclusiones del cuaderno (apartado 8.7). Las cifras provienen de los resultados de los apartados 5 y 8."""

CONC = """
### 8.7 Conclusiones

**1. Los vehículos y sus conductores son lo que más pesa en la gravedad.**
Son el grupo de datos más influyente (31 % del peso del modelo A) y mejoran los resultados en todos los años.
Cuando hay una moto, el 34 % de los accidentes es grave o mortal; sin moto, el 18 %.
El modelo B no tiene estos datos, no puede usar la moto y acierta menos (0,41 frente a 0,44).

**2. El número de vehículos y la velocidad máxima de la vía son los factores que más pesan.**
Pero no de forma simple: los accidentes con un solo vehículo son los más graves (28 %), más que los de dos (17 %).
Con un ciclista hay más accidentes graves (22 % frente a 20 %), pero menos mortales (0,6 % frente a 1,5 %).

**3. El modelo sirve para priorizar, no para detectar con seguridad.**
Si se revisa primero el 5 % de accidentes con más riesgo, se captura un tercio de los mortales (34 %; el azar captaría solo 5 %).
Aun así, de cada 10 alertas solo 1 es realmente mortal. Además, se le escapan sobre todo los mortales de ciudad: son el 48 % de los que no detecta.

**4. Con los años, el modelo se queda corto en los porcentajes, aunque sigue ordenando bien.**
La pandemia no lo estropea. Pero en 2025 estima un 18,8 % de accidentes graves o mortales, cuando la cifra real fue 26,2 %.
Una parte viene de que, desde 2016, muchas fuerzas registran la gravedad según las lesiones: las fuerzas que cambiaron subieron unos 3 puntos más que las que no.
El resto es un aumento que ocurre también sin cambio de registro.

**5. Se puede corregir: ajustar los porcentajes y reentrenar con años recientes.**
Ajustar los porcentajes baja el error de 0,071 a 0,008. Reentrenar con datos recientes mejora el orden de los casos.
Lo mejor es hacer las dos cosas.

**6. Conocer el lugar ayuda poco, y solo donde ya hay historia.**
Mejora el acierto entre una y dos centésimas y no ayuda en zonas que el modelo no había visto.

**7. Un ranking de zonas por proporción de accidentes graves cambia mucho con los años.**
Entre periodos coincide poco (0,2 a 0,5), mientras que el ranking por cantidad de accidentes casi no cambia (0,94 a 0,98). Conviene no tomar decisiones solo con el primero.
"""

NOTA_TABLA = ("*Nota. Los accidentes mortales están incluidos en la columna «% graves o mortales»; la diferencia entre ambas columnas "
              "corresponde a los accidentes graves que no fueron mortales.*")
