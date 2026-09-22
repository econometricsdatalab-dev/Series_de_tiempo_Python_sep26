# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Introducción a series de tiempo
Sesión: 01
Fecha: 21/09/2026
Docente: Alexis Adonai Morales Albero
"""

# Modulos a utilizar 

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math 
import warnings 

# Quitar mensajes de precaución 

warnings.filterwarnings('ignore')

# Importación/Lectura de datos 

PIB = pd.read_csv(
    "Datos\\PIB_MEX.csv",
    encoding="utf-16le",
    usecols = [0,1,2]
    )

# Indagación: Nombres de columnas 

PIB.columns

# Modificar nombres 

PIB.columns = ["Fecha", "CVE_ENT", "PIB"]

# Quitar de la fila 186 a 189

PIB = (
    PIB
    .drop(index = range(186,190))
    .reset_index(drop = True)
    )

# Quitar de las fechas los caracteres "r1"

PIB["Fecha"] = (
    PIB["Fecha"]
    .str.replace("r1", "", regex = False)
    .str.strip()
    )

# Revisar los tipos de datos 

PIB.dtypes

# Convertir 1980/01 - 1980Q1

## Paso 1) Definir el diccionario de replazo de meses correspondientes
## a los trimestres

mes_trimestre = {
    "01" : "01",
    "02" : "04",
    "03" : "07",
    "04" : "10"
    }

## Paso 2) Sustracción de texto basado en posiciones

partes = PIB["Fecha"].str.extract(
    r"(?P<anio>\d{4})/(?P<trimestre>\d{1,2})"
    )

## Paso 3) Crear objeto auxiliar de mes correspondiente al trimestre

mes = partes["trimestre"].str.zfill(2).map(mes_trimestre)

## Paso 4) Crear columna de trimestre concatenando el año,
## el mes y seguido de 01 para transformalo en formato de fecha

PIB["Trimestre"] = pd.to_datetime(
    partes["anio"]+ "-" + mes + "-01",
    format = "%Y-%m-%d"
    )

## Paso 5) Convertir la columna Fecha en el formato trimestral
## mediante la columna trimestre (que esta enformato datetime)

PIB["Fecha"] = PIB["Trimestre"].dt.to_period("Q")

# Gráficar la serie de tiempo 

plt.figure(figsize=(10,6), dpi = 500)
plt.plot(PIB.Trimestre, PIB.PIB)
plt.title("PIB de 1980 a 2026\ntrimestral\nmillones de pesos del 2018")
plt.xlabel("Fecha")
plt.ylabel("PIB")
plt.show()











