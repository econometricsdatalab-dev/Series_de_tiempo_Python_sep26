# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Introducción a series de tiempo
Sesión: 02
Fecha: 23/09/2026
Docente: Alexis Adonai Morales Albero
"""

# NOTA: Ejecutar previamente el script S01.py

# Modulos a cargar

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math 
import warnings 

# Clases o "funciones" 

from statsmodels.tsa.seasonal import seasonal_decompose

# Quitar mensajes de precaución 

warnings.filterwarnings('ignore')

# Convertir PIB a valores numericos 

PIB["PIB"] = pd.to_numeric(
    PIB["PIB"] 
    )

# Crear serie temporal ordenada 

serie_pib = (
    PIB
    .sort_values("Trimestre")
    .set_index("Trimestre")["PIB"]
    )

# Validación necesaria para la descomposición de la serie de tiempo 

## Revisar que no haya datos duplicados (en las fechas)

if serie_pib.index.has_duplicates:
    raise ValueError("Hay trimestres duplicados en la serie del PIB")
else:
    print("No hay duplicados en la serie de PIB")

## Valores perdidos o NA's

if serie_pib.isna().any():
    raise ValueError("Hay valores de PIB vacíos, no numéricos o nulos")
else:
    print("La serie de tiempo PIB esta completa")

## Mínimo de observaciones que requiere el método de descomposición 

if len(serie_pib) < 8:
    raise ValueError("Se necesitan al menos 8 observaciones trimestrales")
else:
    print("La serie trimestral cumple con el mínimo de observaciones")
    
## Faltantes o fechas fuera de orden 

trimestres_esperados = pd.date_range(
    start = serie_pib.index.min(),
    end = serie_pib.index.max(),
    freq = "QS")

if not serie_pib.index.equals(trimestres_esperados):
    raise ValueError(
        "La serie tiene trimestres faltantes o fechas fuera de orden"
        )
else:
    print("La serie tiene trimestres completos o fechas en orden esperado")

# Descomposiciones temporales 

## Método aditivo

descomposicion_aditiva = seasonal_decompose(
    serie_pib,
    model = "additive",
    period = 4,
    extrapolate_trend="freq"
    )

## Método multiplicativo 

### Valoración de número positivos en toda la serie 

if(serie_pib <= 0).any():
    raise ValueError(
        "La descomposición temporal no puede realizarse con el método multiplicativo; requiere valores positivos"
        )
else:
    print("Se puede realizar descomposición temporal por método multiplicativo")

decomposición_multiplicativa = seasonal_decompose(
    serie_pib,
    model = "multiplicative",
    period = 4,
    extrapolate_trend="freq"
    )

## Definición (proceso) para visualizar la descomposición

def graficar_descomposicion(resultado, metodo, fechas):
    """
    La definción ayudará a gráficar la descomposición temporal
    según el método seleccionado en seasonal_decompose del modulo
    statsmodels.tsa.seasonal. Devolverá una figura de matplotlib
    en la cual tuvo un previo proceso de manipulación de la clase
    tsa.seasonal.DecomposeResult
    """
    
    componentes = [
        ("Observado", np.asarray(resultado.observed, dtype=float)),
        ("Tendencia", np.asarray(resultado.trend, dtype=float)),
        ("Estacionalidad", np.asarray(resultado.seasonal, dtype=float)),
        ("Residuo", np.asarray(resultado.resid, dtype=float))
    ]

    colores = {
        "Observado": "#F04A1D",
        "Tendencia": "#F46F2C",
        "Estacionalidad": "#08989C",
        "Residuo": "#4D565E"
    }

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 10),
        dpi=300,
        sharex=True
    )

    fig.patch.set_facecolor("white")

    for ax, (nombre, valores) in zip(axes, componentes):

        if len(valores) != len(fechas):
            raise ValueError(
                f"{nombre}: el número de valores no coincide "
                "con el número de fechas."
            )

        if not np.isfinite(valores).any():
            raise ValueError(
                f"{nombre}: todos los valores son NaN o infinitos."
            )

        ax.plot(
            fechas,
            valores,
            color=colores[nombre],
            linewidth=1.7
        )

        if nombre in ("Estacionalidad", "Residuo"):
            referencia = 0 if metodo == "aditiva" else 1

            ax.axhline(
                y=referencia,
                color="#9AA0A6",
                linestyle="--",
                linewidth=0.9
            )

        ax.set_ylabel(
            nombre,
            fontsize=11,
            fontweight="bold",
            color="#333333",
            labelpad=15
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.8,
            color="#D5D8DC"
        )

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#D5D8DC")
        ax.tick_params(axis="both", colors="#4D565E", labelsize=9)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Año", fontsize=11, color="#333333")

    fig.text(
        0.08, 0.965,
        f"Descomposición temporal {metodo} del PIB de México",
        fontsize=19,
        fontweight="bold",
        color="#1F2933",
        ha="left",
        va="top"
    )

    fig.text(
        0.08, 0.925,
        "Serie trimestral | Millones de pesos de 2018",
        fontsize=11,
        color="#4D565E",
        ha="left",
        va="top"
    )

    formula = (
        "PIB = tendencia + estacionalidad + residuo"
        if metodo == "aditiva"
        else "PIB = tendencia × estacionalidad × residuo"
    )

    fig.text(
        0.08, 0.025,
        f"Nota: {formula}. Periodicidad estacional: 4 trimestres.",
        fontsize=9,
        color="#4D565E",
        ha="left",
        va="bottom"
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.87,
        bottom=0.10,
        hspace=0.30
    )

    plt.show()

    return fig

# Visualizar ambos métodos 

fig_aditiva = graficar_descomposicion(
    descomposicion_aditiva,
    "aditiva",
    serie_pib.index
)


fig_multiplicativa = graficar_descomposicion(
    decomposición_multiplicativa,
    "multiplicativa",
    serie_pib.index
)















