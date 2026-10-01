# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Estacionariedad en series de tiempo
Sesión: 04
Fecha: 30/09/2026
Docente: Alexis Adonai Morales Albero
"""

# Instalar modulo arch 

pip install arch

# Modulos a cargar

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import arch.data.default
import math 
import warnings
import re
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates


# Clases especificas

from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import kpss
from arch.unitroot import KPSS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose


# Pruebas de estacionariedad 

## Prueba de medias (ADF y Phillips-Perron)

## H0: No estacionariedad (p-value > alpha)
## Ha: Estacionariedad  (p-value < alpha)

## Alpha = 5% ~ 0.05

### Dickey-Fuller aumentado: Statsmodels -----

adf = adfuller(
    Ingresos,
    regression="ct",
    maxlag = 24,
    autolag = 'AIC'
    )

print('Estadístico ADF:', adf[0])
print('Valor p:', adf[1])
print('Valores críticos', adf[4])

print(f"Estadístico ADF: {adf[0]:.3f}")
print(f"Valor p: {adf[1]:.3f}")
print('Valores críticos', adf[4])

### Dickey-Fuller aumentado: arch -----

adf_arch = ADF(
    Ingresos,
    trend = 'ct',
    max_lags=24,
    method = 'aic'
    )

print(adf_arch.summary().as_text())
print(adf_arch.regression.summary().as_text())

### Phillips-Perron (solo en arch) -----

phillips_perron = PhillipsPerron(
    Ingresos,
    trend = "ct",
    test_type="tau"
    )

print(phillips_perron.summary().as_text())

# Descomposiciones temporales 

## Método aditivo

descomposicion_aditiva = seasonal_decompose(
    Ingresos,
    model = "additive",
    period = 12,
    extrapolate_trend="freq"
    )

## Método multiplicativo 

decomposición_multiplicativa = seasonal_decompose(
    Ingresos,
    model = "multiplicative",
    period = 12,
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
        f"Descomposición temporal {metodo}",
        fontsize=19,
        fontweight="bold",
        color="#1F2933",
        ha="left",
        va="top"
    )

    fig.text(
        0.08, 0.925,
        "Serie mensual",
        fontsize=11,
        color="#4D565E",
        ha="left",
        va="top"
    )

    formula = (
        "Ingresos = tendencia + estacionalidad + residuo"
        if metodo == "aditiva"
        else "Ingresos = tendencia × estacionalidad × residuo"
    )

    fig.text(
        0.08, 0.025,
        f"Nota: {formula}. Periodicidad estacional: 12 meses.",
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
    Ingresos.index
)


fig_multiplicativa = graficar_descomposicion(
    decomposición_multiplicativa,
    "multiplicativa",
    Ingresos.index
)

### KPSS: statsmodels ----

## H0: Estacionariedad (p-value > alpha)
## Ha: No estacionariedad (p-value < alpha)

kpss_salida = kpss(
    Ingresos,
    regression = 'ct',
    nlags = 'auto'
    )

print(f"KPSS LM: {kpss_salida[0]:.3f}")
print(f"Valor p: {kpss_salida[1]:.3f}")
print('Valores críticos', kpss_salida[3])

### KPSS: arch -----

kpss_arch = KPSS(
    Ingresos,
    trend = 'ct',
    lags = kpss_salida[2]
    )

print(kpss_arch.summary().as_text())

# Primeras diferencias 















