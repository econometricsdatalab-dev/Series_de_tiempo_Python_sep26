# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Estacionariedad en series de tiempo
Subtema: Primeras diferencias y ordenes de integración
Sesión: 05
Fecha: 02/10/2026
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

# Programación para lectura de archivos xls 

def leer_xls(ruta):
    # Se reemplazan bytes UTF-8 inválidos en las notas; no se alteran los números.
    raw = Path(ruta).read_bytes().decode('utf-8', errors='replace')
    root = ET.fromstring(raw)
    ns = {'s':'urn:schemas-microsoft-com:office:spreadsheet'}
    filas = []
    for row in root.findall('.//s:Worksheet/s:Table/s:Row', ns):
        values = []
        for cell in row.findall('s:Cell', ns):
            index = int(cell.get('{'+ns['s']+'}Index',len(values)+1))
            values.extend(['']*(index-1-len(values)))
            data = cell.find('s:Data', ns)
            values.append(''.join(data.itertext()).strip() if data is not None else '')
        filas.append(values)
    encabezado = next(v for v in filas if 'XAB - Ingresos presupuestarios' in v)
    col = encabezado.index('XAB - Ingresos presupuestarios')
    obs = [(v[0],v[col]) for v in filas if v and re.fullmatch(r'\d{2}/\d{4}', v[0])]
    df = pd.DataFrame(obs,columns=['Fecha','Ingresos'])
    df['Fecha'] = pd.to_datetime(df['Fecha'],format='%m/%Y')
    df['Ingresos'] = pd.to_numeric(df['Ingresos'],errors='raise')
    if df.Fecha.duplicated().any():
        raise ValueError('Fechas duplicadas: revisar el archivo.')
    y = df.set_index('Fecha').Ingresos.sort_index().asfreq('MS')
    if y.isna().any():
        raise ValueError('Faltan meses; no se imputan datos.')
    return y


# Lectura de datos xls 

Ingresos = leer_xls("Datos\\EstadisticasOportunas.xls")


# Especificaciones previas a la diferenciación 

## Nivel de signficancia estadística para las pruebas 

alpha = 0.05

## Especificación de los modelos auxiliares 

### ct - constante + tendencia 
### c - constante 

especificacion_diferencias = "c"

# Primeras diferencias 

## NOTA: Se requiere que los datos esten en un formato
## de series provenientes de pandas.

type(Ingresos)

## Diferencias sin logaritmos (cambios nominales o tasa de cambio nominal)

Ingresos_d1 = Ingresos.diff().dropna()

# En primeras diferencias se pierde 1 observación con respecto
# a la serie originial.

## Visualización de la serie de tiempo 

fig_d1, ax_d1 = plt.subplots(
    figsize = (14,6),
    dpi = 500
    )

fig_d1.patch.set_facecolor("white")

ax_d1.plot(
    Ingresos_d1.index,
    Ingresos_d1,
    color = "#08989C",
    linewidth = 1.7
    )

ax_d1.axhline(
    y = 0,
    color = "#9AA0A6",
    linestyle = "--",
    linewidth = 0.9
    )

ax_d1.set_ylabel(
    "Diferencia de ingresos",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d1.set_xlabel(
    "Año",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_d1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax_d1.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    color="#D5D8DC"
)

ax_d1.spines[["top", "right"]].set_visible(False)
ax_d1.spines[["left", "bottom"]].set_color("#D5D8DC")
ax_d1.tick_params(axis="both", colors="#4D565E", labelsize=9)

fig_d1.text(
    0.10, 0.965,
    "Ingresos: primeras diferencias",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)

fig_d1.text(
    0.10, 0.905,
    "Serie mensual | ΔYₜ = Yₜ − Yₜ₋₁",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig_d1.text(
    0.10, 0.025,
    "Nota: diferenciación regular de orden 1. Se pierden 1 observación inicial.",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()

Ingresos_d1.mean()


## Diferencia en términos logarítmicos (tasa de crecimiento porcentual)

Ingresos_log = np.log(Ingresos)
Ingresos_d1_tc = Ingresos_log.diff().dropna()

# En primeras diferencias se pierde 1 observación con respecto
# a la serie originial.

## Visualización de la serie de tiempo 

fig_d1, ax_d1 = plt.subplots(
    figsize = (14,6),
    dpi = 500
    )

fig_d1.patch.set_facecolor("white")

ax_d1.plot(
    Ingresos_d1_tc.index,
    Ingresos_d1_tc,
    color = "#08989C",
    linewidth = 1.7
    )

ax_d1.axhline(
    y = 0,
    color = "#9AA0A6",
    linestyle = "--",
    linewidth = 0.9
    )

ax_d1.set_ylabel(
    "Tasa de crecimiento de los ingresos",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d1.set_xlabel(
    "Año",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_d1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax_d1.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    color="#D5D8DC"
)

ax_d1.spines[["top", "right"]].set_visible(False)
ax_d1.spines[["left", "bottom"]].set_color("#D5D8DC")
ax_d1.tick_params(axis="both", colors="#4D565E", labelsize=9)

fig_d1.text(
    0.10, 0.965,
    "Ingresos: primeras diferencias logarítmicas",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)

fig_d1.text(
    0.10, 0.905,
    "Serie mensual | Δlog(Yₜ) = log(Yₜ) − log(Yₜ₋₁) = TC",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig_d1.text(
    0.10, 0.025,
    "Nota: diferenciación regular de orden 1. Se pierden 1 observación inicial.",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()

Ingresos_d1_tc.mean()


## Evaluación de estacionariedad 

### Método gráfico (ACF y PACF)

rezagos_d1 = min(24, len(Ingresos_d1_tc) //2 -2)

### ACF 

fig_acf_d1, ax_acf_d1 = plt.subplots(
    figsize=(14, 6),
    dpi=500
)

fig_acf_d1.patch.set_facecolor("white")

plot_acf(
    Ingresos_d1_tc,
    ax=ax_acf_d1,
    lags=rezagos_d1,
    alpha=alpha,
    zero=False,
    title="",
    color="#08989C",
    vlines_kwargs={"colors": "#08989C", "linewidth": 1.5}
)

ax_acf_d1.set_xlabel("Rezago (meses)", fontsize=11, color="#4D565E")
ax_acf_d1.set_ylabel("Autocorrelación", fontsize=11, color="#4D565E")
ax_acf_d1.set_xticks(np.arange(1, rezagos_d1 + 1))
ax_acf_d1.grid(axis="y", linestyle=":", linewidth=0.8, color="#D5D8DC")
ax_acf_d1.spines[["top", "right"]].set_visible(False)
ax_acf_d1.spines[["left", "bottom"]].set_color("#D5D8DC")
ax_acf_d1.tick_params(axis="both", colors="#4D565E", labelsize=9)

fig_acf_d1.text(
    0.10, 0.965,
    "ACF: primeras diferencias de la tasa de crecimiento de losingresos",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)

fig_acf_d1.text(
    0.10, 0.905,
    f"Serie mensual | {rezagos_d1} rezagos | Intervalos de confianza del 95%",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig_acf_d1.text(
    0.10, 0.025,
    "Nota: el rezago cero se omite. Las bandas muestran intervalos aproximados.",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig_acf_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()


### Función PACF 

fig_pacf_d1, ax_pacf_d1 = plt.subplots(
    figsize=(14, 6),
    dpi=300
)

fig_pacf_d1.patch.set_facecolor("white")

plot_pacf(
    Ingresos_d1_tc,
    ax=ax_pacf_d1,
    lags=rezagos_d1,
    alpha=alpha,
    method="ywm",
    zero=False,
    title="",
    color="#08989C",
    vlines_kwargs={"colors": "#08989C", "linewidth": 1.5}
)

ax_pacf_d1.set_xlabel("Rezago (meses)", fontsize=11, color="#4D565E")
ax_pacf_d1.set_ylabel("Autocorrelación parcial", fontsize=11, color="#4D565E")
ax_pacf_d1.set_xticks(np.arange(1, rezagos_d1 + 1))
ax_pacf_d1.grid(axis="y", linestyle=":", linewidth=0.8, color="#D5D8DC")
ax_pacf_d1.spines[["top", "right"]].set_visible(False)
ax_pacf_d1.spines[["left", "bottom"]].set_color("#D5D8DC")
ax_pacf_d1.tick_params(axis="both", colors="#4D565E", labelsize=9)

fig_pacf_d1.text(
    0.10, 0.965,
    "PACF: primeras diferencias de la tasa de crecimiento de los ingresos",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)

fig_pacf_d1.text(
    0.10, 0.905,
    f"Serie mensual | {rezagos_d1} rezagos | Intervalos de confianza del 95%",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig_pacf_d1.text(
    0.10, 0.025,
    "Nota: el rezago cero se omite. Las bandas muestran intervalos aproximados.",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig_pacf_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()

# Pruebas de estacionariedad 

terminos_deterministas = 1 if especificacion_diferencias == "c" else 2

max_rezagos_d1 = min(
    24,
    len(Ingresos_d1_tc) // 2 - terminos_deterministas - 1
)

### Dickey-Fuller aumentado: statsmodels -----

adf_d1 = adfuller(
    Ingresos_d1_tc,
    regression=especificacion_diferencias,
    maxlag=max_rezagos_d1,
    autolag="AIC"
)

print("\nADF statsmodels: primeras diferencias")
print(f"Estadístico ADF: {adf_d1[0]:.3f}")
print(f"Valor p: {adf_d1[1]:.6f}")
print("Rezagos seleccionados:", adf_d1[2])
print("Valores críticos:", adf_d1[4])

### Dickey-Fuller aumentado: arch -----

adf_arch_d1 = ADF(
    Ingresos_d1_tc,
    trend=especificacion_diferencias,
    max_lags=max_rezagos_d1,
    method="aic"
)

print(adf_arch_d1.summary().as_text())
print(adf_arch_d1.regression.summary().as_text())

### Phillips-Perron: arch -----

phillips_perron_d1 = PhillipsPerron(
    Ingresos_d1_tc,
    trend=especificacion_diferencias,
    test_type="tau"
)

print(phillips_perron_d1.summary().as_text())

### KPSS: statsmodels -----

kpss_salida_d1 = kpss(
    Ingresos_d1_tc,
    regression=especificacion_diferencias,
    nlags="auto"
)

print("\nKPSS statsmodels: primeras diferencias")
print(f"KPSS LM: {kpss_salida_d1[0]:.3f}")
print(f"Valor p tabulado (puede ser un límite): {kpss_salida_d1[1]:.6f}")
print("Rezagos utilizados:", kpss_salida_d1[2])
print("Valores críticos:", kpss_salida_d1[3])

### KPSS: arch -----

kpss_arch_d1 = KPSS(
    Ingresos_d1_tc,
    trend=especificacion_diferencias,
    lags=kpss_salida_d1[2]
)

print(kpss_arch_d1.summary().as_text())

### Resumen de las pruebas -----

resumen_d1 = pd.DataFrame({
    "Prueba": [
        "ADF statsmodels", "ADF arch", "Phillips-Perron arch",
        "KPSS statsmodels", "KPSS arch"
    ],
    "Estadistico": [
        adf_d1[0], adf_arch_d1.stat, phillips_perron_d1.stat,
        kpss_salida_d1[0], kpss_arch_d1.stat
    ],
    "Valor_p": [
        adf_d1[1], adf_arch_d1.pvalue, phillips_perron_d1.pvalue,
        kpss_salida_d1[1], kpss_arch_d1.pvalue
    ],
    "H0": [
        "Raíz unitaria", "Raíz unitaria", "Raíz unitaria",
        "Estacionariedad", "Estacionariedad"
    ]
})

resumen_d1["Decision_5pct"] = np.where(
    resumen_d1["Valor_p"] < alpha,
    "Rechazar H0",
    "No rechazar H0"
)

print("\nResumen: primeras diferencias")
print(resumen_d1.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

## Segunda diferencia 

Ingresos_d2 = Ingresos_d1.diff().dropna()

## Visualización de la serie de tiempo 

fig_d2, ax_d2 = plt.subplots(
    figsize = (14,6),
    dpi = 500
    )

fig_d2.patch.set_facecolor("white")

ax_d2.plot(
    Ingresos_d2.index,
    Ingresos_d2,
    color = "#08989C",
    linewidth = 1.7
    )

ax_d2.axhline(
    y = 0,
    color = "#9AA0A6",
    linestyle = "--",
    linewidth = 0.9
    )

ax_d2.set_ylabel(
    "Diferencia de ingresos",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d2.set_xlabel(
    "Año",
    fontsize = 11,
    color = "#4d565e"
    )

ax_d2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_d2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax_d2.grid(
    axis="y",
    linestyle=":",
    linewidth=0.8,
    color="#D5D8DC"
)

ax_d2.spines[["top", "right"]].set_visible(False)
ax_d2.spines[["left", "bottom"]].set_color("#D5D8DC")
ax_d2.tick_params(axis="both", colors="#4D565E", labelsize=9)

fig_d2.text(
    0.10, 0.965,
    "Ingresos: segundas diferencias",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)

fig_d2.text(
    0.10, 0.905,
    "Serie mensual | Δ²Yₜ = Yₜ − 2Yₜ₋₁ + Yₜ₋₂",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig_d2.text(
    0.10, 0.025,
    "Nota: diferenciación regular de orden 2. Se pierden 2 observación inicial.",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig_d2.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()

Ingresos_d2.mean()
