# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Estacionariedad en series de tiempo
Sesión: 03
Fecha: 25/09/2026
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

# Clases especificas

from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import kpss
from arch.unitroot import KPSS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import jarque_bera
from pathlib import Path

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

# Visualización inicial 

fig, ax = plt.subplots(1,1,
                   figsize=(10,6), dpi = 500)
ax.plot(Ingresos.index , Ingresos.values,
         color = "#F04A1D")
ax.set_xlabel("Fecha")
ax.set_ylabel("Ingresos presupuestarios")
fig.text(
    0.08, 0.965,
    "Ingresos presupuestarios del Sector Público en México",
    fontsize=19,
    fontweight="bold",
    color="#1F2933",
    ha="left",
    va="top"
)

fig.text(
    0.08, 0.90,
    "Serie mensual | Millones de pesos",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig.text(
    0.08, 0.025,
    f"Nota: Los datos mensuales son los flujos obtenidos en el periodo\nFuente. SHCP. Estadísticas Oportunas (ESTOPOR).",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.85,
    bottom=0.15,
    hspace=0.30
)

plt.show()

# Función de autocorrelación simple (Prueba gráfica de estacionariedad)

fig, ax = plt.subplots(1,1,
                   figsize=(10,6), dpi = 500)
plot_acf(Ingresos, ax = ax, color = "#F04A1D",
         vlines_kwargs={'colors': "#1F2933"})
ax.set_title("")
ax.set_xlabel("Rezagos temporales (k)")
ax.set_ylabel("Autocorrelación (rho)")
fig.text(
    0.08, 0.965,
    "Función de autocorrelación (ACF) - Ingresos presupuestarios",
    fontsize=19,
    fontweight="bold",
    color="#1F2933",
    ha="left",
    va="top"
)

fig.text(
    0.08, 0.90,
    "Correlación entre -1 y 1",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)

fig.text(
    0.08, 0.025,
    f"Nota: Las bandas azul claro representar el intervalo de confianza",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)

fig.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.85,
    bottom=0.15,
    hspace=0.30
)

plt.show()
