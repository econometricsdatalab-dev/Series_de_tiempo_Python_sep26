# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Modelos estacionarios y no estacionarios
Subtema: Modelo ARIMA 
Sesión: 06
Fecha: 05/10/2026
Docente: Alexis Adonai Morales Albero
"""

# Modulos a utilizar 

import warnings 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
import re
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates

# Clases especificas a importar

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from itertools import product
from matplotlib.ticker import FuncFormatter
from scipy import stats
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

# Configuración inicial 

HORIZONTE = 12
RESERVA = 0
ALPHA = 0.05
MAX_P = 4
MAX_Q = 4
MAX_REZAGOS_ADF = 24
D_MANUAL = None

NARANJA = '#F04A1D'
CIRUELA = '#84517F'
OLIVA = '#65773E'
CREMA = '#F1ECE7'
NEGRO = '#090909'
GRIS = '#6C6259'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': '#222222',
    'axes.labelcolor': GRIS,
    'xtick.color': GRIS,
    'ytick.color': GRIS,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'axes.axisbelow': True,
    'svg.fonttype': 'none'
})

# Transformaciones 

Ingresos_log = np.log(Ingresos)

Transformaciones = {
    0 : Ingresos_log,
    1 : Ingresos_log.diff().dropna(),
    2 : Ingresos_log.diff().diff().dropna()
    }

# Pruebas de estacionariedad 

filas_pruebas = []
for d_candidato, serie in Transformaciones.items():
    deterministico = 'ct' if d_candidato == 0 else 'c'
    adf = adfuller(
        serie, regression = deterministico,
        maxlag = MAX_REZAGOS_ADF, autolag = 'AIC'
    )
    pp = PhillipsPerron(serie, trend = deterministico)
    with warnings.catch_warnings(record = True) as avisos:
        warnings.simplefilter('always')
        kpss_salida = kpss(serie, regression = deterministico, nlags = 'auto')

    nota = ' | '.join(str(aviso.message) for aviso in avisos)
    p_kpss = kpss_salida[1]
    texto_kpss = ('≤0.010' if p_kpss <= 0.01 else
                 '≥0.100' if p_kpss >= 0.10 else f'{p_kpss:.4f}')
    filas_pruebas.append({
        'd': d_candidato, 'Deterministicos': deterministico, 'N': len(serie),
        'ADF_est': adf[0], 'ADF_p': adf[1], 'ADF_rezagos': adf[2],
        'PP_est': pp.stat, 'PP_p': pp.pvalue, 'PP_banda': pp.lags,
        'KPSS_est': kpss_salida[0], 'KPSS_p': p_kpss,
        'KPSS_p_texto': texto_kpss, 'KPSS_banda': kpss_salida[2],
        'ADF_rechaza_raiz': adf[1] < ALPHA,
        'PP_rechaza_raiz': pp.pvalue < ALPHA,
        'KPSS_no_rechaza_est': p_kpss >= ALPHA,
        'Advertencia_KPSS': nota
    })

Pruebas = pd.DataFrame(filas_pruebas)
Pruebas['Compatible_segun_regla'] = (
    Pruebas['ADF_rechaza_raiz'] & Pruebas['PP_rechaza_raiz'] &
    Pruebas['KPSS_no_rechaza_est']
)
print('\nTABLA ÚNICA DE ESTACIONARIEDAD · MUESTRA DE ENTRENAMIENTO')
print(Pruebas[['d', 'Deterministicos', 'ADF_p', 'PP_p',
               'KPSS_p_texto', 'Compatible_segun_regla']].to_string(index = False))

SALIDA = Path.cwd()

Pruebas.to_csv(SALIDA / '01_estacionariedad.csv', 
               index = False, encoding = 'utf-8-sig')


# Identificación de los ordenes p, d, q del modelo ARIMA 

d_admisibles = Pruebas.loc[Pruebas['Compatible_segun_regla'], 'd']
if D_MANUAL is not None:
    if D_MANUAL not in (0, 1, 2):
        raise ValueError('D_MANUAL debe ser 0, 1, 2 o None.')
    d = int(D_MANUAL)
elif not d_admisibles.empty:
    d = int(d_admisibles.min())
else:
    raise ValueError('Las pruebas discrepan para todos los d. Revisar antes de definir D_MANUAL.')

Serie_identificacion = Transformaciones[d]
print('\nOrden d seleccionado:', d)

# ACF y PACF de la serie en el orden seleccionado 

Fuente = "Fuente: SCHP. Estadísticas oportunas (ESTOPOR)."

fig, axes = plt.subplots(2, 1, 
                         figsize = (14, 10), 
                         dpi = 500)
fig.subplots_adjust(left = 0.08, 
                    right = 0.95, 
                    bottom = 0.14, 
                    top = 0.76, 
                    hspace = 0.48)
plot_acf(Serie_identificacion, 
         lags = 36, zero = False, 
         ax = axes[0],
         color = NARANJA, 
         vlines_kwargs = {'colors': NARANJA}, title = '')
plot_pacf(Serie_identificacion, 
          lags = 36, zero = False, method = 'ywm',
          ax = axes[1], color = CIRUELA,
          vlines_kwargs = {'colors': CIRUELA}, title = '')
for ax, color, nombre in zip(axes, [NARANJA, CIRUELA], ['ACF · Autocorrelación', 'PACF · Autocorrelación parcial']):
    ax.set_title(nombre, loc = 'left', 
                 fontsize = 12, weight = 'bold')
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel('Rezago (meses)')
    ax.grid(axis = 'y', color = CREMA)
    for banda in ax.collections:
        if hasattr(banda, 'set_facecolor'):
            banda.set_facecolor(color)
fig.text(0.08, 0.94, 
         'ECONOMETRICS DATA LAB', 
         color = NARANJA, weight = 'bold')
fig.text(0.08, 0.87, 
         'Identificar la memoria de los ingresos',
         fontsize = 24, weight = 'bold')
fig.text(0.08, 0.82, 
         f'Logaritmo con d = {d} · Sólo entrenamiento · Bandas puntuales del 95%', color = GRIS)
fig.text(0.08, 0.065,
         Fuente, fontsize = 8.5, color = GRIS)
fig.text(0.08, 0.035, 
         'Las funciones de correlación orientan los órdenes; la selección formal usa AICc.', fontsize = 9, color = GRIS)
fig.savefig(SALIDA / '01_identificacion.png', dpi = 500)
fig.savefig(SALIDA / '01_identificacion.svg')
plt.show()
plt.close(fig)

# Estimación de modelos ARIMA 

tendencias = ['c', 'ct'] if d == 0 else ['n', 'c'] if d == 1 else ['n']
Modelos = {}
filas_modelos = []

for p, q, tendencia in product(range(MAX_P + 1), 
                               range(MAX_Q + 1), 
                               tendencias):
    identificador = f'ARIMA({p},{d},{q}) / {tendencia}'
    try:
        with warnings.catch_warnings(record = True) as avisos:
            warnings.simplefilter('always')
            modelo = ARIMA(
                Ingresos_log,
                order = (p, d, q), 
                trend = tendencia,
                enforce_stationarity = True,
                enforce_invertibility = True
            ).fit(method_kwargs = {'maxiter': 500})
        convergio = bool(modelo.mle_retvals.get('converged', False))
        raiz_ar = float(np.min(np.abs(modelo.arroots))) if p else np.inf
        raiz_ma = float(np.min(np.abs(modelo.maroots))) if q else np.inf
        elegible = convergio and np.isfinite(modelo.aicc) and raiz_ar > 1 and raiz_ma > 1
        filas_modelos.append({
            'Modelo': identificador, 'p': p,
            'd': d, 'q': q, 'Tendencia': tendencia,
            'AIC': modelo.aic, 'AICc': modelo.aicc,
            'BIC': modelo.bic,
            'Convergio': convergio, 'Elegible': elegible,
            'Min_raiz_AR': raiz_ar, 'Min_raiz_MA': raiz_ma,
            'Avisos': ' | '.join(str(aviso.message) for aviso in avisos)
        })
        if elegible:
            Modelos[identificador] = modelo
    except (ValueError, np.linalg.LinAlgError) as error:
        filas_modelos.append({'Modelo': identificador, 'p': p, 'd': d, 'q': q,
                              'Tendencia': tendencia, 'Elegible': False, 'Avisos': str(error)})

Comparacion = pd.DataFrame(filas_modelos).sort_values('AICc', na_position = 'last')
Comparacion.to_csv(SALIDA / '02_comparacion_arima.csv', index = False, encoding = 'utf-8-sig')

Ranking = Comparacion.loc[Comparacion['Elegible'] == True].copy()
if Ranking.empty:
    raise RuntimeError('No hay modelos convergentes y admisibles: revisar la especificación.')

Mejor = Ranking.iloc[0]
p = int(Mejor['p'])
q = int(Mejor['q'])
tendencia = str(Mejor['Tendencia'])
Modelo_entrenamiento = Modelos[Mejor['Modelo']]
print('\nMEJORES MODELOS POR AICc EN ENTRENAMIENTO')
print(Ranking[['Modelo', 'AICc', 'BIC']].head(10).to_string(index = False))
print(Modelo_entrenamiento.summary())

# Validación de modelo 

inicio = max(12, int(Modelo_entrenamiento.loglikelihood_burn))
Residuos = pd.Series(
    Modelo_entrenamiento.filter_results.standardized_forecasts_error[0],
    index = Ingresos.index, name = 'Innovacion_estandarizada'
).iloc[inicio:].dropna()

Ljung1 = acorr_ljungbox(Residuos, lags = [1],
                        return_df = True)

Ljung2 = acorr_ljungbox(Residuos, lags = [2],
                        return_df = True)

Ljung = acorr_ljungbox(Residuos, lags = [12, 24, 36],
                       model_df = p + q, return_df = True)
ARCH_1 = het_arch(
    Residuos,
    nlags = 1,
    ddof = p + q
)

ARCH_2 = het_arch(
    Residuos,
    nlags = 2,
    ddof = p + q
)

ARCH_12 = het_arch(
    Residuos,
    nlags = 12,
    ddof = p + q
)

JB = stats.jarque_bera(Residuos)

Diagnosticos = pd.DataFrame({
    'Prueba': [
        'Ljung-Box 1',
        'Ljung-Box 2',
        'Ljung-Box 12',
        'Ljung-Box 24',
        'Ljung-Box 36',
        'ARCH-LM 1',
        'ARCH-LM 2',
        'ARCH-LM 12',
        'Jarque-Bera'
    ],

    'H0': [
        'Sin autocorrelación hasta 1',
        'Sin autocorrelación hasta 2',
        'Sin autocorrelación hasta 12',
        'Sin autocorrelación hasta 24',
        'Sin autocorrelación hasta 36',
        'Sin efectos ARCH hasta 1',
        'Sin efectos ARCH hasta 2',
        'Sin efectos ARCH hasta 12',
        'Normalidad'
    ],

    'Estadistico': (
        list(Ljung1['lb_stat'])+
        list(Ljung2['lb_stat'])+
        list(Ljung['lb_stat'])
        + [
            ARCH_1[0],
            ARCH_2[0],
            ARCH_12[0],
            JB.statistic
        ]
    ),

    'p_valor': (
        list(Ljung1['lb_pvalue'])+
        list(Ljung2['lb_pvalue'])+
        list(Ljung['lb_pvalue'])
        + [
            ARCH_1[1],
            ARCH_2[1],
            ARCH_12[1],
            JB.pvalue
        ]
    )
})


Diagnosticos['Decision_5'] = np.where(Diagnosticos['p_valor'] < ALPHA, 
                                      'Rechazar H0', 'No rechazar H0')

print('\nDIAGNÓSTICO DE INNOVACIONES')
print(Diagnosticos.to_string(index = False))
print('Media y desviación de innovaciones:', Residuos.mean(), Residuos.std())

Raices = pd.DataFrame({
    'Componente': ['AR'] * len(Modelo_entrenamiento.arroots) + ['MA'] * len(Modelo_entrenamiento.maroots),
    'Modulo': np.r_[np.abs(Modelo_entrenamiento.arroots), np.abs(Modelo_entrenamiento.maroots)]
})
print('\nRAÍCES: módulo > 1 para AR estacionario / MA invertible, después de diferenciar')
print(Raices)
if not (Raices['Modulo'] > 1).all():
    raise RuntimeError('Raíces no admisibles en el ajuste final.')

Diagnosticos.to_csv(SALIDA / '07_diagnosticos.csv', index = False, encoding = 'utf-8-sig')
Raices.to_csv(SALIDA / '08_raices.csv', index = False)
Residuos.to_csv(SALIDA / '09_residuos.csv')

# No normalidad: afecta interpretación gaussiana de intervalos, no implica
# por sí misma autocorrelación. ARCH-LM sólo detecta el tipo de heterocedasticidad
# contrastado; no rechazar no demuestra varianza constante en todo sentido.
autocorrelacion = bool((Ljung['lb_pvalue'] < ALPHA).any())
alerta = ('DIAGNÓSTICO: autocorrelación residual; pronóstico provisional.' if autocorrelacion
          else 'Sin rechazo de autocorrelación en los rezagos evaluados; revisar los demás diagnósticos.')
print('\n' + alerta)
if autocorrelacion:
    print('Revisar estacionalidad de 12 meses, cambios estructurales y extensión SARIMA.')

# pronostico 

prediccion = Modelo_entrenamiento.get_forecast(steps = HORIZONTE)
intervalo = np.exp(prediccion.conf_int(alpha = ALPHA))
Pronostico = pd.DataFrame({
    'Media': np.exp(prediccion.predicted_mean + 0.5 * prediccion.var_pred_mean),
    'Mediana': np.exp(prediccion.predicted_mean),
    'Inferior_95': intervalo.iloc[:, 0], 'Superior_95': intervalo.iloc[:, 1]
})

Pronostico.index.name = 'Fecha'
print('\nPRONÓSTICO · MILLONES DE PESOS')
print(Pronostico.round(2).to_string())
Pronostico.to_csv(SALIDA / '10_pronostico.csv', encoding = 'utf-8-sig')

# Ajustes de un paso dentro de muestra: cada punto usa observaciones previas
# y parámetros estimados con la muestra completa; no es evaluación fuera de muestra.
pred_ajuste = Modelo_entrenamiento.get_prediction(start = inicio,
                                                  end = len(Ingresos) - 1, dynamic = False)
Ajuste = np.exp(pred_ajuste.predicted_mean + 0.5 * pred_ajuste.var_pred_mean)
Ajuste.name = 'Ajuste_un_paso_media'
pd.concat([Ingresos, Ajuste], axis = 1).to_csv(SALIDA / '11_serie_y_ajuste.csv')

# Visualización 

for vista in ['historia', 'detalle']:
    fig, ax = plt.subplots(figsize = (15, 8.5))
    fig.subplots_adjust(left = 0.08, right = 0.95, bottom = 0.19, top = 0.75)
    observada = Ingresos if vista == 'historia' else Ingresos.iloc[-60:]
    ajustada = Ajuste.loc[observada.index.min():]
    ax.plot(observada, color = NARANJA, lw = 1.5, label = 'Serie original')
    ax.plot(ajustada, color = CIRUELA, lw = 1.3, alpha = 0.9, label = 'Ajuste de un paso')
    ax.plot(Pronostico['Media'], color = OLIVA, lw = 2.2, label = 'Pronóstico · media', marker = 'o', markersize = 3)
    ax.fill_between(Pronostico.index, Pronostico['Inferior_95'], Pronostico['Superior_95'],
                    color = OLIVA, alpha = 0.15, label = 'Intervalo de predicción 95%')
    ax.axvline(Ingresos.index[-1], color = GRIS, lw = 1, ls = '--')
    ax.set_ylim(bottom = 0)
    ax.grid(axis = 'y', color = CREMA)
    ax.tick_params(length = 0, pad = 8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f'{valor:,.0f}'.replace(',', ' ')))
    ax.xaxis.set_major_locator(mdates.YearLocator(5 if vista == 'historia' else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(frameon = False, ncol = 2, loc = 'upper left', fontsize = 9)
    fig.text(0.08, 0.94, 'ECONOMETRICS DATA LAB', color = NARANJA, weight = 'bold')
    fig.text(0.08, 0.87, 'Ingresos: trayectoria, ajuste y pronóstico', fontsize = 24, weight = 'bold')
    fig.text(0.08, 0.82, f'ARIMA({p},{d},{q}) sobre logaritmos · Millones de pesos · Horizonte de {HORIZONTE} meses', color = GRIS)
    fig.text(0.08, 0.105, alerta, fontsize = 10, color = '#A52F0D', weight = 'bold')
    fig.text(0.08, 0.065, Fuente, fontsize = 8.5, color = GRIS)
    fig.text(0.08, 0.035, '2026 preliminar. Media e intervalos bajo supuesto lognormal. Ajuste dentro de muestra ≠ validación predictiva.', fontsize = 8.5, color = GRIS)
    fig.savefig(SALIDA / f'02_pronostico_{vista}.png', dpi = 300)
    fig.savefig(SALIDA / f'02_pronostico_{vista}.svg')
    plt.show()
    plt.close(fig)
