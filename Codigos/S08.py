# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Modelos estacionales
Subtema: Modelo SARIMA pt2 (estimación, evaluación y prónostico)
Sesión: 08
Fecha: 14/10/2026
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
import shutil
import subprocess
import tempfile
import os
import json
import re
import statsmodels.api as sm
from getpass import getpass, GetPassWarning


# Clases directas 

from pathlib import Path
from itertools import product
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from pmdarima.arima import nsdiffs

# Extra: Modulos y clases para función de api series de INEGI

import requests
from datetime import datetime, timezone

# Configuración 

FUENTE = 'BIE'                  
ID_SERIE = '735879'              
TOKEN_INEGI = os.environ.get('INEGI_TOKEN', "af847734-746b-8eb8-f0e6-4070cc851e47").strip()  
ETIQUETA_SERIE = 'PIB nacional | cifras originales | precios de 2018'
UNIDAD_PIB = 'Millones de pesos anualizados, a precios de 2018'
ARCHIVO = None                 
HOJA = 0                        
FECHA = 'Fecha'
VARIABLE = 'PIB'
SEP = ','                      
DECIMAL = '.'
DAYFIRST = True                  
FRECUENCIA = 'Q-DEC'             
s = 4                          
USAR_LOG = True                  
ALPHA = 0.05
HORIZONTE = 10
D_MANUAL = None                  
d_MANUAL = None                  
ORDEN_MANUAL = None              
P_ORDINARIOS = range(3)         
P_ESTACIONALES = range(2)       
MAX_DINAMICOS = 4                
TENDENCIAS = ('n', 'c')          
RSCRIPT = shutil.which('Rscript')  
HEGY_OBLIGATORIO = True          
HEGY_DETERMINISTICOS = (1, 1, 1) 
REPLICAS_DHF = 4999
SEMILLA = 2026
MOSTRAR = True
SALIDAS = Path(__name__).resolve().parent / 'salidas_sarima_PIB'
assert (FRECUENCIA == 'M' and s == 12) or (FRECUENCIA.startswith('Q') and s == 4), \
    'Este flujo está configurado para series mensuales o trimestrales.'
SALIDAS.mkdir(parents=True, exist_ok=True)
VERDE, GRIS, NARANJA = '#08989C', '#4D565E', '#F04A1D'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': GRIS, 'xtick.color': GRIS, 'ytick.color': GRIS,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})

# Definiciones para la exportación 

def guardar_figura(fig, nombre):
    fig.savefig(SALIDAS / f'{nombre}.png', dpi=500, bbox_inches='tight')
    fig.savefig(SALIDAS / f'{nombre}.svg', bbox_inches='tight')
    if MOSTRAR:
        plt.show()
    plt.close(fig)

# Definición de consulta mediante BIE 

def Series_INEGI_BIE(id_serie='', token='', periodo='Trimestral'):
    """Consulta nacional completa. Retorna Fecha/Serie y metadatos en df.attrs.
    Se conserva la interfaz del archivo original. Mensual: AAAA/MM.
    Trimestral: AAAA/01 ... AAAA/04 o AAAAQ1 ... AAAAQ4.
    Los códigos trimestrales NO se interpretan como enero/febrero/marzo/abril.
    """
    if periodo not in ('Mensual', 'Trimestral'):
        raise ValueError("periodo debe ser 'Mensual' o 'Trimestral'.")
    id_serie = str(id_serie).strip()
    if not id_serie.isdigit():
        raise ValueError('id_serie debe identificar un solo indicador numérico.')
    token = str(token).strip()
    if not token:
        raise ValueError('Falta el token de la API de INEGI.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', token):
        raise ValueError('El token contiene caracteres no válidos. Revisa lo copiado.')
    url_data = (
        'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/'
        f'jsonxml/INDICATOR/{id_serie}/es/00/false/BIE-BISE/2.0/{token}'
    )
    try:
        response = requests.get(url_data, params={'type':'json'}, timeout=(10,60))
        response.raise_for_status()
        data_json = response.json()
    except requests.RequestException:
        # Las excepciones requests pueden incluir la URL con token. No se imprimen.
        raise RuntimeError('No se pudo consultar el BIE. Revisa conexión, token e indicador.') from None
    except ValueError:
        raise RuntimeError('El BIE no devolvió JSON válido. Revisa el servicio y el token.') from None
    series = data_json.get('Series') if isinstance(data_json,dict) else None
    if not isinstance(series,list) or len(series)!=1 or not isinstance(series[0],dict):
        raise ValueError('La API no devolvió exactamente una serie. Revisa token e indicador.')
    estructura = series[0]
    id_devuelto = estructura.get('INDICADOR', estructura.get('INDICATOR'))
    if id_devuelto is not None and str(id_devuelto)!=id_serie:
        raise ValueError('El identificador recibido no coincide con el solicitado.')
    observaciones = estructura.get('OBSERVATIONS')
    if not isinstance(observaciones,list) or not observaciones:
        raise ValueError('La serie no tiene observaciones disponibles.')
    originales = pd.DataFrame(observaciones)
    if not {'TIME_PERIOD','OBS_VALUE'}.issubset(originales.columns):
        raise ValueError('Faltan TIME_PERIOD/OBS_VALUE en la respuesta del BIE.')
    df = originales[['TIME_PERIOD','OBS_VALUE']].copy()
    df.columns = ['Periodo_BIE','Serie']
    df['Serie'] = pd.to_numeric(df['Serie'],errors='coerce')
    if not np.isfinite(df['Serie'].to_numpy(dtype=float)).all():
        raise ValueError('El BIE devuelve valores faltantes/no numéricos. Revisa OBS_EXCEPTION; '
                         'no se eliminan ni interpolan automáticamente.')
    texto = df['Periodo_BIE'].astype(str).str.strip()
    texto = texto.str.replace(r'[-/]?[QT]', '/', regex=True)
    partes = texto.str.extract(r'^(?P<anio>\d{4})[-/](?P<parte>\d{1,2})$')
    if partes.isna().any().any():
        raise ValueError('Formato temporal BIE no reconocido. Esperado AAAA/NN o AAAAQN.')
    anio, parte = partes['anio'].astype(int), partes['parte'].astype(int)
    limite = 12 if periodo=='Mensual' else 4
    if not parte.between(1,limite).all():
        raise ValueError(f'Los periodos recibidos no son {periodo.lower()}es; revisa el indicador.')
    mes = parte if periodo=='Mensual' else (parte-1)*3+1
    df['Fecha'] = pd.to_datetime({'year':anio,'month':mes,'day':1},errors='raise')
    df = df[['Fecha','Serie']].sort_values('Fecha').reset_index(drop=True)
    if df['Fecha'].duplicated().any():
        raise ValueError('La API contiene periodos duplicados; revisa la serie solicitada.')
    df.attrs['metadatos'] = {k:v for k,v in estructura.items() if k!='OBSERVATIONS'}
    df.attrs['observaciones_BIE'] = observaciones  # Incluye notas/estatus originales.
    return df

# Construcción de consulta 

if FUENTE == 'BIE':
    if FRECUENCIA!='Q-DEC' or s!=4:
        raise ValueError('La consulta del PIB requiere FRECUENCIA=Q-DEC y s=4.')
    if not TOKEN_INEGI:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', GetPassWarning)
                TOKEN_INEGI = getpass('Token de INEGI (entrada oculta): ').strip()
        except (EOFError, KeyboardInterrupt, GetPassWarning):
            raise RuntimeError('No se recibió el token por una entrada oculta. Configura INEGI_TOKEN '
                               'o ejecuta el script en una terminal.') from None
    print(f'Consultando BIE: indicador {ID_SERIE} | PIB original trimestral | nacional...')
    Datos_BIE = Series_INEGI_BIE(id_serie=ID_SERIE, token=TOKEN_INEGI, periodo='Trimestral')
    PIB = Datos_BIE.rename(columns={'Serie':VARIABLE,'Fecha':FECHA})
    Datos = PIB[[FECHA,VARIABLE]].copy()
    Metadatos_BIE = dict(Datos_BIE.attrs['metadatos'])
    Metadatos_BIE.update({'serie_configurada':ID_SERIE,'descripcion_configurada':ETIQUETA_SERIE,
                         'unidad_configurada':UNIDAD_PIB,'periodicidad':'Trimestral',
                         'consulta_UTC':datetime.now(timezone.utc).isoformat(),
                         'cobertura':'Nacional (00)','historia_completa':True,
                         'fuente':'https://www.inegi.org.mx/servicios/feeds.html'})
    (SALIDAS/'PIB_BIE_metadatos.json').write_text(json.dumps(Metadatos_BIE,ensure_ascii=False,indent=2),
                                               encoding='utf-8')
    pd.DataFrame(Datos_BIE.attrs['observaciones_BIE']).to_csv(
        SALIDAS/'PIB_BIE_observaciones_originales.csv',index=False,encoding='utf-8-sig')
    Datos.to_csv(SALIDAS/'PIB_BIE_consulta.csv',index=False,encoding='utf-8-sig')
    Origen_datos = f'INEGI BIE | {ID_SERIE} | cifras originales | base 2018'
elif FUENTE == 'ARCHIVO':
    if ARCHIVO is None:
        raise ValueError('Configura ARCHIVO para leer el PIB descargado.')
    ruta = Path(ARCHIVO).expanduser()
    if ruta.suffix.lower() in ('.xlsx', '.xls'):
        Datos = pd.read_excel(ruta, sheet_name=HOJA)  # .xls necesita xlrd.
    elif ruta.suffix.lower() in ('.csv', '.txt'):
        Datos = pd.read_csv(ruta, sep=SEP, decimal=DECIMAL)
    else:
        raise ValueError('Usa CSV o Excel; adapta solo este bloque para otra fuente.')
    PIB = Datos[[FECHA,VARIABLE]].copy()
    Origen_datos = str(ruta)
else:
    raise ValueError("FUENTE debe ser 'BIE' o 'ARCHIVO'.")

# Indice trimestral y regularidad 

if not {FECHA, VARIABLE}.issubset(Datos.columns):
    raise ValueError(f'Faltan columnas: {FECHA}, {VARIABLE}. Disponibles: {Datos.columns.tolist()}')
Fechas = pd.to_datetime(Datos[FECHA], errors='raise', dayfirst=DAYFIRST)
Periodos = pd.PeriodIndex(Fechas, freq=FRECUENCIA)
Valores = pd.to_numeric(Datos[VARIABLE], errors='raise').to_numpy(dtype=float)
Serie = pd.Series(Valores, index=Periodos, name=VARIABLE).sort_index()
if Serie.index.has_duplicates:
    raise ValueError('Hay más de una observación por periodo. Define una agregación económica.')
Indice_completo = pd.period_range(Serie.index.min(), Serie.index.max(), freq=FRECUENCIA)
Serie = Serie.reindex(Indice_completo)
if not np.isfinite(Serie.to_numpy()).all():
    raise ValueError('Hay periodos faltantes, NA o infinitos. No se eliminan ni interpolan automáticamente.')
if len(Serie) < 6*s or Serie.nunique() < 3:
    raise ValueError('Se necesitan al menos 6 ciclos y una serie no constante para este flujo.')
if USAR_LOG and (Serie <= 0).any():
    raise ValueError('El logaritmo exige valores estrictamente positivos.')
Y = np.log(Serie) if USAR_LOG else Serie.copy()
Y.name = f'log({VARIABLE})' if USAR_LOG else VARIABLE
Serie.to_csv(SALIDAS / 'serie_utilizada.csv', encoding='utf-8-sig')
print(Serie.head(), '\nN =', len(Serie), '| s =', s,
      '| Historia:', Serie.index[0], 'a', Serie.index[-1], '\nUnidad:',UNIDAD_PIB)
print('Toda la muestra se utiliza en la estimación; se pronostican 10 trimestres.')


# Visualización inicial 

fig_d1, ax_d1 = plt.subplots(
    figsize = (14,6),
    dpi = 500
    )
fig_d1.patch.set_facecolor("white")
ax_d1.plot(
    Datos[["Fecha"]],
    Datos[["PIB"]],
    color = "#08989C",
    linewidth = 1.7
    )
ax_d1.axhline(
    y = np.mean(Datos[["PIB"]]),
    color = "#9AA0A6",
    linestyle = "--",
    linewidth = 0.9
    )
ax_d1.set_ylabel(
    "PIB: Millones de pesos a precios del 2018",
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
    "Producto Interno Bruto de México",
    fontsize=19,
    fontweight="bold",
    color="#003057",
    ha="left",
    va="top"
)
fig_d1.text(
    0.10, 0.905,
    "Serie trimestral: 1980 Q1 - 2026 Q2",
    fontsize=11,
    color="#4D565E",
    ha="left",
    va="top"
)
fig_d1.text(
    0.10, 0.025,
    "Fuente. INEGI. Banco de Información Económica (BIE)",
    fontsize=9,
    color="#4D565E",
    ha="left",
    va="bottom"
)
fig_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
plt.show()

# Grárfico polar 

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(Serie.index.to_timestamp(), Serie, color=VERDE, linewidth=1.8)
ax.set(title=ETIQUETA_SERIE, xlabel='Trimestre', ylabel=UNIDAD_PIB)
ax.grid(axis='y', alpha=.15)
guardar_figura(fig, '01_serie')

# Ángulo = mes/trimestre; radio = nivel original, una curva por año.
Estacion = Serie.index.month if s == 12 else Serie.index.quarter
Polar = pd.DataFrame({'anio': Serie.index.year, 'estacion': Estacion, 'valor': Serie.values})
Polar = Polar.pivot(index='anio', columns='estacion', values='valor').reindex(columns=range(1,s+1))
Angulos = np.linspace(0, 2*np.pi, s, endpoint=False)
Angulos = np.r_[Angulos, Angulos[0]]
Desplazamiento = max(0.0, -float(Serie.min()))
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
for anio, fila in Polar.iterrows():
    radio = fila.to_numpy() + Desplazamiento
    ax.plot(Angulos, np.r_[radio, radio[0]], color=VERDE, alpha=.18, linewidth=1)
Perfil = Polar.mean(axis=0).to_numpy() + Desplazamiento
ax.plot(Angulos, np.r_[Perfil, Perfil[0]], color=NARANJA, linewidth=2.8,
        label='Promedio por estación')
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
Etiquetas = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'] \
    if s == 12 else ['T1','T2','T3','T4']
ax.set_xticks(Angulos[:-1], Etiquetas)
ax.set_title('Perfil estacional | una curva por año', pad=25)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.05))
if Desplazamiento:
    fig.text(.08, .02, f'Radio = valor + {Desplazamiento:.2f} (para admitir valores negativos).')
guardar_figura(fig, '02_polar')
STL_resultado = STL(Y.to_numpy(), period=s, robust=True).fit()
fig, ejes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for ax, componente, etiqueta in zip(ejes,
        [Y.values, STL_resultado.trend, STL_resultado.seasonal, STL_resultado.resid],
        ['Serie', 'Tendencia', 'Estacional', 'Resto']):
    ax.plot(Y.index.to_timestamp(), componente, color=VERDE, linewidth=1.3)
    ax.set_ylabel(etiqueta)
fig.suptitle('Descomposición STL | escala de estimación')
fig.tight_layout()
guardar_figura(fig, '03_STL')

# Prueba de raíz unitaria (ADF)

ADF_nivel = adfuller(Y, regression='ct', autolag='BIC')
ADF = pd.DataFrame([{'Transformacion': 'Nivel; constante y tendencia',
                     'ADF': ADF_nivel[0], 'p_ADF': ADF_nivel[1], 'rezagos': ADF_nivel[2],
                     'H0': 'Raíz unitaria ordinaria (frecuencia cero)'}])
print('\nADF ordinario (solo frecuencia cero):\n', ADF.to_string(index=False))

# Dickey-Hasza-Fuller (Raices unitarias estacionales) 

Dummies_DHF = np.eye(s)[np.arange(s, len(Y)) % s]
Proyector_DHF = np.linalg.pinv(Dummies_DHF)
def estadistico_dhf(valores):
    a = np.asarray(valores, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    x, dy = a[:-s], a[s:] - a[:-s]
    xr = x - Dummies_DHF @ (Proyector_DHF @ x)
    dyr = dy - Dummies_DHF @ (Proyector_DHF @ dy)
    Sxx = (xr*xr).sum(axis=0)
    gamma = (xr*dyr).sum(axis=0) / Sxx
    residuo = dyr - xr*gamma
    sigma2 = (residuo*residuo).sum(axis=0)/(len(dy)-s-1)
    return gamma / np.sqrt(sigma2/Sxx), gamma, residuo

DHF_t, DHF_gamma, DHF_residuos = estadistico_dhf(Y.values)
rng = np.random.default_rng(SEMILLA)
# Mismo tamaño y calendario. Bajo H0: y_t=y_(t-s)+e_t, e iid N(0,1), sin deriva.
Simuladas = np.zeros((len(Y), REPLICAS_DHF))
Innovaciones = rng.normal(size=(len(Y)-s, REPLICAS_DHF))
for t in range(s, len(Y)):
    Simuladas[t] = Simuladas[t-s] + Innovaciones[t-s]
DHF_sim, _, _ = estadistico_dhf(Simuladas)
DHF_p = (1 + np.count_nonzero(DHF_sim <= DHF_t[0]))/(REPLICAS_DHF+1)
DHF = pd.DataFrame([{'Prueba': 'DHF no aumentado; Monte Carlo iid gaussiano sin deriva',
                     'rho_s': 1+DHF_gamma[0], 't': DHF_t[0], 'p_MC': DHF_p,
                     'critico_1pct': np.quantile(DHF_sim,.01),
                     'critico_5pct': np.quantile(DHF_sim,.05),
                     'critico_10pct': np.quantile(DHF_sim,.10),
                     'error_MC_aprox': np.sqrt(DHF_p*(1-DHF_p)/(REPLICAS_DHF+1)),
                     'decision': 'Rechazar H0 conjunta' if DHF_p<ALPHA else 'No rechazar H0 conjunta'}])
DHF_LB = acorr_ljungbox(DHF_residuos[:,0], lags=[s,2*s], return_df=True)
print('\nDHF (calibración restringida):\n', DHF.to_string(index=False))
print('\nAutocorrelación de residuos DHF:\n', DHF_LB)
if (DHF_LB.lb_pvalue < ALPHA).any():
    print('DHF: residuos autocorrelacionados; el p_MC iid NO sustenta inferencia. Prioriza HEGY aumentado.')
DHF.to_csv(SALIDAS/'04_DHF.csv', index=False, encoding='utf-8-sig')
DHF_LB.to_csv(SALIDAS/'04_DHF_residuos_Ljung_Box.csv')
del Simuladas, Innovaciones

# Pruebas y ACF con PACF 

D_sugerido = nsdiffs(Y.to_numpy(), m=s, max_D=1, test='ocsb')
D = D_sugerido if D_MANUAL is None else int(D_MANUAL)
if D not in (0,1):
    raise ValueError('D debe ser 0 o 1 en este flujo.')
Y_estacional = Y.diff(s).dropna() if D==1 else Y.copy()
print('\nD sugerido por OCSB =', D_sugerido, '| D utilizado =', D)

Diferencias, Pruebas_d = {}, []
for candidato_d in (0,1,2):
    z = Y_estacional.copy()
    for _ in range(candidato_d):
        z = z.diff().dropna()
    Diferencias[candidato_d] = z
    a = adfuller(z, regression='c', autolag='BIC')
    # KPSS puede devolver límites tabulados (no p exacto fuera de [.01,.10]).
    with warnings.catch_warnings(record=True) as avisos_kpss:
        warnings.simplefilter('always')
        k = kpss(z, regression='c', nlags='auto')
    Pruebas_d.append({'d': candidato_d, 'D': D, 'ADF': a[0], 'p_ADF': a[1],
                     'rezagos_ADF': a[2], 'KPSS': k[0], 'p_KPSS_tabla': k[1],
                     'nota_KPSS': ' | '.join(str(w.message) for w in avisos_kpss),
                     'compatible_estacionariedad': a[1]<ALPHA and k[1]>=ALPHA})
Pruebas_d = pd.DataFrame(Pruebas_d)
print('\nContrastes después de aplicar D:\n', Pruebas_d.to_string(index=False))
if d_MANUAL is None:
    admisibles = Pruebas_d.loc[Pruebas_d.compatible_estacionariedad, 'd']
    if admisibles.empty:
        raise RuntimeError('ADF/KPSS no concuerdan hasta d=2. Revisa tendencia, HEGY, '
                           'quiebres y estacionalidad determinista; fija d_MANUAL solo con fundamento.')
    d = int(admisibles.iloc[0])
else:
    d = int(d_MANUAL)
if d not in (0,1,2):
    raise ValueError('d debe ser 0, 1 o 2.')
Pruebas_d.to_csv(SALIDAS/'06_diferenciacion_ADF_KPSS.csv', index=False, encoding='utf-8-sig')
ADF.to_csv(SALIDAS/'04_ADF_nivel.csv', index=False, encoding='utf-8-sig')
Z = Diferencias[d]
print('\nÓrdenes de integración utilizados: d =', d, ', D =', D)
fig, ejes = plt.subplots(3,1,figsize=(11,10))
ejes[0].plot(Z.index.to_timestamp(), Z, color=VERDE)
ejes[0].set(title=f'Serie transformada: (1-B)^{d}(1-B^{s})^{D} y', ylabel=Y.name)
max_lags = min(3*s, len(Z)//2-1)
plot_acf(Z, lags=max_lags, zero=False, ax=ejes[1], color=VERDE)
plot_pacf(Z, lags=max_lags, zero=False, method='ywm', ax=ejes[2], color=VERDE)
for ax in ejes[1:]:
    for h in range(s,max_lags+1,s):
        ax.axvline(h, color=NARANJA, linestyle='--', alpha=.5)
    ax.set_xlabel('Rezagos | líneas naranjas: s, 2s, 3s')
fig.tight_layout()
guardar_figura(fig, '07_identificacion_ACF_PACF')

# Estimación del modelo SARIMA con toda la muestra 

Candidatos, Fallos, Ajustes = [], [], {}
for p,q,P,Q,tendencia in product(P_ORDINARIOS,P_ORDINARIOS,P_ESTACIONALES,P_ESTACIONALES,TENDENCIAS):
    if p+q+P+Q > MAX_DINAMICOS:
        continue
    orden, estacional = (p,d,q), (P,D,Q,s)
    try:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            ajuste = SARIMAX(Y, order=orden, seasonal_order=estacional, trend=tendencia,
                            simple_differencing=False, enforce_stationarity=True,
                            enforce_invertibility=True, use_exact_diffuse=True).fit(disp=False,maxiter=300)
        if not ajuste.mle_retvals.get('converged',False) or not np.isfinite(ajuste.bic):
            Fallos.append({'orden': str(orden),'estacional': str(estacional),
                           'tendencia':tendencia,'motivo':'No convergió o BIC no finito'})
            continue
        identificador = len(Candidatos)
        Ajustes[identificador] = ajuste
        Candidatos.append({'id':identificador,'p':p,'d':d,'q':q,'P':P,'D':D,'Q':Q,'s':s,
                           'tendencia':tendencia,'k':len(ajuste.params),'AIC':ajuste.aic,
                           'AICc':ajuste.aicc,'BIC':ajuste.bic,
                           'avisos':' | '.join(str(w.message) for w in avisos)})
    except (ValueError, np.linalg.LinAlgError) as error:
        Fallos.append({'orden':str(orden),'estacional':str(estacional),
                       'tendencia':tendencia,'motivo':str(error)})
Comparacion = pd.DataFrame(Candidatos)
if Comparacion.empty:
    raise RuntimeError('Ningún candidato convergió. Revisa especificación/escala/muestra.')
Comparacion = Comparacion.sort_values(['BIC','k']).reset_index(drop=True)
print('\nCandidatos con menor BIC (d/D fijos):\n', Comparacion.head(10).to_string(index=False))
Comparacion.to_csv(SALIDAS/'08_comparacion_modelos.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(Fallos, columns=['orden','estacional','tendencia','motivo']).to_csv(
    SALIDAS/'08_modelos_descartados.csv', index=False, encoding='utf-8-sig')
Mejor = Comparacion.iloc[0]
if ORDEN_MANUAL is None:
    Resultado = Ajustes[int(Mejor.id)]
else:
    orden, estacional = ORDEN_MANUAL
    if orden[1]!=d or estacional[1]!=D or estacional[3]!=s:
        raise ValueError('ORDEN_MANUAL debe coincidir con d,D,s identificados/configurados.')
    Resultado = SARIMAX(Y, order=orden, seasonal_order=estacional, trend=TENDENCIAS[0],
                       simple_differencing=False, enforce_stationarity=True,
                       enforce_invertibility=True, use_exact_diffuse=True).fit(disp=False,maxiter=300)
    if not Resultado.mle_retvals.get('converged',False):
        raise RuntimeError('El modelo manual no convergió.')
p,d,q = Resultado.model.order
P,D,Q,s = Resultado.model.seasonal_order
print(Resultado.summary())
(SALIDAS/'09_resumen_modelo.txt').write_text(str(Resultado.summary()), encoding='utf-8')

# Circulo unitario 

## Raices del modelo 

Raices_AR = 1/np.asarray(Resultado.arroots, dtype = complex)
Raices_MA = 1/np.asarray(Resultado.maroots, dtype = complex)

orden = Resultado.model.order
estacional = Resultado.model.seasonal_order
angulo = np.linspace(0, 2 * np.pi, 600)

with plt.rc_context({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": "#4D565E",
    "axes.labelcolor": "#4D565E",
    "xtick.color": "#4D565E",
    "ytick.color": "#4D565E"
}):
    fig, ejes = plt.subplots(
        1, 2, figsize=(11, 5.8), constrained_layout=True,
        dpi = 500
    )

    for ax, raices, tipo, color in zip(
        ejes,
        [Raices_AR, Raices_MA],
        ["AR", "MA"],
        ["#08989C", "#F04A1D"]
    ):
        # Círculo unitario y zona interior.
        ax.fill(
            np.cos(angulo), np.sin(angulo),
            color=color, alpha=0.06
        )
        ax.plot(
            np.cos(angulo), np.sin(angulo),
            color="#4D565E", linewidth=1.5
        )

        ax.axhline(0, color="#9AA0A6", linewidth=0.8)
        ax.axvline(0, color="#9AA0A6", linewidth=0.8)

        if raices.size:
            ax.scatter(
                raices.real, raices.imag,
                s=85, color=color, edgecolors="white",
                linewidths=0.9, zorder=3
            )

            max_modulo = np.max(np.abs(raices))
            cumple = bool(np.all(np.abs(raices) < 1))
            estado = "Cumple" if cumple else "No cumple"

            ax.set_title(
                f"Raíces inversas {tipo}\n"
                f"Máximo módulo: {max_modulo:.6f} | {estado}",
                fontsize=12, fontweight="bold", color=color
            )

            print(
                f"{tipo}: máximo módulo inverso = {max_modulo:.8f}"
                f" | Todas dentro del círculo: {cumple}"
            )
        else:
            max_modulo = 0
            ax.set_title(
                f"Raíces inversas {tipo}",
                fontsize=12, fontweight="bold", color=color
            )
            ax.text(
                0, 0.3, f"Sin términos {tipo}",
                ha="center", va="center", color=color
            )
            print(f"{tipo}: el modelo no tiene términos de este tipo.")

        limite = max(1.15, 1.12 * max_modulo)
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Parte real")
        ax.set_ylabel("Parte imaginaria")
        ax.grid(alpha=0.18)

        for borde in ax.spines.values():
            borde.set_visible(False)

    fig.suptitle(
        f"SARIMA{orden} × {estacional}\n"
        "Círculo unitario · Raíces inversas",
        fontsize=15, fontweight="bold", color="#003057"
    )

    SALIDAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDAS / "10_circulo_unitario.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(SALIDAS / "10_circulo_unitario.svg",
                bbox_inches="tight")
    plt.show()

# Innovaciones (residuales) estandarizados

Inicio_residuos = max(Resultado.loglikelihood_burn, Resultado.filter_results.nobs_diffuse)
Residuos = pd.Series(Resultado.filter_results.standardized_forecasts_error[0], index=Y.index)
Residuos = Residuos.iloc[Inicio_residuos:].replace([np.inf,-np.inf],np.nan).dropna()
model_df = p+q+P+Q           # Cantidad de parámetros AR/MA, NO p+q+s*(P+Q).
Lags_LB = sorted({h for h in [1,2,s,2*s,3*s] if h<len(Residuos)})
Ljung = acorr_ljungbox(Residuos, lags=Lags_LB, model_df=model_df, return_df=True)
Filas = []
for h, fila in Ljung.iterrows():
    Filas.append({'Prueba':f'Ljung-Box {h}', 'H0':f'Sin autocorrelación hasta {h}',
                  'Estadistico':fila.lb_stat,'p_valor':fila.lb_pvalue,'gl':h-model_df})
# Agregar un rezago válido si todos los solicitados fueran <= model_df.
if Ljung.lb_pvalue.notna().sum()==0:
    raise RuntimeError('Sin grados de libertad para Ljung-Box; reduce complejidad o amplía muestra.')
for h in sorted({1,2,s}):
    if len(Residuos)-h <= h+model_df+2:
        continue
    lm, pv, _, _ = het_arch(Residuos, nlags=h, ddof=model_df)
    Filas.append({'Prueba':f'ARCH-LM {h}', 'H0':f'Sin efectos ARCH hasta {h}',
                  'Estadistico':lm,'p_valor':pv,'gl':h})
JB = stats.jarque_bera(Residuos)
Filas.append({'Prueba':'Jarque-Bera','H0':'Normalidad de las innovaciones',
              'Estadistico':JB.statistic,'p_valor':JB.pvalue,'gl':2})
# Complemento para heterocedasticidad no ARCH: primer tercio vs último tercio.
Varianza = Resultado.test_heteroskedasticity(method='breakvar',alternative='two-sided',use_f=True)[0]
Filas.append({'Prueba':'Cambio de varianza (tercios)',
              'H0':'Igual varianza al inicio y al final',
              'Estadistico':Varianza[0],'p_valor':Varianza[1],'gl':np.nan})
Diagnosticos = pd.DataFrame(Filas)
Diagnosticos['Decision'] = np.where(Diagnosticos.p_valor.isna(),'No evaluable',
                           np.where(Diagnosticos.p_valor<ALPHA,'Rechazar H0','No rechazar H0'))
print('\nValidación de innovaciones:\n', Diagnosticos.to_string(index=False))
Diagnosticos.to_csv(SALIDAS/'10_diagnosticos.csv', index=False, encoding='utf-8-sig')
Residuos.to_csv(SALIDAS/'10_innovaciones_estandarizadas.csv', encoding='utf-8-sig')
Rechazos = Diagnosticos.loc[Diagnosticos.p_valor<ALPHA,'Prueba'].tolist()
Validacion = ('EXPLORATORIO: rechazos en '+', '.join(Rechazos)) if Rechazos \
    else 'Sin rechazos en los contrastes evaluables; no demuestra validez absoluta.'
print(Validacion)
fig, ejes = plt.subplots(2,2,figsize=(12,7))
ejes[0,0].plot(Residuos.index.to_timestamp(),Residuos,color=VERDE,linewidth=1)
ejes[0,0].axhline(0,color=GRIS,linewidth=.8)
ejes[0,0].set_title('Innovaciones estandarizadas')
plot_acf(Residuos,lags=min(3*s,len(Residuos)//2-1),zero=False,ax=ejes[0,1],color=VERDE)
ejes[1,0].hist(Residuos,bins='auto',density=True,color=VERDE,alpha=.65)
x = np.linspace(Residuos.min(),Residuos.max(),200)
ejes[1,0].plot(x,stats.norm.pdf(x),color=NARANJA)
ejes[1,0].set_title('Distribución | referencia N(0,1)')
sm.qqplot(Residuos,line='s',ax=ejes[1,1],markerfacecolor=VERDE,markeredgecolor=VERDE)
ejes[1,1].set_title('Q–Q de normalidad')
fig.tight_layout()
guardar_figura(fig,'11_diagnosticos_graficos')

# Pronóstico a 10 periodos 

Prediccion = Resultado.get_forecast(steps=HORIZONTE)
Media = np.asarray(Prediccion.predicted_mean)
IC80 = np.asarray(Prediccion.conf_int(alpha=.20))
IC95 = np.asarray(Prediccion.conf_int(alpha=.05))
Futuro = pd.period_range(Serie.index[-1]+1,periods=HORIZONTE,freq=FRECUENCIA)
Pronostico = pd.DataFrame(index=Futuro)
if USAR_LOG:
    Pronostico['Mediana'] = np.exp(Media)
    Pronostico['Pronostico_media'] = np.exp(Media + .5*np.asarray(Prediccion.var_pred_mean))
    IC80, IC95 = np.exp(IC80), np.exp(IC95)
else:
    Pronostico['Pronostico_media'] = Media
Pronostico['LI80'],Pronostico['LS80'] = IC80[:,0],IC80[:,1]
Pronostico['LI95'],Pronostico['LS95'] = IC95[:,0],IC95[:,1]
Pronostico.index.name = 'Periodo'
Pronostico['Estado_validacion'] = Validacion
Pronostico['Origen'] = Origen_datos
Pronostico['Unidad'] = UNIDAD_PIB
print('\nPronóstico (escala original):\n',Pronostico.to_string())
Pronostico.to_csv(SALIDAS/'12_pronostico_10_periodos.csv',encoding='utf-8-sig')
fig,ax = plt.subplots(figsize=(12,5))
Historico = Serie.iloc[-min(len(Serie),5*s):]
ax.plot(Historico.index.to_timestamp(),Historico,color=GRIS,label='Observado')
f = Futuro.to_timestamp()
ax.fill_between(f,Pronostico.LI95,Pronostico.LS95,color=VERDE,alpha=.12,label='Intervalo 95 %')
ax.fill_between(f,Pronostico.LI80,Pronostico.LS80,color=VERDE,alpha=.23,label='Intervalo 80 %')
ax.plot(f,Pronostico.Pronostico_media,color=VERDE,marker='o',label='Pronóstico: media')
ax.axvline(Serie.index[-1].to_timestamp(),color=NARANJA,linestyle='--',alpha=.7)
ax.set(title=f'SARIMA({p},{d},{q})({P},{D},{Q})[{s}] | {HORIZONTE} periodos'
       + (' | EXPLORATORIO' if Rechazos else ''),
       xlabel='Trimestre',ylabel=UNIDAD_PIB)
ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
guardar_figura(fig,'13_pronostico')
print('\nArchivos exportados a:',SALIDAS)





