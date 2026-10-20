# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Modelos estacionales y multivariantes
Subtema: Modelo SARIMAX (V.D = COVID-19)
Sesión: 09
Fecha: 16/10/2026
Docente: Alexis Adonai Morales Albero
"""

# Modulos a utilizar 

import warnings 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
import matplotlib.dates as mdates
import shutil
import subprocess
import tempfile
import os
import json
import re
import statsmodels.api as sm

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
from getpass import getpass, GetPassWarning

# Configuración 

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Series de tiempo\ST_SEP26"
    )
SALIDAS = BASE / 'Resultados' / 'S09_SARIMAX'
SALIDAS.mkdir(parents=True, exist_ok=True)


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
P_ORDINARIOS = range(4)          
P_ESTACIONALES = range(4)       
MAX_DINAMICOS = 4                
TENDENCIA_MANUAL = 'n'           
TENDENCIAS = ('n', 'c')          
MODO_DUMMY = 'trimestres'            
PANDEMIA_INICIO = pd.Period('2020Q1', freq=FRECUENCIA)
PANDEMIA_FIN = pd.Period('2020Q4', freq=FRECUENCIA)
MOSTRAR = True

assert FRECUENCIA == 'Q-DEC' and s == 4, 'El PIB requiere frecuencia trimestral y s=4.'
if ORDEN_MANUAL is not None:
    o, oe = ORDEN_MANUAL
    if len(o)!=3 or len(oe)!=4 or oe[3]!=s or o[1] not in (0,1,2) or oe[1] not in (0,1):
        raise ValueError('ORDEN_MANUAL debe ser ((p,d,q),(P,D,Q,4)); revisa d,D,s.')
    if any(not isinstance(v,int) or v<0 for v in (*o,*oe)):
        raise ValueError('Los órdenes manuales deben ser enteros no negativos.')
VERDE, GRIS, NARANJA = '#08989C', '#4D565E', '#F04A1D'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': GRIS, 'xtick.color': GRIS, 'ytick.color': GRIS,
                     'axes.titleweight': 'bold', 'axes.titlecolor': '#003057',
                     'figure.titlesize': 16, 'figure.titleweight': 'bold',
                     'figure.facecolor': 'white'})


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
    TOKEN_INEGI = ''  # No se guarda la credencial en archivos o metadatos.
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
    if not ruta.is_absolute():
        ruta = BASE / ruta
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
# El CSV exportado usa AAAA-MM-DD: evitar intercambiar mes y día al releerlo.
Fechas_texto = Datos[FECHA].astype(str).str.strip()
if Fechas_texto.str.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[ T].*)?').all():
    Fechas = pd.to_datetime(Fechas_texto, format='ISO8601', errors='raise')
else:
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

fig_d1, ax_d1 = plt.subplots(figsize=(14, 6))
ax_d1.plot(Serie.index.to_timestamp(), Serie, color=VERDE, linewidth=1.7)
ax_d1.axhline(Serie.mean(), color='#9AA0A6', linestyle='--', linewidth=.9)
ax_d1.set_ylabel(UNIDAD_PIB, fontsize=11, color=GRIS)
ax_d1.set_xlabel('Año', fontsize=11, color=GRIS)
ax_d1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_d1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax_d1.grid(axis='y', linestyle=':', linewidth=.8, color='#D5D8DC')
ax_d1.spines[['left', 'bottom']].set_color('#D5D8DC')
ax_d1.tick_params(axis='both', colors=GRIS, labelsize=9)
fig_d1.text(.10, .965, 'Producto Interno Bruto de México', fontsize=19,
            fontweight='bold', color='#003057', ha='left', va='top')
fig_d1.text(.10, .905,
            f'Serie trimestral: {Serie.index[0]} - {Serie.index[-1]}',
            fontsize=11, color=GRIS, ha='left', va='top')
fig_d1.text(.10, .025, 'Fuente. INEGI. Banco de Información Económica (BIE)',
            fontsize=9, color=GRIS, ha='left', va='bottom')
fig_d1.subplots_adjust(left=.10, right=.97, top=.82, bottom=.15)
guardar_figura(fig_d1, '00_visualizacion_inicial')

# Construcción de variables dummy para la pandemia

Trimestres_pandemia = pd.period_range(PANDEMIA_INICIO, PANDEMIA_FIN, freq=FRECUENCIA)
if PANDEMIA_INICIO > PANDEMIA_FIN:
    raise ValueError('El inicio de la intervención es posterior al final.')
if not Trimestres_pandemia.isin(Serie.index).all():
    raise ValueError('La serie debe contener los cuatro trimestres de 2020 para estimar esta intervención.')
if not (Serie.index < PANDEMIA_INICIO).any() or not (Serie.index > PANDEMIA_FIN).any():
    raise ValueError('Se necesitan observaciones anteriores y posteriores a 2020.')
X = pd.DataFrame(index=Y.index)
if MODO_DUMMY == 'bloque':
    X['Covid_2020'] = ((Y.index >= PANDEMIA_INICIO) & (Y.index <= PANDEMIA_FIN)).astype(float)
elif MODO_DUMMY == 'trimestres':
    for trimestre in Trimestres_pandemia:
        X[f'Covid_{trimestre}'] = (Y.index == trimestre).astype(float)
else:
    raise ValueError("MODO_DUMMY debe ser 'bloque' o 'trimestres'.")
if not X.index.equals(Y.index) or np.linalg.matrix_rank(X.to_numpy()) != X.shape[1]:
    raise ValueError('Revisa alineación y rango de las dummies.')
Datos_modelo = pd.concat([Serie, Y.rename('Escala_estimacion'), X], axis=1)
Datos_modelo = Datos_modelo.sort_index()
Datos_modelo.to_csv(SALIDAS/'01_PIB_y_dummies.csv', encoding='utf-8-sig')
print('\nDummies alrededor de la pandemia:\n', Datos_modelo.loc['2019Q1':'2021Q4'])


# Visualización de la pandemia

fig, ejes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ejes[0].plot(Serie.index.to_timestamp(), Serie, color=VERDE, linewidth=1.6)
ejes[0].set(title=ETIQUETA_SERIE, ylabel=UNIDAD_PIB)
ejes[0].axvspan(PANDEMIA_INICIO.start_time, (PANDEMIA_FIN+1).start_time,
               color=NARANJA, alpha=.15, label='Intervención 2020')
for nombre in X.columns:
    ejes[1].step(X.index.to_timestamp(), X[nombre], where='post', label=nombre)
ejes[1].set(ylabel='Dummy', xlabel='Trimestre', yticks=[0,1], ylim=(-.1,1.2))
for ax in ejes:
    ax.grid(axis='y', alpha=.15)
    ax.legend(frameon=False)
fig.tight_layout()
guardar_figura(fig, '02_PIB_intervencion_2020')


# Pruebas de estacionariedad y diferenciación

D_sugerido = nsdiffs(Y.to_numpy(), m=s, max_D=1, test='ocsb') if ORDEN_MANUAL is None and D_MANUAL is None else np.nan
D = int(ORDEN_MANUAL[1][1]) if ORDEN_MANUAL is not None else (D_sugerido if D_MANUAL is None else int(D_MANUAL))
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
if ORDEN_MANUAL is not None:
    d = int(ORDEN_MANUAL[0][1])
elif d_MANUAL is None:
    admisibles = Pruebas_d.loc[Pruebas_d.compatible_estacionariedad, 'd']
    if admisibles.empty:
        raise RuntimeError('ADF/KPSS no concuerdan hasta d=2. Revisa tendencia, '
                           'quiebres y estacionalidad determinista; reutiliza el d del SARIMA validado.')
    d = int(admisibles.iloc[0])
else:
    d = int(d_MANUAL)
if d not in (0,1,2):
    raise ValueError('d debe ser 0, 1 o 2.')
Pruebas_d.to_csv(SALIDAS/'06_diferenciacion_ADF_KPSS.csv', index=False, encoding='utf-8-sig')
Z = Diferencias[d]
print('\nÓrdenes de integración utilizados: d =', d, ', D =', D)

# Pruebas y ACF con PACF

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
# ACF con corte en s sugiere Q; PACF con corte en s sugiere P.
# Cortes en rezagos pequeños orientan q/p. Son pautas, no reglas exactas
# para procesos mixtos/multiplicativos; examinar 2s/3s y evitar sobrediferenciar.


# Estimación del modelo SARIMAX con toda la muestra

Candidatos, Fallos, Ajustes = [], [], {}
Rejilla = (product(P_ORDINARIOS,P_ORDINARIOS,P_ESTACIONALES,P_ESTACIONALES,TENDENCIAS)
           if ORDEN_MANUAL is None else [(ORDEN_MANUAL[0][0], ORDEN_MANUAL[0][2],
                                         ORDEN_MANUAL[1][0], ORDEN_MANUAL[1][2], TENDENCIA_MANUAL)])
for p,q,P,Q,tendencia in Rejilla:
    if ORDEN_MANUAL is None and p+q+P+Q > MAX_DINAMICOS:
        continue
    orden, estacional = (p,d,q), (P,D,Q,s)
    try:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            ajuste = SARIMAX(Y, exog=X[['Covid_2020Q1','Covid_2020Q2', 'Covid_2020Q4']], order=orden, seasonal_order=estacional, trend=tendencia,
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
Resultado = Ajustes[int(Mejor.id)]
p,d,q = Resultado.model.order
P,D,Q,s = Resultado.model.seasonal_order
print(Resultado.summary())
(SALIDAS/'09_resumen_modelo.txt').write_text(str(Resultado.summary()), encoding='utf-8')


# Coeficientes y efecto de la pandemia

IC_parametros = Resultado.conf_int(alpha=ALPHA)
Coeficientes = pd.DataFrame({
    'Variable': Resultado.params.index,
    'Coeficiente': Resultado.params.values,
    'Error_estandar': Resultado.bse.values,
    'z': Resultado.zvalues.values,
    'p_valor': Resultado.pvalues.values,
    'LI95': IC_parametros.iloc[:,0].values,
    'LS95': IC_parametros.iloc[:,1].values
})
Coeficientes['H0'] = 'Coeficiente igual a cero'
Coeficientes['Decision'] = np.where(Coeficientes.p_valor.isna(), 'No evaluable',
                          np.where(Coeficientes.p_valor<ALPHA, 'Rechazar H0', 'No rechazar H0'))
Coeficientes.to_csv(SALIDAS/'09_coeficientes.csv', index=False, encoding='utf-8-sig')
Efectos_pandemia = Coeficientes.loc[Coeficientes.Variable.isin(X.columns)].copy()
if USAR_LOG:
    Efectos_pandemia['Desplazamiento_pct'] = 100*np.expm1(Efectos_pandemia.Coeficiente)
    Efectos_pandemia['LI95_pct'] = 100*np.expm1(Efectos_pandemia.LI95)
    Efectos_pandemia['LS95_pct'] = 100*np.expm1(Efectos_pandemia.LS95)
    print('\nEfectos de 2020 (% de desplazamiento de la mediana condicional):')
else:
    print('\nEfectos de 2020 (unidades del PIB):')
print(Efectos_pandemia.to_string(index=False))
Efectos_pandemia.to_csv(SALIDAS/'09_efectos_pandemia.csv', index=False, encoding='utf-8-sig')

# Prueba de significancia conjunta de las variables dummy

# Matriz R: cada fila restringe un coeficiente dummy a cero.

X = X[["Covid_2020Q1","Covid_2020Q2", "Covid_2020Q4"]]

R = np.zeros((X.shape[1], len(Resultado.params)))
for fila, nombre in enumerate(X.columns):
    R[fila, Resultado.params.index.get_loc(nombre)] = 1.0
Wald_pandemia = Resultado.wald_test(R, scalar=True)
Wald = pd.DataFrame([{'Prueba':'Wald conjunto de pandemia',
                     'H0':'Todos los coeficientes de pandemia son cero',
                     'Chi2':float(Wald_pandemia.statistic), 'gl':X.shape[1],
                     'p_valor':float(Wald_pandemia.pvalue)}])
Wald['Decision'] = np.where(Wald.p_valor<ALPHA, 'Rechazar H0', 'No rechazar H0')
Wald.to_csv(SALIDAS/'09_Wald_pandemia.csv', index=False, encoding='utf-8-sig')
print('\n', Wald.to_string(index=False))
print('La asociación de las dummies no identifica por sí sola un efecto causal de la pandemia.')

# Comparación del modelo sin dummy y con dummy

with warnings.catch_warnings(record=True) as avisos_base:
    warnings.simplefilter('always')
    Resultado_sin_dummy = SARIMAX(Y, order=(p,d,q), seasonal_order=(P,D,Q,s),
        trend=Resultado.model.trend, simple_differencing=False,
        enforce_stationarity=True, enforce_invertibility=True,
        use_exact_diffuse=True).fit(disp=False, maxiter=300)
Base_converge = bool(Resultado_sin_dummy.mle_retvals.get('converged', False))
Comparacion_intervencion = pd.DataFrame([
    {'Modelo':'SARIMA sin dummy', 'N':Resultado_sin_dummy.nobs,
     'k':len(Resultado_sin_dummy.params), 'AIC':Resultado_sin_dummy.aic,
     'BIC':Resultado_sin_dummy.bic, 'Converge':Base_converge},
    {'Modelo':f'SARIMAX + pandemia ({MODO_DUMMY})', 'N':Resultado.nobs,
     'k':len(Resultado.params), 'AIC':Resultado.aic,
     'BIC':Resultado.bic, 'Converge':True}
])
Comparacion_intervencion.to_csv(SALIDAS/'09_comparacion_sin_con_dummy.csv',index=False,encoding='utf-8-sig')
print('\nComparación con los mismos órdenes y tendencia:\n',Comparacion_intervencion.to_string(index=False))
if not Base_converge:
    print('El SARIMA de referencia no convergió; sus criterios no sustentan una comparación válida.')

# Visualización del PIB observado y ajustado

# Ajustes de un paso, condicionales a la información pasada.
Inicio_ajuste = max(Resultado.loglikelihood_burn, Resultado.filter_results.nobs_diffuse)
Ajustados = pd.DataFrame(index=Serie.index)
Ajustados['Observado'] = Serie
Ajustados['Ajustado_condicional_mediana'] = (np.exp(Resultado.fittedvalues)
                                           if USAR_LOG else Resultado.fittedvalues)
Ajustados.iloc[:Inicio_ajuste, Ajustados.columns.get_loc('Ajustado_condicional_mediana')] = np.nan
Ajustados.to_csv(SALIDAS/'09_observado_ajustado.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(Ajustados.index.to_timestamp(), Ajustados.Observado, color=GRIS, label='Observado')
ax.plot(Ajustados.index.to_timestamp(), Ajustados.Ajustado_condicional_mediana,
        color=VERDE, linewidth=1.2, alpha=.8, label='Ajustado de un paso (mediana)' if USAR_LOG else 'Ajustado de un paso')
ax.axvspan(PANDEMIA_INICIO.start_time, (PANDEMIA_FIN+1).start_time,
           color=NARANJA,alpha=.15,label='Intervención 2020')
ax.set(title='PIB observado y ajuste SARIMAX', xlabel='Trimestre', ylabel=UNIDAD_PIB)
ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False)
fig.tight_layout()
guardar_figura(fig, '09_observado_ajustado')


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
        dpi = 150
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
        f"SARIMAX{orden} × {estacional}\n"
        "Círculo unitario · Raíces inversas",
        fontsize=15, fontweight="bold", color="#003057"
    )

    guardar_figura(fig, "10_circulo_unitario")

# Innovaciones (residuales) estandarizados

Inicio_residuos = max(Resultado.loglikelihood_burn, Resultado.filter_results.nobs_diffuse)
Residuos = pd.Series(Resultado.filter_results.standardized_forecasts_error[0], index=Y.index)
Residuos = Residuos.iloc[Inicio_residuos:].replace([np.inf,-np.inf],np.nan).dropna()
Residuos_brutos = pd.Series(Resultado.resid, index=Y.index).reindex(Residuos.index)
Tabla_residuos = pd.DataFrame({'Innovacion_escala_estimacion':Residuos_brutos,
                              'Innovacion_estandarizada':Residuos,
                              'Innovacion_cuadrada':Residuos**2})
Tabla_residuos.to_csv(SALIDAS/'10_residuos_completos.csv', encoding='utf-8-sig')
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

# Gráficos de los residuales

fig, ejes = plt.subplots(3,2,figsize=(12,10))
ejes[0,0].plot(Residuos_brutos.index.to_timestamp(),Residuos_brutos,color=VERDE,linewidth=1)
ejes[0,0].set_title('Innovaciones en log(PIB)' if USAR_LOG else 'Innovaciones en unidades del PIB')
ejes[0,1].plot(Residuos.index.to_timestamp(),Residuos,color=VERDE,linewidth=1)
ejes[0,1].set_title('Innovaciones estandarizadas')
for ax in ejes[0]:
    ax.axhline(0,color=GRIS,linewidth=.8)
    ax.axvspan(PANDEMIA_INICIO.start_time,(PANDEMIA_FIN+1).start_time,color=NARANJA,alpha=.12)
max_lags_res = min(3*s,len(Residuos)//2-1)
plot_acf(Residuos,lags=max_lags_res,zero=False,ax=ejes[1,0],color=VERDE)
plot_pacf(Residuos,lags=max_lags_res,zero=False,method='ywm',ax=ejes[1,1],color=VERDE)
ejes[2,0].hist(Residuos,bins='auto',density=True,color=VERDE,alpha=.65)
x = np.linspace(Residuos.min(),Residuos.max(),200)
ejes[2,0].plot(x,stats.norm.pdf(x),color=NARANJA)
ejes[2,0].set_title('Distribución | referencia N(0,1)')
sm.qqplot(Residuos,line='s',ax=ejes[2,1],markerfacecolor=VERDE,markeredgecolor=VERDE)
ejes[2,1].set_title('Q–Q de normalidad')
fig.suptitle(f'Residuos SARIMAX | pandemia: {MODO_DUMMY}',fontweight='bold')
fig.tight_layout()
guardar_figura(fig,'11_diagnosticos_graficos')

# Varianza de los residuales

fig, ejes = plt.subplots(2,1,figsize=(12,6))
ejes[0].plot(Residuos.index.to_timestamp(),Residuos**2,color=NARANJA,linewidth=1.1)
ejes[0].set(title='Innovaciones estandarizadas al cuadrado',ylabel='Innovación²')
ejes[0].axvspan(PANDEMIA_INICIO.start_time,(PANDEMIA_FIN+1).start_time,color=NARANJA,alpha=.12)
plot_acf(Residuos**2,lags=max_lags_res,zero=False,ax=ejes[1],color=VERDE)
ejes[1].set(title='ACF de innovaciones cuadradas | apoyo al diagnóstico ARCH',xlabel='Rezago')
fig.tight_layout()
guardar_figura(fig,'11_varianza_residual')

# Pronóstico a 10 periodos

# Se pronostica desde el último dato; nunca se separó una muestra de prueba.
Futuro = pd.period_range(Serie.index[-1]+1,periods=HORIZONTE,freq=FRECUENCIA)
X_futuro = pd.DataFrame(0.0, index=Futuro, columns=X.columns)
if not X_futuro.columns.equals(X.columns) or len(X_futuro)!=HORIZONTE:
    raise ValueError('Las variables exógenas futuras no coinciden con las de la estimación.')
X_futuro.to_csv(SALIDAS/'12_exog_futuro.csv', encoding='utf-8-sig')
Prediccion = Resultado.get_forecast(steps=HORIZONTE, exog=X_futuro)
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
Pronostico['Escenario'] = 'Sin nueva intervención; dummies futuras iguales a cero'
print('\nPronóstico (escala original):\n',Pronostico.to_string())
Pronostico.to_csv(SALIDAS/'12_pronostico_10_periodos.csv',encoding='utf-8-sig')
fig,ax = plt.subplots(figsize=(12,5))
# Mostrar 2020 y los años siguientes cuando estén disponibles.
Inicio_grafico = min(pd.Period('2018Q1',freq=FRECUENCIA), Serie.index[-min(len(Serie),5*s)])
Historico = Serie.loc[Serie.index>=Inicio_grafico]
ax.plot(Historico.index.to_timestamp(),Historico,color=GRIS,label='Observado')
f = Futuro.to_timestamp()
ax.fill_between(f,Pronostico.LI95,Pronostico.LS95,color=VERDE,alpha=.12,label='Intervalo 95 %')
ax.fill_between(f,Pronostico.LI80,Pronostico.LS80,color=VERDE,alpha=.23,label='Intervalo 80 %')
ax.plot(f,Pronostico.Pronostico_media,color=VERDE,marker='o',label='Pronóstico: media')
ax.axvline(Serie.index[-1].to_timestamp(),color=NARANJA,linestyle='--',alpha=.7)
ax.set(title=f'SARIMAX({p},{d},{q})({P},{D},{Q})[{s}] | {HORIZONTE} periodos'
       + (' | EXPLORATORIO' if Rechazos else ''),
       xlabel='Trimestre',ylabel=UNIDAD_PIB)
ax.axvspan(PANDEMIA_INICIO.start_time, (PANDEMIA_FIN+1).start_time, color=NARANJA, alpha=.10, label='Intervención 2020')
ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
fig.savefig(SALIDAS/'13_pronostico.pdf',bbox_inches='tight')
guardar_figura(fig,'13_pronostico')
print('\nArchivos exportados a:',SALIDAS)


# Se pronostica desde el último dato; nunca se separó una muestra de prueba.
Futuro = pd.period_range(Serie.index[-1]+1,periods=HORIZONTE,freq=FRECUENCIA)
X_futuro = pd.DataFrame(0, index=Futuro, columns=X.columns)
X_futuro.iloc[0,0] = 1
X_futuro.iloc[1,1] = 1
X_futuro.iloc[2,2] = 1

if not X_futuro.columns.equals(X.columns) or len(X_futuro)!=HORIZONTE:
    raise ValueError('Las variables exógenas futuras no coinciden con las de la estimación.')
X_futuro.to_csv(SALIDAS/'14_exog_futuro.csv', encoding='utf-8-sig')
Prediccion = Resultado.get_forecast(steps=HORIZONTE, exog=X_futuro)
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
Pronostico['Escenario'] = 'Sin nueva intervención; dummies futuras iguales a cero'
print('\nPronóstico (escala original):\n',Pronostico.to_string())
Pronostico.to_csv(SALIDAS/'15_pronostico_10_periodos.csv',encoding='utf-8-sig')
fig,ax = plt.subplots(figsize=(12,5))
# Mostrar 2020 y los años siguientes cuando estén disponibles.
Inicio_grafico = min(pd.Period('2018Q1',freq=FRECUENCIA), Serie.index[-min(len(Serie),5*s)])
Historico = Serie.loc[Serie.index>=Inicio_grafico]
ax.plot(Historico.index.to_timestamp(),Historico,color=GRIS,label='Observado')
f = Futuro.to_timestamp()
ax.fill_between(f,Pronostico.LI95,Pronostico.LS95,color=VERDE,alpha=.12,label='Intervalo 95 %')
ax.fill_between(f,Pronostico.LI80,Pronostico.LS80,color=VERDE,alpha=.23,label='Intervalo 80 %')
ax.plot(f,Pronostico.Pronostico_media,color=VERDE,marker='o',label='Pronóstico: media')
ax.axvline(Serie.index[-1].to_timestamp(),color=NARANJA,linestyle='--',alpha=.7)
ax.set(title=f'SARIMAX({p},{d},{q})({P},{D},{Q})[{s}] | {HORIZONTE} periodos'
       + (' | SIMULACIÓN PANDEMIA 26T3-27T1' if Rechazos else ''),
       xlabel='Trimestre',ylabel=UNIDAD_PIB)
ax.axvspan(PANDEMIA_INICIO.start_time, (PANDEMIA_FIN+1).start_time, color=NARANJA, alpha=.10, label='Intervención 2020')
ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False,ncol=2)
fig.tight_layout()
fig.savefig(SALIDAS/'16_pronostico.pdf',bbox_inches='tight')
guardar_figura(fig,'16_pronostico')
print('\nArchivos exportados a:',SALIDAS)
