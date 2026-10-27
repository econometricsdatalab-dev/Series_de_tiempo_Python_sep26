# -*- coding: utf-8 -*-
"""
Curso Integral de econometría
Series de tiempo con Python
Tema: Modelos multivariantes de series de tiempo
Subtema: Modelo VAR y VEC
Sesión: 13
Fecha: 26/10/2026
Docente: Alexis Adonai Morales Alberto
"""

# Modulos a utilizar (completos)

import os
import re
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import requests
import scipy
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm

# Clases o submodulos a cargar

from pathlib import Path
from datetime import datetime, timezone
from getpass import getpass
from itertools import permutations
from scipy.linalg import eigvals
from scipy.stats import jarque_bera
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, coint_johansen
from statsmodels.stats.diagnostic import het_arch, het_breuschpagan, breaks_cusumolsresid
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.tsaplots import plot_acf

# Configuraciones 

matplotlib.use('Agg')

# Configuración general del procedimiento 

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Series de tiempo\ST_SEP26"
    )
TOKEN_INEGI = os.environ.get('INEGI_TOKEN', "af847734-746b-8eb8-f0e6-4070cc851e47").strip()
FUENTE = os.environ.get('S13_FUENTE', globals().get('FUENTE', 'BIE')).upper()
INICIO = globals().get('INICIO', '1993Q1')
FIN = globals().get('FIN', None)
MAX_REZAGOS = int(globals().get('MAX_REZAGOS', 8))
HORIZONTE = 10
SIMULACIONES = 5000
SEMILLA = 13026
ALFA = 0.05
SERIES = {'PIB': '736181', 'Consumo': '738020', 'Inversion': '738065'}
TOPICOS = {'PIB': '600609', 'Consumo': '633186', 'Inversion': '633258'}
NOMBRES = list(SERIES)
VARIABLES_VEC = ['PIB', 'Inversion']  # Sistema reducido exploratorio, motivado por actividad-inversión.
RUTA_DATOS = BASE / 'Datos'
SALIDA = BASE / 'Resultados' / 'S13_VAR_VEC'
for carpeta in [RUTA_DATOS, SALIDA]:
    carpeta.mkdir(parents=True, exist_ok=True)
COLORES = ['#003057', '#08989C', '#F04A1D']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .18, 'figure.dpi': 120, 'savefig.dpi': 220})


# Definiciones para las exportaciones 

def texto(nombre, contenido):
    (SALIDA / (nombre + '.txt')).write_text(str(contenido), encoding='utf-8')

def tabla(nombre, datos, index=True):
    datos.to_csv(SALIDA / (nombre + '.csv'), index=index, encoding='utf-8-sig')

def figura(nombre, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout(rect=(0,.025,1,1))
    fig.text(.01,.005,'Econometrics Data Lab · Fuente: INEGI, BIE · Elaboración de la sesión 13',fontsize=8,color='#4D565E')
    for extension in ['png', 'svg']:
        fig.savefig(SALIDA / f'{nombre}.{extension}', bbox_inches='tight')
    plt.close(fig)

def trimestre(periodo):
    """INEGI YYYY/01..04: el sufijo es trimestre, no mes."""
    m = re.fullmatch(r'(\d{4})[/-](0?[1-4])', str(periodo))
    if not m:
        raise ValueError(f'Periodo trimestral no reconocido: {periodo}')
    return pd.Period(f'{m[1]}Q{int(m[2])}', freq='Q-DEC')

# Definición de consulta mediante BIE
def Series_INEGI_BIE(id_serie, token, periodo='Trimestral'):
    """Adapta bie_inegi.py y API_BIE_INEGI.R; conserva observaciones y metadatos."""
    if periodo != 'Trimestral':
        raise ValueError('Esta sesión requiere frecuencia trimestral.')
    url = ('https://www.inegi.org.mx/app/api/indicadores/desarrolladores/'
           f'jsonxml/INDICATOR/{id_serie}/es/00/false/BIE-BISE/2.0/{token}?type=json')
    try:
        respuesta = requests.get(url, timeout=(10, 45))
        respuesta.raise_for_status()
        datos = respuesta.json()
    except (requests.RequestException, ValueError):
        # Las excepciones de red pueden contener el token en la URL.
        raise RuntimeError('Consulta BIE fallida. Revise conexión y token; no se exporta la URL privada.') from None
    candidatas = [s for s in datos.get('Series', []) if str(s.get('INDICADOR')) == str(id_serie)]
    if len(candidatas) != 1:
        raise ValueError('Respuesta BIE sin una serie inequívoca para la clave solicitada.')
    serie = candidatas[0]
    if str(serie.get('FREQ')) != '4' or str(serie.get('UNIT')) != '1054':
        raise ValueError('La serie no corresponde a frecuencia trimestral y millones de pesos de 2018.')
    observaciones = pd.DataFrame(serie['OBSERVATIONS'])
    observaciones['Trimestre'] = observaciones.TIME_PERIOD.map(trimestre)
    observaciones['Valor'] = pd.to_numeric(observaciones.OBS_VALUE, errors='coerce')
    if observaciones.Trimestre.duplicated().any():
        raise ValueError('Periodos duplicados en la respuesta BIE.')
    valores = observaciones.set_index('Trimestre').Valor.sort_index()
    return valores, datos

# Consultas del BIE INEGI para las series de tiempo 

def cargar_datos():
    ruta = RUTA_DATOS / 'BIE_VAR_VEC.csv'
    if FUENTE == 'CSV':
        if not ruta.exists():
            raise FileNotFoundError(f'No existe {ruta}; use FUENTE=BIE para descargar.')
        datos = pd.read_csv(ruta, dtype={'Trimestre': str})
        datos.index = pd.PeriodIndex(datos.pop('Trimestre'), freq='Q-DEC')
        return datos[NOMBRES].apply(pd.to_numeric, errors='raise'), 'CSV BIE conservado'
    if FUENTE != 'BIE':
        raise ValueError('FUENTE debe ser BIE o CSV.')
    token = os.environ.get('INEGI_TOKEN', '').strip()
    if not token:
        try:
            token = getpass('Token de INEGI (entrada oculta): ').strip()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError('Falta INEGI_TOKEN. Para la descarga incluida use S13_FUENTE=CSV.') from None
    if not token:
        raise RuntimeError('Token vacío. Configure INEGI_TOKEN o use S13_FUENTE=CSV.')
    columnas = {}
    for nombre, clave in SERIES.items():
        valores, bruto = Series_INEGI_BIE(clave, token)
        s = bruto['Series'][0]
        if str(s.get('TOPIC')) != TOPICOS[nombre]:
            raise ValueError(f'Cambió el tópico de {nombre}; revise metadatos antes de estimar.')
        columnas[nombre] = valores
        (RUTA_DATOS / f'BIE_{nombre}_{clave}.json').write_text(json.dumps(bruto, ensure_ascii=False, indent=2), encoding='utf-8')
    datos = pd.concat(columnas, axis=1)
    datos.index.name = 'Trimestre'
    datos.to_csv(ruta, encoding='utf-8-sig')
    return datos, 'API BIE-BISE INEGI'

# Indice trimestral y regularidad 

def depurar(datos):
    tabla('01_faltantes_fuente', datos.isna().sum().to_frame('Faltantes'))
    completos = datos.dropna()
    if completos.empty:
        raise ValueError('No existe ventana común de datos completos.')
    datos = datos.loc[completos.index.min():completos.index.max()]
    datos = datos.reindex(pd.period_range(datos.index.min(), datos.index.max(), freq='Q-DEC'))
    datos = datos.loc[pd.Period(INICIO, freq='Q-DEC'):pd.Period(FIN, freq='Q-DEC') if FIN else None]
    datos.index.name = 'Trimestre'
    if datos.index.duplicated().any() or datos.isna().any().any():
        raise ValueError('Hay duplicados, huecos o faltantes interiores. No se interpola ni se comprime el calendario.')
    if len(datos) < 80 or not np.isfinite(datos.to_numpy()).all() or (datos <= 0).any().any():
        raise ValueError('Se requieren al menos 80 trimestres, todos positivos y finitos.')
    return datos

# Pruebas de raíz unitaria (ADF y KPSS)

def integracion(logs):
    registros = []
    for nombre in logs:
        for transformacion, x, tendencia in [('Nivel_ct', logs[nombre], 'ct'), ('Nivel_c', logs[nombre], 'c'),
                                             ('Diferencia_c', logs[nombre].diff().dropna(), 'c')]:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter('always')
                adf = adfuller(x, regression=tendencia, autolag='BIC')
                kp = kpss(x, regression=tendencia, nlags='auto')
            registros.append([nombre, transformacion, adf[0], adf[1], adf[2], kp[0], kp[1], kp[2],
                              ' | '.join(str(a.message) for a in avisos)])
    resultados = pd.DataFrame(registros, columns=['Variable','Transformacion','ADF','p_ADF','Lags_ADF',
                                                  'KPSS','p_KPSS','Lags_KPSS','Advertencia'])
    tabla('03_integracion', resultados, False)
    # Filtro prudente; no prueba mecánica de I(1). Discordancias requieren revisión.
    compatible = {}
    for nombre in logs:
        n = resultados.query('Variable == @nombre and Transformacion == "Nivel_ct"').iloc[0]
        d = resultados.query('Variable == @nombre and Transformacion == "Diferencia_c"').iloc[0]
        compatible[nombre] = bool(n.p_ADF > ALFA and d.p_ADF < ALFA and d.p_KPSS >= ALFA)
    return resultados, compatible'


# Validación del modelo

def diagnosticos(ajuste, etiqueta, indice, parametros, nombres=None):
    nombres = nombres or NOMBRES
    residuos = pd.DataFrame(np.asarray(ajuste.resid), columns=nombres, index=indice[-len(ajuste.resid):])
    tabla(etiqueta + '_residuales', residuos)
    tabla(etiqueta + '_covarianza', pd.DataFrame(ajuste.sigma_u, index=nombres, columns=nombres))
    registros = []
    orden = ajuste.k_ar
    for lag in sorted(set([max(8, orden + 1), max(12, orden + 1)])):
        prueba = ajuste.test_whiteness(nlags=lag, adjusted=True)
        registros.append(['Portmanteau multivariado', 'Sistema', lag, prueba.test_statistic, prueba.pvalue])
    normal = ajuste.test_normality()
    registros.append(['Normalidad multivariada', 'Sistema', 0, normal.test_statistic, normal.pvalue])
    for nombre in nombres:
        u = residuos[nombre].to_numpy()
        for lag in [4, 8]:
            arch = het_arch(u, nlags=lag, ddof=parametros)
            registros.append(['ARCH-LM marginal', nombre, lag, arch[0], arch[1]])
        jb = jarque_bera(u)
        registros.append(['Jarque-Bera marginal', nombre, 0, jb.statistic, jb.pvalue])
        # BP: heterocedasticidad asociada al tiempo y al valor ajustado.
        ajustados = np.asarray(ajuste.fittedvalues)[:, nombres.index(nombre)]
        z = sm.add_constant(np.column_stack([np.linspace(-1, 1, len(u)), ajustados]))
        bp = het_breuschpagan(u, z, robust=True)
        registros.append(['Breusch-Pagan Koenker marginal', nombre, 0, bp[0], bp[1]])
        cs = breaks_cusumolsresid(u, ddof=parametros)
        registros.append(['CUSUM marginal exploratorio', nombre, 0, cs[0], cs[1]])
    reporte = pd.DataFrame(registros, columns=['Prueba','Variable','Rezagos','Estadistico','p_valor'])
    reporte['p_Holm_familia'] = reporte.groupby('Prueba').p_valor.transform(lambda p: multipletests(p, method='holm')[1])
    reporte['Rechaza_5pct'] = reporte.p_valor < ALFA
    tabla(etiqueta + '_validacion', reporte, False)
    fig, axs = plt.subplots(len(nombres), 3, figsize=(14, 9))
    for j, nombre in enumerate(nombres):
        u = residuos[nombre]
        axs[j,0].plot(u.index.to_timestamp(), u, color=COLORES[j]); axs[j,0].set_title(f'Residuales · {nombre}')
        plot_acf(u, lags=16, ax=axs[j,1], title='ACF residual')
        sm.qqplot(u, line='s', ax=axs[j,2]); axs[j,2].set_title('Q-Q normal: referencia')
    figura(etiqueta + '_residuales', fig)
    advertencia = 'Exploratorio: revise los rechazos y los supuestos antes de inferencia económica.' if reporte.Rechaza_5pct.any() else 'Sin rechazos en diagnósticos considerados; no demuestra especificación correcta.'
    texto(etiqueta + '_estado', advertencia + '\nARCH y BP son marginales, no un test ARCH multivariado. CUSUM es orientativo con regresores dinámicos.\nLos p-valores convencionales pueden distorsionarse ante rupturas o heterocedasticidad.')
    return reporte


# Círculo unitario y estabilidad dinámica

def estabilidad(coefs, etiqueta, raices_unitarias_esperadas=0):
    p, k, _ = coefs.shape
    companera = np.zeros((k*p, k*p))
    companera[:k] = np.concatenate(coefs, axis=1)
    if p > 1:
        companera[k:,:-k] = np.eye(k*(p-1))
    valores = eigvals(companera)
    resultado = pd.DataFrame({'Real':valores.real, 'Imaginaria':valores.imag, 'Modulo':np.abs(valores)})
    resultado['Unitaria_1'] = np.abs(valores-1) < 1e-6
    tabla(etiqueta + '_autovalores', resultado, False)
    estacionarios = np.abs(valores[np.abs(valores-1) >= 1e-6])
    correcto = int(resultado.Unitaria_1.sum()) == raices_unitarias_esperadas and np.all(estacionarios < 1-1e-7)
    fig, ax = plt.subplots(figsize=(5,5)); angulo = np.linspace(0,2*np.pi,400)
    ax.plot(np.cos(angulo), np.sin(angulo), color='#4D565E')
    ax.scatter(valores.real, valores.imag, color=COLORES[2]); ax.set_aspect('equal'); ax.set_title(etiqueta.replace('_',' ') + ' · Autovalores de la compañera',fontsize=11)
    ax.set_xlabel('Parte real'); ax.set_ylabel('Parte imaginaria'); figura(etiqueta + '_estabilidad', fig)
    texto(etiqueta + '_estabilidad', f'Estructura compatible: {correcto}\nRaíces +1 esperadas: {raices_unitarias_esperadas}.\nSe representan autovalores, inversos de las raíces del polinomio VAR.')
    return bool(correcto)

# Funciones impulso-respuesta y descomposición de varianza

def dinamica(coefs, sigma, etiqueta, escala=1, orden=None, nombres=None):
    """FEVD ortogonal finita. No implica causalidad estructural sin identificar."""
    nombres = nombres or NOMBRES
    orden = orden or nombres
    indices = [nombres.index(n) for n in orden]
    A = coefs[:,indices][:,:,indices]
    S = np.asarray(sigma)[np.ix_(indices,indices)]
    k = len(orden); psi = np.zeros((HORIZONTE+1,k,k)); psi[0] = np.eye(k)
    for h in range(1,HORIZONTE+1):
        for lag in range(1,min(len(A),h)+1):
            psi[h] += A[lag-1] @ psi[h-lag]
    P = np.linalg.cholesky(S)
    ort = psi @ P
    girf = (psi @ S) / np.sqrt(np.diag(S))[None,None,:]
    acumulada = np.cumsum(ort,axis=0)
    registros = []; fracciones=[]
    for h in range(HORIZONTE+1):
        for i,n in enumerate(orden):
            for j,choque in enumerate(orden):
                registros.append([h,n,choque,escala*ort[h,i,j],escala*girf[h,i,j],escala*acumulada[h,i,j]])
    for h in range(1,HORIZONTE+1):
        aportes = (ort[:h]**2).sum(axis=0)
        descomp = aportes/aportes.sum(axis=1,keepdims=True)
        for i,n in enumerate(orden):
            for j,choque in enumerate(orden):fracciones.append([h,n,choque,descomp[i,j]])
    tabla(etiqueta+'_IRF',pd.DataFrame(registros,columns=['h','Respuesta','Choque','IRF_ortogonal','IRF_generalizada','IRF_ort_acumulada']),False)
    tabla(etiqueta+'_FEVD',pd.DataFrame(fracciones,columns=['h','Respuesta','Choque','Proporcion']),False)
    fig,axs=plt.subplots(k,k,figsize=(12,8))
    for i in range(k):
        for j in range(k):
            axs[i,j].plot(range(HORIZONTE+1),escala*ort[:,i,j],color=COLORES[i]);axs[i,j].axhline(0,color='gray',lw=.7)
            axs[i,j].set_title(f'{orden[j]} → {orden[i]}',fontsize=10)
    fig.suptitle(etiqueta.replace('_',' ')+' · choque de una desviación estándar; sin bandas de confianza',y=1.02)
    figura(etiqueta+'_IRF',fig)
    fig,axs=plt.subplots(1,k,figsize=(13,4))
    f=pd.DataFrame(fracciones,columns=['h','Respuesta','Choque','Proporcion'])
    for i,n in enumerate(orden):
        v=f.query('Respuesta == @n').pivot(index='h',columns='Choque',values='Proporcion').reindex(columns=orden)
        axs[i].stackplot(v.index,*[100*v[c] for c in orden],labels=orden,colors=COLORES)
        axs[i].set_title(n);axs[i].set_ylim(0,100);axs[i].set_xlabel('Horizonte trimestral')
    axs[0].legend(fontsize=8);figura(etiqueta+'_FEVD',fig)
    return psi

# Pronóstico a 10 periodos

def grafico_pronostico(historia, media, inferior, superior, etiqueta, unidad):
    nombres = list(historia.columns)
    fig,axs=plt.subplots(len(nombres),1,figsize=(11,4*len(nombres)))
    fechas=pd.period_range(historia.index[-1]+1,periods=HORIZONTE,freq='Q-DEC')
    registros=[]
    for j,nombre in enumerate(nombres):
        ax=axs[j];ax.plot(historia.index[-24:].to_timestamp(),historia[nombre].iloc[-24:],color=COLORES[j])
        ax.plot(fechas.to_timestamp(),media[:,j],color=COLORES[2],label='Pronóstico puntual')
        ax.fill_between(fechas.to_timestamp(),inferior[:,j],superior[:,j],color=COLORES[2],alpha=.16,label='Intervalo marginal 95 %')
        ax.set_title(nombre.replace('Inversion','Inversión')+' · '+unidad)
        if 'Millones' in unidad:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'.replace(',', ' ')))
        ax.axvline(historia.index[-1].start_time,color='gray',ls='--');ax.legend(fontsize=8)
        for h,f in enumerate(fechas):registros.append([str(f),nombre,media[h,j],inferior[h,j],superior[h,j]])
    fig.suptitle(etiqueta.replace('_',' ')+'\nProyección exploratoria; parámetros tratados como conocidos',y=1.01)
    figura(etiqueta,fig)
    tabla(etiqueta,pd.DataFrame(registros,columns=['Trimestre','Variable','Puntual','LI95','LS95']),False)

# Exploración de datos y transformaciones

def ejecutar():
    global Datos, Logaritmos, Crecimientos, Pruebas_integracion, Modelo_VAR, Modelo_VEC
    # Archivar resultados anteriores para evitar mezclar ejecuciones y rangos.
    existentes=[p for p in SALIDA.iterdir() if p.is_file() and re.match(r'^\d{2}_',p.name)]
    if existentes:
        archivo=SALIDA/'Historial'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archivo.mkdir(parents=True,exist_ok=True)
        for p in existentes:p.rename(archivo/p.name)
    Datos, origen = cargar_datos(); Datos = depurar(Datos)
    Logaritmos=np.log(Datos);Crecimientos=100*Logaritmos.diff().dropna()
    tabla('01_datos_niveles',Datos);tabla('01_logaritmos',Logaritmos);tabla('01_crecimientos',Crecimientos)
    tabla('02_descriptivos_niveles',Datos.describe().T);tabla('02_descriptivos_crecimientos',Crecimientos.describe().T)
    tabla('02_correlaciones_crecimiento',Crecimientos.corr())
    fig,axs=plt.subplots(3,2,figsize=(13,10))
    for j,nombre in enumerate(NOMBRES):
        for ax,serie,titulo in [(axs[j,0],100*Datos[nombre]/Datos[nombre].iloc[0],'Índice primer trimestre = 100'),(axs[j,1],Crecimientos[nombre],'100 Δ log; aproximación porcentual trimestral')]:
            ax.plot(serie.index.to_timestamp(),serie,color=COLORES[j]);ax.set_title(nombre+' · '+titulo,fontsize=10)
            ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2020-12-31'),color='gray',alpha=.12)
    figura('02_exploracion',fig)
    Pruebas_integracion, compatibles=integracion(Logaritmos)
    texto('03_decision_integracion',json.dumps(compatibles,ensure_ascii=False,indent=2)+'\nCompatible significa evidencia orientativa, no demostración de I(1). No incluye raíces estacionales ni pruebas de rupturas.')

    # Identificación de los órdenes del VAR en crecimientos
    
    maxlags=min(MAX_REZAGOS,(len(Crecimientos)-12)//(len(NOMBRES)+1))
    Seleccion_VAR=VAR(Crecimientos).select_order(maxlags=maxlags,trend='c')
    texto('04_VAR_criterios',Seleccion_VAR.summary())
    tabla('04_VAR_criterios',pd.DataFrame(Seleccion_VAR.ics).rename_axis('Rezago'))
    p_var=int(Seleccion_VAR.selected_orders['bic'])
    # VAR(0) es admisible: no se inventan rezagos para garantizar dinámica.
    Modelo_VAR=VAR(Crecimientos).fit(p_var,trend='c')
    texto('05_VAR_resumen',Modelo_VAR.summary());tabla('05_VAR_coeficientes',Modelo_VAR.params)
    tabla('05_VAR_p_valores',Modelo_VAR.pvalues)
    # Derivación MCO verificada con el diseño realmente usado por statsmodels.
    X=Modelo_VAR.endog_lagged;Y=np.asarray(Crecimientos)[p_var:]
    B_manual=np.linalg.lstsq(X,Y,rcond=None)[0]
    error_mco=float(np.max(np.abs(B_manual-np.asarray(Modelo_VAR.params))))
    texto('05_VAR_MCO_manual',f'Máxima diferencia frente a statsmodels: {error_mco:.3e}')
    diagnosticos(Modelo_VAR,'06_VAR',Crecimientos.index,1+3*p_var)
    estable_var=estabilidad(Modelo_VAR.coefs,'06_VAR') if p_var else True
    causales=[]
    if p_var:
        for causa,efecto in permutations(NOMBRES,2):
            prueba=Modelo_VAR.test_causality(efecto,[causa],kind='f')
            causales.append([causa,efecto,prueba.test_statistic,prueba.pvalue])
    gr=pd.DataFrame(causales,columns=['Causa_predictiva','Efecto','F','p_valor'])
    if len(gr):gr['p_Holm']=multipletests(gr.p_valor,method='holm')[1]
    tabla('07_VAR_Granger',gr,False)
    if p_var:
        dinamica(Modelo_VAR.coefs,Modelo_VAR.sigma_u,'08_VAR')
        dinamica(Modelo_VAR.coefs,Modelo_VAR.sigma_u,'08_VAR_orden_inverso',orden=NOMBRES[::-1])
    else:
        dinamica(np.zeros((1,3,3)),Modelo_VAR.sigma_u,'08_VAR')
    inicial=np.asarray(Crecimientos)[-p_var:] if p_var else np.empty((0,3))
    pr,li,ls=Modelo_VAR.forecast_interval(inicial,steps=HORIZONTE,alpha=ALFA)
    grafico_pronostico(Crecimientos,pr,li,ls,'09_VAR_pronostico_crecimientos','100 Δ log')
    # Trayectorias conjuntas para integrar crecimientos; evita sumar límites marginales.
    rng=np.random.default_rng(SEMILLA)
    historia=np.repeat(inicial[None,:,:],SIMULACIONES,axis=0)
    tray=np.zeros((SIMULACIONES,HORIZONTE,3))
    for h in range(HORIZONTE):
        m=np.repeat(np.asarray(Modelo_VAR.intercept)[None,:],SIMULACIONES,axis=0)
        for lag in range(p_var):m+=historia[:,-lag-1,:] @ Modelo_VAR.coefs[lag].T
        tray[:,h]=m+rng.multivariate_normal(np.zeros(3),Modelo_VAR.sigma_u,SIMULACIONES)
        historia=np.concatenate([historia,tray[:,h:h+1,:]],axis=1)
    logtray=np.asarray(Logaritmos.iloc[-1])[None,None,:]+np.cumsum(tray/100,axis=1)
    niveles=np.exp(logtray)
    mediana=np.exp(np.asarray(Logaritmos.iloc[-1])+np.cumsum(pr/100,axis=0))
    grafico_pronostico(Datos,mediana,np.quantile(niveles,.025,axis=0),np.quantile(niveles,.975,axis=0),'09_VAR_pronostico_niveles','Millones de pesos de 2018; mediana condicional')

    # Evidencia del sistema completo: no se omite su ausencia de cointegración.
    p_tres=max(1,int(VAR(Logaritmos).select_order(maxlags=MAX_REZAGOS).selected_orders['bic']))
    auditoria=[]
    for cols in [NOMBRES, ['PIB','Consumo'], VARIABLES_VEC]:
        p_a=max(1,int(VAR(Logaritmos[cols]).select_order(maxlags=MAX_REZAGOS).selected_orders['bic']))
        for det in [-1,0,1]:
            for metodo in ['trace','maxeig']:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ra=select_coint_rank(Logaritmos[cols],det,p_a-1,method=metodo).rank
                auditoria.append([' + '.join(cols),p_a,det,metodo,ra])
    tabla('10_auditoria_sistemas_explorados',pd.DataFrame(auditoria,columns=['Sistema','p_niveles','det_order','Metodo','Rango']),False)
    texto('10_seleccion_sistema','El sistema triple no presenta cointegración con constante base. Se estudia PIB-Inversion por su hipótesis actividad-inversión. La reducción se decidió tras explorar los datos: evidencia exploratoria, no confirmatoria. Se conservan todos los sistemas y determinísticos examinados. No se cambia a det_order=1 para obtener un rechazo.')
    Niveles_VEC=Logaritmos[VARIABLES_VEC]
    K_VEC=len(VARIABLES_VEC)

    # Identificación de rezagos y prueba de cointegración de Johansen
    # Caso III: constante no restringida en VEC (co), det_order=0 en Johansen.
    # La muestra para seleccionar p es común a todos los candidatos.
    maxp=min(MAX_REZAGOS,(len(Niveles_VEC)-12)//4)
    seleccion=VAR(Niveles_VEC).select_order(maxlags=maxp,trend='c')
    texto('10_VEC_criterios_VAR_niveles',seleccion.summary())
    # Para VEC I(1) se requiere p >= 1. Se minimiza BIC entre órdenes admisibles.
    ics=pd.DataFrame(seleccion.ics).rename_axis('p_VAR_niveles')
    tabla('10_VEC_criterios_VAR_niveles',ics)
    p_vec=int(ics.loc[ics.index>=1,'bic'].idxmin());k_diff=p_vec-1
    sensibilidad=[];rangos={}
    for etiqueta,muestra in [('Completa',Niveles_VEC),('Hasta_2019Q4',Niveles_VEC.loc[:'2019Q4'])]:
        if len(muestra)<50:continue
        for k in sorted(set([max(0,k_diff-1),k_diff,k_diff+1])):
            for metodo in ['trace','maxeig']:
                resultado=select_coint_rank(muestra,det_order=0,k_ar_diff=k,method=metodo,signif=ALFA)
                sensibilidad.append([etiqueta,k+1,k,metodo,resultado.rank])
                if etiqueta=='Completa' and k==k_diff:rangos[metodo]=int(resultado.rank)
        joh=coint_johansen(muestra,0,k_diff)
        tabla('10_Johansen_'+etiqueta,pd.DataFrame({'H0_rango_max':range(K_VEC),'Traza':joh.lr1,'Critico_traza_95':joh.cvt[:,1],
                'Max_autovalor':joh.lr2,'Critico_max_95':joh.cvm[:,1]}),False)
    tabla('10_Johansen_sensibilidad',pd.DataFrame(sensibilidad,columns=['Muestra','p_niveles','k_diferencias','Metodo','Rango']),False)
    rango=rangos['trace'];Modelo_VEC=None
    motivos=[]
    if not all(compatibles[n] for n in VARIABLES_VEC):motivos.append('Evidencia de integración discordante: revisar antes de VEC.')
    if rango==0:motivos.append('Johansen no encuentra cointegración: mantener VAR en diferencias.')
    if rango==K_VEC:motivos.append('Rango completo: revisar estacionariedad y determinísticos; no imponer VEC reducido.')
    if motivos:
        texto('11_VEC_decision','\n'.join(motivos)+'\nSe exportaron pruebas; no se fuerza rango 1.')
    else:
        # Estimación del modelo VEC mediante máxima verosimilitud de rango reducido
        Modelo_VEC=VECM(Niveles_VEC,k_ar_diff=k_diff,coint_rank=rango,deterministic='co').fit()
        texto('11_VEC_resumen',Modelo_VEC.summary())
        columnas=[f'EC{j+1}' for j in range(rango)]
        tabla('11_VEC_beta',pd.DataFrame(Modelo_VEC.beta,index=VARIABLES_VEC,columns=columnas))
        tabla('11_VEC_alpha',pd.DataFrame(Modelo_VEC.alpha,index=VARIABLES_VEC,columns=columnas))
        tabla('11_VEC_alpha_p',pd.DataFrame(Modelo_VEC.pvalues_alpha,index=VARIABLES_VEC,columns=columnas))
        tabla('11_VEC_constante',pd.DataFrame(Modelo_VEC.det_coef,index=VARIABLES_VEC))
        tabla('11_VEC_gamma',pd.DataFrame(Modelo_VEC.gamma,index=VARIABLES_VEC))
        tabla('11_VEC_Pi',pd.DataFrame(Modelo_VEC.alpha @ Modelo_VEC.beta.T,index=VARIABLES_VEC,columns=VARIABLES_VEC))
        ecu=Niveles_VEC.to_numpy() @ Modelo_VEC.beta
        tabla('11_VEC_equilibrios',pd.DataFrame(ecu,index=Niveles_VEC.index,columns=columnas))
        tabla('11_VEC_equilibrios_centrados',pd.DataFrame(ecu-ecu.mean(axis=0),index=Niveles_VEC.index,columns=columnas))
        tabla('11_VEC_media_combinaciones',pd.DataFrame({'Media_muestral':ecu.mean(axis=0)},index=columnas))
        fig,axs=plt.subplots(rango,1,figsize=(11,3*rango),squeeze=False)
        for j in range(rango):axs[j,0].plot(Niveles_VEC.index.to_timestamp(),ecu[:,j]-ecu[:,j].mean(),color=COLORES[j]);axs[j,0].axhline(0,color='gray',lw=.7);axs[j,0].set_title(f'β{j+1}′ log(Y) menos su media muestral: centrado descriptivo')
        figura('11_VEC_equilibrios',fig)
        diagnosticos(Modelo_VEC,'12_VEC',Niveles_VEC.index,K_VEC*k_diff+rango+1,nombres=VARIABLES_VEC)
        estable_vec=estabilidad(Modelo_VEC.var_rep,'12_VEC',K_VEC-rango)
        registros=[]
        for causa,efecto in permutations(VARIABLES_VEC,2):
            prueba=Modelo_VEC.test_granger_causality(caused=efecto,causing=[causa])
            registros.append([causa,efecto,prueba.test_statistic,prueba.pvalue])
        gr=pd.DataFrame(registros,columns=['Causa_predictiva','Efecto','F','p_valor']);gr['p_Holm']=multipletests(gr.p_valor,method='holm')[1]
        tabla('13_VEC_Granger',gr,False)
        dinamica(Modelo_VEC.var_rep,Modelo_VEC.sigma_u,'14_VEC',escala=100,nombres=VARIABLES_VEC)
        dinamica(Modelo_VEC.var_rep,Modelo_VEC.sigma_u,'14_VEC_orden_inverso',escala=100,orden=VARIABLES_VEC[::-1],nombres=VARIABLES_VEC)
        # VEC está en log-niveles: su IRF directa *100 es cambio porcentual aproximado.
        # NO acumular otra vez su IRF para interpretar el nivel.
        m,l,u=Modelo_VEC.predict(steps=HORIZONTE,alpha=ALFA)
        grafico_pronostico(Datos[VARIABLES_VEC],np.exp(m),np.exp(l),np.exp(u),'15_VEC_pronostico_niveles','Millones de pesos de 2018; mediana condicional')
        texto('11_VEC_decision',f'Rango traza={rango}; rango máximo autovalor={rangos["maxeig"]}; p={p_vec}; k_ar_diff={k_diff}; constante co.\nDivergencias de rango requieren sensibilidad; el rango elegido no demuestra una relación estructural.\nEstabilidad de componentes transitorios compatible: {estable_vec}')

    # Exportación de resultados y trazabilidad
    manifiesto={'origen':origen,'generado_UTC':datetime.now(timezone.utc).isoformat(),'series':SERIES,
        'inicio':str(Datos.index[0]),'fin':str(Datos.index[-1]),'n_trimestres':len(Datos),'p_VAR_crecimientos':p_var,
        'p_VAR_niveles_VEC':p_vec,'k_ar_diff':k_diff,'rango_traza':rango,'rango_maxeig':rangos['maxeig'],
        'VEC_estimado':Modelo_VEC is not None,'variables_VEC':VARIABLES_VEC,'p_sistema_triple':p_tres,'compatibles_I1':compatibles,'VAR_estable':estable_var,
        'horizonte':HORIZONTE,'simulaciones':SIMULACIONES,'semilla':SEMILLA,
        'sin_division_entrenamiento_prueba':True,'orden_Cholesky':NOMBRES,
        'versiones':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        'sha256_csv_fuente':hashlib.sha256((RUTA_DATOS/'BIE_VAR_VEC.csv').read_bytes()).hexdigest()}
    (SALIDA/'00_ejecucion.json').write_text(json.dumps(manifiesto,ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(manifiesto,ensure_ascii=False,indent=2))
    print('Resultados exportados en:',SALIDA)
    return manifiesto

if __name__ == '__main__':
    Ejecucion=ejecutar()
