# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Modelos de heterocedasticidad condicional
Subtema: Modelos ARCH, GARCH y algunas extensiones pt1
Sesión: 10
Fecha: 19/10/2026
Docente: Alexis Adonai Morales Alberto
"""

pip install yfinance

# Modulos a utilizar 

import os
import json
import warnings
import contextlib
import io
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import arch.data.sp500

# Clases directas 

from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy import stats, optimize, signal, special
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.numdiff import approx_hess3
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from arch.univariate import GARCH

# Configuración del directorio de trabajo 

BASE = Path(
    "E:\Econometrics Data Lab\Curso integral de econometría\Series de tiempo\ST_SEP26"
    )
SALIDAS = BASE / 'Resultados' / 'S10_Volatilidad'
SALIDAS.mkdir(parents=True, exist_ok=True)

# Parámetros para los modelos 

FUENTE = 'YAHOO'                 
TICKER = '^GSPC'
INICIO = '1999-01-01'
FIN = (datetime.now(timezone.utc).date()+timedelta(days=1)).isoformat()   
PERMITIR_ARCHIVO_YAHOO = False    
ARCHIVO = '00_precios_yahoo.csv' 
ALPHA = .05
INICIALIZACION = 20              
HORIZONTE = 10                  
SIMULACIONES = 20000
SEMILLA = 2026
ANUALIZACION = 252
MOSTRAR = True
DPI = 500
INCLUIR_FIGARCH = False         
VERDE, GRIS, NARANJA, AZUL = '#08989C', '#4D565E', '#F04A1D', '#003057'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':GRIS,'xtick.color':GRIS,'ytick.color':GRIS,
                     'axes.titlecolor':AZUL,'axes.titleweight':'bold','figure.facecolor':'white'})


# Definiciones para la exportación y los diagnósticos repetidos

def guardar_figura(fig, nombre):
    fig.savefig(SALIDAS/f'{nombre}.png',dpi=DPI,bbox_inches='tight')
    fig.savefig(SALIDAS/f'{nombre}.svg',bbox_inches='tight')
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def diagnosticos(nombre, eps, sigma, nu=np.inf, df_media=0):
    """Misma batería para cada ajuste; z se alinea antes de crear sus rezagos."""
    tabla = pd.concat([pd.Series(eps).rename('eps'),pd.Series(sigma).rename('sigma')],axis=1).dropna()
    tabla = tabla.loc[np.isfinite(tabla).all(axis=1) & (tabla.sigma>0)]
    z = tabla.eps/tabla.sigma
    filas = []
    for serie, etiqueta, gl_ajuste in [(z,'Ljung-Box z',df_media),(z*z,'Ljung-Box z²',0)]:
        lb=acorr_ljungbox(serie,lags=[5,10,20],model_df=gl_ajuste,return_df=True)
        for h, f in lb.iterrows():
            filas.append({'Modelo':nombre,'Prueba':f'{etiqueta} ({h})',
                          'H0':'Sin autocorrelación hasta el rezago indicado',
                          'Estadistico':f.lb_stat,'p_valor':f.lb_pvalue,'Evaluacion':'Diagnóstico'})
    for h in [5,10,20]:
        lm,pv,_,_=het_arch(z,nlags=h,ddof=df_media)
        filas.append({'Modelo':nombre,'Prueba':f'ARCH-LM remanente ({h})',
                      'H0':'Sin efectos ARCH remanentes','Estadistico':lm,'p_valor':pv,'Evaluacion':'Diagnóstico'})
    jb=stats.jarque_bera(z)
    filas.append({'Modelo':nombre,'Prueba':'Jarque-Bera','H0':'Normalidad',
                  'Estadistico':jb.statistic,'p_valor':jb.pvalue,
                  'Evaluacion':'Diagnóstico' if np.isinf(nu) else 'Referencia normal; modelo t'})
    previo=tabla.eps.shift(1)
    neg=(previo<0).astype(float)
    aux=pd.DataFrame({'z2':z*z,'Negativo_previo':neg,
                      'Tamano_negativo':neg*previo,'Tamano_positivo':(1-neg)*previo}).dropna()
    regresion=sm.OLS(aux.z2,sm.add_constant(aux.drop(columns='z2'))).fit(cov_type='HC3')
    w=regresion.wald_test(np.column_stack([np.zeros(3),np.eye(3)]),scalar=True)
    filas.append({'Modelo':nombre,'Prueba':'Sesgo de signo/tamaño (Wald robusto)',
                  'H0':'Sin asimetría residual en las tres variables',
                  'Estadistico':float(w.statistic),'p_valor':float(w.pvalue),'Evaluacion':'Diagnóstico aproximado'})
    d=pd.DataFrame(filas)
    d['Decision']=np.where(d.p_valor.isna(),'No evaluable',np.where(d.p_valor<ALPHA,'Rechazar H0','No rechazar H0'))
    return d,z


def grafico_residuos(z, nu, nombre, archivo):
    fig,ejes=plt.subplots(3,2,figsize=(12,10))
    ejes[0,0].plot(z.index,z,color=VERDE,linewidth=.65)
    ejes[0,0].axhline(0,color=GRIS,linewidth=.8)
    ejes[0,0].set_title('Innovaciones estandarizadas')
    plot_acf(z,lags=30,zero=False,ax=ejes[0,1],color=VERDE)
    ejes[0,1].set_title('ACF de z')
    plot_acf(z*z,lags=30,zero=False,ax=ejes[1,0],color=VERDE)
    ejes[1,0].set_title('ACF de z² | varianza remanente')
    probs=(np.arange(1,len(z)+1)-.5)/len(z)
    if np.isfinite(nu):
        teorico=stats.t.ppf(probs,nu)*np.sqrt((nu-2)/nu)
        pit=stats.t.cdf(z*np.sqrt(nu/(nu-2)),nu)
    else:
        teorico=stats.norm.ppf(probs);pit=stats.norm.cdf(z)
    ejes[1,1].scatter(teorico,np.sort(z),s=7,color=VERDE,alpha=.45)
    lim=[min(teorico.min(),z.min()),max(teorico.max(),z.max())]
    ejes[1,1].plot(lim,lim,color=NARANJA,linewidth=1)
    ejes[1,1].set(title='Q–Q con distribución ajustada',xlabel='Cuantil teórico',ylabel='Cuantil observado')
    ejes[2,0].hist(z,bins=60,density=True,color=VERDE,alpha=.6)
    x=np.linspace(z.min(),z.max(),500)
    dens=stats.t.pdf(x*np.sqrt(nu/(nu-2)),nu)*np.sqrt(nu/(nu-2)) if np.isfinite(nu) else stats.norm.pdf(x)
    ejes[2,0].plot(x,dens,color=NARANJA,label='Densidad ajustada')
    ejes[2,0].set_title('Distribución de z');ejes[2,0].legend(frameon=False)
    ejes[2,1].hist(pit,bins=20,density=True,color=VERDE,alpha=.6,range=(0,1))
    ejes[2,1].axhline(1,color=NARANJA)
    ejes[2,1].set(title='PIT | referencia uniforme',xlabel='Probabilidad acumulada')
    fig.suptitle(nombre,color=AZUL,fontweight='bold');fig.tight_layout()
    guardar_figura(fig,archivo)


# Descarga de precios desde Yahoo Finance

Error_descarga=''
if FUENTE=='YAHOO':
    try:
        # yfinance puede emitir mensajes adicionales: conservarlos en el registro.
        registro=io.StringIO()
        with contextlib.redirect_stderr(registro):
            Descarga=yf.download(TICKER,start=INICIO,end=FIN,auto_adjust=True,
                                 progress=False,threads=False,timeout=15)
        if Descarga.empty:
            raise RuntimeError('Yahoo devolvió una tabla vacía. '+registro.getvalue()[:300])
        cierre=Descarga['Close']
        if isinstance(cierre,pd.DataFrame):
            cierre=cierre[TICKER] if TICKER in cierre.columns else cierre.iloc[:,0]
        Precios=cierre.rename('Precio')
        Origen='Yahoo Finance en línea; cierre con auto_adjust=True'
    except Exception as error:
        Error_descarga=f'{type(error).__name__}: {str(error)[:400]}'
        if not PERMITIR_ARCHIVO_YAHOO or TICKER!='^GSPC':
            raise RuntimeError('No fue posible descargar Yahoo. Configura CSV o revisa la conexión. '+Error_descarga) from None
        print('Yahoo en línea no disponible. Se utilizará el histórico REAL Yahoo 1999-2018 incluido en arch.')
        FUENTE='YAHOO_ARCHIVADO'
if FUENTE=='YAHOO_ARCHIVADO':
    if TICKER!='^GSPC':
        raise ValueError('El respaldo de arch es únicamente el S&P 500 (^GSPC).')
    Historico_Yahoo=arch.data.sp500.load()
    Precios=Historico_Yahoo['Adj Close'].rename('Precio')
    Origen='Yahoo Finance; histórico REAL archivado en arch.data.sp500 (1999-2018)'
elif FUENTE=='CSV':
    ruta=Path(ARCHIVO).expanduser()
    if not ruta.is_absolute(): ruta=BASE/ruta
    Datos_CSV=pd.read_csv(ruta)
    fechas=pd.to_datetime(Datos_CSV['Fecha'],format='ISO8601',errors='raise')
    Precios=pd.Series(pd.to_numeric(Datos_CSV.Precio,errors='raise').to_numpy(),index=fechas,name='Precio')
    Origen=f'CSV suministrado por el usuario: {ruta.name}'
elif FUENTE not in ('YAHOO','YAHOO_ARCHIVADO'):
    raise ValueError('FUENTE debe ser YAHOO, YAHOO_ARCHIVADO o CSV.')

# Preparación de los rendimientos

Precios=pd.to_numeric(Precios,errors='raise').astype(float).sort_index()
Precios.index=pd.DatetimeIndex(Precios.index)
if Precios.index.tz is not None: Precios.index=Precios.index.tz_localize(None)
Precios.index.name='Fecha'
if Precios.index.has_duplicates or not np.isfinite(Precios).all() or (Precios<=0).any():
    raise ValueError('Revisa precios: duplicados, faltantes, infinitos o valores no positivos.')
if len(Precios)<1000: raise ValueError('Para esta práctica se requieren al menos 1000 cotizaciones diarias.')
Rendimientos=(100*np.log(Precios/Precios.shift(1))).iloc[1:].rename('Rendimiento_pct')
r=Rendimientos.to_numpy(dtype=float)
Indice=Rendimientos.index
if INICIALIZACION<5 or len(r)-INICIALIZACION<500: raise ValueError('Revisa inicialización o tamaño de muestra.')
Backcast_comun=float(np.var(r[:75],ddof=0))
Precios.to_csv(SALIDAS/'00_precios_yahoo.csv',encoding='utf-8-sig')
Rendimientos.to_csv(SALIDAS/'01_rendimientos.csv',encoding='utf-8-sig')
Metadatos={'ticker':TICKER,'origen':Origen,'desde':str(Precios.index[0].date()),
           'hasta':str(Precios.index[-1].date()),'precios':len(Precios),'rendimientos':len(r),
           'inicio_verosimilitud':INICIALIZACION,'n_efectivo':len(r)-INICIALIZACION,
           'consulta_UTC':datetime.now(timezone.utc).isoformat(),'fallo_descarga_online':Error_descarga,
           'arch':importlib.metadata.version('arch'),'yfinance':importlib.metadata.version('yfinance'),
           'retorno':'100*log(P_t/P_(t-1))','ajuste_online':'auto_adjust=True'}
(SALIDAS/'00_origen_datos.json').write_text(json.dumps(Metadatos,ensure_ascii=False,indent=2),encoding='utf-8')
print('\nORIGEN:',Origen,'\nPeriodo real:',Metadatos['desde'],'a',Metadatos['hasta'])
print('N rendimientos:',len(r),'| N efectivo:',len(r)-INICIALIZACION,'| No hay división train/test.')

# Visualización de precios, rendimientos y agrupamiento de volatilidad

fig,ejes=plt.subplots(3,1,figsize=(13,10),sharex=True)
ejes[0].plot(Precios.index,Precios,color=VERDE,linewidth=1)
ejes[0].set(title=f'S&P 500 ({TICKER}) | cierre ajustado',ylabel='Puntos del índice')
ejes[1].plot(Indice,r,color=VERDE,linewidth=.55)
ejes[1].set(title='Rendimientos logarítmicos diarios',ylabel='% diario')
Rodante=Rendimientos.rolling(21).std()*np.sqrt(ANUALIZACION)
ejes[2].plot(Rodante.index,Rodante,color=NARANJA,linewidth=1)
ejes[2].set(title='Desviación estándar móvil de 21 sesiones | referencia descriptiva',ylabel='% anualizado')
for ax in ejes: ax.grid(axis='y',alpha=.15)
fig.tight_layout();guardar_figura(fig,'02_precios_rendimientos')
fig,ejes=plt.subplots(2,2,figsize=(12,7))
plot_acf(Rendimientos,lags=30,zero=False,ax=ejes[0,0],color=VERDE)
plot_pacf(Rendimientos,lags=30,zero=False,method='ywm',ax=ejes[0,1],color=VERDE)
plot_acf(Rendimientos**2,lags=30,zero=False,ax=ejes[1,0],color=NARANJA)
ejes[1,0].set_title('ACF de rendimientos cuadrados')
ejes[1,1].hist(r,bins=80,density=True,color=VERDE,alpha=.6)
ejes[1,1].set_title('Distribución de rendimientos')
fig.tight_layout();guardar_figura(fig,'03_dependencia_rendimientos')



# Pruebas iniciales de estacionariedad y efectos ARCH

ADF=adfuller(r,regression='c',autolag='BIC')
with warnings.catch_warnings(record=True) as avisos:
    warnings.simplefilter('always');KPSS=kpss(r,regression='c',nlags='auto')
Pruebas_iniciales=[{'Prueba':'ADF','H0':'Raíz unitaria','Estadistico':ADF[0],'p_valor':ADF[1]},
                  {'Prueba':'KPSS','H0':'Estacionariedad en nivel','Estadistico':KPSS[0],'p_valor':KPSS[1]}]
for h in [5,10,20]:
    lm,pv,_,_=het_arch(r-r.mean(),nlags=h,ddof=1)
    Pruebas_iniciales.append({'Prueba':f'ARCH-LM inicial ({h})','H0':'Sin efectos ARCH',
                             'Estadistico':lm,'p_valor':pv})
Pruebas_iniciales=pd.DataFrame(Pruebas_iniciales)
Pruebas_iniciales['Decision']=np.where(Pruebas_iniciales.p_valor<ALPHA,'Rechazar H0','No rechazar H0')
Pruebas_iniciales.to_csv(SALIDAS/'04_pruebas_iniciales.csv',index=False,encoding='utf-8-sig')
print('\n',Pruebas_iniciales.to_string(index=False))
(SALIDAS/'04_notas_KPSS.txt').write_text('\n'.join(str(w.message) for w in avisos),encoding='utf-8')

