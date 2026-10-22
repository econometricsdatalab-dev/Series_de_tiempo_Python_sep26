# -*- coding: utf-8 -*-
"""
Curso Integral de econometría 
Series de tiempo con Python
Tema: Modelos de heterocedasticidad condicional
Subtema: Modelos ARCH, GARCH y algunas extensiones pt2
Sesión: 11
Fecha: 21/10/2026
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

# Modelo ARCH(1) y ARCH(5)

ARCH_1=arch_model(Rendimientos,mean='Constant',
                  vol='ARCH',p=1,
                  dist='t',
                  hold_back=INICIALIZACION,
                  rescale=False)

ARCH_5=arch_model(Rendimientos,
                  mean='Constant',
                  vol='ARCH',
                  p=5,
                  dist='t',
                  hold_back=INICIALIZACION,
                  rescale=False)

# Modelo GARCH(1,1): distribución normal y t-student 

GARCH_Normal=arch_model(Rendimientos,
                        mean='Constant',
                        vol='GARCH',
                        p=1,q=1,
                        dist='normal',
                        hold_back=INICIALIZACION,
                        rescale=False)

GARCH_t=arch_model(Rendimientos,
                   mean='Constant',
                   vol='GARCH',
                   p=1,q=1,
                   dist='t',
                   hold_back=INICIALIZACION,
                   rescale=False)

# Modelo EGARCH(1,1) asimétrico 

EGARCH_t = arch_model(Rendimientos,
                    mean = 'Constant',
                    vol = "EGARCH",
                    p=1, o=1, q=1,
                    dist = 't',
                    hold_back=INICIALIZACION,
                    rescale=False)

# Modelo GJR-GARCH: umbral en la varianza 

GJR_t = arch_model(Rendimientos,
                   mean = 'Constant',
                   vol = "GARCH",
                   p = 1, o = 1, q = 1,
                   power = 2,
                   dist = 't',
                   hold_back=INICIALIZACION,
                   rescale=False)

# Modelo TGARCH / TARCH / ZARCH: umbral en la desviación estándar

TGARCH_t = arch_model(Rendimientos,
                    mean = 'Constant',
                    vol = "GARCH",
                    p = 1, o = 1, q = 1,
                    power = 1,
                    dist = 't',
                    hold_back=INICIALIZACION,
                    rescale=False)

# Modelo APARCH: potencia y asimetría estimadas

APARCH_t = arch_model(Rendimientos,
                    mean = 'Constant',
                    vol = "APARCH",
                    p = 1, o = 1, q = 1,
                    dist = 't',
                    hold_back=INICIALIZACION,
                    rescale=False)

# Extensión opcional FIGARCH: memoria larga 

Modelos={'ARCH_1_t':ARCH_1,
         'ARCH_5_t':ARCH_5,
         'GARCH_11_Normal':GARCH_Normal,
         'GARCH_11_t':GARCH_t,
         'EGARCH_111_t':EGARCH_t,
         'GJR_GARCH_111_t':GJR_t,
         'TGARCH_111_t':TGARCH_t,
         'APARCH_111_t':APARCH_t}
if INCLUIR_FIGARCH:
    Modelos['FIGARCH_11_t']=arch_model(Rendimientos,
                                       mean='Constant',
                                       vol='FIGARCH',
                                       p=1,q=1,
                                       dist='t',
                                       hold_back=INICIALIZACION,
                                       rescale=False)
    

# Estimación y comparación con la misma muestra

Ajustes,Fallos,Comparacion,Coeficientes={},[],[],[]
for nombre,modelo in Modelos.items():
    try:
        with warnings.catch_warnings(record=True) as notas:
            warnings.simplefilter('always')
            resultado=modelo.fit(disp='off',update_freq=0,cov_type='robust',backcast=Backcast_comun,
                                 options={'maxiter':2000,'ftol':1e-9})
        if resultado.convergence_flag!=0 or not np.isfinite(resultado.loglikelihood):
            Fallos.append({'Modelo':nombre,'Motivo':'Sin convergencia o verosimilitud no finita'});continue
        if resultado.nobs!=len(r)-INICIALIZACION:
            raise ValueError('Las muestras efectivas no coinciden.')
        Ajustes[nombre]=resultado
        Comparacion.append({'Modelo':nombre,'N':resultado.nobs,'k':len(resultado.params),
                             'LogL':resultado.loglikelihood,'AIC':resultado.aic,'BIC':resultado.bic})
        ic=resultado.conf_int(alpha=ALPHA)
        for parametro in resultado.params.index:
            Coeficientes.append({'Modelo':nombre,'Parametro':parametro,'Estimacion':resultado.params[parametro],
                                  'Error_estandar_robusto':resultado.std_err[parametro],'p_valor_aprox':resultado.pvalues[parametro],
                                  'LI95':ic.loc[parametro].iloc[0],'LS95':ic.loc[parametro].iloc[1]})
        (SALIDAS/f'05_{nombre}_resumen.txt').write_text(str(resultado.summary())+'\nAvisos:\n'+
              '\n'.join(str(w.message) for w in notas),encoding='utf-8')
        print(nombre,'| BIC:',round(resultado.bic,2))
    except Exception as error:
        Fallos.append({'Modelo':nombre,'Motivo':f'{type(error).__name__}: {error}'})
if not Ajustes: raise RuntimeError('No convergió ningún modelo. Revisa datos y configuración.')
Comparacion=pd.DataFrame(Comparacion).sort_values('BIC').reset_index(drop=True)
Comparacion.to_csv(SALIDAS/'05_comparacion_modelos.csv',index=False,encoding='utf-8-sig')
Coeficientes=pd.DataFrame(Coeficientes)
# nu=0 no pertenece al espacio paramétrico de una t estandarizada (nu>2).
Coeficientes.loc[Coeficientes.Parametro.eq('nu'),'p_valor_aprox']=np.nan
Coeficientes.to_csv(SALIDAS/'05_coeficientes.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(Fallos,columns=['Modelo','Motivo']).to_csv(SALIDAS/'05_fallos.csv',index=False,encoding='utf-8-sig')
print('\nComparación (mínimo BIC = candidato, no validación automática):\n',Comparacion.to_string(index=False))


# Validación de todos los modelos estimados

Diagnosticos,Volatilidades=[],pd.DataFrame(index=Indice)
for nombre,resultado in Ajustes.items():
    nu=float(resultado.params.get('nu',np.inf))
    diag,z=diagnosticos(nombre,resultado.resid,resultado.conditional_volatility,nu,df_media=0)
    Diagnosticos.append(diag)
    Volatilidades[nombre]=resultado.conditional_volatility
Diagnosticos=pd.concat(Diagnosticos,ignore_index=True)
Diagnosticos.to_csv(SALIDAS/'06_diagnosticos.csv',index=False,encoding='utf-8-sig')
Volatilidades.to_csv(SALIDAS/'06_volatilidades_diarias_pct.csv',encoding='utf-8-sig')
Nombre_mejor=Comparacion.iloc[0].Modelo
Mejor=Ajustes[Nombre_mejor]
Diagnostico_mejor=Diagnosticos.loc[Diagnosticos.Modelo==Nombre_mejor].copy()
Rechazos=Diagnostico_mejor.loc[Diagnostico_mejor.p_valor.lt(ALPHA) &
                              Diagnostico_mejor.Evaluacion.str.startswith('Diagnóstico'),'Prueba'].tolist()
print('\nCandidato por BIC:',Nombre_mejor,'\n',Diagnostico_mejor.to_string(index=False))
print('Rechazos diagnósticos (no incluye JB como descarte de una distribución t):',Rechazos)
_,z_mejor=diagnosticos(Nombre_mejor,Mejor.resid,Mejor.conditional_volatility,float(Mejor.params.get('nu',np.inf)))
grafico_residuos(z_mejor,float(Mejor.params.get('nu',np.inf)),Nombre_mejor,'07_diagnosticos_mejor')
fig,ax=plt.subplots(figsize=(13,5))
for nombre in Volatilidades:
    ax.plot(Indice,Volatilidades[nombre]*np.sqrt(ANUALIZACION),linewidth=.8,alpha=.7,label=nombre)
ax.set(title='Volatilidad condicional estimada | misma serie y muestra',ylabel='% anualizado',xlabel='Fecha')
ax.legend(frameon=False,ncol=2,fontsize=8);ax.grid(axis='y',alpha=.15)
fig.tight_layout();guardar_figura(fig,'07_comparacion_volatilidades')


# Persistencia: fórmulas específicas, no aplicar alpha+beta a todos los modelos

Persistencia=[]
for nombre,ajuste in Ajustes.items():
    p=ajuste.params;nu=float(p.get('nu',np.inf))
    a=float(p.get('alpha[1]',0));b=float(p.get('beta[1]',0));g=float(p.get('gamma[1]',0))
    if nombre.startswith('ARCH'):
        valor=sum(float(v) for k,v in p.items() if k.startswith('alpha['))
        medida='Suma alpha; <1 para segundo momento finito'
    elif nombre.startswith('GJR'):
        valor=a+.5*g+b;medida='alpha+gamma/2+beta; simetría de z y E[z²]=1'
    elif nombre.startswith('TGARCH'):
        eabs=np.sqrt(2/np.pi) if np.isinf(nu) else np.exp(np.log(2)+.5*np.log(nu-2)+special.gammaln((nu+1)/2)-np.log(nu-1)-.5*np.log(np.pi)-special.gammaln(nu/2))
        valor=b*b+2*b*eabs*(a+.5*g)+.5*(a*a+(a+g)**2)
        medida='E[(beta+(alpha+gamma*I)*|z|)^2]; <1 para E[sigma²] finita'
    elif nombre.startswith('EGARCH'):
        valor=abs(b);medida='|beta|; estabilidad de log(h), no garantiza E[h] finita'
    elif nombre.startswith('APARCH'):
        delta=float(p['delta'])
        if np.isfinite(nu) and nu<=delta: momento=np.inf
        elif np.isfinite(nu): momento=np.exp(.5*delta*np.log(nu-2)+special.gammaln((delta+1)/2)+special.gammaln((nu-delta)/2)-.5*np.log(np.pi)-special.gammaln(nu/2))
        else: momento=np.exp(.5*delta*np.log(2)+special.gammaln((delta+1)/2)-.5*np.log(np.pi))
        valor=b+.5*a*((1-g)**delta+(1+g)**delta)*momento
        medida=f'Momento de sigma^delta (delta={delta:.3f}); NO equivale a segundo momento si delta!=2'
    elif nombre.startswith('FIGARCH'):
        valor=float(p.get('d',np.nan));medida='d fraccional; no comparar con alpha+beta'
    else:
        valor=a+b;medida='alpha+beta; <1 para segundo momento finito'
    Persistencia.append({'Modelo':nombre,'Medida':medida,'Valor':valor})
pd.DataFrame(Persistencia).to_csv(SALIDAS/'07_persistencia_y_momentos.csv',index=False,encoding='utf-8-sig')


# Pronósticos de volatilidad a 10 sesiones: simulación para todas las familias

Pronosticos=[];Pronostico_mejor=None
for j,(nombre,ajuste) in enumerate(Ajustes.items()):
    gen=np.random.default_rng(SEMILLA+j)
    nu=float(ajuste.params.get('nu',np.inf))
    if np.isfinite(nu):
        generador=lambda size,gen=gen,nu=nu: gen.standard_t(nu,size=size)*np.sqrt((nu-2)/nu)
    else: generador=gen.standard_normal
    f=ajuste.forecast(horizon=HORIZONTE,method='simulation',simulations=SIMULACIONES,rng=generador,reindex=False)
    h=np.asarray(f.residual_variance.iloc[-1],dtype=float)
    sigma_paths=np.sqrt(np.asarray(f.simulations.residual_variances)[-1])
    bandas=np.quantile(sigma_paths,[.025,.975],axis=0)
    tabla=pd.DataFrame({'Modelo':nombre,'Horizonte_sesion':np.arange(1,HORIZONTE+1),
                        'E_varianza_innovacion':h,'Volatilidad_diaria_RMS_pct':np.sqrt(h),
                        'Volatilidad_anualizada_RMS_pct':np.sqrt(h*ANUALIZACION),
                        'Q025_trayectoria_sigma_diaria_pct':bandas[0],
                        'Q975_trayectoria_sigma_diaria_pct':bandas[1]})
    Pronosticos.append(tabla)
    if nombre==Nombre_mejor: Pronostico_mejor=tabla.copy()
Pronosticos=pd.concat(Pronosticos,ignore_index=True)
Pronosticos.to_csv(SALIDAS/'08_pronosticos_volatilidad.csv',index=False,encoding='utf-8-sig')
fig,ejes=plt.subplots(1,2,figsize=(13,4.8))
for nombre,tabla in Pronosticos.groupby('Modelo'):
    ejes[0].plot(tabla.Horizonte_sesion,tabla.Volatilidad_anualizada_RMS_pct,marker='.',label=nombre)
ejes[0].set(title='Pronóstico de volatilidad | comparación',xlabel='Sesión futura h',ylabel='% anualizado')
ejes[0].legend(frameon=False,fontsize=7,ncol=2)
q=Pronostico_mejor
for ax in ejes: ax.grid(axis='y',alpha=.15)
ejes[1].plot(q.Horizonte_sesion,q.Volatilidad_diaria_RMS_pct,color=VERDE,marker='o',label='sqrt(E[h])')
ejes[1].fill_between(q.Horizonte_sesion,q.Q025_trayectoria_sigma_diaria_pct,q.Q975_trayectoria_sigma_diaria_pct,
                     color=VERDE,alpha=.18,label='95 % trayectorias de sigma')
ejes[1].set(title=Nombre_mejor+(' | revisar diagnósticos' if Rechazos else ''),xlabel='Sesión futura h',ylabel='% diario')
ejes[1].legend(frameon=False,fontsize=8)
fig.tight_layout();guardar_figura(fig,'08_pronostico_volatilidad')


# ARMA(1,1)-GARCH(1,1)-t

with warnings.catch_warnings(record=True):
    warnings.simplefilter('always')
    ARMA_inicial=ARIMA(Rendimientos,
                       order=(1,0,1),
                       trend='c',
                       enforce_stationarity=True
                       ,enforce_invertibility=True).fit()
(SALIDAS/'09_ARMA_solo_inicializacion.txt').write_text(str(ARMA_inicial.summary()),encoding='utf-8')
Vol_proceso=GARCH(p=1,o=0,q=1)
Limites_varianza=Vol_proceso.variance_bounds((r-r[INICIALIZACION:].mean())[INICIALIZACION:])


# Funciones necesarias para expresar la verosimilitud conjunta; el resto es paso a paso.
def transformar_parametros(v):
    logden=np.logaddexp(0,np.logaddexp(v[4],v[5]))
    return np.array([v[0],np.tanh(v[1]),np.tanh(v[2]),np.exp(v[3]),
                     (1-1e-6)*np.exp(v[4]-logden),(1-1e-6)*np.exp(v[5]-logden),2+np.exp(v[6])])


def filtrar_arma_garch(v):
    mu,phi,theta,omega,alpha,beta,nu=transformar_parametros(v)
    eps=signal.lfilter([1,-phi],[1,theta],r-mu)
    # arch inicia su recursión de varianza en la muestra efectiva hold_back:
    # alinear ese inicio y el backcast permite una comparación de verosimilitudes.
    eps_efectivos=eps[INICIALIZACION:]
    h_efectiva=Vol_proceso.compute_variance(np.array([omega,alpha,beta]),eps_efectivos,
                                           np.empty_like(eps_efectivos),Backcast_comun,Limites_varianza)
    h=np.r_[np.full(INICIALIZACION,np.nan),h_efectiva]
    return eps,h


def logdensidades(v):
    nu=transformar_parametros(v)[-1]
    eps,h=filtrar_arma_garch(v)
    h=h[INICIALIZACION:];eps=eps[INICIALIZACION:]
    return (special.gammaln((nu+1)/2)-special.gammaln(nu/2)-.5*np.log(np.pi*(nu-2))
            -.5*np.log(h)-.5*(nu+1)*np.log1p(eps*eps/(h*(nu-2))))


def objetivo_conjunto(v):
    valores=logdensidades(v)
    return -float(np.sum(valores)) if np.isfinite(valores).all() else 1e100

# Inicialización GARCH t (si ese ajuste falló se usa una semilla conservadora).
semilla_garch=Ajustes.get('GARCH_11_t')
mu0=float(ARMA_inicial.params.get('const',r.mean()))
a0=float(semilla_garch.params['alpha[1]']) if semilla_garch is not None else .08
b0=float(semilla_garch.params['beta[1]']) if semilla_garch is not None else .88
if a0+b0>=.999: a0,b0=.08,.90
w0=float(semilla_garch.params['omega']) if semilla_garch is not None else np.var(r)*.04
nu0=float(semilla_garch.params['nu']) if semilla_garch is not None else 8
vbase=np.array([mu0,0,0,np.log(max(w0,1e-7)),np.log(a0/(1-1e-6-a0-b0)),
                np.log(b0/(1-1e-6-a0-b0)),np.log(nu0-2)])
phi0=float(ARMA_inicial.params.get('ar.L1',0));theta0=float(ARMA_inicial.params.get('ma.L1',0))
Inicios=[(phi0,theta0),(0,0),(.25,-.10),(-.25,.10)]
Limites=[(-2,2),(-4,4),(-4,4),(-20,5),(-12,12),(-12,12),(np.log(.05),np.log(198))]
Optimos=[];Registro_optimos=[]
for i,(p0,t0) in enumerate(Inicios):
    vi=vbase.copy();vi[1:3]=np.arctanh(np.clip([p0,t0],-.95,.95))
    opt=optimize.minimize(objetivo_conjunto,vi,method='L-BFGS-B',bounds=Limites,
                          options={'maxiter':1500,'ftol':1e-11,'gtol':1e-6,'maxls':40})
    Registro_optimos.append({'Inicio':i+1,'Converge':bool(opt.success),'NegLogL':float(opt.fun),
                             'Iteraciones':int(opt.nit),'Mensaje':str(opt.message)})
    if opt.success and np.isfinite(opt.fun): Optimos.append(opt)
pd.DataFrame(Registro_optimos).to_csv(SALIDAS/'09_ARMA_GARCH_optimizacion.csv',index=False,encoding='utf-8-sig')
if not Optimos: raise RuntimeError('No convergió el ARMA-GARCH conjunto. Revisa los registros e inicializaciones.')
Optimo=min(Optimos,key=lambda x:x.fun)
vhat=Optimo.x
mu,phi,theta,omega,alpha,beta,nu=transformar_parametros(vhat)
EPS,H=filtrar_arma_garch(vhat)
N_efectivo=len(r)-INICIALIZACION
logL=-Optimo.fun;k=len(vhat)
Resumen_conjunto=pd.DataFrame([{'Modelo':'ARMA(1,1)-GARCH(1,1)-t conjunto','N':N_efectivo,'k':k,
                                'LogL':logL,'AIC':-2*logL+2*k,'BIC':-2*logL+k*np.log(N_efectivo),
                                'alpha_mas_beta':alpha+beta,'Varianza_incondicional':omega/(1-alpha-beta),
                                'Converge':bool(Optimo.success)}])
Resumen_conjunto.to_csv(SALIDAS/'09_ARMA_GARCH_criterios.csv',index=False,encoding='utf-8-sig')
print('\nARMA-GARCH conjunto:\n',Resumen_conjunto.to_string(index=False))

# Inferencia sandwich numérica y delta: no inventar SE si hay identificación débil.
SE=np.full(k,np.nan);Nota_inferencia=''
try:
    Hess=approx_hess3(vhat,objetivo_conjunto)
    Hess=.5*(Hess+Hess.T)
    autovalores=np.linalg.eigvalsh(Hess)
    if autovalores.min()<=1e-6 or np.linalg.cond(Hess)>1e12:
        raise ValueError('Hessiano singular/indefinido o mal condicionado; revisar cancelación AR-MA.')
    pasos=1e-5*np.maximum(1,np.abs(vhat))
    Score=np.empty((N_efectivo,k));J=np.empty((k,k))
    for j in range(k):
        mas=vhat.copy();menos=vhat.copy();mas[j]+=pasos[j];menos[j]-=pasos[j]
        Score[:,j]=(logdensidades(mas)-logdensidades(menos))/(2*pasos[j])
        J[:,j]=(transformar_parametros(mas)-transformar_parametros(menos))/(2*pasos[j])
    inversa=np.linalg.inv(Hess)
    cov_raw=inversa@(Score.T@Score)@inversa
    Cov=J@cov_raw@J.T
    if not np.isfinite(Cov).all() or (np.diag(Cov)<=0).any(): raise ValueError('Covarianza no válida.')
    SE=np.sqrt(np.diag(Cov));Nota_inferencia='Sandwich numérico en parámetros transformados + método delta.'
except (ValueError,np.linalg.LinAlgError) as error:
    Nota_inferencia='Inferencia no disponible: '+str(error)
Coef_conjunto=pd.DataFrame({'Parametro':['mu','phi','theta','omega','alpha','beta','nu'],
                            'Estimacion':transformar_parametros(vhat),'SE_robusto_aprox':SE})
Coef_conjunto['z_aprox']=Coef_conjunto.Estimacion/SE
Coef_conjunto['p_valor_aprox']=2*stats.norm.sf(np.abs(Coef_conjunto.z_aprox))
Coef_conjunto.loc[Coef_conjunto.Parametro.eq('nu'),['z_aprox','p_valor_aprox']]=np.nan
Coef_conjunto['LI95']=Coef_conjunto.Estimacion-stats.norm.ppf(.975)*SE
Coef_conjunto['LS95']=Coef_conjunto.Estimacion+stats.norm.ppf(.975)*SE
Coef_conjunto.to_csv(SALIDAS/'09_ARMA_GARCH_coeficientes.csv',index=False,encoding='utf-8-sig')
(SALIDAS/'09_ARMA_GARCH_inferencia.txt').write_text(Nota_inferencia+'\nLos z son orientativos para parámetros con H0 en frontera; nu=0 no es una hipótesis admisible.',encoding='utf-8')
print(Coef_conjunto.to_string(index=False),'\n',Nota_inferencia)

# Validación de ARMA-GARCH sobre innovaciones estandarizadas, no retornos brutos

Serie_eps=pd.Series(EPS[INICIALIZACION:],index=Indice[INICIALIZACION:])
Serie_sigma=pd.Series(np.sqrt(H[INICIALIZACION:]),index=Indice[INICIALIZACION:])
Diagnostico_conjunto,Z=diagnosticos('ARMA_GARCH_conjunto_t',Serie_eps,Serie_sigma,nu,df_media=2)
Rechazos_conjunto=Diagnostico_conjunto.loc[Diagnostico_conjunto.p_valor.lt(ALPHA) &
                 Diagnostico_conjunto.Evaluacion.str.startswith('Diagnóstico'),'Prueba'].tolist()
Estado_conjunto=('EXPLORATORIO: revisar '+', '.join(Rechazos_conjunto)) if Rechazos_conjunto \
                 else 'Sin rechazos en los contrastes evaluados; no garantiza validez.'
Diagnostico_conjunto.to_csv(SALIDAS/'10_ARMA_GARCH_diagnosticos.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'Rendimiento_pct':Rendimientos,'Media_condicional_pct':r-EPS,
              'Innovacion_pct':EPS,'Volatilidad_diaria_pct':np.sqrt(H),'z':EPS/np.sqrt(H)},index=Indice).iloc[INICIALIZACION:].to_csv(
                  SALIDAS/'10_ARMA_GARCH_residuos_y_volatilidad.csv',encoding='utf-8-sig')
print('\nDiagnóstico ARMA-GARCH:\n',Diagnostico_conjunto.to_string(index=False))
grafico_residuos(Z,nu,'ARMA(1,1)-GARCH(1,1)-t | estimación conjunta','10_ARMA_GARCH_residuos')
fig,ejes=plt.subplots(2,1,figsize=(13,7),sharex=True)
ejes[0].plot(Indice[INICIALIZACION:],Rendimientos.iloc[INICIALIZACION:],color=GRIS,linewidth=.5,label='Rendimiento')
ejes[0].plot(Indice[INICIALIZACION:],(r-EPS)[INICIALIZACION:],color=VERDE,linewidth=.8,label='Media condicional')
ejes[0].set(ylabel='% diario',title='ARMA: media condicional');ejes[0].legend(frameon=False)
ejes[1].plot(Serie_sigma.index,Serie_sigma*np.sqrt(ANUALIZACION),color=NARANJA,linewidth=1)
ejes[1].set(title='GARCH: volatilidad condicional',ylabel='% anualizado')
fig.tight_layout();guardar_figura(fig,'10_ARMA_GARCH_media_volatilidad')

# Pronóstico conjunto: media ARMA, varianza GARCH y trayectorias t

Media_futura=np.empty(HORIZONTE);H_futura=np.empty(HORIZONTE)
r_anterior=float(r[-1]);e_anterior=float(EPS[-1]);h_anterior=float(H[-1])
for j in range(HORIZONTE):
    Media_futura[j]=mu+phi*(r_anterior-mu)+theta*(e_anterior if j==0 else 0)
    H_futura[j]=omega+alpha*e_anterior**2+beta*h_anterior if j==0 else omega+(alpha+beta)*H_futura[j-1]
    r_anterior=Media_futura[j]
# Varianza predictiva de r_(T+h): propagación ARMA de las innovaciones futuras.
Psi=np.r_[1.0,(phi+theta)*phi**np.arange(max(0,HORIZONTE-1))]
Var_r_futura=np.array([sum(Psi[h-i]**2*H_futura[i] for i in range(h+1)) for h in range(HORIZONTE)])
gen=np.random.default_rng(SEMILLA+100)
R_paths=np.empty((SIMULACIONES,HORIZONTE));H_paths=np.empty_like(R_paths)
r_pre=np.full(SIMULACIONES,r[-1]);e_pre=np.full(SIMULACIONES,EPS[-1]);h_pre=np.full(SIMULACIONES,H[-1])
for j in range(HORIZONTE):
    h_pre=omega+alpha*e_pre**2+beta*h_pre
    m_pre=mu+phi*(r_pre-mu)+theta*e_pre
    e_pre=np.sqrt(h_pre)*gen.standard_t(nu,size=SIMULACIONES)*np.sqrt((nu-2)/nu)
    r_pre=m_pre+e_pre
    R_paths[:,j]=r_pre;H_paths[:,j]=h_pre
Cuantiles_r=np.quantile(R_paths,[.025,.10,.90,.975],axis=0)
Cuantiles_sigma=np.quantile(np.sqrt(H_paths),[.025,.975],axis=0)
Pronostico_conjunto=pd.DataFrame({'Horizonte_sesion':np.arange(1,HORIZONTE+1),
    'Media_rendimiento_pct':Media_futura,'E_varianza_innovacion':H_futura,
    'Varianza_predictiva_rendimiento':Var_r_futura,'Volatilidad_innovacion_RMS_diaria_pct':np.sqrt(H_futura),
    'Desv_predictiva_rendimiento_pct':np.sqrt(Var_r_futura),
    'LI80_rendimiento_pct':Cuantiles_r[1],'LS80_rendimiento_pct':Cuantiles_r[2],
    'LI95_rendimiento_pct':Cuantiles_r[0],'LS95_rendimiento_pct':Cuantiles_r[3],
    'Q025_trayectoria_sigma_diaria_pct':Cuantiles_sigma[0],
    'Q975_trayectoria_sigma_diaria_pct':Cuantiles_sigma[1]})
Pronostico_conjunto['Estado_validacion']=Estado_conjunto
Pronostico_conjunto.to_csv(SALIDAS/'11_ARMA_GARCH_pronostico.csv',index=False,encoding='utf-8-sig')
print('\nPronóstico conjunto:\n',Pronostico_conjunto.to_string(index=False))
fig,ejes=plt.subplots(1,2,figsize=(13,5))
h=Pronostico_conjunto.Horizonte_sesion
q=Pronostico_conjunto
ejes[0].fill_between(h,q.LI95_rendimiento_pct,q.LS95_rendimiento_pct,color=VERDE,alpha=.12,label='95 % predictivo simulado')
ejes[0].fill_between(h,q.LI80_rendimiento_pct,q.LS80_rendimiento_pct,color=VERDE,alpha=.25,label='80 % predictivo simulado')
ejes[0].plot(h,Media_futura,color=VERDE,marker='o',label='Media condicional')
ejes[0].set(title='ARMA-GARCH | pronóstico del rendimiento',ylabel='% diario',xlabel='Sesión futura h')
ejes[0].legend(frameon=False,fontsize=8)
ejes[1].plot(h,np.sqrt(H_futura),color=NARANJA,marker='o',label='Volatilidad innovación: sqrt(E[h])')
ejes[1].plot(h,np.sqrt(Var_r_futura),color=VERDE,linestyle='--',label='Desv. predictiva del rendimiento')
ejes[1].fill_between(h,Cuantiles_sigma[0],Cuantiles_sigma[1],color=NARANJA,alpha=.15,label='95 % trayectorias de sigma')
ejes[1].set(title='Volatilidad y propagación de la media ARMA',ylabel='% diario',xlabel='Sesión futura h')
ejes[1].legend(frameon=False,fontsize=8)
for ax in ejes: ax.grid(axis='y',alpha=.15)
if Rechazos_conjunto:
    fig.suptitle('Pronóstico exploratorio | hay diagnósticos rechazados',color=NARANJA,fontsize=11)
fig.tight_layout();guardar_figura(fig,'11_ARMA_GARCH_pronostico')

