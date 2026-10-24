# -*- coding: utf-8 -*-
"""
Curso Integral de econometría
Series de tiempo con Python
Tema: Filtros de series de tiempo
Subtema: Medias móviles (centradas y no centradas), Hodrick-Prescott y Holt-Winters
Sesión: 11
Fecha: 23/10/2026
Docente: Alexis Adonai Morales Alberto
"""

# Modulos a utilizar

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import os
import json
from getpass import getpass, GetPassWarning

# Clases directas

from pathlib import Path
from datetime import datetime, timezone
import requests
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
HORIZONTE = 10                       # Horizonte de pronóstico: 10 trimestres
LAMB_HP = 1600                       # Lambda estándar de Ravn-Uhlig para trimestral
LAMB_HP_RU = 6.25                    # Lambda Ravn-Uhlig (ciclo de ~6 años)
VENTANA_MA = s                       # Ventana de las medias móviles: 4 trimestres
REPLICAS_HW = 2000                   # Trayectorias simuladas para bandas HW
SEMILLA = 2026
ALPHA = 0.05
MOSTRAR = True
SALIDAS = Path.cwd() / 'salidas_filtros_PIB'
assert (FRECUENCIA == 'M' and s == 12) or (FRECUENCIA.startswith('Q') and s == 4), \
    'Este flujo está configurado para series mensuales o trimestrales.'
SALIDAS.mkdir(parents=True, exist_ok=True)

# Paleta de colores del curso (sistema descrito en S07.py)

VERDE, GRIS, NARANJA = '#08989C', '#4D565E', '#F04A1D'
AZUL_TITULO, GRIS_CLARO, LINEA_SUAVE = '#003057', '#9AA0A6', '#D5D8DC'
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

def encabezado(fig, titulo, subtitulo):
    fig.text(0.10, 0.965, titulo, fontsize=19, fontweight='bold',
             color=AZUL_TITULO, ha='left', va='top')
    fig.text(0.10, 0.905, subtitulo, fontsize=11, color=GRIS, ha='left', va='top')

def fuente(fig):
    fig.text(0.10, 0.025, 'Fuente. INEGI. Banco de Información Económica (BIE)',
             fontsize=9, color=GRIS, ha='left', va='bottom')

def estilo_ejes(ax):
    ax.grid(axis='y', linestyle=':', linewidth=0.8, color=LINEA_SUAVE)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(LINEA_SUAVE)
    ax.tick_params(axis='both', colors=GRIS, labelsize=9)

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
    series = data_json.get('Series') if isinstance(data_json, dict) else None
    if not isinstance(series, list) or len(series) != 1 or not isinstance(series[0], dict):
        raise ValueError('La API no devolvió exactamente una serie. Revisa token e indicador.')
    estructura = series[0]
    id_devuelto = estructura.get('INDICADOR', estructura.get('INDICATOR'))
    if id_devuelto is not None and str(id_devuelto) != id_serie:
        raise ValueError('El identificador recibido no coincide con el solicitado.')
    observaciones = estructura.get('OBSERVATIONS')
    if not isinstance(observaciones, list) or not observaciones:
        raise ValueError('La serie no tiene observaciones disponibles.')
    originales = pd.DataFrame(observaciones)
    if not {'TIME_PERIOD','OBS_VALUE'}.issubset(originales.columns):
        raise ValueError('Faltan TIME_PERIOD/OBS_VALUE en la respuesta del BIE.')
    df = originales[['TIME_PERIOD','OBS_VALUE']].copy()
    df.columns = ['Periodo_BIE','Serie']
    df['Serie'] = pd.to_numeric(df['Serie'], errors='coerce')
    if not np.isfinite(df['Serie'].to_numpy(dtype=float)).all():
        raise ValueError('El BIE devuelve valores faltantes/no numéricos. Revisa OBS_EXCEPTION; '
                         'no se eliminan ni interpolan automáticamente.')
    texto = df['Periodo_BIE'].astype(str).str.strip()
    texto = texto.str.replace(r'[-/]?[QT]', '/', regex=True)
    partes = texto.str.extract(r'^(?P<anio>\d{4})[-/](?P<parte>\d{1,2})$')
    if partes.isna().any().any():
        raise ValueError('Formato temporal BIE no reconocido. Esperado AAAA/NN o AAAAQN.')
    anio, parte = partes['anio'].astype(int), partes['parte'].astype(int)
    limite = 12 if periodo == 'Mensual' else 4
    if not parte.between(1, limite).all():
        raise ValueError(f'Los periodos recibidos no son {periodo.lower()}es; revisa el indicador.')
    mes = parte if periodo == 'Mensual' else (parte-1)*3+1
    df['Fecha'] = pd.to_datetime({'year': anio, 'month': mes, 'day': 1}, errors='raise')
    df = df[['Fecha','Serie']].sort_values('Fecha').reset_index(drop=True)
    if df['Fecha'].duplicated().any():
        raise ValueError('La API contiene periodos duplicados; revisa la serie solicitada.')
    df.attrs['metadatos'] = {k: v for k, v in estructura.items() if k != 'OBSERVATIONS'}
    df.attrs['observaciones_BIE'] = observaciones  # Incluye notas/estatus originales.
    return df

# Construcción de consulta

if FUENTE == 'BIE':
    if FRECUENCIA != 'Q-DEC' or s != 4:
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
    PIB = Datos_BIE.rename(columns={'Serie': VARIABLE, 'Fecha': FECHA})
    Datos = PIB[[FECHA, VARIABLE]].copy()
    Metadatos_BIE = dict(Datos_BIE.attrs['metadatos'])
    Metadatos_BIE.update({'serie_configurada': ID_SERIE, 'descripcion_configurada': ETIQUETA_SERIE,
                          'unidad_configurada': UNIDAD_PIB, 'periodicidad': 'Trimestral',
                          'consulta_UTC': datetime.now(timezone.utc).isoformat(),
                          'cobertura': 'Nacional (00)', 'historia_completa': True,
                          'flujo': 'filtros (medias móviles, HP, Holt-Winters)',
                          'fuente': 'https://www.inegi.org.mx/servicios/feeds.html'})
    (SALIDAS/'PIB_BIE_metadatos.json').write_text(json.dumps(Metadatos_BIE, ensure_ascii=False, indent=2),
                                                  encoding='utf-8')
    pd.DataFrame(Datos_BIE.attrs['observaciones_BIE']).to_csv(
        SALIDAS/'PIB_BIE_observaciones_originales.csv', index=False, encoding='utf-8-sig')
    Datos.to_csv(SALIDAS/'PIB_BIE_consulta.csv', index=False, encoding='utf-8-sig')
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
    PIB = Datos[[FECHA, VARIABLE]].copy()
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
Serie.to_csv(SALIDAS / 'serie_utilizada.csv', encoding='utf-8-sig')
print(Serie.head(), '\nN =', len(Serie), '| s =', s,
      '| Historia:', Serie.index[0], 'a', Serie.index[-1], '\nUnidad:', UNIDAD_PIB)
print('Origen:', Origen_datos)
print(f'Los filtros pronosticables proyectan {HORIZONTE} trimestres.')

Textos_periodo = pd.PeriodIndex(Serie.index, freq=FRECUENCIA).astype(str)
Historia_txt = f'{Textos_periodo[0]} - {Textos_periodo[-1]}'
Fechas_ts = Serie.index.to_timestamp()

# Visualización inicial

fig_d1, ax_d1 = plt.subplots(figsize=(14, 6), dpi=500)
fig_d1.patch.set_facecolor('white')
ax_d1.plot(Fechas_ts, Serie.to_numpy(), color=VERDE, linewidth=1.7)
ax_d1.axhline(y=float(np.mean(Serie)), color=GRIS_CLARO, linestyle='--', linewidth=0.9)
ax_d1.set_ylabel('PIB: Millones de pesos a precios del 2018', fontsize=11, color=GRIS)
ax_d1.set_xlabel('Año', fontsize=11, color=GRIS)
ax_d1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_d1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
estilo_ejes(ax_d1)
encabezado(fig_d1, 'Producto Interno Bruto de México',
           f'Serie trimestral: {Historia_txt}')
fuente(fig_d1)
fig_d1.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.15)
guardar_figura(fig_d1, '01_serie')

# Filtro 1: Medias moviles no centradas (trailing) y centradas

# a) No centrada: solo PESOS EN EL PASADO. Última media disponible en t = final.
#    Pronóstico admisible: plano en la última estimación (nivel sin tendencia).
MA_no_centrada = Serie.rolling(window=VENTANA_MA, min_periods=VENTANA_MA).mean()
MA_no_centrada.name = 'MA_no_centrada'

# b) Centrada (2x4 clásica para ventana par): usa PASADO Y FUTURO;
#    por eso pierde dos extremos y NO admite pronóstico en t+k.
MA_centrada = (Serie.shift(2) + 2*Serie.shift(1) + 2*Serie + 2*Serie.shift(-1) + Serie.shift(-2)) / 8
MA_centrada.name = 'MA_centrada_2x4'

print('\nMedia móvil no centrada (últimos 5):')
print(MA_no_centrada.tail().to_string())
print('\nMedia móvil centrada 2x4 (últimos 5; observa los faltantes del extremo):')
print(MA_centrada.tail().to_string())

# Grafico comparativo de medias moviles

fig_ma, ax_ma = plt.subplots(figsize=(14, 6), dpi=500)
fig_ma.patch.set_facecolor('white')
ax_ma.plot(Fechas_ts, Serie.to_numpy(), color=LINEA_SUAVE, linewidth=1.1,
           label='PIB original')
ax_ma.plot(Fechas_ts, MA_centrada.to_numpy(), color=NARANJA, linewidth=2.0,
           label='MA centrada 2x4 (usa futuro: sin pronóstico)')
ax_ma.plot(Fechas_ts, MA_no_centrada.to_numpy(), color=VERDE, linewidth=2.0,
           label=f'MA no centrada {VENTANA_MA}T (solo pasado)')
ax_ma.set_ylabel('PIB: Millones de pesos a precios del 2018', fontsize=11, color=GRIS)
ax_ma.set_xlabel('Año', fontsize=11, color=GRIS)
ax_ma.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_ma.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
estilo_ejes(ax_ma)
ax_ma.legend(loc='upper left', frameon=False, fontsize=10)
encabezado(fig_ma, 'Filtro 1: Medias móviles del PIB',
           f'Ventana = {VENTANA_MA} trimestres | diferencia entre centrada y no centrada')
fuente(fig_ma)
fig_ma.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.14)
guardar_figura(fig_ma, '02_medias_moviles')

# Pronóstico de la MA no centrada: plano (supuesto de nivel constante)

Ultima_MA = float(MA_no_centrada.iloc[-1])
Indice_pron = pd.period_range(Serie.index[-1] + 1, periods=HORIZONTE, freq=FRECUENCIA)
Pron_MA = pd.Series(np.repeat(Ultima_MA, HORIZONTE), index=Indice_pron, name='MA_pronostico')
Pron_MA.to_csv(SALIDAS/'pronostico_MA_no_centrada.csv', encoding='utf-8-sig')
print('\nPronóstico MA no centrada (plano, supuesto de nivel sin tendencia). Primeros 3:')
print(Pron_MA.head(3).to_string())

fig_ma_p, ax_map = plt.subplots(figsize=(14, 6), dpi=500)
fig_ma_p.patch.set_facecolor('white')
ax_map.plot(Fechas_ts, Serie.to_numpy(), color=LINEA_SUAVE, linewidth=1.1, label='PIB original')
ax_map.plot(Fechas_ts, MA_no_centrada.to_numpy(), color=VERDE, linewidth=2.0,
            label='MA no centrada (ajuste histórico)')
Ts_pron = Pron_MA.index.to_timestamp()
ax_map.plot(Ts_pron, Pron_MA.to_numpy(), color=NARANJA, linewidth=2.2,
            linestyle='--', label=f'Pronóstico plano a {HORIZONTE} trimestres')
ax_map.axvline(Serie.index[-1].to_timestamp(), color=GRIS, linewidth=0.9,
               linestyle=':', alpha=.6)
ax_map.set_ylabel('PIB: Millones de pesos a precios del 2018', fontsize=11, color=GRIS)
ax_map.set_xlabel('Año', fontsize=11, color=GRIS)
ax_map.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_map.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
estilo_ejes(ax_map)
ax_map.legend(loc='upper left', frameon=False, fontsize=10)
encabezado(fig_ma_p, 'Pronóstico con media móvil no centrada',
           f'Última MA = {Ultima_MA:,.1f} millones de pesos | horizonte: {HORIZONTE} trimestres | '
           'la centrada no admite pronóstico')
fuente(fig_ma_p)
fig_ma_p.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.14)
guardar_figura(fig_ma_p, '03_pronostico_MA_no_centrada')

# Filtro 2: Hodrick-Prescott (bilateral: tendencia de toda la muestra,
# NO pronosticable directamente)

Ciclo_HP, Tendencia_HP = hpfilter(Serie, lamb=LAMB_HP)
Tendencia_HP.name = 'Tendencia_HP'
Ciclo_HP.name = 'Ciclo_HP'
_, Tendencia_HP_RU = hpfilter(Serie, lamb=LAMB_HP_RU)
Tendencia_HP_RU.name = 'Tendencia_HP_RavnUhlig'

fig_hp, ejes_hp = plt.subplots(2, 1, figsize=(14, 9), dpi=500, sharex=True)
fig_hp.patch.set_facecolor('white')
ejes_hp[0].plot(Fechas_ts, Serie.to_numpy(), color=LINEA_SUAVE, linewidth=1.1, label='PIB original')
ejes_hp[0].plot(Fechas_ts, Tendencia_HP.to_numpy(), color=NARANJA, linewidth=2.2,
                label=f'Tendencia HP (lambda = {LAMB_HP:g})')
ejes_hp[0].plot(Fechas_ts, Tendencia_HP_RU.to_numpy(), color=VERDE, linewidth=1.8,
                linestyle='--', label=f'Tendencia HP Ravn-Uhlig (lambda = {LAMB_HP_RU:g})')
ejes_hp[0].set_ylabel('PIB niveles', fontsize=11, color=GRIS)
ejes_hp[0].legend(loc='upper left', frameon=False, fontsize=10)
estilo_ejes(ejes_hp[0])
ejes_hp[1].plot(Fechas_ts, Ciclo_HP.to_numpy(), color=VERDE, linewidth=1.8, label='Ciclo HP')
ejes_hp[1].axhline(0, color=GRIS, linewidth=0.9, linestyle='--')
ejes_hp[1].set_ylabel('Componente cíclico', fontsize=11, color=GRIS)
ejes_hp[1].set_xlabel('Año', fontsize=11, color=GRIS)
ejes_hp[1].legend(loc='lower left', frameon=False, fontsize=10)
estilo_ejes(ejes_hp[1])
ejes_hp[1].xaxis.set_major_locator(mdates.AutoDateLocator())
ejes_hp[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
encabezado(fig_hp, 'Filtro 2: Hodrick-Prescott',
           f'Filtro bilateral (toda la muestra interviene): la tendencia NO admite pronóstico | '
           f'{Historia_txt}')
fuente(fig_hp)
fig_hp.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.10, hspace=0.25)
guardar_figura(fig_hp, '04_hodrick_prescott')

# Filtro 3: Holt-Winters (aditivo en log; suavizamiento alfa, beta, gamma)

Usar_log = bool((Serie > 0).all())
Base_HW = np.log(Serie) if Usar_log else Serie.copy()
Escala_HW = f'log({VARIABLE})' if Usar_log else VARIABLE

Modelo_HW = ExponentialSmoothing(Base_HW.to_numpy(), trend='add', seasonal='add',
                                 seasonal_periods=s, damped_trend=False,
                                 initialization_method='estimated')
Ajuste_HW = Modelo_HW.fit(optimized=True)
Parametros_HW = dict(Ajuste_HW.params)
Alfa, Beta, Gamma = (float(Parametros_HW.get(k, np.nan))
                     for k in ('smoothing_level', 'smoothing_trend', 'smoothing_seasonal'))
print('\nHolt-Winters estimado en', Escala_HW, '| alfa =', round(Alfa, 4),
      '| beta =', round(Beta, 4), '| gamma =', round(Gamma, 4),
      '| SSE =', float(Ajuste_HW.sse))

# Pronostico a 10 trimestres con bandas por simulacion (cuantiles 2.5 y 97.5)

Tendencia_prev = pd.Series(Ajuste_HW.fittedvalues, index=Serie.index, name='HW_ajuste')
try:
    rng_hw = np.random.default_rng(SEMILLA)
    Trayectorias = Ajuste_HW.simulate(HORIZONTE, anchor='end', repetitions=REPLICAS_HW,
                                      random_errors='bootstrap', random_state=rng_hw)
except Exception:
    rng_hw = np.random.default_rng(SEMILLA)
    Trayectorias = Ajuste_HW.simulate(HORIZONTE, anchor='end', repetitions=REPLICAS_HW,
                                      random_errors='gaussian', random_state=rng_hw)
Media_log = pd.Series(np.asarray(Trayectorias, dtype=float).mean(axis=1), index=Indice_pron)
Bandas_log = pd.DataFrame({
    'li': np.quantile(np.asarray(Trayectorias, dtype=float), ALPHA/2, axis=1),
    'ls': np.quantile(np.asarray(Trayectorias, dtype=float), 1-ALPHA/2, axis=1)},
    index=Indice_pron)

def volver(valor):
    return np.exp(pd.Series(valor)) if Usar_log else pd.Series(valor)

Pron_HW = volver(Media_log.to_numpy())
Pron_HW.index = Indice_pron
Pron_HW.name = 'HW_pronostico'
Pron_HW_li = volver(Bandas_log['li'].to_numpy())
Pron_HW_li.index = Indice_pron
Pron_HW_li.name = 'HW_li_95'
Pron_HW_ls = volver(Bandas_log['ls'].to_numpy())
Pron_HW_ls.index = Indice_pron
Pron_HW_ls.name = 'HW_ls_95'
Pronostico_HW = pd.concat([Pron_HW, Pron_HW_li, Pron_HW_ls], axis=1)
Pronostico_HW.to_csv(SALIDAS/'pronostico_Holt_Winters.csv', encoding='utf-8-sig')
print('\nPronóstico Holt-Winters a', HORIZONTE, 'trimestres (bandas simuladas',
      REPLICAS_HW, f'trayectorias, {int((1-ALPHA)*100)}%):')
print(Pronostico_HW.to_string())

# Grafico del ajuste y pronostico Holt-Winters

fig_hw, ax_hw = plt.subplots(figsize=(14, 6), dpi=500)
fig_hw.patch.set_facecolor('white')
Ajuste_niveles = volver(Tendencia_prev.to_numpy())
Ajuste_niveles.index = Serie.index
ax_hw.plot(Fechas_ts, Serie.to_numpy(), color=LINEA_SUAVE, linewidth=1.1, label='PIB original')
ax_hw.plot(Fechas_ts, Ajuste_niveles.to_numpy(), color=VERDE, linewidth=2.0,
           label='Holt-Winters ajustado (en log, back-transformado)')
ax_hw.fill_between(Ts_pron, Pron_HW_li.to_numpy(), Pron_HW_ls.to_numpy(),
                   color=NARANJA, alpha=.18, label=f'Banda {int((1-ALPHA)*100)}% (simulación)')
ax_hw.plot(Ts_pron, Pron_HW.to_numpy(), color=NARANJA, linewidth=2.2,
           linestyle='--', label=f'Pronóstico HW a {HORIZONTE} trimestres')
ax_hw.axvline(Serie.index[-1].to_timestamp(), color=GRIS, linewidth=0.9, linestyle=':', alpha=.6)
ax_hw.set_ylabel('PIB: Millones de pesos a precios del 2018', fontsize=11, color=GRIS)
ax_hw.set_xlabel('Año', fontsize=11, color=GRIS)
ax_hw.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_hw.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
estilo_ejes(ax_hw)
ax_hw.legend(loc='upper left', frameon=False, fontsize=10)
encabezado(fig_hw, 'Filtro 3: Holt-Winters y pronóstico',
           f'alfa={Alfa:.2f} | beta={Beta:.2f} | gamma={Gamma:.2f} | horizonte: {HORIZONTE} trimestres')
fuente(fig_hw)
fig_hw.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.14)
guardar_figura(fig_hw, '05_pronostico_holt_winters')

# Componentes de Holt-Winters (nivel, tendencia y estacional estimados)

Nivel_HW = volver(np.asarray(Ajuste_HW.level, dtype=float))
Nivel_HW.index = Serie.index
Nivel_HW.name = 'Nivel'
Tendencia_HW = volver(np.asarray(Ajuste_HW.trend, dtype=float))
if Usar_log:  # En log, la tendencia puntual se suma al nivel, no se exponencia sola.
    Tendencia_HW = pd.Series(np.asarray(Ajuste_HW.trend, dtype=float), index=Serie.index)
Tendencia_HW.name = 'Tendencia'
Estacional_HW = pd.Series(np.asarray(Ajuste_HW.season, dtype=float), index=Serie.index)
Estacional_HW.name = 'Estacional'

fig_hw_c, ejes_hwc = plt.subplots(3, 1, figsize=(13, 9), dpi=500, sharex=True)
fig_hw_c.patch.set_facecolor('white')
Componentes_hw = [('Nivel (suavizado)', Nivel_HW, VERDE, 'PIB niveles'),
                  ('Tendencia (beta)', Tendencia_HW, NARANJA, 'Aporte por periodo'),
                  (f'Estacional (gamma, s = {s})', Estacional_HW, GRIS, 'Efecto en escala de estimación')]
for ax, (titulo, obj, color, ylabel) in zip(ejes_hwc, Componentes_hw):
    ax.plot(Fechas_ts, obj.to_numpy(), color=color, linewidth=1.8)
    ax.set_title(titulo, fontsize=11, loc='left', color=GRIS)
    ax.set_ylabel(ylabel, fontsize=10, color=GRIS)
    estilo_ejes(ax)
ejes_hwc[-1].set_xlabel('Año', fontsize=11, color=GRIS)
ejes_hwc[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
ejes_hwc[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
encabezado(fig_hw_c, 'Componentes de Holt-Winters',
           f'Nivel, tendencia y estacionalidad estimados sobre {Escala_HW} | {Historia_txt}')
fuente(fig_hw_c)
fig_hw_c.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.10, hspace=0.45)
guardar_figura(fig_hw_c, '06_componentes_holt_winters')

# Consolidado de filtros y pronosticos

Tablas = pd.DataFrame({'PIB': Serie, 'MA_centrada_2x4': MA_centrada,
                       'MA_no_centrada': MA_no_centrada, 'Tendencia_HP': Tendencia_HP,
                       'Ciclo_HP': Ciclo_HP, 'Tendencia_HP_RU': Tendencia_HP_RU,
                       'HW_ajuste': Ajuste_niveles})
Tablas.to_csv(SALIDAS/'filtros_consolidado.csv', encoding='utf-8-sig')
Resumen = pd.DataFrame({
    'filtro': ['MA no centrada', 'MA centrada 2x4', 'Hodrick-Prescott', 'Holt-Winters'],
    'pronostico': ['Sí (plano en la última media)', 'No (usa observaciones futuras)',
                   'No (filtro bilateral, toda la muestra)',
                   f'Sí (banda simulada {int((1-ALPHA)*100)}%)'],
    'horizonte': [HORIZONTE, np.nan, np.nan, HORIZONTE],
    'primer_periodo': [str(Pron_MA.index[0]), np.nan, np.nan,
                       str(Pron_HW.index[0])],
    'ultimo_periodo': [str(Pron_MA.index[-1]), np.nan, np.nan,
                       str(Pron_HW.index[-1])]})
Resumen.to_csv(SALIDAS/'resumen_filtros_pronostico.csv', index=False, encoding='utf-8-sig')
print('\nResumen de filtros:\n', Resumen.to_string(index=False))
print('\nSalidas en:', SALIDAS)
