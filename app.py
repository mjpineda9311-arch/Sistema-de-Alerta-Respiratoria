import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.figure import Figure

# Configuración inicial de la página
st.set_page_config(
    page_title="Sistema de Alerta Respiratoria", 
    layout="wide",
    page_icon="🫁"
)

# Constantes del dataset BIDMC
SUJETOS = [f"{i:02d}" for i in range(1, 54)]
CARPETA_DATOS = "bidmc_data"
BASE_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"
FS_SIG = 125
VENTANA_S = 30
RR_MIN, RR_MAX = 12, 20
SPO2_MIN = 95
RELLENO_FFT = 8
MAX_RMSSD_RELATIVO = 0.15

# Cargar modelo y configuración con respaldo automático
@st.cache_resource
def cargar_recursos():
    modelo = None
    if os.path.exists('modelo_alerta_respiratoria.joblib'):
        modelo = joblib.load('modelo_alerta_respiratoria.joblib')
    
    cfg = {
        'modelo': 'Decision Tree',
        'auc_por_paciente': 0.539,
        'umbral': 0.373,
        'features': ['rr_fuera_rango', 'rr_rpm'],
        'n_confirmar': 2,
        'n_liberar': 2
    }
    
    if os.path.exists('config_alerta_respiratoria.json'):
        with open('config_alerta_respiratoria.json', 'r', encoding='utf-8') as f:
            cfg_disk = json.load(f)
            cfg.update(cfg_disk)
            
    return modelo, cfg

modelo, cfg = cargar_recursos()

if modelo is None:
    st.error("No se encontró el archivo 'modelo_alerta_respiratoria.joblib' en la carpeta del proyecto.")
    st.stop()

MEJOR_MODELO = cfg.get('modelo', 'Decision Tree')
AUC_MEJOR = cfg.get('auc_por_paciente', 0.539)
UMBRAL_FINAL = cfg.get('umbral', 0.373)
FEATURES = cfg.get('features', ['rr_fuera_rango', 'rr_rpm'])

# Funciones de procesamiento de señales
def filtro_pasa_banda(signal, fs, low=0.5, high=8.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def normalizar(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def agregar_features(m):
    rr = m.get('rr_rpm', np.nan)
    m['rr_fuera_rango'] = np.nan if rr is None or pd.isna(rr) else max(0.0, RR_MIN - rr) + max(0.0, rr - RR_MAX)
    return m

def extraer_metricas_ventana(ppg_win, fs):
    picos, _ = find_peaks(ppg_win, distance=fs*0.4)
    hr, sdnn, rmssd = np.nan, np.nan, np.nan
    if len(picos) > 2:
        intervalos_s = np.diff(picos) / fs
        hr = 60 / intervalos_s.mean()
        intervalos_ms = intervalos_s * 1000
        sdnn = np.std(intervalos_ms, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(intervalos_ms) ** 2))

    rr = np.nan
    if len(picos) > 3:
        amp_picos = ppg_win[picos]
        t_picos = picos / fs
        t_uniforme = np.arange(0, len(ppg_win) / fs, 1 / fs)
        amp_interp = np.interp(t_uniforme, t_picos, amp_picos)
        amp_interp = amp_interp - amp_interp.mean()
        n_fft = RELLENO_FFT * len(amp_interp)
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(amp_interp, n=n_fft))
        banda_resp = (freqs >= 0.1) & (freqs <= 0.6)
        if banda_resp.sum() > 0:
            rr = freqs[banda_resp][np.argmax(fft_vals[banda_resp])] * 60

    return agregar_features({'hr_bpm': hr, 'hrv_sdnn_ms': sdnn, 'hrv_rmssd_ms': rmssd, 'rr_rpm': rr})

def calidad_ok(m):
    hr, rmssd = m.get('hr_bpm'), m.get('hrv_rmssd_ms')
    if hr is None or pd.isna(hr) or not (30 <= hr <= 200): return False
    if rmssd is None or pd.isna(rmssd): return False
    return rmssd / (60000 / hr) < MAX_RMSSD_RELATIVO

# Descarga y lectura de datos BIDMC
def descargar_archivo(nombre):
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    ruta_local = os.path.join(CARPETA_DATOS, nombre)
    if os.path.exists(ruta_local):
        return ruta_local
    import requests
    r = requests.get(BASE_URL + nombre, timeout=60)
    r.raise_for_status()
    with open(ruta_local, 'wb') as f:
        f.write(r.content)
    return ruta_local

def cargar_ppg(sid):
    descargar_archivo(f"bidmc_{sid}_Signals.csv")
    df = pd.read_csv(os.path.join(CARPETA_DATOS, f"bidmc_{sid}_Signals.csv"))
    df.columns = [c.strip() for c in df.columns]
    return df['PLETH'].values

def cargar_referencia(sid):
    descargar_archivo(f"bidmc_{sid}_Numerics.csv")
    df = pd.read_csv(os.path.join(CARPETA_DATOS, f"bidmc_{sid}_Numerics.csv"))
    df.columns = [c.strip() for c in df.columns]
    return df

# Clase del Sistema de Alerta con registro de eventos
class SistemaAlertaRespiratoria:
    def __init__(self, modelo, umbral, paciente='?'):
        self.modelo, self.umbral, self.paciente = modelo, umbral, paciente
        self.reset()

    def reset(self):
        self.alarma, self.critico = False, False
        self._pos, self._neg = 0, 0
        self.historial = []
        self.eventos = []

    def actualizar_metricas(self, metricas, t=None, spo2=None):
        metricas = agregar_features(dict(metricas))
        rr, hr = metricas.get('rr_rpm', np.nan), metricas.get('hr_bpm', np.nan)
        ok = calidad_ok(metricas)
        prob, positivo = np.nan, None
        
        if ok:
            fila = pd.DataFrame([{f: metricas.get(f, np.nan) for f in FEATURES}])
            prob = float(self.modelo.predict_proba(fila)[0, 1])
            positivo = prob >= self.umbral or (spo2 is not None and not pd.isna(spo2) and spo2 < SPO2_MIN)
            if positivo:
                self._pos, self._neg = self._pos + 1, 0
            else:
                self._neg, self._pos = self._neg + 1, 0

        estado = {'t_s': t, 'rr_rpm': rr, 'hr_bpm': hr, 'spo2': spo2, 'probabilidad': prob,
                  'calidad': 'OK' if ok else 'Señal pobre', 'positivo': positivo}

        # Detección de cambios de estado para el registro de alertas
        anterior_alarma = self.alarma
        if not self.alarma and self._pos >= cfg.get('n_confirmar', 2):
            self.alarma = True
            self.eventos.append({'tipo': 'ALERTA', 't_s': t, 'rr': rr, 'prob': prob})
        elif self.alarma and self._neg >= cfg.get('n_liberar', 2):
            self.alarma, self.critico = False, False
            self.eventos.append({'tipo': 'FIN DE ALERTA', 't_s': t, 'rr': rr, 'prob': prob})

        if self.critico: nivel = 'CRÍTICO'
        elif self.alarma: nivel = 'ALERTA'
        elif positivo: nivel = 'Vigilancia'
        else: nivel = 'Normal'
        
        estado.update(alarma=int(self.alarma), nivel=nivel)
        self.historial.append(estado)
        return estado

    def actualizar(self, ppg_ventana_cruda, fs, t=None, spo2=None):
        x = normalizar(filtro_pasa_banda(np.asarray(ppg_ventana_cruda, dtype=float), fs))
        return self.actualizar_metricas(extraer_metricas_ventana(x, fs), t=t, spo2=spo2)

    def tabla(self):
        return pd.DataFrame(self.historial)

# Generador de gráficas de 3 paneles
COLORES_NIVEL = {'Normal': '#2e7d32', 'Vigilancia': '#f9a825', 'ALERTA': '#ef6c00', 'CRÍTICO': '#c62828'}

def figura_png(hist, umbral, titulo):
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots(3, 1, sharex=True)
    if len(hist):
        t = hist['t_s']
        ax[0].plot(t, hist['rr_rpm'], 'k-o', ms=4)
        ax[1].plot(t, hist['probabilidad'], 'b-o', ms=4)
        ax[2].bar(t, 1, width=VENTANA_S * 0.95, align='edge', color=hist['nivel'].map(COLORES_NIVEL))
    ax[0].axhspan(RR_MIN, RR_MAX, color='green', alpha=0.1)
    ax[0].set_ylabel('RR (rpm)')
    ax[1].axhline(umbral, color='red', ls='--', lw=1, label='Umbral')
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel('P(riesgo)')
    ax[1].legend(loc='upper right', fontsize=8)
    ax[2].set_yticks([])
    ax[2].set_ylabel('Nivel')
    ax[2].set_xlabel('Tiempo (s)')
    ax[0].set_title(titulo, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    return buf.getvalue()

# Interfaz visual principal en Streamlit
st.markdown(f"""
    <div style='font-family:sans-serif'>
        <h2 style='margin:4px 0'>🫁 Sistema de alerta respiratoria (PPG + {MEJOR_MODELO})</h2>
        <span style='color:#555'>Modelo: <b>{MEJOR_MODELO}</b> · AUC por paciente: <b>{AUC_MEJOR:.3f}</b> · 
        umbral de Youden: <b>{UMBRAL_FINAL:.3f}</b> · 53 pacientes BIDMC configurados</span>
    </div>
    <hr style='margin-top: 10px; margin-bottom: 20px;'>
""", unsafe_allow_html=True)

# 4 pestañas restauradas completamente
tab_mon, tab_man, tab_ana, tab_cfg = st.tabs(["Monitor en vivo", "Evaluación manual", "Analizar mi señal", "Configuración"])

# 1. Pestaña Monitor en vivo
with tab_mon:
    col1, col2 = st.columns([2, 1])
    with col1:
        paciente_sel = st.selectbox("Seleccionar Paciente BIDMC", SUJETOS, index=0)
    with col2:
        simular_spo2 = st.checkbox("Simular sensor de SpO2 con monitor")

    if st.button("▶ Iniciar Monitoreo del Paciente", type="primary"):
        with st.spinner(f"Descargando y procesando señales del paciente {paciente_sel}..."):
            try:
                ppg_stream = cargar_ppg(paciente_sel)
                num = cargar_referencia(paciente_sel)
                tam = VENTANA_S * FS_SIG
                
                sistema = SistemaAlertaRespiratoria(modelo, UMBRAL_FINAL, paciente=paciente_sel)
                
                for w in range(len(ppg_stream) // tam):
                    spo2_val = None
                    if simular_spo2:
                        sub_num = num.loc[(num['Time [s]'] >= w*VENTANA_S) & (num['Time [s]'] < (w+1)*VENTANA_S), 'SpO2']
                        if not sub_num.empty:
                            spo2_val = float(sub_num.mean())
                    sistema.actualizar(ppg_stream[w*tam:(w+1)*tam], FS_SIG, t=w * VENTANA_S, spo2=spo2_val)
                
                hist_df = sistema.tabla()
                st.success(f"Monitoreo completado para el paciente {paciente_sel}. Ventanas analizadas: {len(hist_df)}")
                
                # Gráfica de 3 paneles
                img_bytes = figura_png(hist_df, UMBRAL_FINAL, f'Paciente {paciente_sel} — {MEJOR_MODELO}')
                st.image(img_bytes, use_container_width=True)
                
                # Reporte / Registro de alertas en vivo detectadas
                st.markdown("### 📋 Registro de Alertas del Monitoreo")
                if sistema.eventos:
                    for ev in sistema.eventos:
                        icono = "🟠" if ev['tipo'] == 'ALERTA' else "🟢"
                        st.markdown(f"{icono} **{ev['tipo']}** en $t = {ev['t_s']}s$ — RR: `{ev['rr']:.1f} rpm` | Prob: `{ev['prob']:.3f}`")
                else:
                    st.info("No se registraron cambios de estado crítico o alertas en este paciente durante la simulación.")
                
                with st.expander("Ver tabla detallada de ventanas"):
                    st.dataframe(hist_df)
                    
            except Exception as e:
                st.error(f"Error al procesar los datos del paciente: {e}")

# 2. Pestaña Evaluación manual (Reparada)
with tab_man:
    st.subheader("Simulador de Parámetros Clínicos Individuales")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        rr_val = st.slider("Frecuencia Respiratoria (RR - rpm)", 4.0, 40.0, 16.0, 0.5)
        hr_val = st.slider("Frecuencia Cardíaca (HR - lpm)", 30.0, 180.0, 80.0, 1.0)
        sdnn_val = st.slider("SDNN (ms)", 0.0, 200.0, 30.0, 1.0)
    with c_m2:
        rmssd_val = st.slider("RMSSD (ms)", 0.0, 200.0, 25.0, 1.0)
        usar_spo2_man = st.checkbox("Incluir SpO2 (Evaluación manual)")
        spo2_val = st.slider("SpO2 (%)", 80.0, 100.0, 97.0, 0.5, disabled=not usar_spo2_man)

    m_dict = agregar_features({'rr_rpm': rr_val, 'hr_bpm': hr_val, 'hrv_sdnn_ms': sdnn_val, 'hrv_rmssd_ms': rmssd_val})
    fila_pred = pd.DataFrame([{f: m_dict[f] for f in FEATURES}])
    prob_val = float(modelo.predict_proba(fila_pred)[0, 1])
    
    es_positivo = prob_val >= UMBRAL_FINAL or (usar_spo2_man and spo2_val < SPO2_MIN)
    color_tarjeta = "#d32f2f" if es_positivo else "#2e7d32"
    estado_texto = "🚨 ALERTA DE RIESGO RESPIRATORIO" if es_positivo else "🟢 ESTADO NORMAL"

    st.markdown(f"""
        <div style='border-radius:10px; padding:16px 20px; background:{color_tarjeta}; color:white; margin-top:20px;'>
            <div style='font-size:22px; font-weight:bold;'>{estado_texto}</div>
            <div style='margin-top:6px;'>Probabilidad de riesgo ({MEJOR_MODELO}): <b>{prob_val:.3f}</b> · Umbral Youden: {UMBRAL_FINAL:.3f}</div>
        </div>
    """, unsafe_allow_html=True)

# 3. Pestaña Analizar mi señal (Restaurada)
with tab_ana:
    st.subheader("Análisis por lote de archivos de señal PPG personalizados")
    st.markdown("Sube un archivo CSV con una columna de PPG crudo. El sistema dividirá la señal en ventanas de 30 segundos.")
    
    archivo_subido = st.file_uploader("Subir archivo de registro (.csv)", type=["csv"])
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        txt_columna = st.text_input("Nombre de columna PPG", value="PLETH")
    with col_a2:
        int_fs = st.number_input("Frecuencia de muestreo (Hz)", value=125, step=1)

    if st.button("Analizar Señal Subida", type="primary"):
        if archivo_subido is not None:
            try:
                df_sub = pd.read_csv(archivo_subido)
                df_sub.columns = [str(c).strip() for c in df_sub.columns]
                col = txt_columna.strip()
                
                if col not in df_sub.columns:
                    st.error(f"No se encontró la columna '{col}' en el archivo CSV. Columnas disponibles: {list(df_sub.columns)}")
                else:
                    ppg_arr = df_sub[col].astype(float).interpolate().bfill().values
                    fs_val = int(int_fs)
                    tam_ventana = VENTANA_S * fs_val
                    
                    if len(ppg_arr) < tam_ventana:
                        st.error(f"La señal es demasiado corta ({len(ppg_arr)/fs_val:.1f}s). Se requieren al menos {VENTANA_S} segundos.")
                    else:
                        sistema_custom = SistemaAlertaRespiratoria(modelo, UMBRAL_FINAL, paciente="Personalizado")
                        for w in range(len(ppg_arr) // tam_ventana):
                            sistema_custom.actualizar(ppg_arr[w*tam_ventana:(w+1)*tam_ventana], fs_val, t=w * VENTANA_S)
                        
                        hist_custom = sistema_custom.tabla()
                        st.success(f"Análisis completado. Ventanas analizadas: {len(hist_custom)}")
                        
                        img_custom = figura_png(hist_custom, UMBRAL_FINAL, f'Señal Personalizada — {MEJOR_MODELO}')
                        st.image(img_custom, use_container_width=True)
            except Exception as ex:
                st.error(f"Ocurrió un error al procesar el archivo: {ex}")
        else:
            st.warning("Por favor, sube un archivo CSV válido primero.")

# 4. Pestaña Configuración
with tab_cfg:
    st.subheader("Información del Sistema Entrenado")
    st.write(f"* **Modelo activo:** {MEJOR_MODELO}")
    st.write(f"* **AUC obtenido:** {AUC_MEJOR:.3f}")
    st.write(f"* **Umbral de decisión:** {UMBRAL_FINAL:.3f}")
    st.write(f"* **Variables de entrada utilizadas:** {', '.join(FEATURES)}")