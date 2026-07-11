"""
============================================================================
 api_irm.py  -  Backend IRM (Indice de Riesgo Minero) - PULSARIX
============================================================================
 API en FastAPI que:
   1. Recibe lecturas del ESP32 por WiFi (HTTP POST)
   2. Las guarda agrupadas por muestra_id en lecturas_esp32.csv
   3. Al finalizar la muestra calcula 18 caracteristicas, valida la
      exhalacion con CO2, aplica el modelo de regresion logistica entrenado
      (modelo_IRM_prototipo.pkl) y devuelve el IRM + nivel de riesgo
   4. Guarda el resultado en resultados_irm.csv
   5. La app celular consulta el ultimo resultado o uno por muestra_id

 Correr local:
   pip install -r requirements.txt
   uvicorn api_irm:app --host 0.0.0.0 --port 8000 --reload

 Correr en la nube (Render / Railway):
   uvicorn api_irm:app --host 0.0.0.0 --port $PORT
============================================================================
"""

import os
import csv
import threading
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from caracteristicas import extraer_caracteristicas, COLUMNAS_MODELO

# ----------------------------------------------------------------------
# Rutas de archivos (siempre relativas a la carpeta de este script)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_IRM_prototipo.pkl")
RUTA_LECTURAS = os.path.join(BASE_DIR, "lecturas_esp32.csv")
RUTA_RESULTADOS = os.path.join(BASE_DIR, "resultados_irm.csv")

# Umbral de CO2 para considerar que SI hubo exhalacion
CO2_DELTA_MINIMO = 500  # ppm

# ======================================================================
# ==========   MODO DE CALCULO DEL IRM   (IMPORTANTE)   =================
# ======================================================================
# "relativo" -> calcula el IRM con el CAMBIO RELATIVO de TUS sensores
#               (BME688 y CO2) respecto a la linea base de la propia
#               muestra. RESPONDE al soplido: cada exhalacion da un valor
#               distinto y salen Bajo/Medio/Alto. Ideal para la demo en vivo.
#               NO depende del modelo .pkl, asi que nunca se rompe.
#
# "modelo"   -> usa tu modelo entrenado (modelo_IRM_prototipo.pkl).
#               Sirve para mostrar la parte de machine learning, pero con
#               datos reales del prototipo puede saturarse (solo Bajo/Alto).
#
# Para la SUSTENTACION EN VIVO deja "relativo".
MODO_IRM = "relativo"

# Pesos de la formula del IRM relativo (tu formula original PULSARIX)
PESO_VOC = 0.6
PESO_CO2 = 0.4
# La %CO2 sube muchisimo al exhalar; la suavizamos para que no domine todo
ESCALA_CO2 = 3.0
# Umbrales de riesgo (0-100): <=BAJO Bajo, <=MEDIO Medio, resto Alto
UMBRAL_BAJO = 33
UMBRAL_MEDIO = 66
# ======================================================================

MENSAJE_SEGURIDAD = (
    "Este resultado no representa un diagnostico medico. "
    "Es una estimacion preliminar basada en senales de sensores de aliento."
)

# Columnas de los CSV
COLUMNAS_LECTURAS = [
    "participante_id", "muestra_id", "clase", "condicion", "indice", "tiempo_ms",
    "BME688_gas_kohm", "BME688_temp_C", "BME688_humidity_pct", "BME688_pressure_hPa",
    "MiCS6814_NH3_v", "MiCS6814_RED_v", "MiCS6814_OX_v", "MHZ19_CO2_ppm",
]
COLUMNAS_RESULTADOS = [
    "muestra_id", "participante_id", "muestra_valida", "CO2_delta",
    "IRM", "riesgo", "mensaje",
]

# Lock para que escrituras simultaneas al CSV no se pisen
_lock = threading.Lock()

# ----------------------------------------------------------------------
# Carga del modelo (una sola vez, al arrancar)
# ----------------------------------------------------------------------
modelo = None
modelo_error = None
try:
    if os.path.exists(RUTA_MODELO):
        modelo = joblib.load(RUTA_MODELO)
        print(f"[OK] Modelo cargado desde {RUTA_MODELO}")
    else:
        modelo_error = (
            f"No se encontro el modelo en {RUTA_MODELO}. "
            f"Coloca modelo_IRM_prototipo.pkl en esta carpeta "
            f"(o corre primero: python entrenar_modelo_demo.py)."
        )
        print(f"[ADVERTENCIA] {modelo_error}")
except Exception as e:  # pragma: no cover
    modelo_error = f"Error cargando el modelo: {e}"
    print(f"[ERROR] {modelo_error}")


def _indice_clase_positiva():
    """Devuelve el indice de la clase 'patron alterado' (clase 1) en model.classes_."""
    try:
        clases = list(modelo.classes_)
        if 1 in clases:
            return clases.index(1)
    except Exception:
        pass
    # por defecto, la segunda columna de predict_proba
    return 1


def _crear_csv_si_no_existe(ruta, columnas):
    if not os.path.exists(ruta):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columnas)


def _clasificar_riesgo(irm):
    if irm <= UMBRAL_BAJO:
        return "Bajo", "Medicion dentro del rango bajo de alerta respiratoria"
    if irm <= UMBRAL_MEDIO:
        return "Medio", "Medicion en rango medio. Se sugiere repetir y observar."
    return "Alto", "Medicion en rango alto. Se sugiere evaluacion adicional."


def _calcular_irm_relativo(df_muestra):
    """
    Calcula el IRM con el CAMBIO RELATIVO de los sensores propios respecto
    a la linea base de la muestra (primera lectura). Responde al soplido.
      IRM = PESO_VOC * dVOC%  +  PESO_CO2 * (dCO2% / ESCALA_CO2)
    Devuelve el IRM (0-100), dVOC% y dCO2%.
    """
    df = df_muestra.copy()
    if "indice" in df.columns:
        df = df.sort_values("indice")
    elif "tiempo_ms" in df.columns:
        df = df.sort_values("tiempo_ms")

    # VOC: BME688 gas (kOhm). Cambio maximo respecto al valor inicial.
    dVOC = 0.0
    if "BME688_gas_kohm" in df.columns and len(df) > 0:
        gas = df["BME688_gas_kohm"].astype(float)
        gas0 = gas.iloc[0]
        if gas0 and gas0 > 0:
            dVOC = float((gas - gas0).abs().max() / gas0 * 100.0)

    # CO2: subida respecto al valor inicial.
    dCO2 = 0.0
    if "MHZ19_CO2_ppm" in df.columns and len(df) > 0:
        co2 = df["MHZ19_CO2_ppm"].astype(float)
        co2_0 = co2.iloc[0]
        if co2_0 and co2_0 > 0:
            dCO2 = float((co2.max() - co2_0) / co2_0 * 100.0)

    irm = PESO_VOC * dVOC + PESO_CO2 * (dCO2 / ESCALA_CO2)
    irm = max(0.0, min(100.0, irm))   # dejar entre 0 y 100
    return round(irm, 1), round(dVOC, 1), round(dCO2, 1)


# Asegurar que los CSV existan al arrancar
_crear_csv_si_no_existe(RUTA_LECTURAS, COLUMNAS_LECTURAS)
_crear_csv_si_no_existe(RUTA_RESULTADOS, COLUMNAS_RESULTADOS)

# ----------------------------------------------------------------------
# Modelos de datos (validacion automatica de los JSON que llegan)
# ----------------------------------------------------------------------
class Lectura(BaseModel):
    participante_id: str
    muestra_id: str
    clase: Optional[int] = None
    condicion: Optional[str] = ""
    indice: Optional[int] = 0
    tiempo_ms: Optional[float] = 0
    BME688_gas_kohm: float
    BME688_temp_C: Optional[float] = None
    BME688_humidity_pct: Optional[float] = None
    BME688_pressure_hPa: Optional[float] = None
    MiCS6814_NH3_v: float
    MiCS6814_RED_v: float
    MiCS6814_OX_v: Optional[float] = None
    MHZ19_CO2_ppm: float


class FinalizarMuestra(BaseModel):
    muestra_id: str = Field(..., description="ID de la muestra a finalizar")


# --- Modelos para enviar TODA la muestra en UNA sola peticion (mas rapido) ---
class LecturaBatch(BaseModel):
    indice: Optional[int] = 0
    tiempo_ms: Optional[float] = 0
    BME688_gas_kohm: float
    BME688_temp_C: Optional[float] = None
    BME688_humidity_pct: Optional[float] = None
    BME688_pressure_hPa: Optional[float] = None
    MiCS6814_NH3_v: float
    MiCS6814_RED_v: float
    MiCS6814_OX_v: Optional[float] = None
    MHZ19_CO2_ppm: float


class MuestraCompleta(BaseModel):
    participante_id: str
    muestra_id: str
    clase: Optional[int] = None
    condicion: Optional[str] = ""
    lecturas: List[LecturaBatch]


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
app = FastAPI(
    title="API IRM - PULSARIX",
    description="Backend que recibe lecturas del ESP32, calcula el IRM y lo entrega a la app.",
    version="1.0.0",
)

# CORS abierto para que la app celular pueda consultar sin problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# 1. GET /  -> ping de salud
# ----------------------------------------------------------------------
@app.get("/")
def raiz():
    return {
        "estado": "API IRM funcionando",
        "mensaje": "Backend listo para recibir datos del ESP32",
    }


# ----------------------------------------------------------------------
# 2. POST /guardar_lectura  -> guarda una lectura del ESP32
# ----------------------------------------------------------------------
@app.post("/guardar_lectura")
def guardar_lectura(lectura: Lectura):
    try:
        fila = [
            lectura.participante_id,
            lectura.muestra_id,
            lectura.clase if lectura.clase is not None else "",
            lectura.condicion or "",
            lectura.indice if lectura.indice is not None else "",
            lectura.tiempo_ms if lectura.tiempo_ms is not None else "",
            lectura.BME688_gas_kohm,
            lectura.BME688_temp_C if lectura.BME688_temp_C is not None else "",
            lectura.BME688_humidity_pct if lectura.BME688_humidity_pct is not None else "",
            lectura.BME688_pressure_hPa if lectura.BME688_pressure_hPa is not None else "",
            lectura.MiCS6814_NH3_v,
            lectura.MiCS6814_RED_v,
            lectura.MiCS6814_OX_v if lectura.MiCS6814_OX_v is not None else "",
            lectura.MHZ19_CO2_ppm,
        ]
        with _lock:
            _crear_csv_si_no_existe(RUTA_LECTURAS, COLUMNAS_LECTURAS)
            with open(RUTA_LECTURAS, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fila)

        return {
            "estado": "ok",
            "mensaje": "Lectura guardada",
            "muestra_id": lectura.muestra_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando la lectura: {e}")


# ----------------------------------------------------------------------
# 3. POST /finalizar_muestra  -> calcula features, valida CO2 y predice IRM
# ----------------------------------------------------------------------
@app.post("/finalizar_muestra")
def finalizar_muestra(datos: FinalizarMuestra):
    muestra_id = datos.muestra_id

    # 3.1 Leer todas las lecturas de esa muestra
    try:
        df = pd.read_csv(RUTA_LECTURAS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo leer {RUTA_LECTURAS}: {e}")

    df_muestra = df[df["muestra_id"] == muestra_id]
    if df_muestra.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay lecturas para la muestra '{muestra_id}'.",
        )

    participante_id = str(df_muestra["participante_id"].iloc[0])
    return _calcular_resultado(df_muestra, muestra_id, participante_id)


def _calcular_resultado(df_muestra, muestra_id, participante_id):
    """Calcula features, valida CO2, aplica el modelo y guarda el resultado."""
    # Calcular las 18 caracteristicas + CO2
    try:
        features, co2 = extraer_caracteristicas(df_muestra)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando caracteristicas: {e}")

    co2_delta = round(co2["delta"], 1)

    # Validar exhalacion con CO2
    if co2_delta < CO2_DELTA_MINIMO:
        resultado = {
            "muestra_id": muestra_id,
            "muestra_valida": False,
            "CO2_delta": co2_delta,
            "IRM": None,
            "riesgo": "No calculado",
            "mensaje": "No se detecto una exhalacion adecuada. Repetir la medicion.",
        }
        _guardar_resultado(resultado, participante_id)
        return resultado

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=modelo_error or "Modelo no disponible. Coloca modelo_IRM_prototipo.pkl.",
        )

    try:
        X = pd.DataFrame([[features[c] for c in COLUMNAS_MODELO]], columns=COLUMNAS_MODELO)
        proba = modelo.predict_proba(X)[0]
        irm = float(proba[_indice_clase_positiva()] * 100.0)
        irm = round(irm, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aplicando el modelo: {e}")

    riesgo, mensaje_riesgo = _clasificar_riesgo(irm)
    resultado = {
        "muestra_id": muestra_id,
        "muestra_valida": True,
        "CO2_delta": co2_delta,
        "IRM": irm,
        "riesgo": riesgo,
        "mensaje": mensaje_riesgo,
    }
    _guardar_resultado(resultado, participante_id)
    return resultado


# ----------------------------------------------------------------------
# 3-bis. POST /procesar_muestra  -> recibe TODA la muestra de una vez
#         (guarda + calcula IRM en UNA sola peticion). MAS RAPIDO para el ESP32.
# ----------------------------------------------------------------------
@app.post("/procesar_muestra")
def procesar_muestra(muestra: MuestraCompleta):
    if not muestra.lecturas:
        raise HTTPException(status_code=400, detail="La muestra no tiene lecturas.")

    # Construir el DataFrame directo desde las lecturas (sin releer el CSV)
    filas = []
    for i, lec in enumerate(muestra.lecturas):
        filas.append({
            "participante_id": muestra.participante_id,
            "muestra_id": muestra.muestra_id,
            "clase": muestra.clase if muestra.clase is not None else "",
            "condicion": muestra.condicion or "",
            "indice": lec.indice if lec.indice is not None else i,
            "tiempo_ms": lec.tiempo_ms if lec.tiempo_ms is not None else 0,
            "BME688_gas_kohm": lec.BME688_gas_kohm,
            "BME688_temp_C": lec.BME688_temp_C,
            "BME688_humidity_pct": lec.BME688_humidity_pct,
            "BME688_pressure_hPa": lec.BME688_pressure_hPa,
            "MiCS6814_NH3_v": lec.MiCS6814_NH3_v,
            "MiCS6814_RED_v": lec.MiCS6814_RED_v,
            "MiCS6814_OX_v": lec.MiCS6814_OX_v,
            "MHZ19_CO2_ppm": lec.MHZ19_CO2_ppm,
        })
    df_muestra = pd.DataFrame(filas)

    # Guardar todas las lecturas en el CSV (para el registro) en UN solo bloque
    try:
        with _lock:
            _crear_csv_si_no_existe(RUTA_LECTURAS, COLUMNAS_LECTURAS)
            df_muestra[COLUMNAS_LECTURAS].to_csv(
                RUTA_LECTURAS, mode="a", header=False, index=False
            )
    except Exception as e:
        # aunque falle guardar, igual calculamos el IRM
        print(f"[ADVERTENCIA] no se pudo guardar en CSV: {e}")

    return _calcular_resultado(df_muestra, muestra.muestra_id, muestra.participante_id)


def _guardar_resultado(resultado, participante_id):
    fila = [
        resultado["muestra_id"],
        participante_id,
        resultado["muestra_valida"],
        resultado["CO2_delta"],
        resultado["IRM"] if resultado["IRM"] is not None else "",
        resultado["riesgo"],
        resultado["mensaje"],
    ]
    with _lock:
        _crear_csv_si_no_existe(RUTA_RESULTADOS, COLUMNAS_RESULTADOS)
        with open(RUTA_RESULTADOS, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fila)


def _limpio(v):
    """Convierte NaN/None de pandas en None (JSON-safe) y deja el resto igual."""
    if v is None:
        return None
    try:
        # pd.isna detecta NaN tanto en floats como en strings vacios leidos como NaN
        if isinstance(v, float) and (v != v):  # NaN
            return None
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _fila_resultado_a_json(fila):
    irm = _limpio(fila.get("IRM"))
    try:
        irm = float(irm) if irm is not None else None
    except (TypeError, ValueError):
        irm = None

    co2 = _limpio(fila.get("CO2_delta"))
    try:
        co2 = float(co2) if co2 is not None else None
    except (TypeError, ValueError):
        co2 = None

    valida = str(_limpio(fila.get("muestra_valida"))).strip().lower() in ("true", "1", "verdadero")
    return {
        "muestra_id": _limpio(fila.get("muestra_id")),
        "participante_id": _limpio(fila.get("participante_id")),
        "muestra_valida": valida,
        "CO2_delta": co2,
        "IRM": irm,
        "riesgo": _limpio(fila.get("riesgo")),
        "mensaje": _limpio(fila.get("mensaje")),
        "aviso_seguridad": MENSAJE_SEGURIDAD,
    }


# ----------------------------------------------------------------------
# 4. GET /ultimo_resultado  -> ultimo resultado guardado (lo usa la app)
# ----------------------------------------------------------------------
@app.get("/ultimo_resultado")
def ultimo_resultado():
    try:
        df = pd.read_csv(RUTA_RESULTADOS)
    except Exception:
        df = pd.DataFrame(columns=COLUMNAS_RESULTADOS)

    if df.empty:
        raise HTTPException(status_code=404, detail="Aun no hay resultados guardados.")

    fila = df.iloc[-1].to_dict()
    return _fila_resultado_a_json(fila)


# ----------------------------------------------------------------------
# 5. GET /resultado/{muestra_id}  -> resultado de una muestra especifica
# ----------------------------------------------------------------------
@app.get("/resultado/{muestra_id}")
def resultado_por_muestra(muestra_id: str):
    try:
        df = pd.read_csv(RUTA_RESULTADOS)
    except Exception:
        df = pd.DataFrame(columns=COLUMNAS_RESULTADOS)

    df_m = df[df["muestra_id"] == muestra_id]
    if df_m.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay resultado para la muestra '{muestra_id}'.",
        )

    fila = df_m.iloc[-1].to_dict()
    return _fila_resultado_a_json(fila)


# ----------------------------------------------------------------------
# ESTADO EN VIVO (buzon para el Monitor de la app)
# ----------------------------------------------------------------------
# El ESP32 deja el estado actual con POST /estado_vivo (cada ~1 seg).
# La app lo recoge con GET /estado_vivo para mostrarlo en vivo.
# Se guarda en memoria (el ultimo estado). No necesita persistir.
# ----------------------------------------------------------------------
_estado_vivo = {}


@app.post("/estado_vivo")
def guardar_estado_vivo(datos: dict):
    """Recibe el estado en vivo del ESP32 y lo guarda (sobrescribe el anterior)."""
    global _estado_vivo
    _estado_vivo = datos or {}
    return {"estado": "ok", "mensaje": "Estado en vivo actualizado"}


@app.get("/estado_vivo")
def obtener_estado_vivo():
    """Devuelve el ultimo estado en vivo (lo consulta la app cada segundo)."""
    if not _estado_vivo:
        return {
            "estado": "sin_datos",
            "mensaje": "Aun no hay estado en vivo. El equipo no ha enviado datos.",
        }
    return _estado_vivo


# ----------------------------------------------------------------------
# Permite correr con: python api_irm.py  (ademas de uvicorn)
# En la nube respeta el puerto que asigna el host ($PORT)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    puerto = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_irm:app", host="0.0.0.0", port=puerto, reload=False)
