"""
caracteristicas.py
==================
Extraccion de las 18 caracteristicas que espera el modelo IRM.

Este modulo lo usan TANTO la API (api_irm.py) COMO el entrenador
(entrenar_modelo_demo.py). Asi se garantiza que las features se calculen
EXACTAMENTE igual al entrenar y al predecir. Si cambias algo aqui,
cambia para los dos a la vez (que es justo lo que queremos).

Orden EXACTO de las 18 variables que entran al modelo:

    VOC_like_mean, VOC_like_max, VOC_like_min, VOC_like_std, VOC_like_delta, VOC_like_auc
    NH3_like_mean, NH3_like_max, NH3_like_min, NH3_like_std, NH3_like_delta, NH3_like_auc
    RED_like_mean, RED_like_max, RED_like_min, RED_like_std, RED_like_delta, RED_like_auc

Correspondencia sensor real -> variable del modelo:
    BME688_gas_kohm  -> VOC_like
    MiCS6814_NH3_v   -> NH3_like
    MiCS6814_RED_v   -> RED_like
"""

import numpy as np

# ----------------------------------------------------------------------
# Orden exacto de las columnas que el modelo espera. NO cambiar el orden
# sin reentrenar el modelo.
# ----------------------------------------------------------------------
COLUMNAS_MODELO = [
    "VOC_like_mean", "VOC_like_max", "VOC_like_min", "VOC_like_std", "VOC_like_delta", "VOC_like_auc",
    "NH3_like_mean", "NH3_like_max", "NH3_like_min", "NH3_like_std", "NH3_like_delta", "NH3_like_auc",
    "RED_like_mean", "RED_like_max", "RED_like_min", "RED_like_std", "RED_like_delta", "RED_like_auc",
]

# Mapeo: nombre de la senal del modelo -> columna real del sensor en el CSV
MAPEO_SENSORES = {
    "VOC_like": "BME688_gas_kohm",
    "NH3_like": "MiCS6814_NH3_v",
    "RED_like": "MiCS6814_RED_v",
}

# Como calcular el AUC (area bajo la curva):
#   True  -> np.trapezoid(valores, x=tiempo_en_segundos)  (area real en el tiempo)
#   False -> np.trapezoid(valores)  (espaciado = 1, solo por indice de muestra)
# IMPORTANTE: tiene que coincidir con como entrenaste el modelo. El entrenador
# demo usa este mismo flag, asi que con el modelo demo siempre calza.
AUC_USA_TIEMPO = True


def _trapz(y, x=None):
    """Compatibilidad numpy: usa np.trapezoid (numpy>=2) o np.trapz (numpy<2)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    if x is None:
        return float(fn(y))
    return float(fn(y, x=x))


def _features_de_senal(valores, tiempos_s):
    """Calcula las 6 caracteristicas (mean, max, min, std, delta, auc) de una senal."""
    v = np.asarray(valores, dtype=float)
    if v.size == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "delta": 0.0, "auc": 0.0}

    mean = float(np.mean(v))
    vmax = float(np.max(v))
    vmin = float(np.min(v))
    # std poblacional (ddof=0); si quieres muestral cambia a ddof=1
    std = float(np.std(v, ddof=0))
    delta = float(vmax - v[0])  # maximo - valor inicial

    if AUC_USA_TIEMPO and tiempos_s is not None and len(tiempos_s) == len(v) and len(v) > 1:
        x = np.asarray(tiempos_s, dtype=float)
        # normalizamos el tiempo para que arranque en 0 (no afecta el area)
        x = x - x[0]
        auc = _trapz(v, x=x)
    else:
        auc = _trapz(v)

    return {"mean": mean, "max": vmax, "min": vmin, "std": std, "delta": delta, "auc": auc}


def extraer_caracteristicas(df_muestra, ordenar_por="indice"):
    """
    Recibe un DataFrame con TODAS las lecturas de UNA muestra y devuelve:
      - features: dict con las 18 variables (en el orden de COLUMNAS_MODELO)
      - co2: dict con {'inicial', 'max', 'delta'}

    df_muestra debe tener al menos las columnas de MAPEO_SENSORES, ademas de
    MHZ19_CO2_ppm y (opcionalmente) 'tiempo_ms' e 'indice' para ordenar.
    """
    df = df_muestra.copy()

    # Ordenar las lecturas en el tiempo (importante para 'delta' y 'auc')
    if ordenar_por in df.columns:
        df = df.sort_values(ordenar_por)
    elif "tiempo_ms" in df.columns:
        df = df.sort_values("tiempo_ms")

    tiempos_s = None
    if "tiempo_ms" in df.columns:
        tiempos_s = (df["tiempo_ms"].astype(float) / 1000.0).tolist()

    features = {}
    for senal, col_sensor in MAPEO_SENSORES.items():
        if col_sensor not in df.columns:
            raise ValueError(f"Falta la columna '{col_sensor}' en las lecturas de la muestra.")
        valores = df[col_sensor].astype(float).tolist()
        f = _features_de_senal(valores, tiempos_s)
        features[f"{senal}_mean"] = f["mean"]
        features[f"{senal}_max"] = f["max"]
        features[f"{senal}_min"] = f["min"]
        features[f"{senal}_std"] = f["std"]
        features[f"{senal}_delta"] = f["delta"]
        features[f"{senal}_auc"] = f["auc"]

    # Validacion de exhalacion con CO2
    co2 = {"inicial": 0.0, "max": 0.0, "delta": 0.0}
    if "MHZ19_CO2_ppm" in df.columns and len(df) > 0:
        co2_vals = df["MHZ19_CO2_ppm"].astype(float)
        co2["inicial"] = float(co2_vals.iloc[0])
        co2["max"] = float(co2_vals.max())
        co2["delta"] = float(co2["max"] - co2["inicial"])

    # Devolver features en el orden EXACTO del modelo
    features_ordenado = {k: features[k] for k in COLUMNAS_MODELO}
    return features_ordenado, co2
