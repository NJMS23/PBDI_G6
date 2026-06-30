"""
entrenar_modelo_demo.py
=======================
Genera un modelo_IRM_prototipo.pkl de DEMO para que puedas probar TODA la API
de punta a punta sin tener todavia el modelo real entrenado.

  >> Corre esto UNA vez:   python entrenar_modelo_demo.py
  >> Crea el archivo:       modelo_IRM_prototipo.pkl

Cuando tengas tu modelo real entrenado con datos de verdad, SOLO reemplaza
modelo_IRM_prototipo.pkl por el tuyo. La API no cambia, siempre que:
  - sea un modelo de sklearn con .predict_proba()
  - haya sido entrenado con las 18 columnas de COLUMNAS_MODELO en ESE orden
  - la clase "patron alterado" sea la clase 1

Este demo entrena una regresion logistica con datos sinteticos: muestras
"normales" (VOC/NH3/RED bajos) vs "alteradas" (mas altos). No tiene validez
clinica, es solo para que la demo academica corra.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from caracteristicas import COLUMNAS_MODELO

RNG = np.random.default_rng(42)


def _muestra_sintetica(alterada: bool):
    """Genera una 'curva' de exhalacion sintetica y devuelve sus 18 features."""
    n = RNG.integers(8, 20)          # numero de lecturas en la exhalacion
    base = 1.0 if not alterada else 1.8  # las alteradas suben mas

    def curva(amp, ruido):
        t = np.linspace(0, 1, n)
        pico = amp * (base + 0.6 * np.sin(np.pi * t))   # sube y baja
        return pico + RNG.normal(0, ruido, n)

    voc = curva(amp=300, ruido=15)
    nh3 = curva(amp=1.2, ruido=0.05)
    red = curva(amp=0.9, ruido=0.04)

    feats = {}
    for nombre, serie in (("VOC_like", voc), ("NH3_like", nh3), ("RED_like", red)):
        feats[f"{nombre}_mean"] = float(np.mean(serie))
        feats[f"{nombre}_max"] = float(np.max(serie))
        feats[f"{nombre}_min"] = float(np.min(serie))
        feats[f"{nombre}_std"] = float(np.std(serie))
        feats[f"{nombre}_delta"] = float(np.max(serie) - serie[0])
        feats[f"{nombre}_auc"] = float(np.trapezoid(serie, x=np.linspace(0, 1, n)))
    return feats


def main():
    filas, etiquetas = [], []
    for _ in range(200):
        alterada = RNG.random() < 0.5
        filas.append(_muestra_sintetica(alterada))
        etiquetas.append(1 if alterada else 0)

    X = pd.DataFrame(filas)[COLUMNAS_MODELO]   # ORDEN exacto
    y = np.array(etiquetas)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X, y)

    joblib.dump(modelo, "modelo_IRM_prototipo.pkl")
    print("[OK] modelo_IRM_prototipo.pkl creado.")
    print("     Clases:", list(modelo.classes_), "(1 = patron alterado)")
    print("     Exactitud en entrenamiento:", round(modelo.score(X, y), 3))


if __name__ == "__main__":
    main()
