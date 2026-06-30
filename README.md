# API IRM — PULSARIX

Backend en FastAPI para el **Índice de Riesgo Minero (IRM)**.
El ESP32 mide y envía lecturas por WiFi → la API guarda, calcula 18
características, valida la exhalación con CO₂, aplica un modelo de regresión
logística y devuelve el IRM → el LCD y la app celular muestran el resultado.

> ⚠️ **Aviso:** este resultado NO es un diagnóstico médico. Es una estimación
> preliminar basada en señales de sensores de aliento.

---

## 📁 Archivos

```
IRM_API/
├── api_irm.py                 ← la API (FastAPI)
├── caracteristicas.py         ← cálculo de las 18 features (compartido)
├── entrenar_modelo_demo.py    ← genera un modelo de DEMO para probar
├── modelo_IRM_prototipo.pkl   ← el modelo (demo incluido; reemplázalo por el real)
├── requirements.txt
├── lecturas_esp32.csv         ← se crea solo
└── resultados_irm.csv         ← se crea solo
```

---

## ▶️ Correr en local

```bash
pip install -r requirements.txt

# (opcional) si todavía no tienes el modelo real, genera el de demo:
python entrenar_modelo_demo.py

uvicorn api_irm:app --host 0.0.0.0 --port 8000 --reload
```

Abre en el navegador:
- `http://localhost:8000/` → debe responder que la API funciona
- `http://localhost:8000/docs` → **documentación interactiva** (puedes probar todos los endpoints desde ahí, súper útil para la demo)

> 💡 El modelo `.pkl` incluido es un **modelo de demo** entrenado con datos
> sintéticos (sin validez clínica). Cuando tengas el real, solo reemplaza el
> archivo `modelo_IRM_prototipo.pkl` — la API no cambia, siempre que el modelo:
> tenga `.predict_proba()`, esté entrenado con las 18 columnas en el mismo
> orden, y use la clase `1` como "patrón alterado".

---

## 🧪 Pruebas con curl

**Guardar una lectura:**
```bash
curl -X POST http://localhost:8000/guardar_lectura \
  -H "Content-Type: application/json" \
  -d '{
    "participante_id": "P001",
    "muestra_id": "P001_M001",
    "clase": 0,
    "condicion": "normal_no_expuesto",
    "indice": 0,
    "tiempo_ms": 12500,
    "BME688_gas_kohm": 327.5,
    "BME688_temp_C": 28.4,
    "BME688_humidity_pct": 55.2,
    "BME688_pressure_hPa": 1008.3,
    "MiCS6814_NH3_v": 1.25,
    "MiCS6814_RED_v": 0.91,
    "MiCS6814_OX_v": 1.46,
    "MHZ19_CO2_ppm": 1800
  }'
```

**Finalizar la muestra (calcula el IRM):**
```bash
curl -X POST http://localhost:8000/finalizar_muestra \
  -H "Content-Type: application/json" \
  -d '{"muestra_id": "P001_M001"}'
```

**Último resultado (lo que consulta la app):**
```bash
curl http://localhost:8000/ultimo_resultado
```

**Resultado de una muestra específica:**
```bash
curl http://localhost:8000/resultado/P001_M001
```

---

## 🔌 Conectar el ESP32

1. El ESP32 toma varias lecturas durante la exhalación. Por **cada** lectura
   hace un `HTTP POST` a `/guardar_lectura` con el JSON de la lectura.
   Usa el **mismo `muestra_id`** para todas las lecturas de esa exhalación.
2. Cuando la exhalación termina, hace un `HTTP POST` a `/finalizar_muestra`
   con `{"muestra_id": "..."}`.
3. La respuesta JSON (IRM, riesgo, mensaje) la muestra en el LCD.

Esqueleto en Arduino (ESP32):
```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* SSID = "TU_WIFI";
const char* PASS = "TU_CLAVE";
String API = "http://192.168.1.50:8000";   // IP de la laptop, o la URL de la nube

void enviarLectura(String muestra, int idx, float gas, float nh3,
                   float red, float ox, float co2) {
  HTTPClient http;
  http.begin(API + "/guardar_lectura");
  http.addHeader("Content-Type", "application/json");
  String body = "{";
  body += "\"participante_id\":\"P001\",";
  body += "\"muestra_id\":\"" + muestra + "\",";
  body += "\"indice\":" + String(idx) + ",";
  body += "\"tiempo_ms\":" + String(millis()) + ",";
  body += "\"BME688_gas_kohm\":" + String(gas, 2) + ",";
  body += "\"MiCS6814_NH3_v\":" + String(nh3, 3) + ",";
  body += "\"MiCS6814_RED_v\":" + String(red, 3) + ",";
  body += "\"MiCS6814_OX_v\":" + String(ox, 3) + ",";
  body += "\"MHZ19_CO2_ppm\":" + String(co2, 0);
  body += "}";
  http.POST(body);
  http.end();
}

String finalizar(String muestra) {
  HTTPClient http;
  http.begin(API + "/finalizar_muestra");
  http.addHeader("Content-Type", "application/json");
  http.POST("{\"muestra_id\":\"" + muestra + "\"}");
  String resp = http.getString();   // JSON con IRM y riesgo → mostrar en LCD
  http.end();
  return resp;
}
```

> El ESP32 ya **no** necesita la tarjeta SD ni calcular el IRM: solo mide, envía
> por WiFi y muestra lo que la API le devuelve.

---

## 📱 Conectar la app celular

La app **no calcula nada**. Solo consulta la API y muestra:
`muestra_id`, `IRM`, `riesgo`, `muestra_valida` (estado) y `mensaje` + el aviso
de seguridad.

- Último resultado: `GET /ultimo_resultado`
- Por muestra: `GET /resultado/{muestra_id}`

Respuesta que recibe la app:
```json
{
  "muestra_id": "P001_M001",
  "participante_id": "P001",
  "muestra_valida": true,
  "CO2_delta": 1280,
  "IRM": 18.4,
  "riesgo": "Bajo",
  "mensaje": "Medicion dentro del rango bajo de alerta respiratoria",
  "aviso_seguridad": "Este resultado no representa un diagnostico medico. ..."
}
```

Semáforo sugerido en la app por `riesgo`: **Bajo** = verde, **Medio** = ámbar,
**Alto** = rojo. Si `muestra_valida` es `false`, mostrar "Repetir medición".

---

## ☁️ Subir a la nube (Render)

1. Sube esta carpeta a un repo de GitHub (con el `.pkl` adentro).
2. En Render → **New > Web Service** → conecta el repo.
3. Configura:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api_irm:app --host 0.0.0.0 --port $PORT`
4. Tu API quedará en algo como `https://irm-prototipo.onrender.com`.
   - ESP32 → `https://irm-prototipo.onrender.com/guardar_lectura`
   - App → `https://irm-prototipo.onrender.com/ultimo_resultado`

> ⚠️ **Ojo con la nube:** en el plan gratis de Render el sistema de archivos es
> efímero — los CSV se **borran** cada vez que se reinicia o redespliega el
> servicio. Para una demo está bien, pero si quieres que los datos persistan de
> verdad, conviene una base de datos (ej. SQLite con disco persistente, o
> Postgres). Te lo puedo migrar cuando lo necesiten.
