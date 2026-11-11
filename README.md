# Clasificador Saber 11 - Admisión a Medicina (MLflow MLOps)

Flujo con MLflow para el análisis y clasificación del desempeño en las pruebas de estado **Saber 11** (`data/saber11_muestra_1000.csv`): limpiar los datos, derivar la edad del estudiante, prevenir fuga de información (*data leakage*) de las notas individuales y predecir si el estudiante alcanzará el puntaje de corte de alta exigencia para aspirar a la carrera de **Medicina** (`pasa_medicina`).

El flujo entrena cuatro clasificadores (**Regresión Logística**, **SVM**, **Árbol de Decisión** y **Random Forest**), registra el de mayor `accuracy` con el alias `challenger` en el MLflow Model Registry y lo sirve para inferencia vía API REST. La API recibe las 16 características ya codificadas y escaladas.

Los nombres del experimento, del modelo y del alias salen de `.env` (`MLFLOW_EXPERIMENT_NAME=saber11-medicina`, `MLFLOW_MODEL_NAME=clasificador-medicina`, `MLFLOW_MODEL_ALIAS=challenger`).

---

## Contexto del Problema: Admisión a Medicina

En Colombia, la carrera de **Medicina** es el programa académico con el puntaje de corte más alto en universidades públicas y privadas de alta exigencia que admiten por Saber 11 (UIS, UdeA, Univalle, UPTC, Unicartagena, etc.).
- El puntaje global de corte para aspirar a Medicina usualmente se ubica entre **330 y 350 puntos** (percentil ~94-98 nacional).
- Se define por defecto `PUNTAJE_CORTE_MEDICINA = 330` (configurable en `.env`):
  - `1`: Estudiante con puntaje competitivo para Medicina ($\ge 330$).
  - `0`: Estudiante por debajo del umbral de Medicina ($< 330$).
- **Prevención de Data Leakage**: Se excluyen los puntajes de las materias (`punt_matematicas`, `punt_lectura_critica`, `punt_c_naturales`, `punt_sociales_ciudadanas`, `punt_ingles`, `desemp_ingles`), ya que el puntaje global es su agregación. El modelo evalúa el potencial a partir del contexto socioeconómico, familiar, escolar y la edad.

---

## Estructura

```
├── README.md                 # este documento
├── requirements.txt          # librerías con versión exacta
├── .env                      # variables de configuración local (Saber 11)
├── .env.example              # variables de ejemplo, sin secretos
├── .gitignore                # mlflow.db, mlruns/, .venv/ y .env
├── MLproject                 # mlflow run . --env-manager local
├── src/
│   ├── config.py             # experimento, modelo, umbral y variables
│   ├── data.py               # limpiar, derivar edad, codificar, escalar y partir
│   ├── train.py              # entrenar los 4 clasificadores y loguear en MLflow
│   ├── register.py           # mejor run → alias challenger
│   ├── predict.py            # predecir con columnas ya transformadas
│   └── ejemplo_api.py        # documento de petición y respuesta
├── ejemplos-api/             # generado: especificación API por modelo y alias
├── data/
│   └── saber11_muestra_1000.csv  # muestra de 1.000 observaciones de Saber 11
├── mlflow.db                 # generado: base de datos SQLite de MLflow
└── mlruns/                   # generado: almacenamiento de modelos y artefactos
```

---

## 1. Instalación y Preparación

> [!IMPORTANT]
> Activa siempre el entorno virtual en la terminal antes de ejecutar cualquiera de los comandos del proyecto.

```bash
# 1. Crear el entorno virtual si no existe
python3 -m venv .venv

# 2. Activar el entorno virtual
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Asegurar archivo de configuración .env
cp .env.example .env
```

---

## 2. Interfaz Gráfica de MLflow

En una terminal independiente, con el entorno activado:

```bash
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda disponible en http://127.0.0.1:5000.

---

## 3. Entrenar los Modelos

`DATA_PATH` apunta a `data/saber11_muestra_1000.csv`. Entrena Regresión Logística, SVM, Árbol de Decisión y Random Forest en el experimento `saber11-medicina`:

```bash
source .venv/bin/activate

# Opción A: Ejecutar directamente con Python (Recomendada)
python src/train.py

# Opción B: Ejecutar mediante MLproject
set -a && source .env && set +a
mlflow run . --env-manager local
```

---

## 4. Registrar el Mejor Modelo (Challenger)

Elige el modelo con mayor `accuracy` y le asigna el alias `challenger` en el registro:

```bash
source .venv/bin/activate

# Opción A: Ejecutar directamente con Python (Recomendada)
python src/register.py

# Opción B: Ejecutar mediante MLproject
mlflow run . -e register --env-manager local
```

---

## 5. Probar Inferencia en Local (CLI)

Ejecuta una predicción de prueba usando el modelo con alias `challenger`:

```bash
source .venv/bin/activate

# Opción A: Ejecutar directamente con Python (Recomendada)
python src/predict.py

# Opción B: Ejecutar mediante MLproject
mlflow run . -e predict --env-manager local
```

---

## 6. Ejecución Completa en un Solo Paso (Pipeline)

Si deseas ejecutar **entrenamiento + registro del challenger + predicción** de una sola vez:

```bash
source .venv/bin/activate
set -a && source .env && set +a
mlflow run . -e pipeline --env-manager local
```

---

## 7. Servir el Modelo vía API REST

Levanta el servidor HTTP para inferencias en tiempo real:

```bash
source .venv/bin/activate
mlflow --env-file .env models serve \
  -m "models:/clasificador-medicina@challenger" \
  --port 5001 \
  --env-manager local
```

El servicio escucha en `http://127.0.0.1:5001`.

### Regenerar Documentación del Endpoint
En otra terminal:

```bash
source .venv/bin/activate
python src/ejemplo_api.py --modelo clasificador-medicina --alias challenger --puerto 5001
```

Esto genera/actualiza `ejemplos-api/clasificador-medicina__challenger.md`.

### Petición de Prueba con `curl`

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["cole_area_ubicacion", "cole_calendario", "cole_caracter", "cole_jornada", "cole_naturaleza", "estu_genero", "fami_cuartoshogar", "fami_educacionmadre", "fami_educacionpadre", "fami_estratovivienda", "fami_personashogar", "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet", "fami_tienelavadora", "edad"], "data": [[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 7.0, 10.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -0.17118636511622146]]}}'
```

Respuesta:
```json
{
  "predictions": [
    1
  ]
}
```
*(Donde `1` = aspirante con perfil competitivo para Medicina, `0` = por debajo del umbral).*

---

## 8. Empaquetar y Desplegar en Docker

Construir la imagen del modelo registrado:

```bash
source .venv/bin/activate
mlflow --env-file .env models build-docker \
  -m "models:/clasificador-medicina@challenger" \
  -n "tu_usuario_dockerhub/saber11-medicina:latest"
```

Probar el contenedor localmente en el puerto 5002:

```bash
docker run --rm -p 5002:8080 tu_usuario_dockerhub/saber11-medicina:latest
```

---

## Qué no se versiona

| Ruta | Motivo |
| --- | --- |
| `.env` | Valores locales y configuraciones de entorno |
| `.venv/`, `.envmlops/` | Entornos virtuales de Python |
| `mlflow.db` | Base de datos SQLite generada con metadatos, parámetros y métricas |
| `mlruns/` | Almacenamiento local de modelos y artefactos |

