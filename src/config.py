"""Nombres del experimento y del modelo, semilla y parámetros de Saber 11.

`src/` lee estos valores desde aquí. Salen de `.env`; si falta el
archivo, se usan valores por defecto para Saber 11.

El objetivo es clasificar si un estudiante alcanza el puntaje de corte
para aspirar a la carrera de Medicina (`pasa_medicina`) y elegir el run
con mayor accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "saber11-medicina")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "clasificador-medicina")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = os.getenv("DATA_PATH", "data/saber11_muestra_1000.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

# Variable objetivo y parámetros de corte para Medicina
TARGET = os.getenv("TARGET", "pasa_medicina")
PUNTAJE_CORTE_MEDICINA = int(os.getenv("PUNTAJE_CORTE_MEDICINA", "330"))
TEST_SIZE = 0.3
METRIC = "accuracy"
EDAD_MIN = int(os.getenv("EDAD_MIN", "14"))
EDAD_MAX = int(os.getenv("EDAD_MAX", "80"))

CATEGORICAS = (
    "cole_area_ubicacion",
    "cole_calendario",
    "cole_caracter",
    "cole_jornada",
    "cole_naturaleza",
    "estu_genero",
    "fami_cuartoshogar",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "fami_personashogar",
    "fami_tieneautomovil",
    "fami_tienecomputador",
    "fami_tieneinternet",
    "fami_tienelavadora",
)

COLUMNS_TO_DROP = (
    "periodo",
    "estu_tipodocumento",
    "estu_consecutivo",
    "cole_bilingue",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_codigo_icfes",
    "cole_depto_ubicacion",
    "cole_genero",
    "cole_mcpio_ubicacion",
    "cole_nombre_establecimiento",
    "cole_nombre_sede",
    "cole_sede_principal",
    "estu_cod_depto_presentacion",
    "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto",
    "estu_cod_reside_mcpio",
    "estu_depto_presentacion",
    "estu_depto_reside",
    "estu_estadoinvestigacion",
    "estu_estudiante",
    "estu_mcpio_presentacion",
    "estu_mcpio_reside",
    "estu_nacionalidad",
    "estu_pais_reside",
    "estu_privado_libertad",
    # Descartadas para evitar data leakage del puntaje:
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica",
    "punt_ingles",
    "desemp_ingles",
)

def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException:
        cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        exp = cliente.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is not None and exp.lifecycle_stage == "deleted":
            cliente.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        else:
            raise
