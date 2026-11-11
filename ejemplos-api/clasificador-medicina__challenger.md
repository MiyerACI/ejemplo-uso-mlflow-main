# clasificador-medicina @ challenger

- URI: `models:/clasificador-medicina@challenger`
- Versión: 4
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "long",
    "name": "cole_area_ubicacion",
    "required": true
  },
  {
    "type": "long",
    "name": "cole_calendario",
    "required": true
  },
  {
    "type": "long",
    "name": "cole_caracter",
    "required": true
  },
  {
    "type": "long",
    "name": "cole_jornada",
    "required": true
  },
  {
    "type": "long",
    "name": "cole_naturaleza",
    "required": true
  },
  {
    "type": "long",
    "name": "estu_genero",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_cuartoshogar",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_educacionmadre",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_educacionpadre",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_estratovivienda",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_personashogar",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_tieneautomovil",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_tienecomputador",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_tieneinternet",
    "required": true
  },
  {
    "type": "long",
    "name": "fami_tienelavadora",
    "required": true
  },
  {
    "type": "double",
    "name": "edad",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "long",
    "required": true
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["cole_area_ubicacion", "cole_calendario", "cole_caracter", "cole_jornada", "cole_naturaleza", "estu_genero", "fami_cuartoshogar", "fami_educacionmadre", "fami_educacionpadre", "fami_estratovivienda", "fami_personashogar", "fami_tieneautomovil", "fami_tienecomputador", "fami_tieneinternet", "fami_tienelavadora", "edad"], "data": [[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 7.0, 10.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -0.17118636511622146]]}}'
```

```json
{
  "dataframe_split": {
    "columns": [
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
      "edad"
    ],
    "data": [
      [
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        4.0,
        7.0,
        10.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        -0.17118636511622146
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    1
  ]
}
```
