# Proyecto_RecMusic — guía rápida para que funcione

## 1) Verifica que existen los artefactos en `data/processed`
En este repo ya aparecen:
- `data/processed/items.csv`
- `data/processed/modelo_svd.pkl`

## 2) Configura el frontend (API_URL)
- Copia `frontend/.env.example` a `frontend/.env`
- Pega la salida de Terraform `api_gateway_url` en `VITE_API_URL`

## 3) Despliega infraestructura (AWS)
Desde la carpeta del proyecto:
- `make tf-init`
- `make tf-apply`

Anota `api_gateway_url` (salida de Terraform).

## 4) Prueba el endpoint directamente
Ejemplo (usa un user_id como el frontend):
- `curl "${VITE_API_URL}?user_id=user_demo"`

Debe responder JSON con:
- `recomendaciones: [...]`

## 5) Verifica logs si falla
- CloudWatch Logs: `/aws/lambda/<function_name>`

Errores típicos:
- El zip de Lambda no incluye dependencias (numpy).
- S3 no tiene `processed/items.csv` o `processed/modelo_svd.pkl`.

