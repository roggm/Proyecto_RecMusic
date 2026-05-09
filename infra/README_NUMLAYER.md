# Lambda Layer: numpy (para que el handler funcione en AWS)

## Por qué
El zip actual de Lambda (`infra/lambda.tf`) empaqueta solo `infra/lambda/`.
El handler importa `numpy`, por lo que sin una capa o dependencias empaquetadas, fallará con `ImportError: numpy`.

Este doc describe cómo crear una Lambda Layer para `numpy` usando Docker (compatibilidad Linux).

## Requisitos
- AWS CLI configurado
- Docker instalado
- Docker debe poder ejecutar imágenes Linux

## Crear layer con Docker (Python 3.12)
1) Crear un build folder y estructura para la layer:

- `layer/python/lib/python3.12/site-packages/`

2) Instalar numpy ahí dentro con pip.

Ejemplo de comandos (desde la raíz del repo `Proyecto_RecMusic`):

```bat
mkdir layer
mkdir layer\python
mkdir layer\python\lib
mkdir layer\python\lib\python3.12
mkdir layer\python\lib\python3.12\site-packages

REM Construir entorno Linux con Docker y pip install target
docker run --rm -v "%cd%/layer:/opt" -w /opt python:3.12-slim bash -lc "pip install --upgrade pip && pip install numpy -t python/lib/python3.12/site-packages"
```

3) Empaquetar la layer en zip:

```bat
powershell -Command "Compress-Archive -Path layer\python -DestinationPath layer-numpy.zip -Force"
```

> Nota: el zip de la layer debe contener `python/` en la raíz.

4) Publicar la layer en AWS:

```bat
aws lambda publish-layer-version --layer-name rec-music-numpy --zip-file fileb://layer-numpy.zip --compatible-runtimes python3.12
```

Te dará un `LayerVersionArn`.

## Conectar la layer a Terraform
En Terraform vamos a agregar:
- `variable "numpy_layer_arn"` en `infra/variables.tf`
- `layers = [var.numpy_layer_arn]` en `infra/lambda.tf`

**Luego** ejecutas:
- `make tf-init`
- `make tf-apply`

> Nota: el ARN lo obtienes con `aws lambda publish-layer-version ...`.

## Alternativa
Si ya tienes `LayerVersionArn`, también se puede almacenar en `variables.tf` y pasarlo como variable en `terraform apply`.

