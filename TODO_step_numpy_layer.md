# TODO — Fix numpy con Lambda Layer

- [ ] Crear/usar una Lambda Layer de `numpy` compatible con `python3.12`.
- [ ] Pasar `numpy_layer_arn` a Terraform.
- [ ] Actualizar `infra/lambda.tf` para agregar `layers = [var.numpy_layer_arn]`.
- [ ] Desplegar (`make tf-init && make tf-apply`) y probar `GET /recommendations?user_id=...`.

- [ ] Ver logs en CloudWatch si aún hay fallas.

