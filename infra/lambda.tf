data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/lambda"
  output_path = "${path.module}/.terraform/lambda_package.zip"
}

resource "aws_lambda_function" "recommender" {
  function_name    = "${local.name_prefix}-recommender"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  role             = aws_iam_role.lambda_exec.arn
  handler          = "handler.handler"
  runtime          = "python3.12"

  layers = var.numpy_layer_arn != "" ? [var.numpy_layer_arn] : []

  timeout          = 60      # el modelo tarda en cargarse la primera vez
  memory_size      = 512     # numpy necesita más memoria que el handler anterior

  environment {
    variables = {
      S3_BUCKET  = aws_s3_bucket.data.bucket
      ITEMS_KEY  = "processed/items.csv"
      MODELO_KEY = "processed/modelo_svd.pkl"
      N_RESULTS  = "10"
    }
  }
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.recommender.function_name}"
  retention_in_days = 7
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.recommender.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.recommender.execution_arn}/*/*"
}
