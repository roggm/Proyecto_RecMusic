# ── REST API ──
resource "aws_api_gateway_rest_api" "recommender" {
  name        = "${local.name_prefix}-api"
  description = "API del Sistema Recomendador de Música"

  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# ── Recurso /recommendations ──
resource "aws_api_gateway_resource" "recommendations" {
  rest_api_id = aws_api_gateway_rest_api.recommender.id
  parent_id   = aws_api_gateway_rest_api.recommender.root_resource_id
  path_part   = "recommendations"
}

# ── GET /recommendations ───
resource "aws_api_gateway_method" "get_recommendations" {
  rest_api_id   = aws_api_gateway_rest_api.recommender.id
  resource_id   = aws_api_gateway_resource.recommendations.id
  http_method   = "GET"
  authorization = "NONE"

  request_parameters = {
    "method.request.querystring.user_id" = true
  }
}

resource "aws_api_gateway_integration" "lambda" {
  rest_api_id             = aws_api_gateway_rest_api.recommender.id
  resource_id             = aws_api_gateway_resource.recommendations.id
  http_method             = aws_api_gateway_method.get_recommendations.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.recommender.invoke_arn
}

# ── OPTIONS /recommendations (CORS preflight) ──
resource "aws_api_gateway_method" "options" {
  rest_api_id   = aws_api_gateway_rest_api.recommender.id
  resource_id   = aws_api_gateway_resource.recommendations.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options" {
  rest_api_id = aws_api_gateway_rest_api.recommender.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options.http_method
  type        = "MOCK"

  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_200" {
  rest_api_id = aws_api_gateway_rest_api.recommender.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options" {
  rest_api_id = aws_api_gateway_rest_api.recommender.id
  resource_id = aws_api_gateway_resource.recommendations.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = aws_api_gateway_method_response.options_200.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

# ── Deployment ──
resource "aws_api_gateway_deployment" "prod" {
  rest_api_id = aws_api_gateway_rest_api.recommender.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_method.get_recommendations,
      aws_api_gateway_integration.lambda,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_integration.lambda,
    aws_api_gateway_integration.options
  ]
}

resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.prod.id
  rest_api_id   = aws_api_gateway_rest_api.recommender.id
  stage_name    = var.environment
}
