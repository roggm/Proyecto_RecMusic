output "s3_bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "s3_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "lambda_function_name" {
  value = aws_lambda_function.recommender.function_name
}

output "api_gateway_url" {
  description = "URL del API — pégala en frontend/.env como VITE_API_URL"
  value       = "${aws_api_gateway_stage.prod.invoke_url}/recommendations"
}

output "cloudwatch_log_group" {
  value = aws_cloudwatch_log_group.lambda_logs.name
}
