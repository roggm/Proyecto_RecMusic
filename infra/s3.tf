resource "aws_s3_bucket" "data" {
  bucket = "${local.name_prefix}-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── CSVs ──
resource "aws_s3_object" "items" {
  bucket = aws_s3_bucket.data.id
  key    = "processed/items.csv"
  source = "${var.data_processed_path}/items.csv"
  etag   = filemd5("${var.data_processed_path}/items.csv")
}

resource "aws_s3_object" "users" {
  bucket = aws_s3_bucket.data.id
  key    = "processed/users.csv"
  source = "${var.data_processed_path}/users.csv"
  etag   = filemd5("${var.data_processed_path}/users.csv")
}

resource "aws_s3_object" "interactions" {
  bucket = aws_s3_bucket.data.id
  key    = "processed/interactions.csv"
  source = "${var.data_processed_path}/interactions.csv"
  etag   = filemd5("${var.data_processed_path}/interactions.csv")
}

# ── Modelo SVD ──
resource "aws_s3_object" "modelo" {
  bucket = aws_s3_bucket.data.id
  key    = "processed/modelo_svd.pkl"
  source = "${var.data_processed_path}/modelo_svd.pkl"
  etag   = filemd5("${var.data_processed_path}/modelo_svd.pkl")
}

data "aws_caller_identity" "current" {}
