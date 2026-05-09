variable "aws_region" {
  description = "Región de AWS donde se desplegará la infraestructura"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre base del proyecto"
  type        = string
  default     = "rec-music"
}

variable "environment" {
  description = "Ambiente de despliegue"
  type        = string
  default     = "dev"
}

variable "data_processed_path" {
  description = "Ruta local a los CSVs y modelo procesados"
  type        = string
  default     = "../data/processed"
}
