.PHONY: help init validate-structure clean

BUCKET ?= music-recommender-bucket
REGION ?= us-east-1

# ─────────────────────────────────────────
# DEFAULT
# ─────────────────────────────────────────
help:
	@echo ""
	@echo "Music Recommender — Comandos disponibles:"
	@echo ""
	@echo "  make init                 Crea el skeleton del proyecto"
	@echo "  make validate-structure   Verifica que el skeleton esté completo"
	@echo "  make tf-init              Inicializa Terraform"
	@echo "  make tf-plan              Muestra qué va a crear en AWS"
	@echo "  make tf-apply             Crea la infraestructura en AWS"
	@echo "  make tf-destroy           Destruye toda la infra de AWS"
	@echo "  make clean                Limpia archivos temporales"
	@echo ""

# ─────────────────────────────────────────
# SKELETON
# ─────────────────────────────────────────
init:
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p data/samples
	mkdir -p src/data
	mkdir -p src/model
	mkdir -p src/api
	mkdir -p infrastructure/modules/s3
	mkdir -p infrastructure/modules/personalize
	mkdir -p infrastructure/modules/lambda
	mkdir -p infrastructure/modules/api_gateway
	mkdir -p notebooks
	mkdir -p tests/fixtures
	mkdir -p docs
	touch src/__init__.py
	touch src/data/__init__.py
	touch src/model/__init__.py
	touch src/api/__init__.py
	touch data/samples/.gitkeep
	touch data/raw/.gitkeep
	touch data/processed/.gitkeep
	@echo "Skeleton creado ✅"

validate-structure:
	@echo "Validando estructura del proyecto..."
	@test -d data/raw             && echo "✅ data/raw" 	        || echo "❌ data/raw"
	@test -d data/processed       && echo "✅ data/processed"     || echo "❌ data/processed"
	@test -d data/samples         && echo "✅ data/samples"       || echo "❌ data/samples"
	@test -d src/data             && echo "✅ src/data"           || echo "❌ src/data"
	@test -d src/model            && echo "✅ src/model"          || echo "❌ src/model"
	@test -d src/api              && echo "✅ src/api"            || echo "❌ src/api"
	@test -d infrastructure       && echo "✅ infrastructure"     || echo "❌ infrastructure"
	@test -d notebooks            && echo "✅ notebooks"          || echo "❌ notebooks"
	@test -d tests                && echo "✅ tests"              || echo "❌ tests"
	@echo "Validación completa ✅"

# ─────────────────────────────────────────
# TERRAFORM
# ─────────────────────────────────────────
tf-init:
	cd infrastructure && terraform init

tf-plan:
	cd infrastructure && terraform plan -var="bucket_name=$(BUCKET)" -var="aws_region=$(REGION)"

tf-apply:
	cd infrastructure && terraform apply -var="bucket_name=$(BUCKET)" -var="aws_region=$(REGION)"

tf-destroy:
	cd infrastructure && terraform destroy -var="bucket_name=$(BUCKET)" -var="aws_region=$(REGION)"

# ─────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Limpieza completa ✅"