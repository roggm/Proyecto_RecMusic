.PHONY: help install train validate run run-api run-frontend clean

PYTHON  ?= python3
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

# ─────────────────────────────────────────────────────────────
# AYUDA
# ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "RecMusic — Comandos disponibles:"
	@echo ""
	@echo "  make install      Instala dependencias de Python y Node"
	@echo "  make train        Entrena los tres modelos en orden"
	@echo "  make validate     Verifica que los modelos existen antes de correr"
	@echo "  make run          Levanta API y frontend al mismo tiempo"
	@echo "  make run-api      Solo levanta el API (puerto $(API_PORT))"
	@echo "  make run-frontend Solo levanta el frontend (puerto 5173)"
	@echo "  make clean        Borra __pycache__ y archivos .pyc"
	@echo ""

# ─────────────────────────────────────────────────────────────
# INSTALACIÓN
# ─────────────────────────────────────────────────────────────
install:
	pip install fastapi uvicorn pandas numpy scipy scikit-learn jupyter nbconvert
	cd frontend && npm install
	@echo ""
	@echo "Dependencias instaladas ✅"

# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────
train:
	@echo "Entrenando collaborative → content_based → hybrid..."
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		src/model/collaborative.ipynb
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		src/model/content_based.ipynb
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		src/model/hybrid.ipynb
	@echo ""
	@echo "Modelos entrenados ✅"
	@echo "  data/processed/modelo_svd.pkl"
	@echo "  data/processed/modelo_cb.pkl"
	@echo "  data/processed/modelo_hybrid.pkl"

# ─────────────────────────────────────────────────────────────
# VALIDACIÓN
# ─────────────────────────────────────────────────────────────
validate:
	@echo "Verificando modelos..."
	@test -f data/processed/items.csv         && echo "  ✅ items.csv"          || (echo "  ❌ items.csv no encontrado"         && exit 1)
	@test -f data/processed/interactions.csv  && echo "  ✅ interactions.csv"   || (echo "  ❌ interactions.csv no encontrado"  && exit 1)
	@test -f data/processed/modelo_svd.pkl    && echo "  ✅ modelo_svd.pkl"     || (echo "  ❌ modelo_svd.pkl — corre: make train" && exit 1)
	@test -f data/processed/modelo_cb.pkl     && echo "  ✅ modelo_cb.pkl"      || (echo "  ❌ modelo_cb.pkl  — corre: make train" && exit 1)
	@test -f data/processed/modelo_hybrid.pkl && echo "  ✅ modelo_hybrid.pkl"  || (echo "  ❌ modelo_hybrid.pkl — corre: make train" && exit 1)
	@echo ""
	@echo "Todo listo para correr ✅"

# ─────────────────────────────────────────────────────────────
# CORRER
# ─────────────────────────────────────────────────────────────
run: validate
	uvicorn api_local:app --host $(API_HOST) --port $(API_PORT) --reload &
	cd frontend && npm run dev

run-api:
	uvicorn api_local:app --host $(API_HOST) --port $(API_PORT) --reload

run-frontend:
	cd frontend && npm run dev

# ─────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Limpieza completa ✅"
