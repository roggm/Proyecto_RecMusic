# RecMusic — Sistema Recomendador de Música

Demo de un sistema de recomendación de música que combina filtrado colaborativo (SVD) y content-based filtering en un modelo híbrido. El usuario llena un formulario con su nombre, géneros favoritos y preferencias de audio, y recibe una playlist personalizada.

---

## Arquitectura

```
data/
  processed/
    items.csv             ← catálogo de canciones
    interactions.csv      ← historial de interacciones (se actualiza en runtime)
    modelo_svd.pkl        ← modelo colaborativo (generado por collaborative.ipynb)
    modelo_cb.pkl         ← modelo content-based (generado por content_based.ipynb)
    modelo_hybrid.pkl     ← modelo final usado por el API (generado por hybrid.ipynb)

src/
  data/                   ← notebooks de exploración y procesamiento de datos
  model/
    collaborative.ipynb   ← entrena SVD y guarda modelo_svd.pkl
    content_based.ipynb   ← construye feature matrix y guarda modelo_cb.pkl
    hybrid.ipynb          ← combina ambos y guarda modelo_hybrid.pkl  ← MODELO PRODUCCIÓN
    train_mf.py           ← entrenamiento alternativo con PyTorch (Matrix Factorization)
    evaluación.ipynb      ← métricas y comparación de modelos

frontend/                 ← app React (Vite)
api_local.py              ← API FastAPI local
```

---

## Cómo correr el proyecto

### 1. Prerrequisitos

```bash
pip install fastapi uvicorn pandas numpy scipy scikit-learn
cd frontend && npm install
```

### 2. Entrenar los modelos

Los notebooks deben correrse en orden. Solo necesitas hacerlo una vez (o cuando cambien los datos).

```
src/model/collaborative.ipynb   →  genera data/processed/modelo_svd.pkl
src/model/content_based.ipynb   →  genera data/processed/modelo_cb.pkl
src/model/hybrid.ipynb          →  genera data/processed/modelo_hybrid.pkl
```

### 3. Levantar el backend

```bash
uvicorn api_local:app --reload
```

Corre en `http://localhost:8000`. Endpoints disponibles:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/recommendations/new-user` | Recomendaciones por perfil (nuevo o conocido) |
| `GET`  | `/recommendations?user_id=` | Recomendaciones para usuario ya en el modelo |
| `POST` | `/users/{user_id}/interactions` | Registrar interacción (like / play / skip) |
| `GET`  | `/status` | Estado del modelo y si hay reentrenamiento en curso |

### 4. Levantar el frontend

```bash
cd frontend
npm run dev
```

Abre `http://localhost:5173`.

---

## Cómo funciona el modelo híbrido

El score final combina dos señales:

```
score_final = α × score_contenido + (1 - α) × score_colaborativo
```

- **Content-based:** similitud de coseno entre el perfil de audio del usuario y cada canción.
- **Colaborativo (SVD):** factorización matricial sobre el historial de interacciones.
- **α = 0.5** por defecto (configurable en `hybrid.ipynb` antes de guardar el modelo).

### Nuevo usuario (cold start)

1. Respuesta inmediata via **fold-in**: el perfil del formulario se proyecta al espacio latente del SVD.
2. Las recomendaciones generadas se guardan como interacciones sintéticas en `interactions.csv`.
3. Se dispara un **reentrenamiento en background**: la próxima vez que el mismo usuario use la app, ya tiene un modelo real entrenado con sus preferencias.

---

## Notas

- El nombre que escribe el usuario funciona como `user_id`. Para una demo es suficiente.
- El reentrenamiento en background actualiza solo la parte colaborativa (SVD). La feature matrix de content-based no cambia con nuevas interacciones.
- `train_mf.py` es un entrenador alternativo con PyTorch (Matrix Factorization con embeddings). No está integrado al API actual pero puede reemplazar al SVD si se quiere mayor capacidad.
