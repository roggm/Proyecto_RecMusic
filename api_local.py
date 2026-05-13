# api_local.py
import os, pickle, csv, threading
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data/processed")
ITEMS_PATH      = DATA_DIR / "items.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"
MODEL_PATH      = DATA_DIR / "modelo_svd.pkl"

RATING_MAP   = {"like": 3.0, "play": 2.0, "skip": 1.0}
RATING_SCALE = (1.0, 3.0)
K            = 50
N_RESULTS    = 10

# ── Estado global del modelo ─────────────────────────────────────────────────
_modelo      = None          # dict con los arrays del SVD
_modelo_lock = threading.Lock()
_retraining  = False         # flag para saber si ya hay un reentrenamiento corriendo


def cargar_modelo():
    """Carga el .pkl desde disco al dict global."""
    global _modelo
    with open(MODEL_PATH, "rb") as f:
        _modelo = pickle.load(f)
    print("Modelo cargado desde disco.")


def guardar_modelo(modelo: dict):
    """Guarda el dict del modelo en disco."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(modelo, f)
    print("Modelo guardado en disco.")


def entrenar_modelo() -> dict:
    """
    Replica exactamente la lógica del collaborative.ipynb.
    Lee interactions.csv actualizado y devuelve el dict del modelo.
    """
    print("Iniciando reentrenamiento...")

    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions["rating"] = interactions["EVENT_TYPE"].map(RATING_MAP)

    ratings = (
        interactions
        .groupby(["USER_ID", "ITEM_ID"])["rating"]
        .max()
        .reset_index()
    )

    usuarios  = sorted(ratings["USER_ID"].unique())
    canciones = sorted(ratings["ITEM_ID"].unique())

    user_to_idx = {u: i for i, u in enumerate(usuarios)}
    item_to_idx = {s: i for i, s in enumerate(canciones)}
    idx_to_item = {i: s for s, i in item_to_idx.items()}

    n_users = len(usuarios)
    n_items = len(canciones)

    row_idx = ratings["USER_ID"].map(user_to_idx).to_numpy()
    col_idx = ratings["ITEM_ID"].map(item_to_idx).to_numpy()
    values  = ratings["rating"].to_numpy(dtype=float)

    R_train = coo_matrix((values, (row_idx, col_idx)), shape=(n_users, n_items)).tocsr()

    conteo = np.asarray(R_train.getnnz(axis=1)).ravel()
    suma   = np.asarray(R_train.sum(axis=1)).ravel()
    media_u_train = np.divide(suma, conteo, out=np.zeros_like(suma, dtype=float), where=conteo > 0)

    R_centered = R_train.tocoo(copy=True).astype(float)
    R_centered.data = R_centered.data - media_u_train[R_centered.row]
    R_centered = R_centered.tocsr()

    k_svd = min(K, min(R_centered.shape) - 1)
    U, S, Vt = svds(R_centered, k=k_svd)
    factores_usuario_train = U * S

    print("Reentrenamiento completado.")

    return {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "usuarios": usuarios,
        "canciones": canciones,
        "n_users": n_users,
        "n_items": n_items,
        "U_train": U,
        "S_train": S,
        "Vt_train": Vt,
        "factores_usuario": factores_usuario_train,
        "factores_usuario_train": factores_usuario_train,
        "media_u_train": media_u_train,
        "R_train": R_train,
        "Vt": Vt,
        "media_usuarios": media_u_train,
    }


def reentrenar_en_background():
    """Corre el reentrenamiento en un hilo separado y actualiza _modelo."""
    global _modelo, _retraining
    try:
        nuevo_modelo = entrenar_modelo()
        guardar_modelo(nuevo_modelo)
        with _modelo_lock:
            _modelo = nuevo_modelo
        print("Modelo actualizado en memoria.")
    except Exception as e:
        print(f"Error en reentrenamiento: {e}")
    finally:
        _retraining = False


# ── Helpers de recomendación ─────────────────────────────────────────────────

def cargar_items() -> dict:
    items_db = {}
    with open(ITEMS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items_db[row["ITEM_ID"]] = row
    return items_db


def recomendar_populares(items_db: dict, n: int) -> list:
    items = sorted(
        items_db.values(),
        key=lambda x: float(x.get("popularidad", 0) or 0),
        reverse=True,
    )
    return [
        {
            "item_id":  i["ITEM_ID"],
            "titulo":   i.get("titulo", "?"),
            "artista":  i.get("artista", "?"),
            "genero":   i.get("genero", ""),
            "popularidad": i.get("popularidad", ""),
            "score":    0.0,
            "tipo":     "popular",
        }
        for i in items[:n]
    ]


def recomendar(user_id: str, modelo: dict, items_db: dict, n: int) -> list:
    # Soporta ambas keys que puede tener el pkl (original o nuestro)
    user_to_idx = modelo.get("user_to_idx", {})
    factores    = modelo.get("factores_usuario_train") if modelo.get("factores_usuario_train") is not None else modelo.get("factores_usuario")
    Vt          = modelo.get("Vt_train") if modelo.get("Vt_train") is not None else modelo.get("Vt")
    media       = modelo.get("media_u_train") if modelo.get("media_u_train") is not None else modelo.get("media_usuarios")
    idx_to_item = modelo.get("idx_to_item", {})

    if user_id not in user_to_idx or factores is None:
        return recomendar_populares(items_db, n)

    u_idx  = user_to_idx[user_id]
    scores = np.asarray(factores[u_idx] @ Vt + media[u_idx]).ravel()
    top    = np.argsort(scores)[::-1][:n]

    result = []
    for idx in top:
        item_id = idx_to_item.get(int(idx))
        if not item_id:
            continue
        meta = items_db.get(item_id, {})
        result.append({
            "item_id":    item_id,
            "titulo":     meta.get("titulo", "?"),
            "artista":    meta.get("artista", "?"),
            "genero":     meta.get("genero", ""),
            "popularidad": meta.get("popularidad", ""),
            "score":      round(float(scores[idx]), 4),
            "tipo":       "colaborativo",
        })
    return result


# ── Arranque ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    if MODEL_PATH.exists():
        cargar_modelo()
    else:
        print("No existe modelo_svd.pkl — entrenando desde cero...")
        global _modelo
        nuevo = entrenar_modelo()
        guardar_modelo(nuevo)
        _modelo = nuevo


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/recommendations")
def recommendations(user_id: str = Query(...)):
    global _modelo, _retraining

    items_db = cargar_items()

    with _modelo_lock:
        modelo_actual = _modelo

    if modelo_actual is None:
        return {"error": "Modelo no disponible aún, intenta en unos segundos."}

    user_to_idx = modelo_actual.get("user_to_idx", {})
    usuario_existe = user_id in user_to_idx

    recs = recomendar(user_id, modelo_actual, items_db, N_RESULTS)

    return {
        "user_id":        user_id,
        "usuario_conocido": usuario_existe,
        "recomendaciones": recs,
        "total":          len(recs),
        "reentrenando":   _retraining,
    }


@app.post("/users/{user_id}/interactions")
def agregar_interaccion(
    user_id: str,
    item_id: str = Query(...),
    event_type: str = Query("like"),
):
    """
    Agrega una interacción al CSV y lanza reentrenamiento en background.
    event_type puede ser: like | play | skip
    """
    global _retraining

    if event_type not in RATING_MAP:
        return {"error": f"event_type inválido. Usa: {list(RATING_MAP.keys())}"}

    # Agregar al CSV
    with open(INTERACTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_id, item_id, event_type])

    # Lanzar reentrenamiento si no hay uno ya corriendo
    if not _retraining:
        _retraining = True
        hilo = threading.Thread(target=reentrenar_en_background, daemon=True)
        hilo.start()
        msg = "Interacción guardada. Reentrenamiento iniciado en background."
    else:
        msg = "Interacción guardada. Ya hay un reentrenamiento en curso."

    return {"status": "ok", "mensaje": msg}


@app.get("/status")
def status():
    global _modelo, _retraining
    n_usuarios = len(_modelo.get("user_to_idx", {})) if _modelo else 0
    return {
        "modelo_cargado": _modelo is not None,
        "reentrenando":   _retraining,
        "n_usuarios":     n_usuarios,
    }
