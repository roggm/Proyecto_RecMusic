# api_local.py
import os, pickle, csv, threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR          = Path("data/processed")
ITEMS_PATH        = DATA_DIR / "items.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"
MODEL_PATH        = DATA_DIR / "modelo_svd.pkl"

RATING_MAP     = {"like": 3.0, "play": 2.0, "skip": 1.0}
RATING_SCALE   = (1.0, 3.0)
K              = 50
N_RESULTS      = 10
AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness"]

_modelo      = None
_modelo_lock = threading.Lock()
_retraining  = False


def cargar_modelo():
    global _modelo
    with open(MODEL_PATH, "rb") as f:
        _modelo = pickle.load(f)
    print("Modelo cargado desde disco.")


def guardar_modelo(modelo):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(modelo, f)
    print("Modelo guardado en disco.")


def entrenar_modelo():
    print("Iniciando reentrenamiento...")
    interactions = pd.read_csv(INTERACTIONS_PATH)
    interactions["rating"] = interactions["EVENT_TYPE"].map(RATING_MAP)
    ratings = interactions.groupby(["USER_ID","ITEM_ID"])["rating"].max().reset_index()

    usuarios  = sorted(ratings["USER_ID"].unique())
    canciones = sorted(ratings["ITEM_ID"].unique())
    user_to_idx = {u: i for i, u in enumerate(usuarios)}
    item_to_idx = {s: i for i, s in enumerate(canciones)}
    idx_to_item = {i: s for s, i in item_to_idx.items()}
    n_users, n_items = len(usuarios), len(canciones)

    row_idx = ratings["USER_ID"].map(user_to_idx).to_numpy()
    col_idx = ratings["ITEM_ID"].map(item_to_idx).to_numpy()
    values  = ratings["rating"].to_numpy(dtype=float)
    R_train = coo_matrix((values,(row_idx,col_idx)), shape=(n_users,n_items)).tocsr()

    conteo = np.asarray(R_train.getnnz(axis=1)).ravel()
    suma   = np.asarray(R_train.sum(axis=1)).ravel()
    media_u_train = np.divide(suma, conteo, out=np.zeros_like(suma,dtype=float), where=conteo>0)

    R_c = R_train.tocoo(copy=True).astype(float)
    R_c.data -= media_u_train[R_c.row]
    R_c = R_c.tocsr()

    k_svd = min(K, min(R_c.shape)-1)
    U, S, Vt = svds(R_c, k=k_svd)
    print("Reentrenamiento completado.")
    return dict(user_to_idx=user_to_idx, item_to_idx=item_to_idx, idx_to_item=idx_to_item,
                usuarios=usuarios, canciones=canciones, n_users=n_users, n_items=n_items,
                U_train=U, S_train=S, Vt_train=Vt, factores_usuario_train=U*S,
                media_u_train=media_u_train, R_train=R_train)


def reentrenar_en_background():
    global _modelo, _retraining
    try:
        m = entrenar_modelo()
        guardar_modelo(m)
        with _modelo_lock:
            _modelo = m
        print("Modelo actualizado en memoria.")
    except Exception as e:
        print(f"Error reentrenamiento: {e}")
    finally:
        _retraining = False


def cargar_items():
    items_db = {}
    with open(ITEMS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items_db[row["ITEM_ID"]] = row
    return items_db


def recomendar_populares(items_db, n):
    items = sorted(items_db.values(), key=lambda x: float(x.get("popularidad",0) or 0), reverse=True)
    return [{"item_id":i["ITEM_ID"],"titulo":i.get("titulo","?"),"artista":i.get("artista","?"),
             "genero":i.get("genero",""),"popularidad":i.get("popularidad",""),"score":0.0,"tipo":"popular"}
            for i in items[:n]]


def recomendar_usuario_conocido(user_id, modelo, items_db, n):
    user_to_idx = modelo.get("user_to_idx", {})
    factores    = modelo.get("factores_usuario_train")
    Vt          = modelo.get("Vt_train")
    media       = modelo.get("media_u_train")
    idx_to_item = modelo.get("idx_to_item", {})

    if user_id not in user_to_idx or factores is None:
        return recomendar_populares(items_db, n)

    u_idx  = user_to_idx[user_id]
    scores = np.asarray(factores[u_idx] @ Vt + media[u_idx]).ravel()
    top = np.argsort(scores)[::-1][:n*2]  # agarramos más por si algunos son -inf
    result = []
    for idx in top:
        if len(result) >= n: break
        if scores[idx] == -np.inf: continue
        item_id = idx_to_item.get(int(idx))
        if not item_id: continue
        meta = items_db.get(item_id, {})
        result.append({"item_id":item_id,"titulo":meta.get("titulo","?"),"artista":meta.get("artista","?"),
                       "genero":meta.get("genero",""),"popularidad":meta.get("popularidad",""),
                       "score":round(float(scores[idx]),4),"tipo":"colaborativo"})
    return result


def generar_ratings_foldin(perfil, items_db, modelo):
    generos      = set(perfil.get("generos", []))
    features_usr = perfil.get("features", {})
    ref_id       = perfil.get("cancion_referencia_id")
    item_to_idx  = modelo.get("item_to_idx", {})
    n_items      = modelo.get("n_items", len(item_to_idx))
    r = np.zeros(n_items)

    for item_id, meta in items_db.items():
        if item_id not in item_to_idx: continue
        idx = item_to_idx[item_id]

        genero_match = meta.get("genero","") in generos

        sim = 0.0
        nf  = 0
        for feat in AUDIO_FEATURES:
            vi = meta.get(feat)
            vu = features_usr.get(feat)
            if vi is not None and vu is not None:
                try:
                    sim += 1.0 - abs(float(vi) - float(vu))
                    nf  += 1
                except (ValueError, TypeError):
                    pass
        if nf > 0: sim /= nf

        if not genero_match and sim < 0.4:
            rating = 1.0
        elif genero_match and sim >= 0.6:
            rating = 3.0
        else:
            rating = 2.0

        if ref_id and item_id == ref_id:
            rating = 3.0

        r[idx] = rating
    return r


def foldin_recomendar(perfil, modelo, items_db, n):
    Vt          = modelo.get("Vt_train")
    S           = modelo.get("S_train")
    media_arr   = modelo.get("media_u_train")
    idx_to_item = modelo.get("idx_to_item", {})

    if Vt is None or S is None:
        return recomendar_populares(items_db, n)

    media_global = float(np.mean(media_arr))
    r            = generar_ratings_foldin(perfil, items_db, modelo)
    r_centrado   = np.where(r > 0, r - media_global, 0.0)
    u_nuevo      = r_centrado @ Vt.T / (S + 1e-9)
    scores       = u_nuevo @ Vt + media_global

    # Excluir items ya "vistos" poniendolos en nan
    scores[r == 0] = np.nan

    result = []
    # Ordenar ignorando nan
    valid_idx = np.where(np.isfinite(scores))[0]
    top = valid_idx[np.argsort(scores[valid_idx])[::-1][:n]]

    for idx in top:
        item_id = idx_to_item.get(int(idx))
        if not item_id: continue
        meta = items_db.get(item_id, {})
        result.append({
            "item_id":     item_id,
            "titulo":      meta.get("titulo", "?"),
            "artista":     meta.get("artista", "?"),
            "genero":      meta.get("genero", ""),
            "popularidad": meta.get("popularidad", ""),
            "score":       round(float(scores[idx]), 4),
            "tipo":        "foldin",
        })
    return result

class PerfilUsuario(BaseModel):
    generos: list = []
    features: dict = {}
    cancion_referencia_id: Optional[str] = None


@app.on_event("startup")
def startup():
    global _modelo
    if MODEL_PATH.exists():
        cargar_modelo()
    else:
        print("No existe modelo_svd.pkl, entrenando desde cero...")
        nuevo = entrenar_modelo()
        guardar_modelo(nuevo)
        _modelo = nuevo


@app.get("/recommendations")
def recommendations(user_id: str = Query(...)):
    global _modelo, _retraining
    items_db = cargar_items()
    with _modelo_lock:
        modelo_actual = _modelo
    if modelo_actual is None:
        return {"error": "Modelo no disponible aun."}
    usuario_existe = user_id in modelo_actual.get("user_to_idx", {})
    recs = recomendar_usuario_conocido(user_id, modelo_actual, items_db, N_RESULTS)
    return {"user_id":user_id,"usuario_conocido":usuario_existe,
            "recomendaciones":recs,"total":len(recs),"reentrenando":_retraining}


@app.post("/recommendations/new-user")
def recommendations_new_user(perfil: PerfilUsuario):
    global _modelo
    items_db = cargar_items()
    with _modelo_lock:
        modelo_actual = _modelo
    if modelo_actual is None:
        return {"error": "Modelo no disponible aun."}
    recs = foldin_recomendar(perfil.dict(), modelo_actual, items_db, N_RESULTS)
    return {"usuario_conocido":False,"tipo":"foldin","recomendaciones":recs,"total":len(recs)}


@app.post("/users/{user_id}/interactions")
def agregar_interaccion(user_id: str, item_id: str = Query(...), event_type: str = Query("like")):
    global _retraining
    if event_type not in RATING_MAP:
        return {"error": f"event_type invalido. Usa: {list(RATING_MAP.keys())}"}
    with open(INTERACTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([user_id, item_id, event_type])
    if not _retraining:
        _retraining = True
        threading.Thread(target=reentrenar_en_background, daemon=True).start()
        msg = "Interaccion guardada. Reentrenamiento iniciado."
    else:
        msg = "Interaccion guardada. Ya hay un reentrenamiento en curso."
    return {"status":"ok","mensaje":msg}


@app.get("/status")
def status():
    global _modelo, _retraining
    n_usuarios = len(_modelo.get("user_to_idx",{})) if _modelo else 0
    return {"modelo_cargado":_modelo is not None,"reentrenando":_retraining,"n_usuarios":n_usuarios}
