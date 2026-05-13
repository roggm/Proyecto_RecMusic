# api_local.py
import os, pickle, csv, threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
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
MODEL_PATH        = DATA_DIR / "modelo_hybrid.pkl"

RATING_MAP     = {"like": 3.0, "play": 2.0, "skip": 1.0}
RATING_SCALE   = (1.0, 3.0)
K              = 50
N_RESULTS      = 10
ALPHA          = 0.5   # peso content-based (0=solo collab, 1=solo content)
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
    """
    Reentrena el modelo híbrido completo:
      1. SVD colaborativo sobre las interacciones actuales
      2. Conserva la feature_matrix de content-based del modelo anterior
         (no cambia con nuevas interacciones — solo cambia si se actualizan los items)
    """
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
    media_usuarios = np.divide(suma, conteo, out=np.zeros_like(suma,dtype=float), where=conteo>0)

    R_c = R_train.tocoo(copy=True).astype(float)
    R_c.data -= media_usuarios[R_c.row]
    R_c = R_c.tocsr()

    k_svd = min(K, min(R_c.shape)-1)
    U, S, Vt = svds(R_c, k=k_svd)
    factores_usuario = U * S

    # Reutilizar la parte content-based del modelo anterior (no depende de interacciones)
    modelo_anterior = _modelo or {}
    feature_matrix  = modelo_anterior.get("feature_matrix")
    item_ids        = modelo_anterior.get("item_ids", [])
    cb_item_to_idx  = modelo_anterior.get("cb_item_to_idx", {})
    audio_features  = modelo_anterior.get("audio_features", AUDIO_FEATURES)
    alpha           = modelo_anterior.get("alpha", ALPHA)

    print("Reentrenamiento completado.")
    return dict(
        # SVD colaborativo
        user_to_idx=user_to_idx, item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        n_users=n_users, n_items=n_items,
        factores_usuario=factores_usuario, Vt=Vt, media_usuarios=media_usuarios,
        # Content-based (preservado)
        feature_matrix=feature_matrix, item_ids=item_ids,
        cb_item_to_idx=cb_item_to_idx, audio_features=audio_features,
        # Híbrido
        alpha=alpha,
    )


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


def _normalizar(arr: np.ndarray) -> np.ndarray:
    """Normaliza un array al rango [0, 1]."""
    min_v, max_v = arr.min(), arr.max()
    if max_v == min_v:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def recomendar_usuario_conocido(user_id, modelo, items_db, n):
    user_to_idx    = modelo.get("user_to_idx", {})
    factores       = modelo.get("factores_usuario")
    Vt             = modelo.get("Vt")
    media          = modelo.get("media_usuarios")
    idx_to_item    = modelo.get("idx_to_item", {})
    feature_matrix = modelo.get("feature_matrix")
    item_ids       = modelo.get("item_ids", [])
    cb_item_to_idx = modelo.get("cb_item_to_idx", {})
    alpha          = modelo.get("alpha", ALPHA)

    if user_id not in user_to_idx or factores is None:
        return recomendar_populares(items_db, n)

    u_idx = user_to_idx[user_id]

    # ── Score colaborativo (SVD) ──────────────────────────────────────
    scores_col = np.asarray(factores[u_idx] @ Vt + media[u_idx]).ravel()
    # Mapear a ITEM_ID para poder hacer el merge
    df_col = pd.DataFrame({
        "ITEM_ID": [idx_to_item[i] for i in range(len(scores_col))],
        "score_col": scores_col,
    })

    # ── Score content-based (cosine similarity sobre historial) ───────
    if feature_matrix is not None and item_ids:
        # Obtener historial del usuario desde items_db (likes guardados en interactions.csv)
        interactions = pd.read_csv(INTERACTIONS_PATH)
        likes = interactions[
            (interactions["USER_ID"] == user_id) &
            (interactions["EVENT_TYPE"] == "like")
        ]["ITEM_ID"].tolist()
        if not likes:
            likes = interactions[
                (interactions["USER_ID"] == user_id) &
                (interactions["EVENT_TYPE"] == "play")
            ]["ITEM_ID"].tolist()

        if likes:
            indices = [cb_item_to_idx[i] for i in likes if i in cb_item_to_idx]
            perfil  = np.asarray(feature_matrix[indices].mean(axis=0)).ravel().reshape(1, -1)
            scores_cb_raw = cosine_similarity(perfil, feature_matrix).flatten()
        else:
            scores_cb_raw = np.zeros(len(item_ids))

        df_cb = pd.DataFrame({"ITEM_ID": item_ids, "score_cb": scores_cb_raw})
    else:
        # Sin feature_matrix, degradar a solo colaborativo
        alpha = 0.0
        df_cb = pd.DataFrame({"ITEM_ID": list(idx_to_item.values()), "score_cb": 0.0})

    # ── Combinar ──────────────────────────────────────────────────────
    df = df_cb.merge(df_col, on="ITEM_ID", how="left")
    df["score_col"] = df["score_col"].fillna(0)
    df["score_cb_norm"]  = _normalizar(df["score_cb"].to_numpy())
    df["score_col_norm"] = _normalizar(df["score_col"].to_numpy())
    df["score_final"] = alpha * df["score_cb_norm"] + (1 - alpha) * df["score_col_norm"]

    # Excluir ya escuchadas
    ya_vistas = set(
        pd.read_csv(INTERACTIONS_PATH)
        .query("USER_ID == @user_id")["ITEM_ID"].tolist()
    )
    df = df[~df["ITEM_ID"].isin(ya_vistas)]
    df = df.sort_values("score_final", ascending=False)

    result = []
    for _, row in df.iterrows():
        if len(result) >= n:
            break
        item_id = row["ITEM_ID"]
        meta = items_db.get(item_id, {})
        result.append({
            "item_id":     item_id,
            "titulo":      meta.get("titulo", "?"),
            "artista":     meta.get("artista", "?"),
            "genero":      meta.get("genero", ""),
            "popularidad": meta.get("popularidad", ""),
            "score":       round(float(row["score_final"]), 4),
            "tipo":        "hibrido",
        })
    return result


def foldin_recomendar(perfil, modelo, items_db, n):
    """
    Nuevo usuario: combina content-based (perfil de audio del formulario)
    con fold-in SVD usando alpha del modelo híbrido.
    """
    Vt             = modelo.get("Vt")
    media_arr      = modelo.get("media_usuarios")
    idx_to_item    = modelo.get("idx_to_item", {})
    item_to_idx    = modelo.get("item_to_idx", {})
    feature_matrix = modelo.get("feature_matrix")
    item_ids       = modelo.get("item_ids", [])
    cb_item_to_idx = modelo.get("cb_item_to_idx", {})
    alpha          = modelo.get("alpha", ALPHA)

    if Vt is None or media_arr is None:
        return recomendar_populares(items_db, n)

    generos_usr   = set(perfil.get("generos", []))
    features_usr  = perfil.get("features", {})
    ref_id        = perfil.get("cancion_referencia_id")
    media_global  = float(np.mean(media_arr))

    # ── Score content-based usando feature_matrix del modelo ─────────
    if feature_matrix is not None and item_ids:
        # Construir vector de perfil desde los audio features del formulario
        # usando las mismas columnas que tiene la feature_matrix del modelo
        audio_features = modelo.get("audio_features", AUDIO_FEATURES)

        # Buscar en items_db las canciones de los géneros seleccionados para
        # formar un perfil de referencia (igual que hace el notebook con likes)
        canciones_genero = [
            iid for iid, meta in items_db.items()
            if meta.get("genero", "") in generos_usr and iid in cb_item_to_idx
        ]

        if canciones_genero:
            indices = [cb_item_to_idx[i] for i in canciones_genero]
            perfil_base = np.asarray(feature_matrix[indices].mean(axis=0)).ravel().reshape(1, -1)
            scores_cb_raw = cosine_similarity(perfil_base, feature_matrix).flatten()
        else:
            scores_cb_raw = np.zeros(len(item_ids))

        df_cb = pd.DataFrame({"ITEM_ID": item_ids, "score_cb": scores_cb_raw})
    else:
        alpha = 0.0
        df_cb = pd.DataFrame({"ITEM_ID": list(idx_to_item.values()), "score_cb": 0.0})

    # ── Score colaborativo (fold-in SVD) ──────────────────────────────
    # Generar ratings sintéticos basados en géneros y audio features
    n_items = modelo.get("n_items", len(item_to_idx))
    r = np.zeros(n_items)
    for item_id, meta in items_db.items():
        if item_id not in item_to_idx:
            continue
        idx = item_to_idx[item_id]
        genero_match = meta.get("genero", "") in generos_usr
        sim, nf = 0.0, 0
        for feat in AUDIO_FEATURES:
            vi, vu = meta.get(feat), features_usr.get(feat)
            if vi is not None and vu is not None:
                try:
                    sim += 1.0 - abs(float(vi) - float(vu))
                    nf  += 1
                except (ValueError, TypeError):
                    pass
        if nf > 0:
            sim /= nf
        if ref_id and item_id == ref_id:
            r[idx] = 3.0
        elif genero_match and sim >= 0.6:
            r[idx] = 3.0
        elif not genero_match and sim < 0.4:
            r[idx] = 1.0
        else:
            r[idx] = 2.0

    r_centrado = np.where(r > 0, r - media_global, 0.0)
    u_nuevo    = r_centrado @ Vt.T / (np.diag(Vt @ Vt.T).mean() + 1e-9)
    scores_col = u_nuevo @ Vt + media_global

    df_col = pd.DataFrame({
        "ITEM_ID": [idx_to_item.get(i) for i in range(len(scores_col))],
        "score_col": scores_col,
    }).dropna(subset=["ITEM_ID"])

    # ── Combinar ──────────────────────────────────────────────────────
    df = df_cb.merge(df_col, on="ITEM_ID", how="left")
    df["score_col"] = df["score_col"].fillna(0)
    df["score_cb_norm"]  = _normalizar(df["score_cb"].to_numpy())
    df["score_col_norm"] = _normalizar(df["score_col"].to_numpy())
    df["score_final"] = alpha * df["score_cb_norm"] + (1 - alpha) * df["score_col_norm"]
    df = df.sort_values("score_final", ascending=False)

    result = []
    for _, row in df.iterrows():
        if len(result) >= n:
            break
        item_id = row["ITEM_ID"]
        meta = items_db.get(item_id, {})
        result.append({
            "item_id":     item_id,
            "titulo":      meta.get("titulo", "?"),
            "artista":     meta.get("artista", "?"),
            "genero":      meta.get("genero", ""),
            "popularidad": meta.get("popularidad", ""),
            "score":       round(float(row["score_final"]), 4),
            "tipo":        "hibrido_nuevo_usuario",
        })
    return result

class PerfilUsuario(BaseModel):
    user_id: str = "anonimo"
    generos: list = []
    features: dict = {}
    cancion_referencia_id: Optional[str] = None


@app.on_event("startup")
def startup():
    global _modelo
    if MODEL_PATH.exists():
        cargar_modelo()
    else:
        # modelo_hybrid.pkl no existe todavía — intentar con SVD solo como fallback
        svd_path = DATA_DIR / "modelo_svd.pkl"
        if svd_path.exists():
            print("AVISO: modelo_hybrid.pkl no encontrado. Cargando modelo_svd.pkl como fallback.")
            print("       Ejecuta hybrid.ipynb para generar el modelo híbrido completo.")
            with open(svd_path, "rb") as f:
                _modelo = pickle.load(f)
        else:
            print("No existe modelo_hybrid.pkl ni modelo_svd.pkl — entrenando SVD desde cero...")
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
    global _modelo, _retraining
    items_db = cargar_items()
    with _modelo_lock:
        modelo_actual = _modelo
    if modelo_actual is None:
        return {"error": "Modelo no disponible aun."}

    user_id = perfil.user_id.strip() or "anonimo"
    usuario_conocido = user_id in modelo_actual.get("user_to_idx", {})

    if usuario_conocido:
        # Ya está en el modelo — usar recomendación híbrida normal
        recs = recomendar_usuario_conocido(user_id, modelo_actual, items_db, N_RESULTS)
        return {"user_id": user_id, "usuario_conocido": True, "tipo": "hibrido",
                "recomendaciones": recs, "total": len(recs)}

    # Nuevo usuario — fold-in inmediato
    recs = foldin_recomendar(perfil.dict(), modelo_actual, items_db, N_RESULTS)

    # Guardar interacciones sintéticas para que el reentrenamiento lo aprenda
    generos_set = set(perfil.generos)
    with open(INTERACTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for rec in recs:
            item_id = rec.get("item_id")
            if not item_id:
                continue
            meta = items_db.get(item_id, {})
            genero_match = meta.get("genero", "") in generos_set
            event_type = "like" if genero_match else "play"
            writer.writerow([user_id, item_id, event_type])

    # Reentrenamiento en background para que la próxima visita use el modelo real
    if not _retraining:
        _retraining = True
        threading.Thread(target=reentrenar_en_background, daemon=True).start()

    return {"user_id": user_id, "usuario_conocido": False, "tipo": "hibrido_nuevo_usuario",
            "recomendaciones": recs, "total": len(recs), "reentrenando": True}


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
