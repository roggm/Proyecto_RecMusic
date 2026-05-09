"""
Lambda handler — Sistema Recomendador de Música
Carga el modelo SVD desde S3 y genera recomendaciones localmente con numpy.
No requiere Amazon Personalize.
"""
import json
import os
import csv
import io
import pickle
import boto3
import numpy as np
from functools import lru_cache
 
s3 = boto3.client('s3')
 
S3_BUCKET  = os.environ['S3_BUCKET']
ITEMS_KEY  = os.environ.get('ITEMS_KEY',  'processed/items.csv')
MODELO_KEY = os.environ.get('MODELO_KEY', 'processed/modelo_svd.pkl')
N_RESULTS  = int(os.environ.get('N_RESULTS', '10'))
 
CORS = {
    'Access-Control-Allow-Origin' : '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET,OPTIONS'
}
 
 
def respuesta(status, body):
    return {'statusCode': status, 'headers': CORS, 'body': json.dumps(body, ensure_ascii=False)}
 
 
@lru_cache(maxsize=1)
def cargar_modelo():
    """Descarga y deserializa el modelo SVD desde S3. Se cachea por instancia."""
    print(f'Cargando modelo desde s3://{S3_BUCKET}/{MODELO_KEY}')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODELO_KEY)
    return pickle.loads(obj['Body'].read())
 
 
@lru_cache(maxsize=1)
def cargar_items():
    """Descarga items.csv desde S3 y lo indexa por ITEM_ID."""
    print(f'Cargando items desde s3://{S3_BUCKET}/{ITEMS_KEY}')
    obj     = s3.get_object(Bucket=S3_BUCKET, Key=ITEMS_KEY)
    content = obj['Body'].read().decode('utf-8')
    reader  = csv.DictReader(io.StringIO(content))
    return {row['ITEM_ID']: row for row in reader}
 
 
def recomendar_por_modelo(user_id, modelo, items_db, n):
    """Genera recomendaciones usando el SVD cargado desde S3."""
    user_to_idx      = modelo['user_to_idx']
    idx_to_item      = modelo['idx_to_item']
    factores_usuario = modelo['factores_usuario']
    Vt               = modelo['Vt']
    media_usuarios   = modelo['media_usuarios']
 
    if user_id not in user_to_idx:
        print(f'Usuario {user_id} no encontrado — usando fallback por popularidad')
        return recomendar_populares(items_db, n)
 
    u_idx  = user_to_idx[user_id]
    scores = factores_usuario[u_idx] @ Vt + media_usuarios[u_idx]
 
    top_indices = np.argsort(scores)[::-1][:n]
    resultado   = []
    for idx in top_indices:
        item_id = idx_to_item.get(int(idx))
        if not item_id:
            continue
        meta = items_db.get(item_id, {})
        resultado.append({
            'item_id'    : item_id,
            'titulo'     : meta.get('titulo',     'Desconocido'),
            'artista'    : meta.get('artista',    'Desconocido'),
            'album'      : meta.get('album',      ''),
            'genero'     : meta.get('genero',     ''),
            'popularidad': meta.get('popularidad', ''),
            'score'      : round(float(scores[idx]), 4)
        })
    return resultado
 
 
def recomendar_populares(items_db, n):
    """Fallback: retorna las N canciones más populares del dataset."""
    items_list = list(items_db.values())
    items_list.sort(key=lambda x: float(x.get('popularidad', 0) or 0), reverse=True)
    return [
        {
            'item_id'    : i.get('ITEM_ID', ''),
            'titulo'     : i.get('titulo',  'Desconocido'),
            'artista'    : i.get('artista', 'Desconocido'),
            'genero'     : i.get('genero',  ''),
            'popularidad': i.get('popularidad', ''),
            'score'      : 0.0
        }
        for i in items_list[:n]
    ]
 
 
def handler(event, context):
    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': CORS, 'body': ''}
 
    try:
        params  = event.get('queryStringParameters') or {}
        user_id = params.get('user_id')
 
        if not user_id:
            body    = json.loads(event.get('body') or '{}')
            user_id = body.get('user_id')
 
        if not user_id:
            return respuesta(400, {'error': 'user_id es requerido'})
 
        modelo   = cargar_modelo()
        items_db = cargar_items()
        recs     = recomendar_por_modelo(user_id, modelo, items_db, N_RESULTS)
 
        return respuesta(200, {
            'user_id'         : user_id,
            'recomendaciones' : recs,
            'total'           : len(recs)
        })
 
    except Exception as e:
        print(f'ERROR: {e}')
        return respuesta(500, {'error': 'Error interno del servidor'})