import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import redis.asyncio as redis
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from prometheus_fastapi_instrumentator import Instrumentator
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(base_dir)
from src.utils.common import read_config

class LSTMAutoComplete(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out
    
st_model = None
w2v_model = None
item2vec_model = None
lstm_model = None
word_to_idx = {}
idx_to_word = {}
db_client = None
redis_client = None


def clean_nan(value, default_value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or (isinstance(value, str) and value.lower().strip() == "nan"):
        return default_value
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    global st_model, w2v_model, item2vec_model, lstm_model, word_to_idx, idx_to_word, db_client, redis_client
    config = read_config("config/config.yaml")
    
    print("1. Loading SentenceTransformer (Modern AI)...")
    st_model = SentenceTransformer(config["model"]["name"])
    
    print("2. Loading Word2Vec (Traditional AI)...")
    try:
        w2v_path = os.path.join(base_dir, config["paths"]["word2vec_model"])
        w2v_model = Word2Vec.load(w2v_path)
    except Exception as e:
        print(f"Word2Vec could not be loaded: {e}")

    print("3. Loading Item2Vec (Recommendation System) ...")
    try:
        i2v_path = os.path.join(base_dir, "models", "amazon_item2vec.model")
        item2vec_model = Word2Vec.load(i2v_path)
    except Exception as e:
        print(f"Item2Vec could not be loaded: {e}")

    print("4. LSTM Auto-Complete Is Loading...")
    try:
        vocab_path = os.path.join(base_dir, "models", "lstm_vocab.json")
        model_path = os.path.join(base_dir, "models", "amazon_lstm.pth")
        
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
            word_to_idx = vocab_data["word_to_idx"]
            idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
            
        lstm_model = LSTMAutoComplete(len(word_to_idx), 64, 128)
        lstm_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        lstm_model.eval()
    except Exception as e:
        print(f"LSTM could not be loaded: {e}")
    
    print("5. Connecting to Qdrant and Redis...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    db_client = QdrantClient(host=qdrant_host, port=int(os.getenv("QDRANT_PORT", 6333)))
    try:
        redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), db=0, decode_responses=True)
        await redis_client.ping()
    except:
        redis_client = None
    yield


app = FastAPI(title="Amazon Double Engined AI Search", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    model_type: str = "sentence_transformer"

class AutoCompleteRequest(BaseModel):
    text: str
    top_k: int = 3

@app.post("/autocomplete")
async def autocomplete(request: AutoCompleteRequest):
    if not lstm_model or not word_to_idx:
        return {"suggestions": []}

    tokens = request.text.lower().split()
    if not tokens: return {"suggestions": []}
    MAX_SEQ_LEN = 5
    input_seq = [word_to_idx.get(w, word_to_idx.get("<UNK>", 0)) for w in tokens[-MAX_SEQ_LEN:]]
    padded_input = [word_to_idx.get("<PAD>", 0)] * (MAX_SEQ_LEN - len(input_seq)) + input_seq

    x_tensor = torch.tensor([padded_input], dtype=torch.long)

    with torch.no_grad():
        predictions = lstm_model(x_tensor)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, request.top_k)

    suggestions = []
    for i in range(request.top_k):
        word = idx_to_word.get(top_indices[i].item(), "")
        if word not in ["<PAD>", "<UNK>"]:
            suggestions.append(word)

    return {"suggestions": suggestions}

@app.post("/search")
async def search_products(request: SearchRequest):
    normalized_text = request.query.lower().strip()
    cache_key = f"amazon_{request.model_type}:{normalized_text}:{request.top_k}"
    
    if redis_client:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

    
    if request.model_type == "word2vec" and w2v_model:
        collection_name = "amazon_fashion_w2v"
        tokens = simple_preprocess(normalized_text)
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        search_vector = np.mean(vectors, axis=0).tolist() if vectors else np.zeros(w2v_model.vector_size).tolist()
    else:
        collection_name = "amazon_fashion"
        search_vector = st_model.encode(normalized_text).tolist()

    search_results = db_client.query_points(collection_name=collection_name, query=search_vector, limit=request.top_k)
    
    formatted_results = []
    for item in search_results.points:
        payload = getattr(item, "payload", {}) if not isinstance(item, tuple) else (item[2] if len(item) > 2 else {})
        score = getattr(item, "score", 0.0) if not isinstance(item, tuple) else (item[1] if len(item) > 1 else 0.0)

        safe_asin = clean_nan(payload.get("parent_asin"), "ASIN Doesnt Exist")

        recommendations = []
        if item2vec_model and safe_asin in item2vec_model.wv:
            try:
                similar_items = item2vec_model.wv.most_similar(safe_asin, topn=2)
                recommendations = [rec_item[0] for rec_item in similar_items]
            except:
                pass

        formatted_results.append({
            "score": round(clean_nan(score, 0.0), 4),
            "product_name": clean_nan(payload.get("title"), "İsimsiz Ürün"),
            "price": clean_nan(payload.get("price"), "Fiyat Yok"),
            "asin": safe_asin,
            "rating": clean_nan(payload.get("average_rating"), 0.0),
            "review_count": clean_nan(payload.get("review_count"), 0),
            "recommendations": recommendations
        })
        
    final_response = {"search_word": request.query, "model_used": request.model_type, "results": formatted_results}
    if redis_client and formatted_results:
        await redis_client.setex(cache_key, 3600, json.dumps(final_response))
        
    return final_response