import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
import redis.asyncio as redis
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(base_dir)

from src.pipelines.inference_pipeline import InferencePipeline

pipeline     = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, redis_client

    print("AI Search Engine is initializing via InferencePipeline...")
    try:
        pipeline = InferencePipeline()
        print("InferencePipeline initialized successfully.")
    except Exception as e:
        print(f"CRITICAL: InferencePipeline failed to initialize: {e}")
        pipeline = None

    print("Connecting to Redis...")
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=0,
            decode_responses=True
        )
        await redis_client.ping()
        print("Redis connected successfully!")
    except Exception:
        redis_client = None
        print("Redis connection failed, continuing without cache.")

    yield

    # --- Shutdown cleanup ---
    if redis_client:
        await redis_client.aclose()
        print("Redis connection closed.")
    if pipeline:
        pipeline.close()
        print("Qdrant connection closed.")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Amazon Double Engined AI Search", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)


class SearchRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=500, description="User's search text")
    top_k:      int = Field(5, ge=1, le=50, description="Maximum number of results")
    model_type: Literal["sentence_transformer", "word2vec"] = Field(
        "sentence_transformer", description="The model to be used"
    )

class AutoCompleteRequest(BaseModel):
    text:  str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(3, ge=1, le=10)


@app.get("/health")
async def health():
    pipeline_ok = pipeline is not None
    qdrant_ok   = False
    if pipeline_ok:
        try:
            pipeline.client.get_collections()
            qdrant_ok = True
        except Exception:
            pass
    return {
        "status":   "ok" if (pipeline_ok and qdrant_ok) else "degraded",
        "pipeline": pipeline_ok,
        "qdrant":   qdrant_ok,
        "redis":    redis_client is not None,
    }


@app.post("/autocomplete")
@limiter.limit("60/minute")
async def autocomplete(request: Request, body: AutoCompleteRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service is initializing, please try again.")
    suggestions = await asyncio.to_thread(pipeline.get_autocomplete, body.text, body.top_k)
    return {"suggestions": suggestions}


@app.post("/search")
@limiter.limit("30/minute")
async def search_products(request: Request, body: SearchRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service is initializing, please try again.")

    if body.model_type == "word2vec" and pipeline.w2v_model is None:
        raise HTTPException(
            status_code=400,
            detail="The Word2Vec model is not installed or could not be started on the server."
        )
    elif body.model_type == "sentence_transformer" and pipeline.st_model is None:
        raise HTTPException(
            status_code=500,
            detail="The SentenceTransformer model is not loaded on the server."
        )

    normalized_text = body.query.lower().strip()
    cache_key = f"amazon_{body.model_type}:{normalized_text}:{body.top_k}"

    if redis_client:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

    formatted_results = await asyncio.to_thread(
        pipeline.search_products, normalized_text, body.top_k, body.model_type
    )

    final_response = {
        "search_word": body.query,
        "model_used":  body.model_type,
        "results":     formatted_results,
    }

    if redis_client:
        await redis_client.setex(cache_key, 3600, json.dumps(final_response))

    return final_response
