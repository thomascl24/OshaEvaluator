import logging
from contextlib import asynccontextmanager
from datetime import datetime
from os import getcwd
from os.path import join
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from joblib import load
from redis import asyncio

from pydantic import BaseModel

import os
import numpy as np
import io

from model import ManualChunker, ExplainNLP, Predictor
from qdrant_client import QdrantClient


app = FastAPI()

LOCAL_REDIS_URL = "redis://localhost"

@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    logging.info("Starting up API")

    # Load the Model on Startup
    global model
    model = ExplainNLP.load_from_checkpoint('./epoch=3-step=68672.ckpt')

    # Load the vector datastore
    print('Loading vector store...')
    vector_store_path = "./qdrant_data"
    lock_file = os.path.join(vector_store_path, ".lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)
    os.makedirs(vector_store_path, exist_ok=True)
    
    global qdrant_client
    qdrant_client = QdrantClient(path=vector_store_path)

    # Load the Redis Cache
    HOST_URL = os.getenv("REDIS_URL", LOCAL_REDIS_URL)  
    redis = asyncio.from_url(HOST_URL, port=6379, encoding="utf8", decode_responses=True)

    # We initialize the connection to Redis and declare that all keys in the
    # database will be prefixed with w255-cache-predict. Do not change this
    # prefix for the submission.
    FastAPICache.init(RedisBackend(redis), prefix="nli-cache-prediction")

    yield
    # We don't need a shutdown event for our system, but we could put something
    # here after the yield to deal with things during shutdown
    logging.info("Shutting down NLI API")

subapp = FastAPI(lifespan=lifespan_mechanism)

# Endpoints

class Output(BaseModel):
    predictions: list[str]
    premises: list[str]
    hypotheses: list[str]

@subapp.post("/predict")
async def chunk_manual(file: UploadFile = File(...)):
    """Accept a JSON‑lines upload and return DataFrame rows as JSON."""
    print(f"Received file: {file.filename}, content_type: {file.content_type}")

    if file.content_type not in {"application/json", "text/plain"}:
        raise HTTPException(
            415, detail="Upload must be JSON‑lines (.jsonl) text."
        )

    # Treat the underlying SpooledTemporaryFile as text
    text_stream = io.TextIOWrapper(file.file, encoding="utf-8")

    df = ManualChunker(text_stream).chunk_data()
    predictor = Predictor(
        model=model, 
        batch_size=3, 
        chunked_steps=df, 
        vector_client=qdrant_client
    )

    res = predictor.predict()
    return Output(
        predictions=res['predictions'],
        premises=res['premises'],
        hypotheses=res['hypotheses'],
    )

@subapp.get("/health")
async def health():
    return {"time": f"{datetime.now().isoformat()}"}    
