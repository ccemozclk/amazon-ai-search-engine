import os
import sys
import uuid
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import read_config

class Word2VecIngestion:
    def __init__(self):
        self.config = read_config("config/config.yaml")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.processed_data_path = os.path.join(self.base_dir, self.config["paths"]["processed_data"])
        self.model_path = os.path.join(self.base_dir, self.config["paths"]["word2vec_model"])
        
        
        self.collection_name = "amazon_fashion_w2v"
        
        print("Word2Vec Model is Loading...")
        self.w2v_model = Word2Vec.load(self.model_path)
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def get_document_embedding(self, text):
        """It creates a sentence vector by averaging the vectors of the words (Mean Pooling)."""
        tokens = simple_preprocess(str(text))
        vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
        
        if vectors:
            return np.mean(vectors, axis=0).tolist()
        else:
            return np.zeros(self.w2v_model.vector_size).tolist()

    def run(self, limit=5000):
        print(f"Data Is Reading")
        df = pd.read_parquet(self.processed_data_path).head(limit)

        
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.w2v_model.vector_size, distance=models.Distance.COSINE),
        )

        documents = df["document"].tolist()
        ids = df["parent_asin"].fillna(pd.Series(df.index).astype(str)).tolist()
        payloads = df[["title", "price", "parent_asin", "average_rating", "review_count"]].to_dict(orient="records")

        batch_size = 64
        for i in tqdm(range(0, len(documents), batch_size), desc="Word2Vec Vektörleri Qdrant'a Yükleniyor"):
            batch_docs = documents[i : i + batch_size]
            embeddings = [self.get_document_embedding(doc) for doc in batch_docs]
            
            points = [
                models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx))), vector=vec, payload=pay)
                for idx, vec, pay in zip(ids[i:i+batch_size], embeddings, payloads[i:i+batch_size])
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        print("✅ Word2Vec data has been successfully uploaded to Qdrant!")

if __name__ == "__main__":
    Word2VecIngestion().run(limit=5000)