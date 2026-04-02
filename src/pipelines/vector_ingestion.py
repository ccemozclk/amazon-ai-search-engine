import os
import sys
import uuid
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import read_config

class VectorIngestionPipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config   = read_config(config_path)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.processed_data_path = os.path.join(self.base_dir, self.config["paths"]["processed_data"])
        self.collection_name     = self.config["qdrant"]["collection_st"]
        self.vector_size         = self.config["model"]["vector_size"]

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.model  = SentenceTransformer(self.config["model"]["name"])

    def setup_qdrant(self):
        if self.client.collection_exists(self.collection_name):
            count = self.client.count(self.collection_name).count
            if count > 0:
                print(f"Collection '{self.collection_name}' already has {count} vectors. Skipping ingestion.")
                return False
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )
        print("Qdrant Collection is ready!")
        return True

    def run_pipeline(self, limit=None):
        if not self.setup_qdrant():
            return

        print(f"Reading cleaned data: {self.processed_data_path}")
        df = pd.read_parquet(self.processed_data_path)
        if limit:
            print(f"Only the first {limit} rows taken.")
            df = df.head(limit)

        documents    = df["document"].tolist()
        ids          = df["parent_asin"].fillna(pd.Series(range(len(df)), index=df.index).astype(str)).tolist()
        payloads     = df[["title", "price", "parent_asin", "average_rating", "review_count"]].to_dict(orient="records")
        batch_size   = 64

        print(f"Vectorization and Qdrant loading starting (batch_size={batch_size})...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Uploading to Qdrant"):
            batch_docs     = documents[i : i + batch_size]
            batch_ids      = ids[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            embeddings     = self.model.encode(batch_docs).tolist()

            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx))),
                    vector=vector,
                    payload=payload
                )
                for idx, vector, payload in zip(batch_ids, embeddings, batch_payloads)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"\nSUCCESS! {len(documents)} items loaded into Qdrant.")

if __name__ == "__main__":
    pipeline = VectorIngestionPipeline()
    pipeline.run_pipeline()
