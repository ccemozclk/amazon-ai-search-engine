import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..utils.common import read_config
from ..utils.logger import logger

class InferencePipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)
        
       
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = "amazon_fashion"

        logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        logger.info("Connected to Qdrant successfully!")

        
        model_name = self.config.get("model", {}).get("name", "all-MiniLM-L6-v2")
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Model Loaded! Ready for semantic search.")

    def encode_text(self, text: str) -> list[float]:
        """Converts the user's search text into a vector format."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def search_products(self, query_text: str, top_k: int = 3):
        """It performs a semantic search on Qdrant."""
        logger.info(f"SEARCHING: '{query_text}'")
        try:
            query_vector = self.encode_text(query_text)
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
            
            results = []
            for hit in search_result:
                
                results.append(
                    {
                        "score": hit.score,
                        "title": hit.payload.get("title", "Unknown Title"),
                        "price": hit.payload.get("price", "N/A"),
                        "asin": hit.payload.get("parent_asin", "Unknown ASIN"),
                    }
                )
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []