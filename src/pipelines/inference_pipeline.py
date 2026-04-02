import os
import sys
import json
import math
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(base_dir)
from src.utils.common import read_config
from src.utils.logger import logger
from src.models import LSTMAutoComplete

_lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_nan(value, default_value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or (isinstance(value, str) and value.lower().strip() == "nan"):
        return default_value
    return value

class InferencePipeline:
    def __init__(self, config_path="config/config.yaml"):
        self.config = read_config(config_path)

        self.collection_st  = self.config["qdrant"]["collection_st"]
        self.collection_w2v = self.config["qdrant"]["collection_w2v"]

        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        model_name = self.config.get("model", {}).get("name", "all-MiniLM-L6-v2")
        logger.info(f"Loading SentenceTransformer: {model_name}...")
        try:
            self.st_model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformer could not be loaded: {e}")
            self.st_model = None

        logger.info("Loading Word2Vec...")
        try:
            w2v_path = os.path.join(base_dir, self.config["paths"]["word2vec_model"])
            self.w2v_model = Word2Vec.load(w2v_path)
        except Exception as e:
            logger.warning(f"Word2Vec could not be loaded: {e}")
            self.w2v_model = None

        logger.info("Loading Item2Vec...")
        try:
            i2v_path = os.path.join(base_dir, self.config["paths"]["item2vec_model"])
            self.item2vec_model = Word2Vec.load(i2v_path)
        except Exception as e:
            logger.warning(f"Item2Vec could not be loaded: {e}")
            self.item2vec_model = None

        logger.info(f"Loading LSTM Auto-Complete (device: {_lstm_device})...")
        self.lstm_model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        try:
            vocab_path = os.path.join(base_dir, self.config["paths"]["lstm_vocab"])
            model_path = os.path.join(base_dir, self.config["paths"]["lstm_model"])
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)
                self.word_to_idx = vocab_data["word_to_idx"]
                self.idx_to_word = {int(k): v for k, v in vocab_data["idx_to_word"].items()}
                embed_size  = vocab_data.get("embed_size", 64)
                hidden_size = vocab_data.get("hidden_size", 128)

            self.lstm_model = LSTMAutoComplete(len(self.word_to_idx), embed_size, hidden_size)
            self.lstm_model.load_state_dict(
                torch.load(model_path, map_location=_lstm_device, weights_only=True)
            )
            self.lstm_model.to(_lstm_device)
            self.lstm_model.eval()
        except Exception as e:
            logger.warning(f"LSTM could not be loaded: {e}")

    def close(self):
        self.client.close()

    def get_autocomplete(self, text: str, top_k: int = 3):
        if not self.lstm_model or not self.word_to_idx:
            return []

        tokens = text.lower().split()
        if not tokens:
            return []

        MAX_SEQ_LEN = 5
        input_seq   = [self.word_to_idx.get(w, self.word_to_idx.get("<UNK>", 0)) for w in tokens[-MAX_SEQ_LEN:]]
        padded_input = [self.word_to_idx.get("<PAD>", 0)] * (MAX_SEQ_LEN - len(input_seq)) + input_seq
        x_tensor = torch.tensor([padded_input], dtype=torch.long).to(_lstm_device)

        with torch.no_grad():
            predictions  = self.lstm_model(x_tensor)
            probabilities = F.softmax(predictions[0], dim=0)
            top_k = min(top_k, probabilities.size(0))
            _, top_indices = torch.topk(probabilities, top_k)

        suggestions = []
        for i in range(top_k):
            word = self.idx_to_word.get(top_indices[i].item(), "")
            if word not in ["<PAD>", "<UNK>"]:
                suggestions.append(word)
        return suggestions

    def _resolve_recommendation_names(self, asins: list) -> list:
        """ASIN listesini Qdrant'tan ürün adlarına çevirir."""
        if not asins:
            return []
        try:
            point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, asin)) for asin in asins]
            points    = self.client.retrieve(
                collection_name=self.collection_st,
                ids=point_ids,
                with_payload=["title"]
            )
            id_to_title = {str(p.id): clean_nan(p.payload.get("title"), None) for p in points}
            return [
                id_to_title.get(str(uuid.uuid5(uuid.NAMESPACE_DNS, asin))) or asin
                for asin in asins
            ]
        except Exception as e:
            logger.warning(f"Could not resolve recommendation names: {e}")
            return asins

    def search_products(self, query_text: str, top_k: int = 5, model_type: str = "sentence_transformer"):
        normalized_text = query_text.lower().strip()  # defensive normalize

        if model_type == "word2vec" and self.w2v_model:
            collection_name = self.collection_w2v
            tokens  = simple_preprocess(normalized_text)
            vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
            if not vectors:
                logger.warning(f"No tokens from query '{query_text}' found in Word2Vec vocabulary. Returning zero vector.")
            search_vector = np.mean(vectors, axis=0).tolist() if vectors else np.zeros(self.w2v_model.vector_size).tolist()
        else:
            collection_name = self.collection_st
            search_vector   = self.st_model.encode(normalized_text).tolist()

        search_results = self.client.query_points(collection_name=collection_name, query=search_vector, limit=top_k)

        formatted_results = []
        for item in search_results.points:
            payload   = getattr(item, "payload", {}) if not isinstance(item, tuple) else (item[2] if len(item) > 2 else {})
            score     = getattr(item, "score", 0.0)  if not isinstance(item, tuple) else (item[1] if len(item) > 1 else 0.0)
            safe_asin = clean_nan(payload.get("parent_asin"), "ASIN Doesnt Exist")

            rec_asins = []
            if self.item2vec_model and safe_asin in self.item2vec_model.wv:
                try:
                    rec_asins = [r[0] for r in self.item2vec_model.wv.most_similar(safe_asin, topn=2)]
                except Exception as e:
                    logger.warning(f"Item2Vec lookup failed: {e}")

            formatted_results.append({
                "score":        round(clean_nan(score, 0.0), 4),
                "product_name": clean_nan(payload.get("title"), "Unnamed Product"),
                "price":        clean_nan(payload.get("price"), "No Price Data"),
                "asin":         safe_asin,
                "rating":       clean_nan(payload.get("average_rating"), 0.0),
                "review_count": clean_nan(payload.get("review_count"), 0),
                "recommendations": self._resolve_recommendation_names(rec_asins),
            })

        return formatted_results
