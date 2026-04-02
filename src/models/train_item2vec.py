import json
import os
import sys
from collections import defaultdict
from gensim.models import Word2Vec
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import read_config



def train_item2vec():
    config = read_config("config/config.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    reviews_path = os.path.join(base_dir, config["paths"]["raw_data_dir"], config["files"]["reviews_data"])
    model_save_path = os.path.join(base_dir, config["paths"]["item2vec_model"])

    user_purchases = defaultdict(list)

    print("1. Comment data is being read and user history (carts) is being extracted...")
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Comments are being processed."):
            try:
                data = json.loads(line)
                user = data.get("user_id") or data.get("reviewerID")
                item = data.get("parent_asin") or data.get("asin")
                
                if user and item:
                    user_purchases[user].append(item)
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue
    
    purchase_sequences = [items for items in user_purchases.values() if len(items) > 1]

    print(f"\nTotal number of users who purchased multiple products: {len(purchase_sequences)}")

    if not purchase_sequences: 
        print("Insufficient purchase history found! Model cannot be trained.") 
        return
    
    print("2. Training Item2Vec Model (Using Product ASINs instead of Words)...")
    model = Word2Vec(
        sentences=purchase_sequences,
        vector_size=64,   
        window=10,        
        min_count=2,      
        sg=1,             
        workers=4,        
        epochs=15         
    )

    print(f"3. Saving the model: {model_save_path}")
    model.save(model_save_path)
    print("✅ Item2Vec training completed! Recommendation Robot is ready.")

    vocab = list(model.wv.key_to_index.keys())

    if vocab:
        test_item = vocab[0]
        print(f"\nExample Test: Those who bought the product with ASIN code '{test_item}' also bought:")
        for item, score in model.wv.most_similar(test_item, topn=3):
            print(f"- ASIN: {item} (Similarity Score: {score:.4f})")

if __name__ == "__main__":
    train_item2vec()