import os
import sys
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from src.utils.common import read_config


def train_word2vec():
    config = read_config("config/config.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_path = os.path.join(base_dir, config["paths"]["processed_data"])
    model_save_path = os.path.join(base_dir, config["paths"]["word2vec_model"])

    print(f"Data Reading: {processed_data_path}")
    df = pd.read_parquet(processed_data_path)

    print("The documents are being broken down into words (Tokenization)... This may take a few minutes.")
    documents = df['document'].dropna().astype(str).tolist()
    tokenized_docs = [simple_preprocess(doc) for doc in tqdm(documents, desc="Tokenizing")]

    print("The Word2Vec (Skip-Gram) Model is Training !! ")
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=128,  
        window=5,         
        min_count=3,      
        sg=1,             
        workers=4,        
        epochs=5          
    )

    
    print(f"Saving the model: {model_save_path}")
    model.save(model_save_path)
    print("✅ Training completed! Word embeds are ready.")

    test_word = "dress"
    if test_word in model.wv:
        similar = model.wv.most_similar(test_word, topn=3)
        print(f"\nThe words that are most semantically similar to '{test_word}':")
        for w, score in similar:
            print(f"- {w} (Similarity: {score:.4f})")

if __name__ == "__main__":
    train_word2vec()