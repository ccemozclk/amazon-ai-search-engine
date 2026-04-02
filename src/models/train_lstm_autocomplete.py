import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import read_config
from src.models import LSTMAutoComplete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

EMBED_SIZE = 64
HIDDEN_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 64
MAX_SEQ_LEN = 5


def train_lstm():
    config = read_config("config/config.yaml")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_path = os.path.join(base_dir, config["paths"]["processed_data"])
    
    print("1. Data is Reading From Source Path...")
    df = pd.read_parquet(processed_data_path).head(300000)
    titles = df["title"].dropna().astype(str).str.lower().tolist()

    print("2. Vocabulary is being built...")
    words = " ".join(titles).split()
    vocab = ["<PAD>", "<UNK>"] + sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: str(w) for i, w in enumerate(vocab)}

    print(f"Total Number of Unique Words: {len(vocab)}")

    print("3. Training Data (X and y) is being prepared....")
    sequences = []
    for title in tqdm(titles, desc="Sentences Are Splitting"):
        tokens = title.split()
        for i in range(1, len(tokens)):
            seq = tokens[:i+1]
            sequences.append([word_to_idx.get(w, word_to_idx["<UNK>"]) for w in seq])
    
    X, y = [], []
    for seq in sequences:
        if len(seq) < 2: continue
        input_seq = seq[:-1][-MAX_SEQ_LEN:]
        target = seq[-1]
        padded_input = [word_to_idx["<PAD>"]] * (MAX_SEQ_LEN - len(input_seq)) + input_seq
        X.append(padded_input)
        y.append(target)

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"4. LSTM Model Training on {device}...")
    model = LSTMAutoComplete(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    print("\n5. Model and Vocabulary are being saved...")
    model_save_path = os.path.join(base_dir, config["paths"]["lstm_model"])
    vocab_save_path = os.path.join(base_dir, config["paths"]["lstm_vocab"])
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    torch.save(model.state_dict(), model_save_path)
    with open(vocab_save_path, "w") as f:
        json.dump({
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "embed_size": EMBED_SIZE,
            "hidden_size": HIDDEN_SIZE
        }, f)
        
    print("✅ LSTM Auto-complete Model Training Completed!")

if __name__ == "__main__":
    train_lstm()