# 🛒 E-Commerce AI Search Engine & Recommendation System

## 📌 Project Overview
This project is an end-to-end, production-ready Artificial Intelligence search engine and recommendation system designed for the e-commerce domain. Built using the Amazon Fashion dataset, it demonstrates a complete transition from traditional lexical search architectures to modern semantic (context-aware) search capabilities. 

The system operates on a dual-engine architecture, allowing real-time A/B testing between traditional machine learning (Word2Vec) and modern deep learning (SentenceTransformers) approaches. Furthermore, it features an intelligent auto-complete mechanism and a collaborative filtering-based recommendation engine.

---

## 🏗️ System Architecture & Microservices
The project is built on a scalable, containerized microservices architecture:

* **Frontend (UI):** Streamlit-based interactive dashboard for real-time model comparison.
* **Backend (API):** FastAPI routing search queries, auto-complete requests, and recommendation logic.
* **Vector Database:** Qdrant dual-collection setup (`amazon_fashion` for embeddings, `amazon_fashion_w2v` for baseline).
* **Caching Layer:** Redis implemented for sub-millisecond response times on repeated queries.
* **Data Pipeline:** PySpark utilized for distributed data ingestion and ETL processes of massive JSONL datasets.
* **Deployment:** Fully dockerized (`docker-compose`) for seamless environment replication.

---

## 🚀 Key Features

### 1. Dual-Engine Search (A/B Testing Capability)
* **Modern AI (Semantic Search):** Utilizes `all-MiniLM-L6-v2` (SentenceTransformers) to understand the *intent* and *context* of a query rather than exact keyword matches. Handles complex natural language queries (e.g., "warm winter coat for snowy weather").
* **Traditional AI (Baseline):** A custom-trained **Word2Vec (Skip-Gram)** model built from scratch on 800K+ product descriptions. Generates document vectors via Mean Pooling to serve as a robust baseline for lexical search comparisons.

### 2. Intelligent Auto-Complete (Next-Token Prediction)
* A custom **LSTM (Long Short-Term Memory)** neural network trained using PyTorch.
* Learns user search behavior sequences to predict the next logical word in the search bar (e.g., typing "black leather" suggests "jacket", "shoes").

### 3. "Frequently Bought Together" Recommendation Engine
* **Item2Vec Architecture:** Applied the Skip-Gram algorithm to user purchase histories and shopping carts instead of words.
* Analyzed hundreds of thousands of user sessions to map product ASINs in a latent space, generating highly accurate, collaborative filtering-based product recommendations embedded directly into the search results.

---

## 🛠️ Tech Stack

* **Data Engineering:** PySpark, Pandas, Parquet
* **Deep Learning & NLP:** PyTorch, Gensim (Word2Vec/Item2Vec), SentenceTransformers
* **Backend & API:** FastAPI, Uvicorn, Python 3.x
* **Database & Cache:** Qdrant (Vector DB), Redis
* **Frontend:** Streamlit
* **DevOps & Infrastructure:** Docker, Docker Compose

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ccemozclk/amazon-ai-search-engine.git](https://github.com/ccemozclk/amazon-ai-search-engine.git)
   cd amazon-ai-search-engine
   ```
2. **Start the Microservices:**
   ```bash
   docker compose up -d --build
   ```
3. ** Access the Application:**
   * UI Dashboard: ```http://localhost:8501```
   * FastAPI Docs (Swagger): ```http://localhost:8001/docs```
   * Qdrant Dashboard: ```http://localhost:6333/dashboard```

*(Note: Data files and pre-trained .pth / .model files are excluded from this repository due to size constraints. The repository showcases the architectural codebase and pipeline implementations.)*

## 👨‍💻 Author 
* **Cem Özçelik**
*   *Data Scientist | Hands-On AI Engineer*
*   *Driven by a passion for designing scalable AI systems, deep learning architectures, and big data pipelines.*
