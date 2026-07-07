# My Mini DB

> A tiny vector database powered by a sentence embedding model trained completely from scratch.

## Overview

This project is my attempt to understand how modern semantic search systems work by building every important component myself instead of relying on existing libraries or APIs.

The project contains:

* A custom sentence embedding model
* A synthetic training dataset
* Contrastive learning for training embeddings
* A lightweight in-memory vector database
* Cosine similarity search

The goal was **not** to build a production-ready vector database, but to understand how embeddings are learned and how vector search works internally.

---

## Why?

Most vector databases use pre-trained embedding models from OpenAI, Sentence Transformers, or similar providers.

In this project I wanted to answer a different question:

> **Can I train my own embedding model and use it to perform semantic search?**

Instead of using external APIs, I trained a small sentence encoder from scratch and used its embeddings inside my own vector database.

---

## Project Pipeline

```text
Synthetic Dataset
        │
        ▼
Tokenizer & Vocabulary
        │
        ▼
Sentence Encoder (Bi-LSTM)
        │
        ▼
Contrastive Training
        │
        ▼
Sentence Embeddings
        │
        ▼
Tiny Vector Database
        │
        ▼
Cosine Similarity Search
        │
        ▼
Top-k Retrieved Sentences
```

---

## Features

* Train a sentence embedding model from scratch
* Build embeddings without using external embedding APIs
* Create an in-memory vector database
* Perform semantic search using cosine similarity
* Minimal implementation for educational purposes

---

## Repository Structure

```text
my-mini-db/
│
├── train_embeddings.py          # Train the sentence encoder
├── vectordb.py                  # Tiny in-memory vector database
├── synthetic_embedding_corpus.txt
├── sentence_encoder.pt          # Trained model (generated after training)
├── vocab.json                   # Vocabulary (generated after training)
```

---

## Training

Train the embedding model:

```bash
python3 train_embeddings.py
```

This generates:

* `sentence_encoder.pt`
* `vocab.json`

---

## Vector Search

Run the vector database demo:

```bash
python3 vectordb.py
```

Example output:

```text
Query:
"The kitten was playing."

↓

Top Results:

The little kitten chased a playful puppy...
The playful puppy was running...
...
```

---

## Technologies Used

* Python
* PyTorch
* Bi-directional LSTM
* Contrastive Learning
* Cosine Similarity

---

## Current Limitations

This is Version 1 of the project.

Current limitations include:

* Small synthetic dataset
* LSTM encoder instead of a Transformer encoder
* Brute-force nearest neighbour search
* In-memory storage only
* No indexing or persistence

---

## Future Improvements

* Replace the LSTM encoder with a Transformer encoder
* Train on a much larger corpus
* Improve the contrastive learning objective
* Add approximate nearest neighbour search (HNSW)
* Persist vectors to disk
* Support PDF ingestion and semantic document search

---

## What I Learned

Building this project helped me understand:

* How sentence embeddings are learned
* Why semantic similarity emerges during training
* How contrastive learning works
* How vector databases store and retrieve embeddings
* Why cosine similarity is sufficient when embeddings are meaningful

---

## Disclaimer

This project is an educational implementation created to learn how embedding models and vector databases work internally.

It is **not** intended to replace production systems such as ChromaDB, FAISS, Milvus, or Pinecone.

The purpose is to understand the underlying ideas by implementing them from scratch.
