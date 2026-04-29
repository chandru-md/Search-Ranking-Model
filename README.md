## 📊 Data Processing

- Loaded Istella LETOR dataset
- Implemented custom data loader
- Extracted:
  - Features (X)
  - Labels (y)
  - Query groups (qid)

Supports partial loading for faster experimentation.

## 📂 Dataset

Using Istella LETOR dataset.

- Loaded train/test splits
- Implemented partial loading for scalability
- Verified feature extraction and query grouping

- Verified data loading pipeline using test script
- Confirmed feature extraction and query grouping

- Fixed module import issues using proper project root execution
- Switched to relative file paths for portability

## 🏗️ Project Structure

- Organized modules into separate directories
- Fixed import issues by aligning with standard Python package structure

## 🤖 Ranking Model

- Implemented Learning-to-Rank using LightGBM
- Objective: LambdaRank
- Metric: NDCG@10
- Supports query-grouped training

- Converted dataset to NumPy arrays for compatibility with LightGBM training

## 📊 Evaluation

- Implemented NDCG@10 for ranking quality measurement
- Evaluates model performance per query group
- Provides average ranking effectiveness

## 🔎 Inference Pipeline

- Built ranking pipeline using trained LightGBM model
- Supports query-level ranking
- Returns top-K ranked results

## 🔍 Retrieval System

- Implemented hybrid retrieval:
  - BM25 (keyword-based)
  - Semantic search (embeddings)
- Combines both methods for better candidate selection

## ⚙️ Dependencies

- rank-bm25 (for keyword-based retrieval)
- sentence-transformers (for semantic search)
