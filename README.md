# Persian Text Preprocessing and Information Retrieval System

This project implements text preprocessing and query processing pipelines for Persian text data, particularly for an **Information Retrieval** system. It is organized into two notebooks: **Phase 1** and **Phase 2**.

---

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
  - [phase1.ipynb](#phase1ipynb)
  - [phase2.ipynb](#phase2ipynb)
- [Requirements](#requirements)
- [Usage](#usage)
- [Details of Implementation](#details-of-implementation)
- [License](#license)

---

## Overview

The two phases cover:
1. **Phase 1**: Preprocessing of Persian text documents and creation of an inverted index.
2. **Phase 2**: Implementation of a query processor supporting **Boolean retrieval**, **phrase searching**, and **ranked retrieval** with **TF-IDF weighting** and **Jaccard similarity**.

---

## Notebooks

### `phase1.ipynb`

**Purpose**:
- Load and preprocess Persian text documents.
- Generate an inverted index and store term frequencies.

**Key Steps**:
1. Load a dataset (`IR_data_news_5k.json`).
2. Use **Hazm** and **Parsivar** for:
   - Normalization
   - Tokenization
   - Stopword removal
   - Stemming
3. Build an **inverted index** with positional indexing.

**Core Features**:
- Preprocessing pipeline with stopword removal and stemming.
- Positional indexing for terms.

---

### `phase2.ipynb`

**Purpose**:
- Enhance retrieval with **ranking** and **phrase queries**.
- Support **Boolean operations**, **TF-IDF weighting**, and **Jaccard similarity** for query scoring.

**Key Steps**:
1. Reuse preprocessed documents from Phase 1.
2. Build champion lists for terms (top documents by importance).
3. Implement retrieval techniques:
   - Boolean Retrieval: AND, OR, AND-NOT operations.
   - Ranked Retrieval: Cosine similarity with **TF-IDF weights**.
   - Jaccard Similarity: Measure similarity between query and documents.
4. Return top-ranked documents with metadata (title, URL).

**Core Features**:
- Support for **Boolean queries**.
- Phrase query processing with positional indexes.
- **TF-IDF weighting** and **Cosine similarity** for ranking.
- **Jaccard similarity** for query-document matching.

---

## Requirements

Make sure the following libraries are installed:

- Python >= 3.7
- Jupyter Notebook
- Libraries:
  ```bash
  pip install numpy pandas hazm parsivar
  ```

---

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Run the notebooks:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `phase1.ipynb` and run all cells.
   - Then, open `phase2.ipynb` to process queries.

3. Input queries in `phase2.ipynb`:
   - **Boolean queries** (e.g., `مایکل ! جردن`)
   - **Phrase queries** (e.g., `"سهمیه المپیک"`)
   - Ranked retrieval using **TF-IDF**.

---

## Details of Implementation

1. **Phase 1**:
   - Preprocessing pipeline:
     - Normalization using `Normalizer`.
     - Tokenization using `Tokenizer`.
     - Stemming with `FindStems`.
   - Build inverted index with positional indexes.

2. **Phase 2**:
   - Query Processing:
     - **Boolean Retrieval**: Support for AND, OR, NOT operations.
     - **Phrase Search**: Matching terms' positions for exact phrases.
     - **Ranked Retrieval**:
       - TF-IDF weights combined with cosine similarity.
       - Jaccard similarity for query-document comparison.
   - Champion lists are implemented to optimize retrieval.

3. Example Results:
   Top-ranked documents are displayed with:
   - Document frequency
   - Title and URL of documents
