# Formula 1 Tweet Analysis

A Natural Language Processing project that applies transformer-based topic modeling to discover and categorize discussion themes within a large collection of Formula 1 tweets.

## Overview

This project uses **BERTopic** with **BERTweet** embeddings to automatically identify topics from over 600,000 F1-related tweets. It uncovers what fans and commentators discuss around Formula 1 — from race weekends and driver performance to technical regulations and strategic analysis.

## Features

- Domain-specific text preprocessing with F1 entity recognition (drivers, teams, circuits)
- Transformer-based embeddings via BERTweet (`vinai/bertweet-base`), optimized for social media text
- Automatic topic discovery using HDBSCAN clustering — no need to pre-specify topic counts
- Outlier reassignment via KMeans mini-clustering and silhouette scoring
- Multi-angle analysis: race-specific, technical, temporal, and fan engagement
- Coherence-based model evaluation (C_V metric)
- GPU support (NVIDIA CUDA / Apple MPS) with automatic CPU fallback

## Dataset

- **Source:** [Formula 1 Trending Tweets](https://www.kaggle.com/datasets/kaushiksuresh147/formula-1-trending-tweets) via Kaggle
- **Total tweets:** 632,388
- **Sampled for analysis:** ~45,000
- **Final cleaned set:** ~26,571 English F1-relevant tweets

## Pipeline

```
Raw tweets → Language detection → F1 relevance filtering → Text cleaning
    → BERTweet embeddings → UMAP dimensionality reduction
    → HDBSCAN clustering → Outlier reassignment → Topic labeling & evaluation
```

### Text Cleaning Steps

1. Remove URLs, mentions, hashtags, and non-alphanumeric characters
2. Filter for English tweets using `langdetect`
3. Keep tweets matching F1-specific entities (drivers, teams, circuits)
4. POS-aware lemmatization with domain stopword removal
5. Length filtering: 5–25 words, 30–200 characters
6. Deduplicate

### BERTopic Configuration

| Parameter | Value |
|---|---|
| Embedding model | `vinai/bertweet-base` |
| UMAP `n_neighbors` | 8 |
| UMAP `n_components` | 15 |
| HDBSCAN `min_cluster_size` | 25 |
| CountVectorizer n-gram range | 1–3 |
| Max features | 5,000 |
| Min topic size | 25 tweets |

## Results

| Metric | Value |
|---|---|
| Tweets analyzed | 26,571 |
| Tweets clustered | 22,002 (82.9%) |
| Outliers | 4,569 (17.1%) |
| Coherence score (C_V) | ~0.52 |

### Topic Categories Discovered

- **Race weekend discussions** — GP-specific conversation threads
- **Driver-focused topics** — individual driver performance and news
- **Technical discussions** — safety car, tires, regulations, car performance
- **Strategic analysis** — pit stop strategy, race tactics
- **Fan engagement** — commentary, reactions, and opinions

## Requirements

Install dependencies automatically via the notebook, or manually:

```bash
pip install langdetect kagglehub seaborn pandas numpy scikit-learn nltk emoji \
            bertopic gensim umap-learn hdbscan torch transformers
```

NLTK resources are downloaded automatically on first run.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/itzadx-sudo/formula1---tweet-analysis.git
   cd formula1---tweet-analysis
   ```

2. Open and run the notebook sequentially:
   ```bash
   jupyter notebook f1topicmodelling.ipynb
   ```

3. The notebook will:
   - Automatically download the Kaggle dataset via `kagglehub`
   - Detect available hardware (GPU/CPU)
   - Run the full preprocessing and modeling pipeline
   - Output topic visualizations and evaluation metrics

> **Note:** A Kaggle account and API token (`~/.kaggle/kaggle.json`) are required to download the dataset automatically.

## Project Structure

```
Formula1---Tweet-Analysis/
├── f1topicmodelling.ipynb           # Main analysis notebook
├── f1topicmodellingdocumentation.docx  # Detailed methodology and findings report
└── README.md
```

## Visualizations

The notebook produces:

- Topic size distribution (bar chart)
- Topic assignment confidence (histogram)
- Clustering success rate (pie chart)
- Top topic clusters with representative keywords
- Intertopic distance map (interactive)
- Fan engagement breakdown by category
- Coherence scores per topic

## Acknowledgements

- Dataset by [kaushiksuresh147](https://www.kaggle.com/kaushiksuresh147) on Kaggle
- [BERTopic](https://maartengr.github.io/BERTopic/) by Maarten Grootendorst
- [BERTweet](https://github.com/VinAIResearch/BERTweet) by VinAI Research
