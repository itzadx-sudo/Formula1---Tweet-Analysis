# Formula 1 Tweet Analysis

A Natural Language Processing project that applies transformer-based topic modeling to discover and categorize discussion themes within a large collection of Formula 1 tweets.

## Overview

This project uses **BERTopic**, configured with **BERTweet** (`vinai/bertweet-base`), to identify discussion topics in a cleaned sample drawn from 632,388 F1-related tweets. It uncovers what fans and commentators discuss around Formula 1 — from race weekends and driver performance to technical regulations and strategic analysis.

**Recorded result:** the saved notebook reports **C_V coherence of 0.5702** on **26,571 cleaned tweets**, matching the value cited in the CV and methodology report. The earlier README value of **~0.52** has no traceable run artifact; see [Results and provenance](#results-and-provenance).

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

The saved notebook records the following sample and preprocessing counts:

| Stage | Tweets |
|---|---:|
| Loaded `F1_tweets.csv` | 632,388 |
| Random sample after dropping rows with missing tweet text | 45,000 |
| Detected as English | 42,203 |
| After F1 relevance, raw-length filtering and deduplication | 27,766 |
| After text cleaning and cleaned-length filtering; input to topic modeling | 26,571 |

Sampling uses `df_clean.sample(45000, random_state=42)` without replacement. This is **simple random sampling**; the report and CV describe it as stratified, but the notebook implements no stratification. Language detection uses `DetectorFactory.seed = 0`. The saved run retained enough English tweets that the code's fallback to all sampled languages was not used.

## Pipeline

```
Raw tweets → Drop missing text → Random sample → English detection
    → F1 relevance and raw-length filtering → Deduplication → Text cleaning
    → Cleaned-length filtering → BERTopic fitting (embeddings → UMAP → HDBSCAN)
    → Optional outlier mini-clustering → Topic evaluation
```

### Preprocessing for the recorded 0.5702 result

The implementation is in the notebook's `enhanced_quality_filter`, `enhanced_clean_text` and `achieve_clustering` functions:

1. Keep raw text matching a case-insensitive regex of F1 driver/team names, racing terms and GP hashtags. This is keyword matching, with no word-boundary requirement for the listed terms.
2. Retain raw tweets with **5–25 whitespace-separated words** and **30–200 characters**, inclusive.
3. Deduplicate using raw text lowercased with all characters other than `a`–`z` and whitespace removed.
4. Lowercase; remove URLs and email addresses; preserve the matched content of selected F1 hashtags and remove other hashtags, mentions and retweet markers. Strip characters other than letters, spaces and periods, then normalize whitespace. Although an earlier numeric rule exempts years, the later character filter removes digits too.
5. Tokenize with NLTK, falling back to whitespace splitting on failure. Remove English and custom stopwords; retain tokens of 3–20 characters that are not numeric or repetitions of a single character. Apply WordNet lemmatization with POS tagging and the code's noun/default fallbacks, then retain lemmas of at least three characters.
6. Retain cleaned strings longer than **10 characters**. These 26,571 cleaned strings are used for both model fitting and coherence evaluation.

### Configuration recorded in the notebook

These settings accompany the saved **0.5702** output. They cannot be assigned to the unsupported **~0.52** value as a verified run configuration.

| Component | Explicit settings |
|---|---|
| Embedding setup | `AutoTokenizer` and `AutoModel` loaded from `vinai/bertweet-base`; the model object is passed to BERTopic |
| Device selection | Apple MPS if available, otherwise CUDA, otherwise CPU; selected device is not printed in the saved output |
| UMAP | `n_neighbors=8`, `n_components=15`, `min_dist=0.1`, `metric='cosine'`, `random_state=42` |
| HDBSCAN | `min_cluster_size=25`, `min_samples=2`, `metric='euclidean'`, `cluster_selection_method='eom'`, `prediction_data=True` |
| CountVectorizer | `ngram_range=(1, 3)`, `max_features=5000`, `min_df=2`, `max_df=0.3`, `token_pattern=r'\b[a-zA-Z]{3,}\b'`; English plus custom stopwords |
| BERTopic | `top_n_words=12`, `nr_topics=None`, `min_topic_size=25`, `calculate_probabilities=True`, `verbose=False` |
| Optional outlier handling | If more than 50 outliers exist, select their top half by maximum membership probability; proceed with at least 20 selected tweets |
| Outlier embeddings | BERTweet tokenization with padding, truncation and `max_length=128`; first-token hidden state under `torch.no_grad()` |
| KMeans and acceptance | `n_clusters=min(5, selected_outliers // 10)`, at least 2 clusters, `random_state=42`, `n_init=10`; assign new IDs only if silhouette score exceeds `0.2` |
| Coherence | Gensim `CoherenceModel`, `coherence='c_v'`, `topn=10`, `processes=1` |

The saved output confirms that outlier mini-clustering was attempted, but does not log its silhouette score or whether new assignments were accepted. The code changes the returned topic labels without updating BERTopic's topic representations; a new ID without a representation is omitted from coherence evaluation.

## Results and provenance

### Traceable saved result: 0.5702

The evidence is the saved `TOPIC MODELING RESULTS` output in [f1topicmodelling.ipynb](f1topicmodelling.ipynb), cell ID `63850d96` (execution count `15`). The sample, filtering and fitting outputs are in cells `605af446` (execution count `6`) and `72f2230f` (execution count `8`). [The methodology report](f1topicmodellingdocumentation.docx), under “Unsupervised Topic Modelling Evaluation,” also repeats the same coherence, topic count and clustering counts. These artifacts support the **0.5702** value cited in the CV.

| Metric | Saved notebook output |
|---|---:|
| Tweets analyzed | 26,571 |
| Tweets clustered | 19,248 (72.44%) |
| Topics found, excluding outlier label `-1` | 248 |
| Outliers | 7,323 (27.6%) |
| Coherence score (C_V, printed to four decimals) | **0.5702** |

The notebook calculates coherence on all 26,571 cleaned tweets split by whitespace, with a Gensim dictionary built from those same texts. For each non-outlier topic ID, it takes the **first 10 BERTopic terms**, then discards multiword terms and skips topics with no remaining terms or no representation. The reported score is `CoherenceModel.get_coherence()` over the resulting topic word lists. It is an in-corpus topic coherence measure, not held-out predictive accuracy.

For version identification, the notebook and report were uploaded in commit [`eb311ad`](https://github.com/itzadx-sudo/Formula1---Tweet-Analysis/commit/eb311adcecdf7a798ec0882b669e3938f3e9a86f) on **10 October 2025**. Their contents are byte-identical to the original files under different names in commit `c0cdf0d`. This identifies the saved artifacts, **not the execution date**, which was not recorded.

### Earlier README claim: approximately 0.52 (unverified)

The **~0.52** coherence value first appears in the README added by commit [`d140eb8`](https://github.com/itzadx-sudo/Formula1---Tweet-Analysis/commit/d140eb88f711af70e3a53b4a94828ab9a026e69f) on **20 April 2026**, alongside **22,002 clustered tweets (82.9%)** and **4,569 outliers (17.1%)** out of 26,571 analyzed tweets.

| Provenance detail | What can be established for the ~0.52 claim |
|---|---|
| Run identity | No matching notebook output, run log or model artifact in the repository history |
| Sample | The old README states ~45,000 sampled and 26,571 cleaned tweets, but provides no separate sample IDs or run record |
| Preprocessing | The old README lists English/F1 filtering, cleaning, lemmatization, length limits and deduplication; no run-specific evidence ties these to ~0.52 |
| Configuration | The old README lists BERTweet, UMAP 8 neighbors/15 components, HDBSCAN minimum cluster size 25, 1–3-gram vectorization/5,000 features and minimum topic size 25; no matching execution verifies them for ~0.52 |
| Status | Unsupported historical README claim; replaced above by the traceable saved output |

There is insufficient evidence to call these two different runs or a measured improvement from ~0.52 to 0.5702. The old clustering figures also do not match the saved notebook or report.

### Reproducibility limits

The repository contains no sampled-row manifest, dataset checksum/version pin, saved model or complete dependency lockfile. Notebook metadata identifies Python **3.12.10**, and installation output lists Gensim **4.3.3**, but the full modeling environment, effective BERTopic embedding backend and execution device are not logged. Recorded seeds and source code describe the saved analysis; they do not guarantee an identical rerun. The figures above are taken from existing saved outputs, not a newly executed experiment.

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
