"""Configuration schema for the F1 tweet topic modeling pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HashtagMode = Literal["strip_race", "strip_all", "downweight"]
StanceMode = Literal["hybrid_nli", "sentiment", "off"]


def load_env_file(env_path: str | Path | None = None) -> Path | None:
    """Load non-empty key=value pairs from a .env file without overwriting existing env vars."""
    candidates = (
        [Path(env_path)]
        if env_path is not None
        else [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    )
    resolved = next((p for p in candidates if p.is_file()), None)
    if resolved is None:
        return None

    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if val != "" and not os.environ.get(key, "").strip():
            os.environ[key] = val

    return resolved


@dataclass(slots=True)
class TopicPipelineConfig:
    """End-to-end configuration targeting ~40-60 human-readable F1 topics."""

    # Sampling & quality filtering
    sample_size: int | None = 45_000
    random_seed: int = 42
    language: str = "en"
    min_words: int = 5
    max_words: int = 45
    min_chars: int = 25
    max_chars: int = 280
    min_cleaned_words: int = 4

    # strip_race drops race/series tags; downweight keeps place stems with BM25
    hashtag_mode: HashtagMode = "strip_race"

    # Embedding model
    embedding_model_name: str = "vinai/bertweet-base"
    embedding_batch_size: int = 64
    max_seq_length: int = 128
    device: str | None = None

    # UMAP dimensionality reduction (broader neighborhood than legacy n_neighbors=8)
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # HDBSCAN density clustering (larger min_cluster_size avoids 248 micro-clusters)
    hdbscan_min_cluster_size: int = 60
    hdbscan_min_samples: int = 10
    hdbscan_cluster_selection_method: str = "eom"

    # Vectorizer & c-TF-IDF down-weighting
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5
    max_df: float = 0.25
    max_features: int = 5_000
    top_n_words: int = 12
    bm25_weighting: bool = True
    reduce_frequent_words: bool = True
    mmr_diversity: float = 0.35

    # Post-clustering consolidation & 40-60 topic target
    reassign_outliers: bool = True
    outlier_similarity_threshold: float = 0.55
    merge_similarity_threshold: float = 0.82
    min_topics: int = 40
    max_topics: int = 60
    target_topics: int = 50

    # Coherence evaluation
    coherence_metric: str = "c_v"
    coherence_top_n: int = 10
    min_coherence_words_per_topic: int = 3

    # Google Gemini LLM topic naming
    enable_llm_naming: bool = True
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_batch_size: int = 20
    representative_docs_per_topic: int = 6
    gemini_temperature: float = 0.2
    gemini_max_retries: int = 3

    # Per-post stance, emotion, hubs & controversy index
    stance_mode: StanceMode = "hybrid_nli"
    sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model_name: str = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
    nli_stance_model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    affect_batch_size: int = 64
    controversy_polarization_weight: float = 0.65
    controversy_heat_weight: float = 0.35
    controversy_prior_k: float = 10.0
    min_hub_tweets: int = 5

    def __post_init__(self) -> None:
        load_env_file()
        env_model = os.environ.get("GEMINI_MODEL", "").strip()
        if self.gemini_model == "gemini-2.5-flash" and env_model:
            self.gemini_model = env_model
        if self.hashtag_mode not in ("strip_race", "strip_all", "downweight"):
            raise ValueError(f"Unsupported hashtag_mode: {self.hashtag_mode!r}")
        if self.stance_mode not in ("hybrid_nli", "sentiment", "off"):
            raise ValueError(f"Unsupported stance_mode: {self.stance_mode!r}")
        if self.controversy_prior_k < 0:
            raise ValueError("controversy_prior_k must be >= 0")
        if self.min_topics < 2:
            raise ValueError("min_topics must be >= 2")
        if not (self.min_topics <= self.target_topics <= self.max_topics):
            raise ValueError(
                f"Expected min_topics <= target_topics <= max_topics, got "
                f"{self.min_topics}, {self.target_topics}, {self.max_topics}"
            )
        if not (0.0 < self.merge_similarity_threshold <= 1.0):
            raise ValueError("merge_similarity_threshold must be in (0, 1]")
