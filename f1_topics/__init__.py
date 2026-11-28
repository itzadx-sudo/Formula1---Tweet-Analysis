"""Formula 1 tweet topic modeling package with hashtag-clean clustering and Gemini naming."""

from f1_topics.affect import (
    PostAffect,
    PostAffectAnalyzer,
    StanceLabel,
    StanceScore,
    attach_post_affect_columns,
)
from f1_topics.config import HashtagMode, StanceMode, TopicPipelineConfig, load_env_file
from f1_topics.embeddings import TransformerTweetEmbedder, resolve_device
from f1_topics.evaluation import (
    CoherenceReport,
    calculate_topic_coherence,
    compute_topic_diversity,
    extract_clean_topic_words,
)
from f1_topics.hubs import (
    ControversyMetrics,
    HubSummary,
    InsightCard,
    build_dataset_insights,
    build_entity_hubs,
    compute_controversy_metrics,
)
from f1_topics.labeling import (
    GeminiTopicRepresentation,
    apply_topic_labels,
    format_fallback_label,
    sanitize_topic_label,
)
from f1_topics.merging import (
    consolidate_topics,
    cosine_similarity_rows,
    find_near_duplicate_groups,
)
from f1_topics.pipeline import F1TopicPipeline, PipelineArtifacts, TopicSummary
from f1_topics.preprocessing import (
    build_bow_analyzer,
    clean_text_for_display,
    clean_text_for_embedding,
    extract_driver_mentions,
    extract_hashtags,
    extract_team_mentions,
    infer_race_weekend,
    prepare_tweet_dataframe,
    tokenize_for_topic_bow,
)

__all__ = [
    "CoherenceReport",
    "ControversyMetrics",
    "F1TopicPipeline",
    "GeminiTopicRepresentation",
    "HashtagMode",
    "HubSummary",
    "InsightCard",
    "PipelineArtifacts",
    "PostAffect",
    "PostAffectAnalyzer",
    "StanceLabel",
    "StanceMode",
    "StanceScore",
    "TopicPipelineConfig",
    "TopicSummary",
    "TransformerTweetEmbedder",
    "apply_topic_labels",
    "attach_post_affect_columns",
    "build_bow_analyzer",
    "build_dataset_insights",
    "build_entity_hubs",
    "calculate_topic_coherence",
    "clean_text_for_display",
    "clean_text_for_embedding",
    "compute_controversy_metrics",
    "compute_topic_diversity",
    "consolidate_topics",
    "cosine_similarity_rows",
    "extract_clean_topic_words",
    "extract_driver_mentions",
    "extract_hashtags",
    "extract_team_mentions",
    "find_near_duplicate_groups",
    "format_fallback_label",
    "infer_race_weekend",
    "load_env_file",
    "prepare_tweet_dataframe",
    "resolve_device",
    "sanitize_topic_label",
    "tokenize_for_topic_bow",
]
