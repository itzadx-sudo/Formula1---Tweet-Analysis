"""Outlier reassignment, near-duplicate topic merging, and target topic count consolidation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from f1_topics.config import TopicPipelineConfig


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for row vectors."""
    normed = _l2_normalize_rows(np.asarray(matrix, dtype=np.float64))
    sim = normed @ normed.T
    return np.clip(sim, -1.0, 1.0)


def find_near_duplicate_groups(
    similarity_matrix: np.ndarray,
    topic_ids: Sequence[int],
    threshold: float = 0.82,
) -> list[list[int]]:
    """Group near-duplicate topics via average-linkage cosine clustering to avoid chaining."""
    n = len(topic_ids)
    if n < 2:
        return []
    if similarity_matrix.shape != (n, n):
        raise ValueError(f"Expected ({n}, {n}) similarity matrix, got {similarity_matrix.shape}")

    distances = np.clip(1.0 - np.asarray(similarity_matrix, dtype=np.float64), 0.0, 2.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)

    condensed = squareform(distances, checks=False)
    z = linkage(condensed, method="average")
    max_distance = max(0.0, 1.0 - threshold)
    cluster_assignments = fcluster(z, t=max_distance, criterion="distance")

    grouped: dict[int, list[int]] = {}
    for topic_id, cluster_id in zip(topic_ids, cluster_assignments):
        grouped.setdefault(int(cluster_id), []).append(int(topic_id))

    return [sorted(members) for members in grouped.values() if len(members) >= 2]


def compute_topic_similarity_matrix(topic_model: Any, active_topic_ids: Sequence[int]) -> np.ndarray | None:
    """Blend c-TF-IDF similarity and semantic topic-embedding similarity for active topics."""
    if len(active_topic_ids) < 2:
        return None

    topic_index_map: dict[int, int] = {}
    if getattr(topic_model, "_outliers", None) is not None and getattr(topic_model, "topics_", None) is not None:
        sorted_all = sorted(set(topic_model.topics_))
        topic_index_map = {tid: idx for idx, tid in enumerate(sorted_all)}
    else:
        has_outlier = -1 in set(getattr(topic_model, "topics_", []))
        offset = 1 if has_outlier else 0
        topic_index_map = {tid: tid + offset for tid in active_topic_ids}

    row_indices = [topic_index_map[tid] for tid in active_topic_ids if tid in topic_index_map]
    if len(row_indices) != len(active_topic_ids):
        return None

    matrices: list[np.ndarray] = []
    c_tf_idf = getattr(topic_model, "c_tf_idf_", None)
    if c_tf_idf is not None and max(row_indices) < c_tf_idf.shape[0]:
        dense_ctfidf = c_tf_idf[row_indices].toarray() if hasattr(c_tf_idf, "toarray") else np.asarray(c_tf_idf[row_indices])
        matrices.append(cosine_similarity_rows(dense_ctfidf))

    topic_embs = getattr(topic_model, "topic_embeddings_", None)
    if topic_embs is not None:
        emb_arr = np.asarray(topic_embs)
        if max(row_indices) < emb_arr.shape[0]:
            matrices.append(cosine_similarity_rows(emb_arr[row_indices]))

    if not matrices:
        return None
    if len(matrices) == 1:
        return matrices[0]
    return 0.5 * matrices[0] + 0.5 * matrices[1]


def count_active_topics(topics: Sequence[int]) -> int:
    """Count non-outlier topics (topic_id != -1)."""
    return len({int(t) for t in topics if int(t) != -1})


def consolidate_topics(
    topic_model: Any,
    docs: list[str],
    topics: Sequence[int],
    embeddings: np.ndarray,
    config: TopicPipelineConfig,
    vectorizer_model: Any = None,
    ctfidf_model: Any = None,
) -> tuple[list[int], Any]:
    """Reassign borderline outliers, merge near-duplicates, and reduce into the 40-60 topic band."""
    current_topics = [int(t) for t in topics]

    # Reassign high-similarity outlier tweets and refresh c-TF-IDF.
    if config.reassign_outliers and -1 in current_topics and count_active_topics(current_topics) >= 2:
        try:
            reassigned = topic_model.reduce_outliers(
                docs,
                current_topics,
                strategy="embeddings",
                embeddings=embeddings,
                threshold=config.outlier_similarity_threshold,
            )
            current_topics = [int(t) for t in reassigned]
            topic_model.update_topics(
                docs,
                topics=current_topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                top_n_words=config.top_n_words,
            )
        except Exception:
            pass

    # Merge near-duplicate clusters before applying the 40-60 topic cap.
    active_ids = sorted({t for t in current_topics if t != -1})
    sim_matrix = compute_topic_similarity_matrix(topic_model, active_ids)
    if sim_matrix is not None:
        dup_groups = find_near_duplicate_groups(
            sim_matrix,
            active_ids,
            threshold=config.merge_similarity_threshold,
        )
        if dup_groups:
            topic_model.merge_topics(docs, dup_groups)
            current_topics = [int(t) for t in topic_model.topics_]

    # Hierarchically reduce remaining topics down to target_topics when above max_topics.
    remaining_count = count_active_topics(current_topics)
    if remaining_count > config.max_topics:
        # BERTopic's nr_topics includes the outlier topic (-1) when present.
        has_outlier = -1 in set(current_topics)
        nr_topics_arg = config.target_topics + (1 if has_outlier else 0)
        topic_model.reduce_topics(docs, nr_topics=nr_topics_arg)
        current_topics = [int(t) for t in topic_model.topics_]

    # Refresh keywords after merges.
    if vectorizer_model is not None or ctfidf_model is not None:
        topic_model.update_topics(
            docs,
            topics=current_topics,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            top_n_words=config.top_n_words,
        )

    return current_topics, topic_model
