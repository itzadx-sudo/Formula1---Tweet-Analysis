"""Uninflated topic coherence (C_v) and topic diversity evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from f1_topics.constants import GLUED_RACE_TOKEN_PATTERN
from f1_topics.preprocessing import get_bow_stopwords


@dataclass(slots=True)
class CoherenceReport:
    """Topic coherence and lexical diversity metrics across discovered clusters."""

    coherence_type: str
    average_coherence: float
    per_topic_coherence: dict[int, float] = field(default_factory=dict)
    topic_diversity: float = 0.0
    evaluated_topic_count: int = 0
    skipped_topic_ids: list[int] = field(default_factory=list)


def extract_clean_topic_words(
    topic_terms: Sequence[tuple[str, float]] | Sequence[str],
    topn: int = 10,
    dictionary_tokens: set[str] | None = None,
) -> list[str]:
    """Split n-grams into distinct content unigrams while dropping race-tag tokens."""
    stop_words = get_bow_stopwords()
    words: list[str] = []
    seen: set[str] = set()

    for entry in topic_terms:
        term = entry[0] if isinstance(entry, tuple) else entry
        if not isinstance(term, str):
            continue
        for subword in term.lower().split():
            tok = subword.strip()
            if len(tok) < 3 or not tok.isalpha():
                continue
            if tok in stop_words or GLUED_RACE_TOKEN_PATTERN.fullmatch(tok):
                continue
            if dictionary_tokens is not None and tok not in dictionary_tokens:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            words.append(tok)
            if len(words) >= topn:
                return words

    return words


def compute_topic_diversity(topic_word_lists: Sequence[Sequence[str]], topn: int = 10) -> float:
    """Compute fraction of unique words across all topics' top-N word lists (0.0 to 1.0)."""
    trimmed = [list(words)[:topn] for words in topic_word_lists if words]
    total_slots = sum(len(w) for w in trimmed)
    if total_slots == 0:
        return 0.0
    unique_words = {word for words in trimmed for word in words}
    return len(unique_words) / total_slots


def calculate_topic_coherence(
    topic_model: Any,
    tokenized_tweets: Sequence[Sequence[str]],
    topics: Sequence[int],
    coherence_type: str = "c_v",
    topn: int = 10,
    min_words_per_topic: int = 3,
) -> CoherenceReport:
    """Compute honest Gensim C_v coherence and topic diversity without hashtag or 1-word inflation."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    clean_texts = [[tok for tok in doc if tok] for doc in tokenized_tweets if doc]
    active_topic_ids = sorted({int(t) for t in topics if int(t) != -1})
    if not clean_texts or not active_topic_ids:
        return CoherenceReport(coherence_type=coherence_type, average_coherence=0.0)

    dictionary = Dictionary(clean_texts)
    vocab_set = set(dictionary.token2id.keys())

    evaluated_ids: list[int] = []
    evaluated_word_lists: list[list[str]] = []
    skipped_ids: list[int] = []

    for topic_id in active_topic_ids:
        raw_terms = topic_model.get_topic(topic_id) or []
        clean_words = extract_clean_topic_words(
            raw_terms,
            topn=topn,
            dictionary_tokens=vocab_set,
        )
        # Skip degenerate <3-word topics so Gensim C_v isn't inflated to 1.0.
        if len(clean_words) < min_words_per_topic:
            skipped_ids.append(topic_id)
            continue
        evaluated_ids.append(topic_id)
        evaluated_word_lists.append(clean_words)

    diversity = compute_topic_diversity(evaluated_word_lists, topn=topn)
    if not evaluated_word_lists:
        return CoherenceReport(
            coherence_type=coherence_type,
            average_coherence=0.0,
            topic_diversity=diversity,
            evaluated_topic_count=0,
            skipped_topic_ids=skipped_ids,
        )

    cm = CoherenceModel(
        topics=evaluated_word_lists,
        texts=clean_texts,
        dictionary=dictionary,
        coherence=coherence_type,
        topn=topn,
        processes=1,
    )
    raw_scores = cm.get_coherence_per_topic()
    per_topic: dict[int, float] = {}
    valid_scores: list[float] = []

    for tid, score in zip(evaluated_ids, raw_scores):
        val = float(score)
        if np.isnan(val):
            skipped_ids.append(tid)
            continue
        per_topic[tid] = val
        valid_scores.append(val)

    avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    return CoherenceReport(
        coherence_type=coherence_type,
        average_coherence=avg_score,
        per_topic_coherence=per_topic,
        topic_diversity=diversity,
        evaluated_topic_count=len(valid_scores),
        skipped_topic_ids=skipped_ids,
    )
