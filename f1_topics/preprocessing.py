"""Tweet loading, metadata extraction, hashtag stripping, and two-stage text cleaning."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from f1_topics.config import HashtagMode, TopicPipelineConfig
from f1_topics.constants import (
    BOW_CUSTOM_STOPWORDS,
    CANDIDATE_DATE_COLUMNS,
    CANDIDATE_TEXT_COLUMNS,
    DRIVER_HUB_ALIASES,
    F1_RELEVANCE_ENTITIES,
    GLUED_ENTITY_NORMALIZATIONS,
    GLUED_RACE_TOKEN_PATTERN,
    KAGGLE_DATASET_SLUG,
    NON_F1_SERIES_EXCLUSIONS,
    RACE_NAME_MAPPING,
    RACE_OR_SERIES_HASHTAG_PATTERN,
    SLOGAN_HASHTAGS,
    TEAM_HUB_ALIASES,
    TRAILING_HASHTAG_BLOCK_PATTERN,
)

_URL_PATTERN = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
_RT_PREFIX_PATTERN = re.compile(r"^\s*rt\b[:\s]*|\brt\s+(?=@)", flags=re.IGNORECASE)
_ANY_HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")
_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_ALPHA_PATTERN = re.compile(r"[A-Za-z]{2,}")

# Minimal fallback English stopwords used when NLTK corpora have not been downloaded yet.
_FALLBACK_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "could", "did", "didn", "do",
        "does", "doesn", "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "hasn", "have", "haven", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
        "isn", "it", "its", "itself", "just", "ll", "m", "me", "might", "more", "most",
        "must", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
        "same", "she", "should", "shouldn", "so", "some", "such", "t", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
        "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "won", "would", "wouldn", "you", "your",
        "yours", "yourself", "yourselves",
    }
)

_SORTED_RACE_KEYS: tuple[tuple[str, str], ...] = tuple(
    sorted(RACE_NAME_MAPPING.items(), key=lambda kv: len(kv[0]), reverse=True)
)

_GLUED_ENTITY_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(GLUED_ENTITY_NORMALIZATIONS, key=len, reverse=True))
    + r")\d*\b",
    flags=re.IGNORECASE,
)

_F1_ENTITY_FILTER_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(e) for e in F1_RELEVANCE_ENTITIES) + r")\b"
    r"|#(?:f1|formula1|formulaone|[a-z0-9_]*gp|[a-z0-9_]*grandprix)\b",
    flags=re.IGNORECASE,
)

_NON_F1_EXCLUSION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(e) for e in NON_F1_SERIES_EXCLUSIONS) + r")\b",
    flags=re.IGNORECASE,
)


def _compile_hub_patterns(
    alias_map: dict[str, tuple[str, ...]],
    extra_patterns: dict[str, str] | None = None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for canonical, aliases in alias_map.items():
        parts = [rf"\b{re.escape(alias)}\b" for alias in aliases]
        if extra_patterns and canonical in extra_patterns:
            parts.append(extra_patterns[canonical])
        compiled.append((canonical, re.compile("|".join(parts), flags=re.IGNORECASE)))
    return tuple(compiled)


_DRIVER_MENTION_PATTERNS = _compile_hub_patterns(
    DRIVER_HUB_ALIASES,
    extra_patterns={
        "Max Verstappen": r"\bmax\b(?!\s+(?:speed|pace|downforce|power|points|laps|load|tyre|tire))",
    },
)
_TEAM_MENTION_PATTERNS = _compile_hub_patterns(TEAM_HUB_ALIASES)


def extract_driver_mentions(text: str) -> list[str]:
    """Extract canonical F1 driver hub names mentioned in a tweet (prose, #tags, or @mentions)."""
    if not isinstance(text, str) or not text:
        return []
    searchable = re.sub(r"[#@]+", " ", text)
    return [driver for driver, pattern in _DRIVER_MENTION_PATTERNS if pattern.search(searchable)]


def extract_team_mentions(text: str) -> list[str]:
    """Extract canonical F1 constructor/team hub names mentioned in a tweet."""
    if not isinstance(text, str) or not text:
        return []
    searchable = re.sub(r"[#@]+", " ", text)
    return [team for team, pattern in _TEAM_MENTION_PATTERNS if pattern.search(searchable)]

_CACHED_STOPWORDS: frozenset[str] | None = None
_NLTK_LEMMATIZER = None
_NLTK_CHECKED = False


def get_bow_stopwords() -> frozenset[str]:
    """Return combined English + social + F1 boilerplate stopwords."""
    global _CACHED_STOPWORDS
    if _CACHED_STOPWORDS is not None:
        return _CACHED_STOPWORDS

    base_words = set(_FALLBACK_ENGLISH_STOPWORDS)
    try:
        from nltk.corpus import stopwords

        base_words.update(stopwords.words("english"))
    except Exception:
        pass

    base_words.update(BOW_CUSTOM_STOPWORDS)
    _CACHED_STOPWORDS = frozenset(base_words)
    return _CACHED_STOPWORDS


def _get_lemmatizer():
    """Lazily initialize NLTK WordNetLemmatizer if wordnet data is installed."""
    global _NLTK_LEMMATIZER, _NLTK_CHECKED
    if _NLTK_CHECKED:
        return _NLTK_LEMMATIZER
    _NLTK_CHECKED = True
    try:
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer

        wordnet.synsets("race")
        _NLTK_LEMMATIZER = WordNetLemmatizer()
    except Exception:
        _NLTK_LEMMATIZER = None
    return _NLTK_LEMMATIZER


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora when running the full pipeline."""
    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            continue
    global _CACHED_STOPWORDS, _NLTK_CHECKED
    _CACHED_STOPWORDS = None
    _NLTK_CHECKED = False


def detect_text_column(df: pd.DataFrame) -> str:
    """Locate the primary tweet text column in a DataFrame."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in CANDIDATE_TEXT_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            return col
    raise ValueError(f"Could not find a text column among {list(df.columns)!r}")


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Locate an optional date/timestamp column in a DataFrame."""
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in CANDIDATE_DATE_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def load_tweets_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load tweets from a local CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Tweet CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def load_kaggle_tweets(dataset_slug: str = KAGGLE_DATASET_SLUG) -> pd.DataFrame:
    """Download and load the Formula 1 tweets dataset from KaggleHub."""
    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(dataset_slug))
    csv_files = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() == ".csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in Kaggle dataset {dataset_slug!r} ({dataset_dir})")
    return pd.read_csv(csv_files[0], low_memory=False)


def extract_hashtags(text: str) -> list[str]:
    """Extract lowercase hashtag bodies (without '#') in order of appearance."""
    if not isinstance(text, str) or not text:
        return []
    return [m.group(1).lower() for m in _ANY_HASHTAG_PATTERN.finditer(text)]


def infer_race_weekend(text: str) -> str | None:
    """Infer the Grand Prix weekend from raw hashtags or circuit mentions before stripping."""
    if not isinstance(text, str) or not text:
        return None

    # Check hashtags first since #QatarGP / #BritishGP / #AbuDhabiGP are explicit tags.
    for tag in extract_hashtags(text):
        cleaned_tag = re.sub(r"(?:gp|grandprix|f1|\d+)$", "", tag)
        for key, canonical_race in _SORTED_RACE_KEYS:
            compact_key = key.replace(" ", "")
            if cleaned_tag == compact_key or compact_key in tag:
                return canonical_race

    lowered = text.lower()
    for key, canonical_race in _SORTED_RACE_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return canonical_race
    return None


def _expand_inline_hashtag(match: re.Match[str], hashtag_mode: HashtagMode) -> str:
    """Convert an inline hashtag to natural words or remove it if it is a race/slogan tag."""
    full_tag = match.group(0)
    body = match.group(1)
    lower_body = body.lower()
    alpha_body = re.sub(r"\d+$", "", lower_body)

    if hashtag_mode == "strip_all":
        return " "

    if lower_body in SLOGAN_HASHTAGS or alpha_body in SLOGAN_HASHTAGS:
        if alpha_body in GLUED_ENTITY_NORMALIZATIONS:
            return f" {GLUED_ENTITY_NORMALIZATIONS[alpha_body]} "
        return " "

    if RACE_OR_SERIES_HASHTAG_PATTERN.fullmatch(full_tag):
        if hashtag_mode == "downweight":
            stem = re.sub(r"(?:gp|grandprix|f1|\d+)$", "", lower_body)
            for key, canonical_race in _SORTED_RACE_KEYS:
                if stem == key.replace(" ", ""):
                    return f" {canonical_race.removesuffix(' GP')} "
        return " "

    if alpha_body in GLUED_ENTITY_NORMALIZATIONS:
        return f" {GLUED_ENTITY_NORMALIZATIONS[alpha_body]} "

    # Split CamelCase inline hashtags (#SafetyCar -> Safety Car) when used as sentence words.
    split_camel = _CAMEL_BOUNDARY_PATTERN.sub(" ", body).replace("_", " ")
    split_camel = re.sub(r"\d+$", "", split_camel).strip()
    return f" {split_camel} " if len(split_camel) >= 3 else " "


def _collapse_adjacent_duplicates(text: str) -> str:
    """Collapse stuttered adjacent tokens ('race race', 'lewis lewis') while keeping case."""
    tokens = text.split()
    if len(tokens) < 2:
        return text
    deduped = [tokens[0]]
    prev_norm = re.sub(r"\W+", "", tokens[0]).lower()
    for tok in tokens[1:]:
        curr_norm = re.sub(r"\W+", "", tok).lower()
        if curr_norm and curr_norm == prev_norm:
            continue
        deduped.append(tok)
        prev_norm = curr_norm
    return " ".join(deduped)


def clean_text_for_display(text: str) -> str:
    """Lightly clean a raw tweet for human display and Gemini prompt context."""
    if pd.isna(text):
        return ""
    cleaned = html.unescape(str(text))
    cleaned = _URL_PATTERN.sub("", cleaned)
    cleaned = _RT_PREFIX_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _replace_trailing_hashtag_block(match: re.Match[str], hashtag_mode: HashtagMode) -> str:
    """Strip trailing hashtag spam while preserving explicit driver/team subject entities."""
    if hashtag_mode == "strip_all":
        return " "
    kept_entities: list[str] = []
    for tag_match in _ANY_HASHTAG_PATTERN.finditer(match.group(0)):
        alpha_body = re.sub(r"\d+$", "", tag_match.group(1).lower())
        if alpha_body in GLUED_ENTITY_NORMALIZATIONS:
            entity_name = GLUED_ENTITY_NORMALIZATIONS[alpha_body]
            if entity_name not in kept_entities:
                kept_entities.append(entity_name)
    return (" " + " ".join(kept_entities) + " ") if kept_entities else " "


def clean_text_for_embedding(text: str, hashtag_mode: HashtagMode = "strip_race") -> str:
    """Prepare natural-language tweet prose for transformer embeddings without hashtag clustering."""
    if pd.isna(text):
        return ""

    cleaned = html.unescape(str(text))
    cleaned = cleaned.replace("&", " and ")
    cleaned = _RT_PREFIX_PATTERN.sub(" ", cleaned)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = _MENTION_PATTERN.sub(" ", cleaned)

    # Strip trailing tag footers while keeping explicit driver/team names.
    if hashtag_mode in ("strip_race", "strip_all"):
        cleaned = TRAILING_HASHTAG_BLOCK_PATTERN.sub(
            lambda m: _replace_trailing_hashtag_block(m, hashtag_mode),
            cleaned,
        )

    # Strip or expand remaining hashtags according to hashtag_mode.
    cleaned = _ANY_HASHTAG_PATTERN.sub(
        lambda m: _expand_inline_hashtag(m, hashtag_mode),
        cleaned,
    )

    # Remove un-hashed glued race tokens like "britishgp", "qatargp", "abudhabigp", "qatarf".
    if hashtag_mode in ("strip_race", "strip_all"):
        cleaned = GLUED_RACE_TOKEN_PATTERN.sub(" ", cleaned)
    else:
        cleaned = re.sub(
            r"\b([A-Za-z]+?)(?:gp|grandprix)\b",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Normalize glued driver/team handles ("mercedesamgf" -> "Mercedes", "maxverstappen" -> "Max Verstappen").
    cleaned = _GLUED_ENTITY_PATTERN.sub(
        lambda m: GLUED_ENTITY_NORMALIZATIONS[m.group(1).lower()],
        cleaned,
    )

    # Normalize punctuation and whitespace while preserving sentence syntax and negations.
    cleaned = re.sub(r"[^\w\s'.,!?-]", " ", cleaned)
    cleaned = re.sub(r"([.,!?-])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.,")
    return _collapse_adjacent_duplicates(cleaned)


def _lemmatize_token(token: str) -> str:
    """Lemmatize an English lowercase token with WordNet or a deterministic suffix fallback."""
    # Protect proper F1 names ending in 's' from mistaken singularization ("lewis" -> "lewi", "carlos" -> "carlo").
    if token in {
        "lewis", "carlos", "bottas", "norris", "gasly", "charles",
        "mercedes", "williams", "haas", "vegas", "texas", "pirelli", "masi",
    }:
        return token

    lemmatizer = _get_lemmatizer()
    if lemmatizer is not None:
        lemma = lemmatizer.lemmatize(token, pos="n")
        if lemma == token:
            lemma = lemmatizer.lemmatize(token, pos="v")
        return lemma if len(lemma) >= 3 else token

    # Lightweight fallback when NLTK WordNet is unavailable.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if token.endswith("ed") and len(token) > 4 and token not in {"robbed", "red", "speed"}:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
        return token[:-1]
    return token


def tokenize_for_topic_bow(text: str, hashtag_mode: HashtagMode = "strip_race") -> list[str]:
    """Tokenize tweet text into clean, deduplicated content words for c-TF-IDF and coherence."""
    prose = clean_text_for_embedding(text, hashtag_mode=hashtag_mode).lower()
    if not prose:
        return []

    # Always strip any residual glued race tags from the bag-of-words representation.
    prose = GLUED_RACE_TOKEN_PATTERN.sub(" ", prose)

    stop_words = get_bow_stopwords()
    raw_tokens = _TOKEN_ALPHA_PATTERN.findall(prose)

    tokens: list[str] = []
    prev_lemma: str | None = None
    for tok in raw_tokens:
        if len(tok) < 3 or len(tok) > 22:
            continue
        if len(set(tok)) <= 1:
            continue
        if tok in stop_words:
            continue
        lemma = _lemmatize_token(tok)
        if len(lemma) < 3 or lemma in stop_words:
            continue
        if GLUED_RACE_TOKEN_PATTERN.fullmatch(lemma):
            continue
        # Prevent consecutive identical stems ("race race", "hamilton hamilton").
        if lemma == prev_lemma:
            continue
        tokens.append(lemma)
        prev_lemma = lemma

    return tokens


def normalize_for_dedup(text: str) -> str:
    """Create a canonical fingerprint for near-duplicate tweet removal."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def filter_english_tweets(df: pd.DataFrame, text_col: str, target_lang: str = "en") -> pd.DataFrame:
    """Filter DataFrame to English tweets using precomputed column or deterministic langdetect."""
    if "language" in df.columns:
        filtered = df[df["language"].astype(str).str.lower() == target_lang.lower()].copy()
        if len(filtered) >= 100:
            return filtered

    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0

    def _safe_detect(val: object) -> str:
        if pd.isna(val):
            return "unknown"
        s = str(val).strip()
        if len(s) < 3:
            return "unknown"
        try:
            return detect(s)
        except Exception:
            return "unknown"

    out = df.copy()
    out["language"] = out[text_col].map(_safe_detect)
    english = out[out["language"] == target_lang].copy()
    return english if len(english) >= 100 else out


def filter_f1_relevant_tweets(df: pd.DataFrame, text_col: str, config: TopicPipelineConfig) -> pd.DataFrame:
    """Retain substantive Formula 1 tweets and exclude non-F1 motorsport noise."""
    texts = df[text_col].astype(str)
    has_f1 = texts.str.contains(_F1_ENTITY_FILTER_PATTERN, na=False)
    has_non_f1 = texts.str.contains(_NON_F1_EXCLUSION_PATTERN, na=False) & ~texts.str.contains(
        r"\b(?:f1|formula\s*1|verstappen|hamilton|ferrari|mercedes|red\s*bull|mclaren)\b",
        case=False,
        na=False,
    )

    filtered = df[has_f1 & ~has_non_f1].copy()
    word_counts = filtered[text_col].astype(str).str.split().str.len()
    char_lengths = filtered[text_col].astype(str).str.len()

    filtered = filtered[
        (word_counts >= config.min_words)
        & (word_counts <= config.max_words)
        & (char_lengths >= config.min_chars)
        & (char_lengths <= config.max_chars)
    ].copy()
    return filtered


def prepare_tweet_dataframe(
    df: pd.DataFrame,
    config: TopicPipelineConfig,
    text_col: str | None = None,
    filter_language: bool = True,
) -> pd.DataFrame:
    """Run sampling, language filtering, metadata extraction, hashtag stripping, and deduplication."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    resolved_text_col = text_col or detect_text_column(df)
    working = df.dropna(subset=[resolved_text_col]).copy()

    if config.sample_size is not None and len(working) > config.sample_size:
        working = working.sample(config.sample_size, random_state=config.random_seed).copy()

    if filter_language:
        working = filter_english_tweets(working, resolved_text_col, target_lang=config.language)

    working = filter_f1_relevant_tweets(working, resolved_text_col, config)
    if working.empty:
        return working

    raw_series = working[resolved_text_col].astype(str)
    working["raw_text"] = raw_series
    working["display_text"] = raw_series.map(clean_text_for_display)
    working["raw_hashtags"] = raw_series.map(extract_hashtags)
    working["inferred_race"] = raw_series.map(infer_race_weekend)
    working["mentioned_drivers"] = raw_series.map(extract_driver_mentions)
    working["mentioned_teams"] = raw_series.map(extract_team_mentions)

    date_col = detect_date_column(working)
    if date_col is not None:
        working["date_parsed"] = pd.to_datetime(working[date_col], errors="coerce", utc=True)
    else:
        working["date_parsed"] = pd.NaT

    working["embedding_text"] = raw_series.map(
        lambda t: clean_text_for_embedding(t, hashtag_mode=config.hashtag_mode)
    )
    working["dedup_key"] = working["embedding_text"].map(normalize_for_dedup)
    working = working[working["dedup_key"].str.len() > 0].drop_duplicates(subset=["dedup_key"]).copy()

    working["bow_tokens"] = working["raw_text"].map(
        lambda t: tokenize_for_topic_bow(t, hashtag_mode=config.hashtag_mode)
    )
    working["bow_text"] = working["bow_tokens"].map(" ".join)

    # Drop tweets that were purely hashtags/boilerplate once hashtags were stripped.
    cleaned_word_counts = working["embedding_text"].str.split().str.len()
    bow_word_counts = working["bow_tokens"].map(len)
    working = working[
        (cleaned_word_counts >= config.min_cleaned_words) & (bow_word_counts >= 3)
    ].copy()

    working = working.drop(columns=["dedup_key"]).reset_index(drop=True)
    return working


def build_bow_analyzer(hashtag_mode: HashtagMode = "strip_race", ngram_range: tuple[int, int] = (1, 2)):
    """Create a scikit-learn CountVectorizer analyzer producing clean 1..k-grams without hashtag noise."""
    min_n, max_n = ngram_range

    def _analyzer(doc: str) -> list[str]:
        unigrams = tokenize_for_topic_bow(doc, hashtag_mode=hashtag_mode)
        if max_n <= 1:
            return unigrams if min_n == 1 else []
        features: list[str] = list(unigrams) if min_n == 1 else []
        n_tokens = len(unigrams)
        for n in range(max(2, min_n), max_n + 1):
            for idx in range(n_tokens - n + 1):
                gram_tokens = unigrams[idx : idx + n]
                # Skip n-grams containing repeated words ("lewis max lewis").
                if len(set(gram_tokens)) < len(gram_tokens):
                    continue
                features.append(" ".join(gram_tokens))
        return features

    return _analyzer
