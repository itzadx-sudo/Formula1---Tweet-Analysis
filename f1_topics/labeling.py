"""Gemini-powered BERTopic representation model for human-readable screen titles."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import pandas as pd
from scipy.sparse import csr_matrix

from f1_topics.config import TopicPipelineConfig, load_env_file

try:
    from bertopic.representation._base import BaseRepresentation
except Exception:
    class BaseRepresentation:  # type: ignore[no-redef]
        pass


SYSTEM_INSTRUCTION = """You write concise, vivid Formula 1 topic headlines for analytics screens.
Given each topic's top keywords, optional date/race context, and representative fan tweets,
write ONE specific human-readable headline per topic.

Formatting rules:
1. Aim for 5 to 10 words (maximum 70 characters).
2. Capture what fans are actually talking about or reacting to, not a bag of keywords.
3. Include the race and year before a colon ONLY when the tweets clearly center on a specific race/season event:
   - Good: "Abu Dhabi 2021: fans call result robbery"
   - Good: "Silverstone 2021: Hamilton and Verstappen Lap 1 crash"
   - Good: "Mercedes W13: porpoising and suspension struggles"
   - Good: "Michael Masi: calls to resign after finale"
   - Good: "Pirelli compounds and pit stop tyre strategy"
   - Bad: "lewis, max, abudhabigp, robbed"
   - Bad: "Tweets discussing Formula 1 racing"
4. Never include hashtags (#), @mentions, quotes, or topic numbers.
5. Every topic in a batch must receive a distinct headline."""

SINGLE_TOPIC_PROMPT_TEMPLATE = """Topic keywords: [KEYWORDS]
Context: [CONTEXT]
Representative tweets:
[DOCUMENTS]

Return a single 5-10 word headline for this Formula 1 topic (no hashtags, no quotes):"""


def sanitize_topic_label(raw_label: str | None, fallback: str) -> str:
    """Clean model output into a single crisp screen headline."""
    if not raw_label or not isinstance(raw_label, str):
        return fallback

    label = raw_label.strip().splitlines()[0].strip()
    label = label.strip("\"'`“”‘’")
    label = re.sub(r"^(?:topic\s*#?\d+\s*[:\-–]\s*|headline\s*[:\-–]\s*)", "", label, flags=re.IGNORECASE)
    label = label.strip("\"'`“”‘’")
    label = re.sub(r"#([A-Za-z0-9_]+)", r"\1", label)
    label = re.sub(r"\s+", " ", label).strip(" .;-")
    label = label.strip("\"'`“”‘’")

    if len(label) < 4:
        return fallback
    if len(label) > 76:
        clipped = label[:73].rsplit(" ", 1)[0].rstrip(" ,;:-")
        label = f"{clipped}..."
    return label


def format_fallback_label(
    keywords: Sequence[str],
    dominant_race: str | None = None,
    peak_year: str | None = None,
) -> str:
    """Generate a clean readable fallback label when LLM naming is disabled or unavailable."""
    clean_kws: list[str] = []
    seen_roots: set[str] = set()
    for kw in keywords:
        phrase = kw.strip()
        if not phrase:
            continue
        words = phrase.lower().split()
        if any(w in seen_roots for w in words):
            continue
        seen_roots.update(words)
        clean_kws.append(phrase.title() if len(words) <= 2 else phrase.capitalize())
        if len(clean_kws) == 3:
            break

    core = ", ".join(clean_kws[:2]) if clean_kws else "General F1 Discussion"
    if len(clean_kws) == 3:
        core = f"{clean_kws[0]}, {clean_kws[1]} & {clean_kws[2]}"

    if dominant_race:
        prefix = f"{dominant_race} {peak_year}".strip() if peak_year else dominant_race
        if dominant_race.lower().removesuffix(" gp") not in core.lower():
            return f"{prefix}: {core}"
    return core


def _summarize_cluster_context(
    cluster_rows: pd.DataFrame | None,
) -> tuple[str | None, str | None, str]:
    """Summarize dominant race tag and peak month/year from raw metadata if concentrated."""
    if cluster_rows is None or cluster_rows.empty:
        return None, None, "Season-wide discussion"

    dominant_race: str | None = None
    peak_year: str | None = None
    hints: list[str] = []

    if "inferred_race" in cluster_rows.columns:
        races = cluster_rows["inferred_race"].dropna()
        if len(races) >= 2:
            top_race, count = Counter(races).most_common(1)[0]
            # Only surface race hint when strongly concentrated in this cluster.
            if count / max(len(cluster_rows), 1) >= 0.25 or count / len(races) >= 0.55:
                dominant_race = str(top_race)
                hints.append(f"Primary race tag in raw tweets: {dominant_race}")

    if "date_parsed" in cluster_rows.columns:
        valid_dates = pd.to_datetime(cluster_rows["date_parsed"], errors="coerce", utc=True).dropna()
        if len(valid_dates) >= 2:
            months = valid_dates.dt.strftime("%Y-%m")
            top_month, m_count = Counter(months).most_common(1)[0]
            if m_count / len(valid_dates) >= 0.35:
                peak_year = top_month[:4]
                hints.append(f"Peak tweet period: {top_month}")

    return dominant_race, peak_year, (" | ".join(hints) if hints else "Multi-race / season-wide")


class GeminiTopicRepresentation(BaseRepresentation):
    """BERTopic BaseRepresentation implementation backed by Google's `google-genai` SDK."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        metadata_df: pd.DataFrame | None = None,
        nr_docs: int = 6,
        batch_size: int = 20,
        temperature: float = 0.2,
        max_retries: int = 3,
        prompt: str | None = None,
        client: Any = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.metadata_df = metadata_df
        self.nr_docs = nr_docs
        self.batch_size = max(1, batch_size)
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt = prompt or SINGLE_TOPIC_PROMPT_TEMPLATE
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        load_env_file()
        resolved_key = (
            (self.api_key or "").strip()
            or os.environ.get(self.api_key_env, "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if not resolved_key:
            raise EnvironmentError(
                f"Missing Gemini API key. Set {self.api_key_env} in your environment or pass api_key."
            )
        from google import genai

        self._client = genai.Client(api_key=resolved_key)
        return self._client

    def _select_representative_tweets(
        self,
        topic_model: Any,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix | None,
        topics: Mapping[int, list[tuple[str, float]]],
        topic_id: int,
    ) -> list[str]:
        """Extract readable representative tweets for a topic, preferring raw display_text."""
        selected_docs: list[str] = []

        # Attempt BERTopic's internal c-TF-IDF representative document extractor first.
        if topic_model is not None and c_tf_idf is not None and hasattr(topic_model, "_extract_representative_docs"):
            try:
                repr_docs, _, _, _ = topic_model._extract_representative_docs(
                    c_tf_idf,
                    documents,
                    topics,
                    500,
                    self.nr_docs,
                )
                selected_docs = list(repr_docs.get(topic_id, []))[: self.nr_docs]
            except Exception:
                selected_docs = []

        topic_mask = documents["Topic"] == topic_id
        if not selected_docs:
            selected_docs = documents.loc[topic_mask, "Document"].head(self.nr_docs).astype(str).tolist()

        # Map cleaned embedding documents back to natural `display_text` when metadata_df is aligned.
        if self.metadata_df is not None and len(self.metadata_df) == len(documents) and "display_text" in self.metadata_df.columns:
            emb_to_display: dict[str, str] = {}
            topic_indices = documents.index[topic_mask]
            for idx in topic_indices:
                doc_val = str(documents.at[idx, "Document"])
                disp_val = str(self.metadata_df.iloc[int(idx)]["display_text"])
                if disp_val:
                    emb_to_display[doc_val] = disp_val
            selected_docs = [emb_to_display.get(d, d) for d in selected_docs]

        return [d.strip() for d in selected_docs if d and d.strip()][: self.nr_docs]

    def _generate_text(self, prompt_text: str, json_mode: bool = False) -> str:
        """Invoke Gemini with exponential backoff on transient rate-limit or server errors."""
        client = self._get_client()

        config_obj: Any = None
        try:
            from google.genai import types

            cfg_kwargs: dict[str, Any] = {
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": self.temperature,
            }
            if json_mode:
                cfg_kwargs["response_mime_type"] = "application/json"
            config_obj = types.GenerateContentConfig(**cfg_kwargs)
        except Exception:
            config_obj = {
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": self.temperature,
                **({"response_mime_type": "application/json"} if json_mode else {}),
            }

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_text,
                    config=config_obj,
                )
                return (response.text or "").strip()
            except Exception as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_err}") from last_err

    def _label_batch_json(self, batch_dossiers: list[dict[str, Any]]) -> dict[int, str]:
        """Request distinct headlines for a batch of topics in a single structured JSON call."""
        lines = [
            "Generate a JSON object mapping each topic_id (as a string key) to its 5-10 word headline.",
            'Example output format: {"0": "Abu Dhabi 2021: fans call result robbery", "2": "Mercedes W13: porpoising and suspension struggles"}',
            "",
            "Topics to name:",
        ]
        for item in batch_dossiers:
            tid = item["topic_id"]
            kw_str = ", ".join(item["keywords"][:8])
            doc_lines = "\n".join(f"  - {doc}" for doc in item["docs"])
            lines.append(
                f"\n[Topic {tid}]\nKeywords: {kw_str}\nContext: {item['context']}\nRepresentative tweets:\n{doc_lines}"
            )

        raw_json = self._generate_text("\n".join(lines), json_mode=True)
        # Strip optional markdown code fences if present.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json.strip(), flags=re.IGNORECASE)
        parsed = json.loads(cleaned)

        result: dict[int, str] = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                digits = re.search(r"-?\d+", str(k))
                if digits and isinstance(v, str):
                    result[int(digits.group(0))] = v
        return result

    def _label_single_topic(self, dossier: dict[str, Any]) -> str:
        kw_str = ", ".join(dossier["keywords"][:8])
        doc_str = "\n".join(f"- {doc}" for doc in dossier["docs"])
        filled = (
            self.prompt.replace("[KEYWORDS]", kw_str)
            .replace("[DOCUMENTS]", doc_str)
            .replace("[CONTEXT]", dossier["context"])
        )
        return self._generate_text(filled, json_mode=False)

    def extract_topics(
        self,
        topic_model: Any,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[int, list[tuple[str, float]]],
    ) -> dict[int, list[tuple[str, float]]]:
        """Extract concise Gemini headlines for every active topic in BERTopic."""
        dossiers: list[dict[str, Any]] = []
        fallbacks: dict[int, str] = {}
        updated_topics: dict[int, list[tuple[str, float]]] = {}

        aligned_meta = (
            self.metadata_df
            if self.metadata_df is not None and len(self.metadata_df) == len(documents)
            else None
        )

        for topic_id, kw_tuples in topics.items():
            tid = int(topic_id)
            if tid == -1:
                updated_topics[tid] = [("Outliers / Unclustered", 1.0)]
                continue

            kw_list = [w for w, _ in kw_tuples if isinstance(w, str)]
            cluster_rows = aligned_meta.loc[(documents["Topic"] == tid).values] if aligned_meta is not None else None
            dominant_race, peak_year, context_str = _summarize_cluster_context(cluster_rows)
            fallback_label = format_fallback_label(kw_list, dominant_race=dominant_race, peak_year=peak_year)
            fallbacks[tid] = fallback_label

            rep_docs = self._select_representative_tweets(topic_model, documents, c_tf_idf, topics, tid)
            dossiers.append(
                {
                    "topic_id": tid,
                    "keywords": kw_list,
                    "docs": rep_docs,
                    "context": context_str,
                    "dominant_race": dominant_race,
                    "peak_year": peak_year,
                }
            )

        raw_labels: dict[int, str] = {}
        for start in range(0, len(dossiers), self.batch_size):
            batch = dossiers[start : start + self.batch_size]
            try:
                batch_labels = self._label_batch_json(batch)
                raw_labels.update(batch_labels)
            except Exception:
                batch_labels = {}

            # Per-topic retry for any topic missing from the batch response.
            for item in batch:
                tid = item["topic_id"]
                if tid not in raw_labels:
                    try:
                        raw_labels[tid] = self._label_single_topic(item)
                    except Exception:
                        raw_labels[tid] = fallbacks[tid]

        # Sanitize and guarantee uniqueness across screen headlines.
        used_titles_lower: set[str] = set()
        for item in dossiers:
            tid = item["topic_id"]
            candidate = sanitize_topic_label(raw_labels.get(tid), fallbacks[tid])
            if candidate.lower() in used_titles_lower:
                qualifier = item["dominant_race"] or (item["keywords"][0].title() if item["keywords"] else f"#{tid}")
                if qualifier.lower() not in candidate.lower():
                    candidate = sanitize_topic_label(f"{candidate} ({qualifier})", fallbacks[tid])
                else:
                    candidate = f"{candidate} [{tid}]"
            used_titles_lower.add(candidate.lower())
            updated_topics[tid] = [(candidate, 1.0)]

        return updated_topics


def apply_topic_labels(
    topic_model: Any,
    processed_df: pd.DataFrame,
    topics: Sequence[int],
    config: TopicPipelineConfig,
) -> dict[int, str]:
    """Generate human-readable topic titles (via Gemini or fallback) and register them on BERTopic."""
    documents_df = pd.DataFrame(
        {
            "Document": processed_df["embedding_text"].astype(str).tolist(),
            "ID": range(len(processed_df)),
            "Topic": [int(t) for t in topics],
        }
    )

    active_topic_ids = sorted({int(t) for t in topics if int(t) != -1})
    base_topics_map: dict[int, list[tuple[str, float]]] = {
        tid: (topic_model.get_topic(tid) or []) for tid in active_topic_ids
    }
    base_topics_map[-1] = topic_model.get_topic(-1) or []

    load_env_file()
    has_key = bool(
        (config.gemini_api_key or "").strip()
        or os.environ.get(config.gemini_api_key_env, "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )

    label_map: dict[int, str] = {-1: "Outliers / Unclustered"}
    if config.enable_llm_naming and has_key:
        namer = GeminiTopicRepresentation(
            model_name=config.gemini_model,
            api_key=config.gemini_api_key,
            api_key_env=config.gemini_api_key_env,
            metadata_df=processed_df,
            nr_docs=config.representative_docs_per_topic,
            batch_size=config.gemini_batch_size,
            temperature=config.gemini_temperature,
            max_retries=config.gemini_max_retries,
        )
        extracted = namer.extract_topics(
            topic_model=topic_model,
            documents=documents_df,
            c_tf_idf=getattr(topic_model, "c_tf_idf_", None),
            topics=base_topics_map,
        )
        if hasattr(topic_model, "topic_aspects_"):
            topic_model.topic_aspects_["Gemini"] = extracted
        for tid, tuples in extracted.items():
            if tuples:
                label_map[int(tid)] = tuples[0][0]
    else:
        for tid in active_topic_ids:
            kws = [w for w, _ in base_topics_map.get(tid, []) if isinstance(w, str)]
            cluster_rows = processed_df.loc[[t == tid for t in topics]]
            dom_race, peak_year, _ = _summarize_cluster_context(cluster_rows)
            label_map[tid] = format_fallback_label(kws, dominant_race=dom_race, peak_year=peak_year)

    if hasattr(topic_model, "set_topic_labels"):
        topic_model.set_topic_labels(label_map)

    return label_map
