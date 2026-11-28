"""Batched per-post stance, emotion, and rival-aware entity NLI analysis."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from f1_topics.config import StanceMode, TopicPipelineConfig
from f1_topics.embeddings import resolve_device

StanceLabel = Literal["favor", "against", "neutral"]

_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")


@dataclass(slots=True)
class StanceScore:
    """Soft and discrete stance probabilities toward a target (or overall post polarity)."""

    label: StanceLabel
    prob_favor: float
    prob_neutral: float
    prob_against: float

    @property
    def net_score(self) -> float:
        """Continuous polarity in [-1.0, 1.0]."""
        return float(self.prob_favor - self.prob_against)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["net_score"] = round(self.net_score, 4)
        return data


@dataclass(slots=True)
class PostAffect:
    """Per-tweet stance, emotion probabilities, antagonism heat, and rival entity stances."""

    stance: StanceScore
    emotions: dict[str, float]
    dominant_emotion: str
    antagonism: float
    entity_stances: dict[str, StanceScore] = field(default_factory=dict)


def normalize_tweet_for_roberta(text: str) -> str:
    """Normalize @mentions and URLs following CardiffNLP Twitter-RoBERTa conventions."""
    if not isinstance(text, str):
        return ""
    norm = _MENTION_RE.sub("@user", text)
    norm = _URL_RE.sub("http", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _stance_from_probs(prob_favor: float, prob_neutral: float, prob_against: float) -> StanceScore:
    total = prob_favor + prob_neutral + prob_against
    if total > 0.0:
        prob_favor /= total
        prob_neutral /= total
        prob_against /= total
    else:
        prob_favor, prob_neutral, prob_against = 0.0, 1.0, 0.0

    best_label: StanceLabel = "neutral"
    best_val = prob_neutral
    if prob_favor > best_val:
        best_label = "favor"
        best_val = prob_favor
    if prob_against > best_val:
        best_label = "against"

    return StanceScore(
        label=best_label,
        prob_favor=float(prob_favor),
        prob_neutral=float(prob_neutral),
        prob_against=float(prob_against),
    )


def _map_sentiment_label_indices(id2label: dict[Any, str]) -> tuple[int, int, int]:
    """Identify (against/negative, neutral, favor/positive) logit indices from model config."""
    neg_idx, neu_idx, pos_idx = 0, 1, 2
    for raw_idx, raw_name in id2label.items():
        idx = int(raw_idx)
        name = str(raw_name).lower()
        if "neg" in name or name == "label_0":
            neg_idx = idx
        elif "neu" in name or name == "label_1":
            neu_idx = idx
        elif "pos" in name or name == "label_2":
            pos_idx = idx
    return neg_idx, neu_idx, pos_idx


def _find_entailment_index(id2label: dict[Any, str]) -> int:
    """Locate entailment logit index for NLI models (handles DeBERTa/RoBERTa conventions)."""
    for raw_idx, raw_name in id2label.items():
        if "entail" in str(raw_name).lower():
            return int(raw_idx)
    return 0


class PostAffectAnalyzer:
    """Local batched transformer analyzer for per-post stance, emotion, and rival NLI targeting."""

    def __init__(
        self,
        config: TopicPipelineConfig | None = None,
        stance_mode: StanceMode | None = None,
        sentiment_model_name: str | None = None,
        emotion_model_name: str | None = None,
        nli_stance_model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = config or TopicPipelineConfig()
        self.stance_mode: StanceMode = stance_mode or cfg.stance_mode
        self.sentiment_model_name = sentiment_model_name or cfg.sentiment_model_name
        self.emotion_model_name = emotion_model_name or cfg.emotion_model_name
        self.nli_stance_model_name = nli_stance_model_name or cfg.nli_stance_model_name
        self.batch_size = batch_size or cfg.affect_batch_size
        self.device_name = resolve_device(device if device is not None else cfg.device)
        self._models: dict[str, tuple[Any, Any]] = {}

    def _load_seq_cls(self, model_name: str) -> tuple[Any, Any]:
        if model_name in self._models:
            return self._models[model_name]
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            torch.device(self.device_name)
        )
        model.eval()
        self._models[model_name] = (tokenizer, model)
        return tokenizer, model

    def _predict_sentiment_stance(self, texts: Sequence[str]) -> list[StanceScore]:
        if not texts:
            return []
        import torch

        tokenizer, model = self._load_seq_cls(self.sentiment_model_name)
        device = torch.device(self.device_name)
        neg_idx, neu_idx, pos_idx = _map_sentiment_label_indices(getattr(model.config, "id2label", {}))
        results: list[StanceScore] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = [normalize_tweet_for_roberta(t) for t in texts[start : start + self.batch_size]]
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                results.append(
                    _stance_from_probs(
                        prob_favor=float(row[pos_idx]),
                        prob_neutral=float(row[neu_idx]),
                        prob_against=float(row[neg_idx]),
                    )
                )
        return results

    def _predict_emotions(self, texts: Sequence[str]) -> list[tuple[dict[str, float], str, float]]:
        if not texts:
            return []
        import torch

        tokenizer, model = self._load_seq_cls(self.emotion_model_name)
        device = torch.device(self.device_name)
        id2label = getattr(model.config, "id2label", {})
        is_multilabel = (
            getattr(model.config, "problem_type", "") == "multi_label_classification"
            or "multilabel" in self.emotion_model_name.lower()
        )
        results: list[tuple[dict[str, float], str, float]] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = [normalize_tweet_for_roberta(t) for t in texts[start : start + self.batch_size]]
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probs_t = torch.sigmoid(logits) if is_multilabel else torch.softmax(logits, dim=-1)
                probs = probs_t.cpu().numpy()

            for row in probs:
                emo_dict = {
                    str(id2label.get(i, f"emotion_{i}")).lower(): float(row[i])
                    for i in range(len(row))
                }
                if emo_dict:
                    top_label, top_val = max(emo_dict.items(), key=lambda kv: kv[1])
                    dominant = top_label if top_val >= 0.25 else "neutral"
                else:
                    dominant = "neutral"
                antagonism = float(
                    np.clip(emo_dict.get("anger", 0.0) + 0.75 * emo_dict.get("disgust", 0.0), 0.0, 1.0)
                )
                results.append((emo_dict, dominant, antagonism))
        return results

    def _predict_entity_nli_stances(
        self,
        texts: Sequence[str],
        target_entities_per_row: Sequence[Sequence[str]],
    ) -> list[dict[str, StanceScore]]:
        """Score targeted favor/against/neutral stance per entity on multi-entity rival tweets."""
        per_row: list[dict[str, StanceScore]] = [{} for _ in texts]
        pairs: list[tuple[int, str, str, str]] = []

        for row_idx, (text, entities) in enumerate(zip(texts, target_entities_per_row)):
            if len(entities) < 2:
                continue
            for entity in entities:
                pairs.append(
                    (
                        row_idx,
                        entity,
                        f"The author supports or praises {entity}.",
                        f"The author criticizes, blames, or opposes {entity}.",
                    )
                )

        if not pairs:
            return per_row

        import torch

        tokenizer, model = self._load_seq_cls(self.nli_stance_model_name)
        device = torch.device(self.device_name)
        entail_idx = _find_entailment_index(getattr(model.config, "id2label", {}))

        # Score both hypotheses in flat batches (2*N) for throughput.
        premises: list[str] = []
        hypotheses: list[str] = []
        for row_idx, _, h_pos, h_neg in pairs:
            clean_txt = normalize_tweet_for_roberta(texts[row_idx])
            premises.extend((clean_txt, clean_txt))
            hypotheses.extend((h_pos, h_neg))

        entail_probs: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            encoded = tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            entail_probs.extend(float(row[entail_idx]) for row in probs)

        for idx, (row_idx, entity, _, _) in enumerate(pairs):
            e_pos = entail_probs[2 * idx]
            e_neg = entail_probs[2 * idx + 1]
            e_neu = max(0.0, 1.0 - max(e_pos, e_neg))
            per_row[row_idx][entity] = _stance_from_probs(e_pos, e_neu, e_neg)

        return per_row

    def analyze_dataframe(self, df: pd.DataFrame) -> list[PostAffect]:
        """Run per-post stance and emotion scoring on display_text (and NLI on rival posts)."""
        if df.empty or self.stance_mode == "off":
            return []

        text_series = df["display_text"] if "display_text" in df.columns else df["raw_text"]
        texts = text_series.astype(str).tolist()

        stances = self._predict_sentiment_stance(texts)
        emotions = self._predict_emotions(texts)

        if self.stance_mode == "hybrid_nli":
            drivers_col = df["mentioned_drivers"].tolist() if "mentioned_drivers" in df.columns else [[] for _ in texts]
            teams_col = df["mentioned_teams"].tolist() if "mentioned_teams" in df.columns else [[] for _ in texts]
            targets: list[list[str]] = []
            for drivers, teams in zip(drivers_col, teams_col):
                row_targets: list[str] = []
                if isinstance(drivers, (list, tuple)) and len(drivers) >= 2:
                    row_targets.extend(drivers)
                if isinstance(teams, (list, tuple)) and len(teams) >= 2:
                    for tm in teams:
                        if tm not in row_targets:
                            row_targets.append(tm)
                targets.append(row_targets)
            entity_stances = self._predict_entity_nli_stances(texts, targets)
        else:
            entity_stances = [{} for _ in texts]

        return [
            PostAffect(
                stance=st,
                emotions=emo_dict,
                dominant_emotion=dom_emo,
                antagonism=antag,
                entity_stances=ent_map,
            )
            for st, (emo_dict, dom_emo, antag), ent_map in zip(stances, emotions, entity_stances)
        ]


def attach_post_affect_columns(df: pd.DataFrame, affects: Sequence[PostAffect]) -> pd.DataFrame:
    """Attach per-post stance, emotion, antagonism, and rival entity stance columns to a DataFrame."""
    if len(df) != len(affects):
        raise ValueError(f"Length mismatch: df has {len(df)} rows but got {len(affects)} affect records.")

    out = df.copy()
    out["stance_label"] = [a.stance.label for a in affects]
    out["stance_score"] = [round(a.stance.net_score, 4) for a in affects]
    out["prob_favor"] = [round(a.stance.prob_favor, 4) for a in affects]
    out["prob_neutral"] = [round(a.stance.prob_neutral, 4) for a in affects]
    out["prob_against"] = [round(a.stance.prob_against, 4) for a in affects]
    out["dominant_emotion"] = [a.dominant_emotion for a in affects]
    out["antagonism"] = [round(a.antagonism, 4) for a in affects]
    out["emotion_scores"] = [dict(a.emotions) for a in affects]
    out["entity_stances"] = [
        {entity: score.to_dict() for entity, score in a.entity_stances.items()}
        for a in affects
    ]
    return out
