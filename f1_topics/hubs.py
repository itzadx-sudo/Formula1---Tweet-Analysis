"""Controversy Index scoring and Driver / Team / Race Hub aggregation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from f1_topics.config import TopicPipelineConfig

HubType = Literal["driver", "team", "race"]


@dataclass(slots=True)
class InsightCard:
    """Structured dataset insight card surfaced on the overview screen."""

    id: str
    kind: str
    title: str
    summary: str
    metric_value: float | None = None
    related_topic_id: int | None = None
    related_hub: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ControversyMetrics:
    """Polarization, emotional heat, and sample-shrunk Controversy Index for a slice of tweets."""

    controversy_index: float
    controversy_score_100: float
    polarization: float
    opinionated_share: float
    mean_antagonism: float
    favor_pct: float
    against_pct: float
    neutral_pct: float
    net_stance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HubSummary:
    """Aggregated Driver, Team, or Race Hub record for frontend screens."""

    hub_type: HubType
    name: str
    tweet_count: int
    share_pct: float
    controversy: ControversyMetrics
    dominant_emotion: str
    emotion_distribution: dict[str, float]
    top_topics: list[dict[str, Any]]
    representative_tweets: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["controversy"] = self.controversy.to_dict()
        return data


def compute_controversy_metrics(
    prob_favor: Sequence[float],
    prob_neutral: Sequence[float],
    prob_against: Sequence[float],
    antagonism: Sequence[float],
    polarization_weight: float = 0.65,
    heat_weight: float = 0.35,
    prior_k: float = 10.0,
) -> ControversyMetrics:
    """Compute bimodal Controversy Index in [0, 1] from stance probabilities and antagonism heat."""
    n = len(prob_favor)
    if n == 0:
        return ControversyMetrics(
            controversy_index=0.0,
            controversy_score_100=0.0,
            polarization=0.0,
            opinionated_share=0.0,
            mean_antagonism=0.0,
            favor_pct=0.0,
            against_pct=0.0,
            neutral_pct=0.0,
            net_stance=0.0,
        )

    p_pos = float(np.mean(prob_favor))
    p_neu = float(np.mean(prob_neutral))
    p_neg = float(np.mean(prob_against))
    total_p = p_pos + p_neu + p_neg
    if total_p > 0.0:
        p_pos /= total_p
        p_neu /= total_p
        p_neg /= total_p

    opinionated = float(np.clip(p_pos + p_neg, 0.0, 1.0))
    polarization = float((2.0 * min(p_pos, p_neg)) / (p_pos + p_neg + 1e-9)) if opinionated > 1e-6 else 0.0
    mean_antag = float(np.clip(np.mean(antagonism), 0.0, 1.0)) if len(antagonism) else 0.0

    weight_sum = max(polarization_weight + heat_weight, 1e-9)
    raw_score = (
        polarization_weight * opinionated * polarization
        + heat_weight * opinionated * mean_antag
    ) / weight_sum
    shrinkage = float(n / (n + max(prior_k, 0.0)))
    controversy = float(np.clip(shrinkage * raw_score, 0.0, 1.0))

    return ControversyMetrics(
        controversy_index=round(controversy, 4),
        controversy_score_100=round(controversy * 100.0, 1),
        polarization=round(polarization, 4),
        opinionated_share=round(opinionated, 4),
        mean_antagonism=round(mean_antag, 4),
        favor_pct=round(p_pos * 100.0, 1),
        against_pct=round(p_neg * 100.0, 1),
        neutral_pct=round(p_neu * 100.0, 1),
        net_stance=round(p_pos - p_neg, 4),
    )


def summarize_slice_affect(
    rows: pd.DataFrame,
    config: TopicPipelineConfig,
    target_entity: str | None = None,
) -> tuple[ControversyMetrics, str, dict[str, float]]:
    """Aggregate ControversyMetrics and emotion distribution for a DataFrame slice."""
    has_affect = all(
        col in rows.columns for col in ("prob_favor", "prob_neutral", "prob_against", "antagonism")
    )
    if rows.empty or not has_affect:
        empty_metrics = compute_controversy_metrics([], [], [], [])
        return empty_metrics, "neutral", {}

    p_fav: list[float] = []
    p_neu: list[float] = []
    p_agn: list[float] = []
    antag_vals = rows["antagonism"].astype(float).tolist()

    has_entity_stances = target_entity is not None and "entity_stances" in rows.columns
    if has_entity_stances:
        for row_fav, row_neu, row_agn, ent_map in zip(
            rows["prob_favor"].astype(float),
            rows["prob_neutral"].astype(float),
            rows["prob_against"].astype(float),
            rows["entity_stances"],
        ):
            if isinstance(ent_map, dict) and target_entity in ent_map:
                targeted = ent_map[target_entity]
                p_fav.append(float(targeted["prob_favor"]))
                p_neu.append(float(targeted["prob_neutral"]))
                p_agn.append(float(targeted["prob_against"]))
            else:
                p_fav.append(float(row_fav))
                p_neu.append(float(row_neu))
                p_agn.append(float(row_agn))
    else:
        p_fav = rows["prob_favor"].astype(float).tolist()
        p_neu = rows["prob_neutral"].astype(float).tolist()
        p_agn = rows["prob_against"].astype(float).tolist()

    metrics = compute_controversy_metrics(
        prob_favor=p_fav,
        prob_neutral=p_neu,
        prob_against=p_agn,
        antagonism=antag_vals,
        polarization_weight=config.controversy_polarization_weight,
        heat_weight=config.controversy_heat_weight,
        prior_k=config.controversy_prior_k,
    )

    emotion_sums: dict[str, float] = {}
    emotion_rows = 0
    if "emotion_scores" in rows.columns:
        for emo_map in rows["emotion_scores"]:
            if isinstance(emo_map, dict) and emo_map:
                emotion_rows += 1
                for emo_name, score in emo_map.items():
                    emotion_sums[emo_name] = emotion_sums.get(emo_name, 0.0) + float(score)

    emotion_dist = (
        {
            k: round(v / emotion_rows, 4)
            for k, v in sorted(emotion_sums.items(), key=lambda kv: kv[1], reverse=True)
        }
        if emotion_rows > 0
        else {}
    )

    dominant_emotion = "neutral"
    non_neutral = {k: v for k, v in emotion_dist.items() if k != "neutral"}
    if non_neutral:
        top_emo, top_score = max(non_neutral.items(), key=lambda kv: kv[1])
        if top_score >= 0.20:
            dominant_emotion = top_emo
    if dominant_emotion == "neutral" and "dominant_emotion" in rows.columns:
        counts = Counter(rows["dominant_emotion"].dropna().astype(str))
        if counts:
            dominant_emotion = counts.most_common(1)[0][0]

    return metrics, dominant_emotion, emotion_dist


def _build_hub_group(
    processed_df: pd.DataFrame,
    hub_type: HubType,
    index_map: dict[str, list[int]],
    config: TopicPipelineConfig,
    use_target_entity: bool,
) -> list[HubSummary]:
    total_tweets = max(len(processed_df), 1)
    hubs: list[HubSummary] = []

    for name, row_positions in index_map.items():
        if len(row_positions) < config.min_hub_tweets:
            continue
        slice_df = processed_df.iloc[row_positions]
        controversy, dom_emotion, emotion_dist = summarize_slice_affect(
            slice_df,
            config=config,
            target_entity=name if use_target_entity else None,
        )

        top_topics: list[dict[str, Any]] = []
        if "topic" in slice_df.columns and "topic_label" in slice_df.columns:
            clustered = slice_df[slice_df["topic"] != -1]
            if not clustered.empty:
                topic_pairs = (
                    clustered.groupby(["topic", "topic_label"])
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                for (tid, tlabel), cnt in topic_pairs.items():
                    top_topics.append(
                        {
                            "topic_id": int(tid),
                            "label": str(tlabel),
                            "tweet_count": int(cnt),
                        }
                    )

        text_col = "display_text" if "display_text" in slice_df.columns else "raw_text"
        rep_tweets = (
            slice_df[text_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(config.representative_docs_per_topic)
            .tolist()
        )

        hubs.append(
            HubSummary(
                hub_type=hub_type,
                name=name,
                tweet_count=len(row_positions),
                share_pct=round(len(row_positions) / total_tweets * 100.0, 2),
                controversy=controversy,
                dominant_emotion=dom_emotion,
                emotion_distribution=emotion_dist,
                top_topics=top_topics,
                representative_tweets=rep_tweets,
            )
        )

    hubs.sort(key=lambda h: h.tweet_count, reverse=True)
    return hubs


def build_entity_hubs(
    processed_df: pd.DataFrame,
    config: TopicPipelineConfig,
) -> dict[str, list[HubSummary]]:
    """Aggregate Driver, Team, and Race Hubs with stance splits, emotions, and Controversy Index."""
    driver_index: dict[str, list[int]] = {}
    team_index: dict[str, list[int]] = {}
    race_index: dict[str, list[int]] = {}

    if "mentioned_drivers" in processed_df.columns:
        for pos, drivers in enumerate(processed_df["mentioned_drivers"]):
            if isinstance(drivers, (list, tuple)):
                for d in drivers:
                    driver_index.setdefault(str(d), []).append(pos)

    if "mentioned_teams" in processed_df.columns:
        for pos, teams in enumerate(processed_df["mentioned_teams"]):
            if isinstance(teams, (list, tuple)):
                for t in teams:
                    team_index.setdefault(str(t), []).append(pos)

    if "inferred_race" in processed_df.columns:
        for pos, race in enumerate(processed_df["inferred_race"]):
            if isinstance(race, str) and race:
                race_index.setdefault(race, []).append(pos)

    return {
        "drivers": _build_hub_group(processed_df, "driver", driver_index, config, use_target_entity=True),
        "teams": _build_hub_group(processed_df, "team", team_index, config, use_target_entity=True),
        "races": _build_hub_group(processed_df, "race", race_index, config, use_target_entity=False),
    }


def build_dataset_insights(
    topic_summaries: Sequence[Any],
    hubs: dict[str, list[HubSummary]],
    temporal_series: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[InsightCard]:
    """Generate concise data-driven insight cards from topics, hubs, and temporal trends."""
    insights: list[InsightCard] = []

    scored = [s for s in topic_summaries if getattr(s, "controversy", None) is not None]
    if scored:
        top_flash = max(scored, key=lambda s: s.controversy.controversy_score_100)
        insights.append(
            InsightCard(
                id="flashpoint",
                kind="flashpoint",
                title=top_flash.label,
                summary=(
                    f"Highest controversy ({top_flash.controversy.controversy_score_100:.1f}/100): "
                    f"{top_flash.controversy.favor_pct:.0f}% favor vs {top_flash.controversy.against_pct:.0f}% "
                    f"against across {top_flash.tweet_count} tweets (dominant emotion: {top_flash.dominant_emotion})."
                ),
                metric_value=top_flash.controversy.controversy_score_100,
                related_topic_id=top_flash.topic_id,
            )
        )

        calm = [s for s in scored if s.controversy.controversy_index < 0.25]
        best_calm = max(calm, key=lambda s: s.tweet_count) if calm else min(scored[:10], key=lambda s: s.controversy.controversy_score_100)
        insights.append(
            InsightCard(
                id="consensus",
                kind="consensus",
                title=best_calm.label,
                summary=(
                    f"Broadest consensus ({best_calm.controversy.controversy_score_100:.1f}/100 controversy) "
                    f"across {best_calm.tweet_count} tweets ({best_calm.controversy.favor_pct:.0f}% positive / "
                    f"{best_calm.controversy.neutral_pct:.0f}% neutral)."
                ),
                metric_value=best_calm.controversy.controversy_score_100,
                related_topic_id=best_calm.topic_id,
            )
        )

    drivers = hubs.get("drivers", [])
    if drivers:
        split_driver = max(
            drivers,
            key=lambda h: h.controversy.polarization * math.log1p(h.tweet_count),
        )
        insights.append(
            InsightCard(
                id="driver_split",
                kind="driver_split",
                title=f"{split_driver.name}: most divided driver hub",
                summary=(
                    f"{split_driver.tweet_count} mentions split {split_driver.controversy.favor_pct:.0f}% favor / "
                    f"{split_driver.controversy.against_pct:.0f}% against "
                    f"(controversy {split_driver.controversy.controversy_score_100:.1f}/100)."
                ),
                metric_value=split_driver.controversy.controversy_score_100,
                related_hub=split_driver.name,
            )
        )

    races = hubs.get("races", [])
    if races:
        hot_race = max(races, key=lambda h: h.controversy.controversy_score_100)
        insights.append(
            InsightCard(
                id="race_heat",
                kind="race_heat",
                title=f"{hot_race.name}: most heated race weekend",
                summary=(
                    f"Controversy {hot_race.controversy.controversy_score_100:.1f}/100 across "
                    f"{hot_race.tweet_count} tweets (dominant emotion: {hot_race.dominant_emotion})."
                ),
                metric_value=hot_race.controversy.controversy_score_100,
                related_hub=hot_race.name,
            )
        )

    peak_point: tuple[str, int, str, int] | None = None
    for series in temporal_series:
        for pt in series.get("points", []):
            cnt = int(pt.get("count", 0))
            if peak_point is None or cnt > peak_point[3]:
                peak_point = (str(series.get("label", "")), int(series.get("topic_id", -1)), str(pt.get("period", "")), cnt)
    if peak_point is not None and peak_point[3] > 0:
        label, tid, period, cnt = peak_point
        insights.append(
            InsightCard(
                id="temporal_spike",
                kind="temporal_spike",
                title=f"Peak surge in {period}: {label}",
                summary=f"Single-period peak of {cnt} tweets occurred during {period}.",
                metric_value=float(cnt),
                related_topic_id=tid,
            )
        )

    num_topics = int(metrics.get("num_topics", 0))
    if num_topics > 0:
        insights.append(
            InsightCard(
                id="model_health",
                kind="model_health",
                title=f"{num_topics} consolidated F1 topics",
                summary=(
                    f"{metrics.get('clustering_success_pct', 0.0):.1f}% of tweets clustered | "
                    f"C_v coherence {metrics.get('average_coherence', 0.0):.3f} | "
                    f"diversity {metrics.get('topic_diversity', 0.0):.3f}."
                ),
                metric_value=float(metrics.get("average_coherence", 0.0)),
            )
        )

    return insights
