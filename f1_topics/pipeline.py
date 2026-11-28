"""End-to-end Formula 1 BERTopic pipeline and structured screen-payload exports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from f1_topics.affect import PostAffectAnalyzer, attach_post_affect_columns
from f1_topics.config import TopicPipelineConfig
from f1_topics.constants import ENGAGEMENT_CATEGORIES, TECHNICAL_CATEGORIES
from f1_topics.embeddings import TransformerTweetEmbedder
from f1_topics.evaluation import CoherenceReport, calculate_topic_coherence, extract_clean_topic_words
from f1_topics.hubs import (
    ControversyMetrics,
    HubSummary,
    InsightCard,
    build_dataset_insights,
    build_entity_hubs,
    summarize_slice_affect,
)
from f1_topics.labeling import _summarize_cluster_context, apply_topic_labels
from f1_topics.merging import consolidate_topics
from f1_topics.preprocessing import (
    build_bow_analyzer,
    ensure_nltk_resources,
    load_kaggle_tweets,
    load_tweets_csv,
    prepare_tweet_dataframe,
)


@dataclass(slots=True)
class TopicSummary:
    """Serializable topic record ready for UI screens, APIs, and charts."""

    topic_id: int
    rank: int
    label: str
    keywords: list[str]
    tweet_count: int
    share_pct: float
    coherence: float | None
    dominant_race: str | None
    peak_period: str | None
    technical_category: str | None
    engagement_category: str
    representative_tweets: list[str] = field(default_factory=list)
    controversy: ControversyMetrics | None = None
    dominant_emotion: str | None = None
    emotion_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineArtifacts:
    """Complete output bundle of a fitted F1 topic modeling run."""

    config: TopicPipelineConfig
    total_tweets: int
    clustered_tweets: int
    outlier_tweets: int
    num_topics: int
    coherence_report: CoherenceReport
    topic_summaries: list[TopicSummary]
    processed_df: pd.DataFrame
    topic_model: Any = None
    hubs: dict[str, list[HubSummary]] = field(
        default_factory=lambda: {"drivers": [], "teams": [], "races": []}
    )

    @property
    def clustering_success_rate(self) -> float:
        return (self.clustered_tweets / self.total_tweets * 100.0) if self.total_tweets else 0.0

    @property
    def outlier_rate(self) -> float:
        return (self.outlier_tweets / self.total_tweets * 100.0) if self.total_tweets else 0.0

    def to_screen_payload(self) -> dict[str, Any]:
        """Build a clean JSON-serializable payload for frontend screens and dashboards."""
        race_rollup: dict[str, int] = {}
        tech_rollup: dict[str, int] = {}
        engagement_rollup: dict[str, int] = {}

        for summary in self.topic_summaries:
            if summary.dominant_race:
                race_rollup[summary.dominant_race] = race_rollup.get(summary.dominant_race, 0) + summary.tweet_count
            if summary.technical_category:
                tech_rollup[summary.technical_category] = (
                    tech_rollup.get(summary.technical_category, 0) + summary.tweet_count
                )
            engagement_rollup[summary.engagement_category] = (
                engagement_rollup.get(summary.engagement_category, 0) + summary.tweet_count
            )

        temporal_series: list[dict[str, Any]] = []
        if "date_parsed" in self.processed_df.columns and "topic" in self.processed_df.columns:
            dated = self.processed_df.dropna(subset=["date_parsed"]).copy()
            dated = dated[dated["topic"] != -1]
            if not dated.empty:
                span_days = (dated["date_parsed"].max() - dated["date_parsed"].min()).days
                freq = "M" if span_days > 365 else ("W" if span_days > 30 else "D")
                dated["period"] = dated["date_parsed"].dt.tz_localize(None).dt.to_period(freq).astype(str)
                counts_matrix = dated.groupby(["period", "topic"]).size().unstack(fill_value=0)
                label_lookup = {s.topic_id: s.label for s in self.topic_summaries}
                for summary in self.topic_summaries[:8]:
                    if summary.topic_id in counts_matrix.columns:
                        series_points = [
                            {"period": str(period), "count": int(cnt)}
                            for period, cnt in counts_matrix[summary.topic_id].items()
                        ]
                        temporal_series.append(
                            {
                                "topic_id": summary.topic_id,
                                "label": label_lookup[summary.topic_id],
                                "points": series_points,
                            }
                        )

        scored_topics = [s for s in self.topic_summaries if s.controversy is not None]
        mean_controversy = (
            round(
                sum(s.controversy.controversy_index for s in scored_topics if s.controversy is not None)
                / len(scored_topics),
                4,
            )
            if scored_topics
            else None
        )
        controversy_leaderboard = [
            {
                "topic_id": s.topic_id,
                "label": s.label,
                "tweet_count": s.tweet_count,
                "dominant_emotion": s.dominant_emotion,
                **(s.controversy.to_dict() if s.controversy is not None else {}),
            }
            for s in sorted(
                scored_topics,
                key=lambda item: item.controversy.controversy_index if item.controversy else 0.0,
                reverse=True,
            )[:10]
        ]

        metrics_dict = {
            "total_tweets": self.total_tweets,
            "clustered_tweets": self.clustered_tweets,
            "outlier_tweets": self.outlier_tweets,
            "clustering_success_pct": round(self.clustering_success_rate, 2),
            "outlier_pct": round(self.outlier_rate, 2),
            "num_topics": self.num_topics,
            "coherence_metric": self.coherence_report.coherence_type,
            "average_coherence": round(self.coherence_report.average_coherence, 4),
            "topic_diversity": round(self.coherence_report.topic_diversity, 4),
            "mean_controversy_index": mean_controversy,
        }
        insights = [
            card.to_dict()
            for card in build_dataset_insights(
                self.topic_summaries,
                self.hubs,
                temporal_series,
                metrics_dict,
            )
        ]

        return {
            "metrics": metrics_dict,
            "insights": insights,
            "topics": [s.to_dict() for s in self.topic_summaries],
            "controversy_leaderboard": controversy_leaderboard,
            "hubs": {
                group: [hub.to_dict() for hub in items]
                for group, items in self.hubs.items()
            },
            "race_breakdown": dict(sorted(race_rollup.items(), key=lambda kv: kv[1], reverse=True)),
            "technical_breakdown": dict(sorted(tech_rollup.items(), key=lambda kv: kv[1], reverse=True)),
            "engagement_breakdown": dict(sorted(engagement_rollup.items(), key=lambda kv: kv[1], reverse=True)),
            "temporal_series": temporal_series,
        }

    def to_replay_events(self) -> list[dict[str, Any]]:
        """Build chronological tweet events for the live replay SSE backend."""
        df = self.processed_df.copy()
        if df.empty:
            return []

        has_dates = "date_parsed" in df.columns
        if has_dates:
            df["_ts"] = pd.to_datetime(df["date_parsed"], errors="coerce", utc=True)
            valid = df.dropna(subset=["_ts"]).sort_values("_ts", kind="mergesort")
            if not valid.empty:
                df = valid
            else:
                has_dates = False

        events: list[dict[str, Any]] = []
        base_ms = 1_639_310_400_000  # 2021-12-12T12:00:00Z fallback baseline
        for event_id, (_, row) in enumerate(df.iterrows()):
            if has_dates:
                ts = row["_ts"]
                unix_ms = int(ts.timestamp() * 1000)
                iso_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                unix_ms = base_ms + event_id * 10_000
                iso_str = pd.Timestamp(unix_ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

            race_val = row.get("inferred_race")
            race = str(race_val) if isinstance(race_val, str) and race_val else None
            drivers = row.get("mentioned_drivers")
            teams = row.get("mentioned_teams")
            text_val = row.get("display_text") if pd.notna(row.get("display_text")) else row.get("raw_text", "")

            events.append(
                {
                    "event_id": event_id,
                    "timestamp": iso_str,
                    "unix_ms": unix_ms,
                    "text": str(text_val),
                    "race": race,
                    "drivers": list(drivers) if isinstance(drivers, (list, tuple)) else [],
                    "teams": list(teams) if isinstance(teams, (list, tuple)) else [],
                    "topic_id": int(row.get("topic", -1)),
                    "topic_label": str(row.get("topic_label", "Outliers / Unclustered")),
                    "stance_label": str(row.get("stance_label", "neutral")),
                    "prob_favor": float(row.get("prob_favor", 0.0)),
                    "prob_neutral": float(row.get("prob_neutral", 1.0)),
                    "prob_against": float(row.get("prob_against", 0.0)),
                    "dominant_emotion": str(row.get("dominant_emotion", "neutral")),
                    "antagonism": float(row.get("antagonism", 0.0)),
                }
            )
        return events

    def save(self, output_dir: str | Path, save_model: bool = False) -> Path:
        """Persist screen JSON payloads, replay JSONL, assignments CSV, and optional model."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        payload = self.to_screen_payload()
        (out_path / "topics_screen_payload.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        (out_path / "hubs_screen_payload.json").write_text(
            json.dumps(
                {
                    "hubs": payload["hubs"],
                    "controversy_leaderboard": payload["controversy_leaderboard"],
                    "insights": payload["insights"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        replay_lines = "\n".join(json.dumps(ev) for ev in self.to_replay_events())
        (out_path / "replay_events.jsonl").write_text(
            (replay_lines + "\n") if replay_lines else "", encoding="utf-8"
        )

        export_cols = [
            c
            for c in (
                "raw_text",
                "display_text",
                "embedding_text",
                "inferred_race",
                "mentioned_drivers",
                "mentioned_teams",
                "date_parsed",
                "topic",
                "topic_label",
                "stance_label",
                "stance_score",
                "prob_favor",
                "prob_neutral",
                "prob_against",
                "dominant_emotion",
                "antagonism",
            )
            if c in self.processed_df.columns
        ]
        self.processed_df[export_cols].to_csv(out_path / "tweet_topic_assignments.csv", index=False)

        if save_model and self.topic_model is not None:
            model_dir = out_path / "bertopic_model"
            self.topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)

        return out_path


def _classify_technical_category(tokens: set[str]) -> str | None:
    best_cat: str | None = None
    best_hits = 0
    for category, indicators in TECHNICAL_CATEGORIES.items():
        hits = sum(1 for ind in indicators if ind in tokens)
        if hits > best_hits:
            best_hits = hits
            best_cat = category
    return best_cat if best_hits > 0 else None


def _classify_engagement_category(tokens: set[str]) -> str:
    best_cat = "Race Commentary"
    best_hits = 0
    for category, indicators in ENGAGEMENT_CATEGORIES.items():
        hits = sum(1 for ind in indicators if ind in tokens)
        if hits > best_hits:
            best_hits = hits
            best_cat = category
    return best_cat


class F1TopicPipeline:
    """Modular Formula 1 tweet topic modeling pipeline with Gemini LLM naming."""

    def __init__(
        self,
        config: TopicPipelineConfig | None = None,
        embedder: TransformerTweetEmbedder | None = None,
        affect_analyzer: PostAffectAnalyzer | None = None,
    ) -> None:
        self.config = config or TopicPipelineConfig()
        self.embedder = embedder or TransformerTweetEmbedder(
            model_name=self.config.embedding_model_name,
            batch_size=self.config.embedding_batch_size,
            max_length=self.config.max_seq_length,
            device=self.config.device,
        )
        self.affect_analyzer = affect_analyzer

    def _build_vectorizer_and_ctfidf(self, corpus_size: int) -> tuple[Any, Any]:
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        adaptive_min_df = min(self.config.min_df, max(1, corpus_size // 250))
        analyzer = build_bow_analyzer(
            hashtag_mode=self.config.hashtag_mode,
            ngram_range=self.config.ngram_range,
        )
        vectorizer = CountVectorizer(
            analyzer=analyzer,
            max_features=self.config.max_features,
            min_df=adaptive_min_df,
            max_df=self.config.max_df,
        )
        # bm25_weighting + reduce_frequent_words (sqrt-TF) down-weights ubiquitous domain & location words.
        ctfidf = ClassTfidfTransformer(
            bm25_weighting=self.config.bm25_weighting,
            reduce_frequent_words=self.config.reduce_frequent_words,
        )
        return vectorizer, ctfidf

    def _build_bertopic(self, corpus_size: int) -> tuple[Any, Any, Any]:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        from hdbscan import HDBSCAN
        from umap import UMAP

        # Scale min_cluster_size gracefully if run on a smaller subset than 26k tweets.
        scaled_min_cluster = min(
            self.config.hdbscan_min_cluster_size,
            max(15, corpus_size // max(self.config.max_topics * 2, 1)),
        )
        scaled_min_samples = min(self.config.hdbscan_min_samples, max(2, scaled_min_cluster // 5))

        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.random_seed,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=scaled_min_cluster,
            min_samples=scaled_min_samples,
            metric="euclidean",
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
            prediction_data=True,
        )
        vectorizer_model, ctfidf_model = self._build_vectorizer_and_ctfidf(corpus_size)

        # Diversify keyword terms so near-duplicate n-grams don't crowd the top-N keyword slots.
        keyword_representation = [
            KeyBERTInspired(),
            MaximalMarginalRelevance(diversity=self.config.mmr_diversity),
        ]

        model = BERTopic(
            embedding_model=self.embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=keyword_representation,
            top_n_words=self.config.top_n_words,
            nr_topics=None,
            min_topic_size=scaled_min_cluster,
            calculate_probabilities=False,
            verbose=False,
        )
        return model, vectorizer_model, ctfidf_model

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_col: str | None = None,
        filter_language: bool = True,
    ) -> PipelineArtifacts:
        """Clean tweets, fit BERTopic without hashtag pollution, consolidate to ~40-60 topics, and name with Gemini."""
        ensure_nltk_resources()
        processed_df = prepare_tweet_dataframe(
            df,
            config=self.config,
            text_col=text_col,
            filter_language=filter_language,
        )
        if len(processed_df) < 100:
            raise ValueError(
                f"Insufficient tweets after filtering ({len(processed_df)} remain; need at least 100)."
            )

        docs = processed_df["embedding_text"].tolist()
        embeddings = self.embedder.embed(docs)

        topic_model, vectorizer_model, ctfidf_model = self._build_bertopic(len(docs))
        raw_topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

        # Consolidate clusters into 40-60 topics before naming with Gemini.
        final_topics, topic_model = consolidate_topics(
            topic_model=topic_model,
            docs=docs,
            topics=raw_topics,
            embeddings=embeddings,
            config=self.config,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
        )

        # Refresh diverse keyword representations on the merged clusters.
        try:
            from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

            topic_model.update_topics(
                docs,
                topics=final_topics,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=[
                    KeyBERTInspired(),
                    MaximalMarginalRelevance(diversity=self.config.mmr_diversity),
                ],
                top_n_words=self.config.top_n_words,
            )
        except Exception:
            pass

        # Name final topics with Google Gemini (sets custom_labels_ on topic_model).
        label_map = apply_topic_labels(
            topic_model=topic_model,
            processed_df=processed_df,
            topics=final_topics,
            config=self.config,
        )

        coherence_report = calculate_topic_coherence(
            topic_model=topic_model,
            tokenized_tweets=processed_df["bow_tokens"].tolist(),
            topics=final_topics,
            coherence_type=self.config.coherence_metric,
            topn=self.config.coherence_top_n,
            min_words_per_topic=self.config.min_coherence_words_per_topic,
        )

        processed_df = processed_df.copy()
        processed_df["topic"] = final_topics
        processed_df["topic_label"] = [label_map.get(int(t), f"Topic {t}") for t in final_topics]

        if self.config.stance_mode != "off":
            analyzer = self.affect_analyzer or PostAffectAnalyzer(self.config)
            affects = analyzer.analyze_dataframe(processed_df)
            if affects:
                processed_df = attach_post_affect_columns(processed_df, affects)

        summaries = self._build_topic_summaries(
            topic_model=topic_model,
            processed_df=processed_df,
            topics=final_topics,
            label_map=label_map,
            coherence_report=coherence_report,
        )
        hubs = build_entity_hubs(processed_df, self.config)

        total_tweets = len(final_topics)
        outlier_tweets = sum(1 for t in final_topics if t == -1)
        clustered_tweets = total_tweets - outlier_tweets

        return PipelineArtifacts(
            config=self.config,
            total_tweets=total_tweets,
            clustered_tweets=clustered_tweets,
            outlier_tweets=outlier_tweets,
            num_topics=len(summaries),
            coherence_report=coherence_report,
            topic_summaries=summaries,
            processed_df=processed_df,
            topic_model=topic_model,
            hubs=hubs,
        )

    def _build_topic_summaries(
        self,
        topic_model: Any,
        processed_df: pd.DataFrame,
        topics: list[int],
        label_map: dict[int, str],
        coherence_report: CoherenceReport,
    ) -> list[TopicSummary]:
        total = len(topics)
        counts = Counter(t for t in topics if t != -1)
        ordered_topics = counts.most_common()

        summaries: list[TopicSummary] = []
        for rank, (topic_id, count) in enumerate(ordered_topics, start=1):
            raw_terms = topic_model.get_topic(topic_id) or []
            keywords = [w for w, _ in raw_terms[: self.config.top_n_words] if isinstance(w, str)]
            unigrams = set(extract_clean_topic_words(raw_terms, topn=15))

            cluster_rows = processed_df[processed_df["topic"] == topic_id]
            dom_race, _, _ = _summarize_cluster_context(cluster_rows)

            peak_period: str | None = None
            if "date_parsed" in cluster_rows.columns:
                valid_dates = pd.to_datetime(cluster_rows["date_parsed"], errors="coerce", utc=True).dropna()
                if not valid_dates.empty:
                    peak_period = Counter(valid_dates.dt.strftime("%Y-%m")).most_common(1)[0][0]

            rep_tweets = (
                cluster_rows["display_text"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(self.config.representative_docs_per_topic)
                .tolist()
            )

            has_affect = "prob_favor" in cluster_rows.columns and "antagonism" in cluster_rows.columns
            if has_affect:
                controversy, dom_emotion, emotion_dist = summarize_slice_affect(
                    cluster_rows,
                    config=self.config,
                )
            else:
                controversy, dom_emotion, emotion_dist = None, None, {}

            summaries.append(
                TopicSummary(
                    topic_id=int(topic_id),
                    rank=rank,
                    label=label_map.get(int(topic_id), f"Topic {topic_id}"),
                    keywords=keywords,
                    tweet_count=int(count),
                    share_pct=round((count / total) * 100.0, 2) if total else 0.0,
                    coherence=coherence_report.per_topic_coherence.get(int(topic_id)),
                    dominant_race=dom_race,
                    peak_period=peak_period,
                    technical_category=_classify_technical_category(unigrams),
                    engagement_category=_classify_engagement_category(unigrams),
                    representative_tweets=rep_tweets,
                    controversy=controversy,
                    dominant_emotion=dom_emotion,
                    emotion_distribution=emotion_dist,
                )
            )
        return summaries

    def run_from_csv(self, csv_path: str | Path, text_col: str | None = None) -> PipelineArtifacts:
        df = load_tweets_csv(csv_path)
        return self.fit_transform(df, text_col=text_col)

    def run_from_kaggle(self) -> PipelineArtifacts:
        df = load_kaggle_tweets()
        return self.fit_transform(df)
