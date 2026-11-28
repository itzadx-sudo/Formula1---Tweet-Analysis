"""Dashboard and diagnostic chart builders displaying LLM-generated topic titles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from f1_topics.pipeline import PipelineArtifacts


def _truncate_label(label: str, limit: int = 48) -> str:
    if len(label) <= limit:
        return label
    return label[: limit - 3].rsplit(" ", 1)[0] + "..."


def plot_topic_overview(artifacts: PipelineArtifacts, save_path: str | Path | None = None) -> Any:
    """Plot clustering coverage and the top LLM-named topic clusters."""
    import matplotlib.pyplot as plt

    fig, (ax_pie, ax_bars) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1.8]})

    ax_pie.pie(
        [artifacts.clustering_success_rate, artifacts.outlier_rate],
        labels=["Clustered", "Outliers"],
        colors=["#76c893", "#f28482"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_pie.set_title(f"Clustering Coverage ({artifacts.num_topics} Readable Topics)")

    top_summaries = artifacts.topic_summaries[:10]
    if top_summaries:
        labels = [_truncate_label(s.label, 52) for s in top_summaries]
        counts = [s.tweet_count for s in top_summaries]
        shares = [s.share_pct for s in top_summaries]

        y_pos = list(range(len(labels)))
        bars = ax_bars.barh(y_pos, counts, color="#4ea8de", alpha=0.88)
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, fontsize=9.5)
        ax_bars.invert_yaxis()
        ax_bars.set_xlabel("Tweets")
        ax_bars.set_title("Largest Formula 1 Narratives (Gemini-Named Topics)")
        ax_bars.grid(axis="x", alpha=0.25)

        max_count = max(counts)
        for bar, pct in zip(bars, shares):
            ax_bars.text(
                bar.get_width() + max_count * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
            )

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_f1_dashboard(artifacts: PipelineArtifacts, save_path: str | Path | None = None) -> Any:
    """Render 4-panel Formula 1 screen figure using LLM topic headlines instead of raw keywords."""
    import matplotlib.pyplot as plt

    payload = artifacts.to_screen_payload()
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    ax_top, ax_tech, ax_time, ax_eng = axes.flatten()

    # Top readable narratives
    top_topics = artifacts.topic_summaries[:8]
    if top_topics:
        titles = [_truncate_label(t.label, 46) for t in top_topics]
        counts = [t.tweet_count for t in top_topics]
        palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#F6BD60", "#B5838D", "#84A59D", "#F28482"]
        bars = ax_top.barh(range(len(titles)), counts, color=palette[: len(titles)], alpha=0.88)
        ax_top.set_yticks(range(len(titles)))
        ax_top.set_yticklabels(titles, fontsize=9.5)
        ax_top.invert_yaxis()
        ax_top.set_xlabel("Number of Tweets")
        ax_top.set_title("Top Fan Narratives & Event Discussions")
        ax_top.grid(axis="x", alpha=0.3)
        max_c = max(counts)
        for bar, count in zip(bars, counts):
            ax_top.text(
                bar.get_width() + max_c * 0.012,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontweight="bold",
                fontsize=9,
            )
    else:
        ax_top.text(0.5, 0.5, "No topics available", ha="center", va="center", transform=ax_top.transAxes)

    # Technical and sporting themes
    tech_data = payload["technical_breakdown"]
    if tech_data:
        cats = list(tech_data.keys())
        vals = list(tech_data.values())
        tech_colors = ["#66B2FF", "#FF9999", "#FF99CC", "#FFCC99", "#99FF99", "#99CCFF"]
        ax_tech.pie(
            vals,
            labels=cats,
            colors=tech_colors[: len(cats)],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax_tech.set_title("Technical & Officiating Themes")
    else:
        ax_tech.text(0.5, 0.5, "No technical themes detected", ha="center", va="center", transform=ax_tech.transAxes)
        ax_tech.set_title("Technical & Officiating Themes")

    # Temporal topic evolution labeled with Gemini headlines
    temporal_series = payload["temporal_series"][:3]
    if temporal_series:
        line_colors = ["#2b9348", "#e63946", "#3a86ff"]
        for idx, series in enumerate(temporal_series):
            periods = [pt["period"] for pt in series["points"]]
            counts = [pt["count"] for pt in series["points"]]
            ax_time.plot(
                periods,
                counts,
                marker="o",
                linewidth=2.2,
                color=line_colors[idx % len(line_colors)],
                label=_truncate_label(series["label"], 42),
            )
        ax_time.set_xlabel("Time Period")
        ax_time.set_ylabel("Tweet Count")
        ax_time.set_title("Temporal Topic Evolution")
        ax_time.legend(fontsize=8.5)
        ax_time.tick_params(axis="x", rotation=45)
        ax_time.grid(alpha=0.3)
    else:
        ax_time.text(0.5, 0.5, "No temporal data available", ha="center", va="center", transform=ax_time.transAxes)
        ax_time.set_title("Temporal Topic Evolution")

    # Fan engagement breakdown
    eng_data = payload["engagement_breakdown"]
    total_eng = sum(eng_data.values())
    if total_eng > 0:
        eng_cats = list(eng_data.keys())
        eng_pcts = [v / total_eng * 100.0 for v in eng_data.values()]
        eng_colors = ["#a8dadc", "#95d5b2", "#ffcad4", "#fde4cf", "#cdb4db"]
        bars = ax_eng.bar(
            eng_cats,
            eng_pcts,
            color=eng_colors[: len(eng_cats)],
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax_eng.set_ylabel("Distribution (%)")
        ax_eng.set_title("Fan Engagement Focus")
        ax_eng.tick_params(axis="x", rotation=30)
        top_pct = max(eng_pcts)
        for idx, pct in enumerate(eng_pcts):
            ax_eng.text(idx, pct + top_pct * 0.02, f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=9)
    else:
        ax_eng.text(0.5, 0.5, "No engagement data available", ha="center", va="center", transform=ax_eng.transAxes)

    fig.suptitle("Formula 1 Topic Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_coherence_diagnostics(artifacts: PipelineArtifacts, save_path: str | Path | None = None) -> Any:
    """Plot per-topic C_v coherence distribution alongside topic diversity and cluster counts."""
    import matplotlib.pyplot as plt

    report = artifacts.coherence_report
    fig = plt.figure(figsize=(14, 9))

    ax_coh = fig.add_subplot(2, 1, 1)
    scored_summaries = [s for s in artifacts.topic_summaries if s.coherence is not None]
    if scored_summaries:
        scores = [float(s.coherence) for s in scored_summaries if s.coherence is not None]
        ax_coh.bar(range(len(scores)), scores, color="#7b2cbf", alpha=0.75)
        ax_coh.axhline(
            y=report.average_coherence,
            color="#d00000",
            linestyle="--",
            linewidth=2,
            label=f"Mean C_v Coherence: {report.average_coherence:.3f} | Diversity: {report.topic_diversity:.3f}",
        )
        ax_coh.set_xlabel("Topic Rank (by size)")
        ax_coh.set_ylabel("C_v Coherence")
        ax_coh.set_title("Uninflated Per-Topic Coherence (Hashtags Stripped)")
        ax_coh.set_ylim(0, min(1.05, max(scores) * 1.15))
        ax_coh.legend()
    else:
        ax_coh.text(0.5, 0.5, "Coherence scores unavailable", ha="center", va="center", transform=ax_coh.transAxes)

    ax_counts = fig.add_subplot(2, 2, 3)
    labels = ["Cleaned Tweets", "Clustered", "Outliers", "Topics"]
    vals = [artifacts.total_tweets, artifacts.clustered_tweets, artifacts.outlier_tweets, artifacts.num_topics]
    ax_counts.bar(labels, vals, color=["#90e0ef", "#95d5b2", "#f28482", "#ffd166"])
    ax_counts.set_title("Corpus & Cluster Counts")
    top_v = max(vals) if vals else 1
    for idx, val in enumerate(vals):
        ax_counts.text(idx, val + top_v * 0.015, str(val), ha="center", va="bottom", fontsize=9)

    ax_qual = fig.add_subplot(2, 2, 4)
    q_labels = ["Clustered (%)", "Outliers (%)", "Diversity (%)"]
    q_vals = [
        artifacts.clustering_success_rate,
        artifacts.outlier_rate,
        report.topic_diversity * 100.0,
    ]
    ax_qual.bar(q_labels, q_vals, color=["#95d5b2", "#f28482", "#bdb2ff"])
    ax_qual.set_ylim(0, 110)
    ax_qual.set_title("Model Quality Rates")
    for idx, val in enumerate(q_vals):
        ax_qual.text(idx, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_intertopic_map(artifacts: PipelineArtifacts) -> Any:
    """Return BERTopic's interactive Plotly intertopic map using Gemini custom labels."""
    if artifacts.topic_model is None:
        raise ValueError("PipelineArtifacts does not hold a live BERTopic instance.")
    return artifacts.topic_model.visualize_topics(custom_labels=True)


def plot_controversy_and_hubs(artifacts: PipelineArtifacts, save_path: str | Path | None = None) -> Any:
    """Render 4-panel Controversy Index, Driver Hub Stance, Race/Team Controversy, and Emotions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
    ax_topics, ax_drivers, ax_hubs, ax_emo = axes.flatten()

    # Most controversial topics
    scored_topics = [s for s in artifacts.topic_summaries if s.controversy is not None]
    scored_topics.sort(
        key=lambda s: s.controversy.controversy_score_100 if s.controversy else 0.0,
        reverse=True,
    )
    top_controversial = scored_topics[:8]
    if top_controversial:
        labels = [_truncate_label(s.label, 44) for s in top_controversial]
        scores = [s.controversy.controversy_score_100 for s in top_controversial if s.controversy]
        bars = ax_topics.barh(range(len(labels)), scores, color="#e63946", alpha=0.85)
        ax_topics.set_yticks(range(len(labels)))
        ax_topics.set_yticklabels(labels, fontsize=9.5)
        ax_topics.invert_yaxis()
        ax_topics.set_xlim(0, 105)
        ax_topics.set_xlabel("Controversy Index (0-100)")
        ax_topics.set_title("Most Controversial Topics")
        ax_topics.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, scores):
            ax_topics.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9)
    else:
        ax_topics.text(0.5, 0.5, "Stance/emotion analysis disabled", ha="center", va="center", transform=ax_topics.transAxes)
        ax_topics.set_title("Most Controversial Topics")

    # Driver hub stance split
    driver_hubs = artifacts.hubs.get("drivers", [])[:8]
    if driver_hubs:
        d_names = [h.name for h in driver_hubs]
        favor = [h.controversy.favor_pct for h in driver_hubs]
        neutral = [h.controversy.neutral_pct for h in driver_hubs]
        against = [h.controversy.against_pct for h in driver_hubs]
        y_pos = list(range(len(d_names)))
        ax_drivers.barh(y_pos, favor, color="#2a9d8f", label="Favor (%)")
        ax_drivers.barh(y_pos, neutral, left=favor, color="#adb5bd", label="Neutral (%)")
        left_against = [f + n for f, n in zip(favor, neutral)]
        ax_drivers.barh(y_pos, against, left=left_against, color="#e76f51", label="Against (%)")
        ax_drivers.set_yticks(y_pos)
        ax_drivers.set_yticklabels(d_names, fontsize=9.5)
        ax_drivers.invert_yaxis()
        ax_drivers.set_xlim(0, 100)
        ax_drivers.set_xlabel("Share of Mentions (%)")
        ax_drivers.set_title("Driver Hub Stance Breakdown")
        ax_drivers.legend(loc="lower right", fontsize=8.5)
    else:
        ax_drivers.text(0.5, 0.5, "No driver hubs available", ha="center", va="center", transform=ax_drivers.transAxes)
        ax_drivers.set_title("Driver Hub Stance Breakdown")

    # Race and team hub controversy
    combined_hubs = artifacts.hubs.get("races", []) + artifacts.hubs.get("teams", [])
    combined_hubs.sort(key=lambda h: h.controversy.controversy_score_100, reverse=True)
    top_hubs = combined_hubs[:8]
    if top_hubs:
        h_labels = [f"{h.name} ({h.hub_type})" for h in top_hubs]
        h_scores = [h.controversy.controversy_score_100 for h in top_hubs]
        bars = ax_hubs.barh(range(len(h_labels)), h_scores, color="#457b9d", alpha=0.88)
        ax_hubs.set_yticks(range(len(h_labels)))
        ax_hubs.set_yticklabels(h_labels, fontsize=9.5)
        ax_hubs.invert_yaxis()
        ax_hubs.set_xlim(0, 105)
        ax_hubs.set_xlabel("Controversy Index (0-100)")
        ax_hubs.set_title("Most Polarized Race & Team Hubs")
        ax_hubs.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, h_scores):
            ax_hubs.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9)
    else:
        ax_hubs.text(0.5, 0.5, "No race/team hubs available", ha="center", va="center", transform=ax_hubs.transAxes)
        ax_hubs.set_title("Most Polarized Race & Team Hubs")

    # Overall emotion distribution
    if "dominant_emotion" in artifacts.processed_df.columns:
        emo_counts = artifacts.processed_df["dominant_emotion"].value_counts().head(7)
        ax_emo.bar(emo_counts.index.tolist(), emo_counts.values.tolist(), color="#8e9aaf", alpha=0.88)
        ax_emo.set_ylabel("Tweets")
        ax_emo.set_title("Dominant Per-Post Emotion Distribution")
        ax_emo.tick_params(axis="x", rotation=25)
    else:
        ax_emo.text(0.5, 0.5, "No emotion data available", ha="center", va="center", transform=ax_emo.transAxes)
        ax_emo.set_title("Dominant Per-Post Emotion Distribution")

    fig.suptitle("Formula 1 Hubs, Stance & Controversy Index", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig
