"""Command-line entrypoint for running the Formula 1 topic modeling pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from f1_topics.config import TopicPipelineConfig
from f1_topics.pipeline import F1TopicPipeline
from f1_topics.visualization import (
    plot_coherence_diagnostics,
    plot_controversy_and_hubs,
    plot_f1_dashboard,
    plot_topic_overview,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-topics",
        description="Cluster Formula 1 tweets into ~40-60 hashtag-clean topics named by Google Gemini.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a local tweets CSV file (defaults to KaggleHub download if omitted).",
    )
    source_group.add_argument(
        "--kaggle",
        action="store_true",
        help="Download and load the Formula 1 Trending Tweets dataset via KaggleHub.",
    )

    parser.add_argument("--text-col", type=str, default=None, help="Explicit tweet text column name.")
    parser.add_argument("--sample-size", type=int, default=45_000, help="Tweet sample size (0 to use full dataset).")
    parser.add_argument(
        "--hashtag-mode",
        choices=("strip_race", "strip_all", "downweight"),
        default="strip_race",
        help="How race and event hashtags are handled before embedding and clustering.",
    )
    parser.add_argument("--min-topics", type=int, default=40, help="Lower bound of target topic window.")
    parser.add_argument("--target-topics", type=int, default=50, help="Target topic count when reducing clusters.")
    parser.add_argument("--max-topics", type=int, default=60, help="Upper bound before hierarchical topic reduction.")
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for merging near-duplicate topics.",
    )
    parser.add_argument(
        "--stance-mode",
        choices=("hybrid_nli", "sentiment", "off"),
        default="hybrid_nli",
        help="Per-post stance/emotion inference mode ('hybrid_nli' adds rival-aware NLI targeting).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Google Gemini model name used to generate screen headlines (reads GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Gemini API calls and use deterministic keyword/context fallback titles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write JSON screen payload, assignments CSV, and optional plots.",
    )
    parser.add_argument("--save-plots", action="store_true", help="Render PNG charts into --output-dir.")
    parser.add_argument("--save-model", action="store_true", help="Save serialized BERTopic model into --output-dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = TopicPipelineConfig(
        sample_size=None if args.sample_size == 0 else args.sample_size,
        hashtag_mode=args.hashtag_mode,
        min_topics=args.min_topics,
        target_topics=args.target_topics,
        max_topics=args.max_topics,
        merge_similarity_threshold=args.merge_threshold,
        stance_mode=args.stance_mode,
        enable_llm_naming=not args.no_llm,
        gemini_model=args.gemini_model,
    )

    pipeline = F1TopicPipeline(config=config)
    artifacts = (
        pipeline.run_from_csv(args.csv, text_col=args.text_col)
        if args.csv is not None
        else pipeline.run_from_kaggle()
    )

    out_dir = artifacts.save(args.output_dir, save_model=args.save_model)
    if args.save_plots:
        plot_f1_dashboard(artifacts, save_path=out_dir / "f1_topic_dashboard.png")
        plot_topic_overview(artifacts, save_path=out_dir / "f1_topic_overview.png")
        plot_coherence_diagnostics(artifacts, save_path=out_dir / "f1_coherence_diagnostics.png")
        if args.stance_mode != "off":
            plot_controversy_and_hubs(artifacts, save_path=out_dir / "f1_controversy_and_hubs.png")

    print(
        f"Completed: {artifacts.num_topics} topics across {artifacts.clustered_tweets}/{artifacts.total_tweets} "
        f"tweets ({artifacts.clustering_success_rate:.1f}% clustered) | "
        f"Mean C_v={artifacts.coherence_report.average_coherence:.4f} | "
        f"Diversity={artifacts.coherence_report.topic_diversity:.4f}"
    )
    for summary in artifacts.topic_summaries[:15]:
        coh_str = f"{summary.coherence:.3f}" if summary.coherence is not None else "n/a"
        c_str = (
            f" | Controversy={summary.controversy.controversy_score_100:.1f}"
            if summary.controversy is not None
            else ""
        )
        print(f"  #{summary.rank:02d} [{summary.tweet_count:4d} tweets | C_v={coh_str}{c_str}] {summary.label}")

    top_drivers = artifacts.hubs.get("drivers", [])[:5]
    if top_drivers:
        print("Top Driver Hubs:")
        for hub in top_drivers:
            print(
                f"  - {hub.name}: {hub.tweet_count} tweets | "
                f"Favor {hub.controversy.favor_pct:.0f}% / Against {hub.controversy.against_pct:.0f}% | "
                f"Controversy={hub.controversy.controversy_score_100:.1f} | Emotion={hub.dominant_emotion}"
            )
    print(f"Saved screen payload and assignments to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
