"""Unit tests verifying hashtag stripping, near-duplicate merging, coherence, and Gemini naming."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import tokenize
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from f1_topics.affect import PostAffect, StanceScore, attach_post_affect_columns
from f1_topics.config import TopicPipelineConfig, load_env_file
from f1_topics.evaluation import CoherenceReport, compute_topic_diversity, extract_clean_topic_words
from f1_topics.hubs import build_entity_hubs, compute_controversy_metrics
from f1_topics.labeling import GeminiTopicRepresentation, apply_topic_labels, sanitize_topic_label
from f1_topics.merging import cosine_similarity_rows, find_near_duplicate_groups
from f1_topics.pipeline import PipelineArtifacts, TopicSummary
from f1_topics.preprocessing import (
    build_bow_analyzer,
    clean_text_for_embedding,
    extract_driver_mentions,
    extract_team_mentions,
    infer_race_weekend,
    prepare_tweet_dataframe,
    tokenize_for_topic_bow,
)


class PreprocessingAndHashtagTests(unittest.TestCase):
    def test_race_hashtags_stripped_before_embedding_while_race_metadata_preserved(self) -> None:
        raw = "Lewis was robbed on the final lap restart! &amp; Masi broke the rules #AbuDhabiGP #F1 #TeamLH"
        self.assertEqual(infer_race_weekend(raw), "Abu Dhabi GP")

        emb = clean_text_for_embedding(raw, hashtag_mode="strip_race")
        self.assertNotIn("abudhabigp", emb.lower())
        self.assertNotIn("#", emb)
        self.assertNotIn("amp", emb.lower())
        self.assertIn("Lewis was robbed", emb)

        bow = tokenize_for_topic_bow(raw, hashtag_mode="strip_race")
        self.assertNotIn("abudhabigp", bow)
        self.assertNotIn("britishgp", bow)
        self.assertNotIn("qatargp", bow)
        self.assertIn("robbed", bow)
        self.assertIn("masi", bow)

    def test_glued_race_tokens_and_entity_handles_normalized(self) -> None:
        raw = "What a drive britishgp britishgrandprix qatargp #MaxVerstappen #MercedesAMGF1"
        emb = clean_text_for_embedding(raw, hashtag_mode="strip_race")
        self.assertNotIn("britishgp", emb.lower())
        self.assertNotIn("britishgrandprix", emb.lower())
        self.assertNotIn("qatargp", emb.lower())
        self.assertIn("Max Verstappen", emb)
        self.assertIn("Mercedes", emb)

    def test_stuttered_duplicate_tokens_do_not_form_race_race_bigrams(self) -> None:
        analyzer = build_bow_analyzer(hashtag_mode="strip_race", ngram_range=(1, 2))
        grams = analyzer("Race of the race! Lewis, oh Lewis wins the championship")
        self.assertNotIn("race race", grams)
        self.assertNotIn("lewis lewis", grams)

    def test_hashtag_only_tweets_filtered_out_of_corpus(self) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    "Let's go racing!! #BritishGP #BritishGrandPrix #F1 #SkyF1 #Silverstone",
                    "Lewis Hamilton and Max Verstappen collide at Copse corner on lap one! #BritishGP #F1",
                    "Lewis Hamilton and Max Verstappen collide at Copse corner on lap one! #F1 #BritishGP",
                ],
                "date": ["2021-07-18", "2021-07-18", "2021-07-18"],
                "language": ["en", "en", "en"],
            }
        )
        cfg = TopicPipelineConfig(sample_size=None, min_words=4, min_chars=20, min_cleaned_words=4)
        prepared = prepare_tweet_dataframe(df, cfg, filter_language=False)

        # Hashtag-only cheer tweet is dropped; duplicate copy differing only in trailing hashtags is deduplicated.
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.iloc[0]["inferred_race"], "British GP")
        self.assertNotIn("britishgp", prepared.iloc[0]["embedding_text"].lower())


class MergingAndCoherenceTests(unittest.TestCase):
    def test_near_duplicate_merging_groups_close_clusters_without_chaining(self) -> None:
        # Topics 0 and 4 are near-duplicates (Abu Dhabi finale); Topic 9 (tyres) is distinct.
        vectors = np.array(
            [
                [0.98, 0.20, 0.00],
                [0.96, 0.28, 0.02],
                [0.01, 0.05, 0.99],
            ],
            dtype=np.float64,
        )
        sim = cosine_similarity_rows(vectors)
        groups = find_near_duplicate_groups(sim, topic_ids=[0, 4, 9], threshold=0.85)
        self.assertEqual(groups, [[0, 4]])

    def test_coherence_word_extractor_splits_ngrams_and_blocks_hashtag_inflation(self) -> None:
        # Mirrors legacy Topic 0 where 9 of top 10 terms were multi-word bigrams + 'abudhabigp'.
        legacy_topic_0 = [
            ("lewis max", 0.09),
            ("max lewis", 0.08),
            ("lewis abudhabigp", 0.07),
            ("lewis win", 0.06),
            ("lewis robbed", 0.05),
            ("robbed", 0.04),
            ("amp lewis", 0.03),
        ]
        words = extract_clean_topic_words(legacy_topic_0, topn=10)
        self.assertEqual(words, ["lewis", "max", "win", "robbed"])
        self.assertNotIn("abudhabigp", words)
        self.assertNotIn("amp", words)
        self.assertAlmostEqual(
            compute_topic_diversity([["lewis", "max", "robbed"], ["pirelli", "tyre", "compound"]]),
            1.0,
        )


class GeminiLabelingAndScreenPayloadTests(unittest.TestCase):
    def test_gemini_representation_generates_distinct_screen_titles(self) -> None:
        fake_response_payload = {
            "0": "Abu Dhabi 2021: fans call result robbery",
            "2": "Mercedes W13: porpoising and suspension struggles",
        }

        class FakeModels:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def generate_content(self, model: str, contents: str, config: object = None) -> object:
                self.calls.append(contents)
                return types.SimpleNamespace(text=json.dumps(fake_response_payload))

        fake_models = FakeModels()
        fake_client = types.SimpleNamespace(models=fake_models)

        meta_df = pd.DataFrame(
            {
                "display_text": [
                    "Lewis was robbed on the last lap restart!",
                    "Masi handed Max the title on the final lap.",
                    "Mercedes porpoising down the straight looks brutal.",
                    "Bouncing on the W13 suspension is costing lap time.",
                ],
                "inferred_race": ["Abu Dhabi GP", "Abu Dhabi GP", "Bahrain GP", None],
                "date_parsed": pd.to_datetime(
                    ["2021-12-12", "2021-12-12", "2022-03-20", "2022-03-20"], utc=True
                ),
            }
        )
        docs_df = pd.DataFrame(
            {
                "Document": meta_df["display_text"].tolist(),
                "ID": [0, 1, 2, 3],
                "Topic": [0, 0, 2, 2],
            }
        )
        topics_map = {
            0: [("hamilton", 0.1), ("verstappen", 0.09), ("robbed", 0.08), ("masi", 0.07)],
            2: [("mercedes", 0.12), ("porpoising", 0.1), ("suspension", 0.08)],
        }

        namer = GeminiTopicRepresentation(
            model_name="gemini-2.5-flash",
            metadata_df=meta_df,
            client=fake_client,
        )
        labeled = namer.extract_topics(
            topic_model=None,
            documents=docs_df,
            c_tf_idf=csr_matrix((3, 4)),
            topics=topics_map,
        )

        self.assertEqual(len(fake_models.calls), 1)
        self.assertIn("Abu Dhabi GP", fake_models.calls[0])
        self.assertIn("2021-12", fake_models.calls[0])
        self.assertEqual(labeled[0][0][0], "Abu Dhabi 2021: fans call result robbery")
        self.assertEqual(labeled[2][0][0], "Mercedes W13: porpoising and suspension struggles")

    def test_sanitize_topic_label_strips_hashtags_and_prefixes(self) -> None:
        cleaned = sanitize_topic_label('Topic 0: "Abu Dhabi 2021: fans call #F1 result robbery."', "Fallback")
        self.assertEqual(cleaned, 'Abu Dhabi 2021: fans call F1 result robbery')

    def test_screen_payload_serializes_llm_labels_for_ui(self) -> None:
        df = pd.DataFrame(
            {
                "raw_text": ["Lewis robbed! #AbuDhabiGP"],
                "display_text": ["Lewis robbed! #AbuDhabiGP"],
                "embedding_text": ["Lewis robbed!"],
                "inferred_race": ["Abu Dhabi GP"],
                "date_parsed": pd.to_datetime(["2021-12-12"], utc=True),
                "topic": [0],
                "topic_label": ["Abu Dhabi 2021: fans call result robbery"],
            }
        )
        artifacts = PipelineArtifacts(
            config=TopicPipelineConfig(),
            total_tweets=1,
            clustered_tweets=1,
            outlier_tweets=0,
            num_topics=1,
            coherence_report=CoherenceReport(
                coherence_type="c_v",
                average_coherence=0.49,
                per_topic_coherence={0: 0.49},
                topic_diversity=0.92,
                evaluated_topic_count=1,
            ),
            topic_summaries=[
                TopicSummary(
                    topic_id=0,
                    rank=1,
                    label="Abu Dhabi 2021: fans call result robbery",
                    keywords=["hamilton", "verstappen", "robbed", "masi"],
                    tweet_count=1,
                    share_pct=100.0,
                    coherence=0.49,
                    dominant_race="Abu Dhabi GP",
                    peak_period="2021-12",
                    technical_category="Safety & Race Control",
                    engagement_category="Title & Driver Rivalry",
                    representative_tweets=["Lewis robbed! #AbuDhabiGP"],
                )
            ],
            processed_df=df,
        )
        payload = artifacts.to_screen_payload()
        self.assertEqual(payload["topics"][0]["label"], "Abu Dhabi 2021: fans call result robbery")
        self.assertEqual(payload["temporal_series"][0]["label"], "Abu Dhabi 2021: fans call result robbery")

    def test_load_env_file_reads_gemini_key_and_ignores_blank_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"

            # Blank GEMINI_API_KEY leaves key unset so fallback naming runs cleanly.
            env_path.write_text("GEMINI_API_KEY=\n# comment line\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(env_path)
                self.assertEqual(loaded, env_path)
                self.assertNotIn("GEMINI_API_KEY", os.environ)

                fake_model = types.SimpleNamespace(
                    get_topic=lambda tid: [("hamilton", 0.2), ("robbed", 0.1)] if tid == 0 else []
                )
                processed_df = pd.DataFrame(
                    {
                        "embedding_text": ["Lewis robbed at the end"],
                        "inferred_race": ["Abu Dhabi GP"],
                        "date_parsed": pd.to_datetime(["2021-12-12"], utc=True),
                    }
                )
                with mock.patch("f1_topics.labeling.load_env_file", return_value=env_path):
                    labels = apply_topic_labels(
                        topic_model=fake_model,
                        processed_df=processed_df,
                        topics=[0],
                        config=TopicPipelineConfig(enable_llm_naming=True),
                    )
                self.assertIn(0, labels)
                self.assertIn("Hamilton", labels[0])

            # Non-empty key loads into environment.
            env_path.write_text('GEMINI_API_KEY="test-secret-key"\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "test-secret-key")

            # Existing non-empty env var takes precedence.
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "already-set"}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "already-set")


class AffectHubsAndControversyTests(unittest.TestCase):
    def test_driver_and_team_entity_extraction_with_max_speed_guard(self) -> None:
        rival_tweet = "Max cooked Lewis at the restart #AbuDhabiGP #RedBullRacing"
        drivers = extract_driver_mentions(rival_tweet)
        teams = extract_team_mentions(rival_tweet)
        self.assertEqual(set(drivers), {"Lewis Hamilton", "Max Verstappen"})
        self.assertEqual(teams, ["Red Bull"])

        telemetry_tweet = "Mercedes hit top max speed down the Kemmel straight"
        self.assertEqual(extract_driver_mentions(telemetry_tweet), [])
        self.assertEqual(extract_team_mentions(telemetry_tweet), ["Mercedes"])

    def test_controversy_index_separates_polarized_wars_from_consensus_praise_and_neutral(self) -> None:
        # 50/50 split with high anger/disgust (e.g. Abu Dhabi 2021 debate)
        n_split = 200
        war = compute_controversy_metrics(
            prob_favor=[0.85] * 100 + [0.05] * 100,
            prob_neutral=[0.10] * n_split,
            prob_against=[0.05] * 100 + [0.85] * 100,
            antagonism=[0.78] * n_split,
            prior_k=10.0,
        )
        self.assertGreater(war.controversy_index, 0.65)
        self.assertGreater(war.polarization, 0.9)

        # Unanimous praise (e.g. Kimi farewell) -> low polarization & low antagonism -> near 0 controversy
        praise = compute_controversy_metrics(
            prob_favor=[0.90] * n_split,
            prob_neutral=[0.08] * n_split,
            prob_against=[0.02] * n_split,
            antagonism=[0.04] * n_split,
            prior_k=10.0,
        )
        self.assertLess(praise.controversy_index, 0.10)

        # Pure neutral updates -> near 0 controversy
        neutral = compute_controversy_metrics(
            prob_favor=[0.05] * n_split,
            prob_neutral=[0.90] * n_split,
            prob_against=[0.05] * n_split,
            antagonism=[0.05] * n_split,
            prior_k=10.0,
        )
        self.assertLess(neutral.controversy_index, 0.10)

        # Sample-size shrinkage: identical split over N=2 scores much lower than N=200
        tiny = compute_controversy_metrics(
            prob_favor=[0.85, 0.05],
            prob_neutral=[0.10, 0.10],
            prob_against=[0.05, 0.85],
            antagonism=[0.78, 0.78],
            prior_k=10.0,
        )
        self.assertLess(tiny.controversy_index, war.controversy_index * 0.3)

    def test_rival_aware_entity_stances_route_opposing_stances_to_driver_hubs(self) -> None:
        df = pd.DataFrame(
            {
                "display_text": [
                    "Max deserved it, Lewis choked on the restart",
                    "Verstappen outdrove Hamilton all season",
                    "Hamilton was robbed while Verstappen got gifted",
                    "Lewis got cheated by the stewards, Max lucky",
                ],
                "mentioned_drivers": [
                    ["Lewis Hamilton", "Max Verstappen"],
                    ["Lewis Hamilton", "Max Verstappen"],
                    ["Lewis Hamilton", "Max Verstappen"],
                    ["Lewis Hamilton", "Max Verstappen"],
                ],
                "mentioned_teams": [["Red Bull"], ["Red Bull"], ["Mercedes"], ["Mercedes"]],
                "inferred_race": ["Abu Dhabi GP"] * 4,
                "topic": [0, 0, 0, 0],
                "topic_label": ["Abu Dhabi 2021: fans call result robbery"] * 4,
            }
        )
        affects = [
            PostAffect(
                stance=StanceScore("against", 0.10, 0.15, 0.75),
                emotions={"anger": 0.60, "disgust": 0.30, "joy": 0.05},
                dominant_emotion="anger",
                antagonism=0.82,
                entity_stances={
                    "Max Verstappen": StanceScore("favor", 0.88, 0.08, 0.04),
                    "Lewis Hamilton": StanceScore("against", 0.05, 0.10, 0.85),
                },
            ),
            PostAffect(
                stance=StanceScore("favor", 0.75, 0.15, 0.10),
                emotions={"joy": 0.70, "anger": 0.08},
                dominant_emotion="joy",
                antagonism=0.08,
                entity_stances={
                    "Max Verstappen": StanceScore("favor", 0.90, 0.07, 0.03),
                    "Lewis Hamilton": StanceScore("against", 0.08, 0.12, 0.80),
                },
            ),
            PostAffect(
                stance=StanceScore("against", 0.05, 0.10, 0.85),
                emotions={"anger": 0.80, "disgust": 0.40},
                dominant_emotion="anger",
                antagonism=0.95,
                entity_stances={
                    "Lewis Hamilton": StanceScore("favor", 0.86, 0.09, 0.05),
                    "Max Verstappen": StanceScore("against", 0.04, 0.08, 0.88),
                },
            ),
            PostAffect(
                stance=StanceScore("against", 0.05, 0.10, 0.85),
                emotions={"anger": 0.78, "disgust": 0.35},
                dominant_emotion="anger",
                antagonism=0.90,
                entity_stances={
                    "Lewis Hamilton": StanceScore("favor", 0.85, 0.10, 0.05),
                    "Max Verstappen": StanceScore("against", 0.05, 0.10, 0.85),
                },
            ),
        ]
        enriched = attach_post_affect_columns(df, affects)
        hubs = build_entity_hubs(enriched, TopicPipelineConfig(min_hub_tweets=2, controversy_prior_k=0.0))

        driver_map = {h.name: h for h in hubs["drivers"]}
        race_map = {h.name: h for h in hubs["races"]}

        # Pro-Max/anti-Lewis + pro-Lewis/anti-Max collide in Abu Dhabi GP and split each driver hub ~50/50
        self.assertIn("Max Verstappen", driver_map)
        self.assertIn("Lewis Hamilton", driver_map)
        self.assertGreater(driver_map["Max Verstappen"].controversy.controversy_index, 0.65)
        self.assertGreater(driver_map["Lewis Hamilton"].controversy.controversy_index, 0.65)
        self.assertIn("Abu Dhabi GP", race_map)
        self.assertEqual(race_map["Abu Dhabi GP"].top_topics[0]["label"], "Abu Dhabi 2021: fans call result robbery")

        # Verify targeted entity routing on pure pro-Max / anti-Lewis slice
        pro_max_only = build_entity_hubs(
            enriched.iloc[:2],
            TopicPipelineConfig(min_hub_tweets=2, controversy_prior_k=0.0),
        )
        pro_max_drivers = {h.name: h for h in pro_max_only["drivers"]}
        self.assertGreater(pro_max_drivers["Max Verstappen"].controversy.net_stance, 0.5)
        self.assertLess(pro_max_drivers["Lewis Hamilton"].controversy.net_stance, -0.5)

    def test_insights_and_replay_jsonl_export(self) -> None:
        df = pd.DataFrame(
            {
                "raw_text": ["Lewis robbed! #AbuDhabiGP", "Max won the title! #AbuDhabiGP"],
                "display_text": ["Lewis robbed! #AbuDhabiGP", "Max won the title! #AbuDhabiGP"],
                "embedding_text": ["Lewis robbed", "Max won the title"],
                "inferred_race": ["Abu Dhabi GP", "Abu Dhabi GP"],
                "mentioned_drivers": [["Lewis Hamilton"], ["Max Verstappen"]],
                "mentioned_teams": [["Mercedes"], ["Red Bull"]],
                "date_parsed": pd.to_datetime(["2021-12-12T15:05:00Z", "2021-12-12T15:02:00Z"], utc=True),
                "topic": [0, 0],
                "topic_label": ["Abu Dhabi 2021: fans call result robbery"] * 2,
                "stance_label": ["against", "favor"],
                "prob_favor": [0.05, 0.90],
                "prob_neutral": [0.10, 0.05],
                "prob_against": [0.85, 0.05],
                "dominant_emotion": ["anger", "joy"],
                "antagonism": [0.85, 0.05],
            }
        )
        controversy = compute_controversy_metrics(
            prob_favor=[0.05, 0.90],
            prob_neutral=[0.10, 0.05],
            prob_against=[0.85, 0.05],
            antagonism=[0.85, 0.05],
            prior_k=0.0,
        )
        artifacts = PipelineArtifacts(
            config=TopicPipelineConfig(),
            total_tweets=2,
            clustered_tweets=2,
            outlier_tweets=0,
            num_topics=1,
            coherence_report=CoherenceReport(
                coherence_type="c_v",
                average_coherence=0.51,
                per_topic_coherence={0: 0.51},
                topic_diversity=1.0,
                evaluated_topic_count=1,
            ),
            topic_summaries=[
                TopicSummary(
                    topic_id=0,
                    rank=1,
                    label="Abu Dhabi 2021: fans call result robbery",
                    keywords=["hamilton", "verstappen", "robbed"],
                    tweet_count=2,
                    share_pct=100.0,
                    coherence=0.51,
                    dominant_race="Abu Dhabi GP",
                    peak_period="2021-12",
                    technical_category="Safety & Race Control",
                    engagement_category="Title & Driver Rivalry",
                    representative_tweets=["Lewis robbed! #AbuDhabiGP"],
                    controversy=controversy,
                    dominant_emotion="anger",
                )
            ],
            processed_df=df,
        )

        payload = artifacts.to_screen_payload()
        self.assertTrue(payload["insights"])
        self.assertEqual(payload["insights"][0]["kind"], "flashpoint")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = artifacts.save(tmp_dir)
            replay_lines = (out_dir / "replay_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(replay_lines), 2)
            ev0 = json.loads(replay_lines[0])
            ev1 = json.loads(replay_lines[1])
            self.assertLess(ev0["unix_ms"], ev1["unix_ms"])
            self.assertEqual(ev0["event_id"], 0)
            self.assertEqual(ev0["text"], "Max won the title! #AbuDhabiGP")

    def test_codebase_comment_and_docstring_style(self) -> None:
        root = Path(__file__).resolve().parent.parent
        py_files = sorted((root / "f1_topics").glob("*.py")) + sorted((root / "tests").glob("*.py"))
        for py_path in py_files:
            source = py_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node, clean=False)
                    if doc is not None:
                        self.assertLessEqual(
                            len(doc.splitlines()),
                            2,
                            f"{py_path}: docstring exceeds 2 lines: {doc!r}",
                        )

            with py_path.open("rb") as fh:
                comment_lines = sorted(
                    {
                        tok.start[0]
                        for tok in tokenize.tokenize(fh.readline)
                        if tok.type == tokenize.COMMENT
                    }
                )
            for a, b in zip(comment_lines, comment_lines[1:]):
                self.assertNotEqual(
                    b,
                    a + 1,
                    f"{py_path}: consecutive comment lines at {a} and {b}",
                )

        rs_files = sorted((root / "backend" / "src").glob("*.rs"))
        self.assertTrue(rs_files)
        for rs_path in rs_files:
            lines = rs_path.read_text(encoding="utf-8").splitlines()
            prev_plain = False
            doc_run = 0
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("///"):
                    doc_run += 1
                    prev_plain = False
                    self.assertLessEqual(doc_run, 2, f"{rs_path}:{idx}: doc comment exceeds 2 lines")
                elif stripped.startswith("//"):
                    doc_run = 0
                    self.assertFalse(prev_plain, f"{rs_path}:{idx}: consecutive // comment lines")
                    prev_plain = True
                else:
                    doc_run = 0
                    prev_plain = False


if __name__ == "__main__":
    unittest.main()
