use std::collections::HashMap;

use crate::models::{ControversyMetrics, InsightCard, ReplayPostEvent, ScreenPayload};

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// Compute bimodal Controversy Index metrics over a slice of replay events.
pub fn compute_live_controversy(events: &[&ReplayPostEvent], prior_k: f64) -> ControversyMetrics {
    let n = events.len();
    if n == 0 {
        return ControversyMetrics::default();
    }

    let nf = n as f64;
    let mut sum_fav = 0.0;
    let mut sum_neu = 0.0;
    let mut sum_agn = 0.0;
    let mut sum_antag = 0.0;

    for ev in events {
        sum_fav += ev.prob_favor;
        sum_neu += ev.prob_neutral;
        sum_agn += ev.prob_against;
        sum_antag += ev.antagonism;
    }

    let mut p_pos = sum_fav / nf;
    let mut p_neu = sum_neu / nf;
    let mut p_neg = sum_agn / nf;
    let total = p_pos + p_neu + p_neg;
    if total > 0.0 {
        p_pos /= total;
        p_neu /= total;
        p_neg /= total;
    }

    let opinionated = (p_pos + p_neg).clamp(0.0, 1.0);
    let polarization = if opinionated > 1e-6 {
        (2.0 * p_pos.min(p_neg)) / (p_pos + p_neg + 1e-9)
    } else {
        0.0
    };
    let mean_antag = (sum_antag / nf).clamp(0.0, 1.0);

    let raw_score = 0.65 * opinionated * polarization + 0.35 * opinionated * mean_antag;
    let shrinkage = nf / (nf + prior_k.max(0.0));
    let controversy = (shrinkage * raw_score).clamp(0.0, 1.0);

    ControversyMetrics {
        controversy_index: round4(controversy),
        controversy_score_100: round1(controversy * 100.0),
        polarization: round4(polarization),
        opinionated_share: round4(opinionated),
        mean_antagonism: round4(mean_antag),
        favor_pct: round1(p_pos * 100.0),
        against_pct: round1(p_neg * 100.0),
        neutral_pct: round1(p_neu * 100.0),
        net_stance: round4(p_pos - p_neg),
    }
}

/// Derive structured overview insights in Rust if the loaded payload omitted them.
pub fn ensure_insights(payload: &mut ScreenPayload, events: &[ReplayPostEvent]) {
    if !payload.insights.is_empty() {
        return;
    }

    let mut cards: Vec<InsightCard> = Vec::new();

    if let Some(flash) = payload
        .topics
        .iter()
        .filter_map(|t| t.controversy.as_ref().map(|c| (t, c)))
        .max_by(|a, b| a.1.controversy_score_100.total_cmp(&b.1.controversy_score_100))
    {
        let (topic, c) = flash;
        let dom_emo = topic.dominant_emotion.as_deref().unwrap_or("mixed");
        cards.push(InsightCard {
            id: "flashpoint".into(),
            kind: "flashpoint".into(),
            title: topic.label.clone(),
            summary: format!(
                "Highest controversy ({:.1}/100): {:.0}% favor vs {:.0}% against across {} tweets (emotion: {}).",
                c.controversy_score_100, c.favor_pct, c.against_pct, topic.tweet_count, dom_emo
            ),
            metric_value: Some(c.controversy_score_100),
            related_topic_id: Some(topic.topic_id),
            related_hub: None,
        });
    }

    if let Some(calm) = payload
        .topics
        .iter()
        .filter_map(|t| t.controversy.as_ref().map(|c| (t, c)))
        .filter(|(_, c)| c.controversy_index < 0.25)
        .max_by_key(|(t, _)| t.tweet_count)
        .or_else(|| {
            payload
                .topics
                .iter()
                .take(10)
                .filter_map(|t| t.controversy.as_ref().map(|c| (t, c)))
                .min_by(|a, b| a.1.controversy_score_100.total_cmp(&b.1.controversy_score_100))
        })
    {
        let (topic, c) = calm;
        cards.push(InsightCard {
            id: "consensus".into(),
            kind: "consensus".into(),
            title: topic.label.clone(),
            summary: format!(
                "Broadest consensus ({:.1}/100 controversy) across {} tweets ({:.0}% positive / {:.0}% neutral).",
                c.controversy_score_100, topic.tweet_count, c.favor_pct, c.neutral_pct
            ),
            metric_value: Some(c.controversy_score_100),
            related_topic_id: Some(topic.topic_id),
            related_hub: None,
        });
    }

    if let Some(driver) = payload.hubs.drivers.iter().max_by(|a, b| {
        let sa = a.controversy.polarization * (1.0 + a.tweet_count as f64).ln();
        let sb = b.controversy.polarization * (1.0 + b.tweet_count as f64).ln();
        sa.total_cmp(&sb)
    }) {
        cards.push(InsightCard {
            id: "driver_split".into(),
            kind: "driver_split".into(),
            title: format!("{}: most divided driver hub", driver.name),
            summary: format!(
                "{} mentions split {:.0}% favor / {:.0}% against (controversy {:.1}/100).",
                driver.tweet_count,
                driver.controversy.favor_pct,
                driver.controversy.against_pct,
                driver.controversy.controversy_score_100
            ),
            metric_value: Some(driver.controversy.controversy_score_100),
            related_topic_id: None,
            related_hub: Some(driver.name.clone()),
        });
    }

    if let Some(race) = payload
        .hubs
        .races
        .iter()
        .max_by(|a, b| a.controversy.controversy_score_100.total_cmp(&b.controversy.controversy_score_100))
    {
        cards.push(InsightCard {
            id: "race_heat".into(),
            kind: "race_heat".into(),
            title: format!("{}: most heated race weekend", race.name),
            summary: format!(
                "Controversy {:.1}/100 across {} tweets (dominant emotion: {}).",
                race.controversy.controversy_score_100, race.tweet_count, race.dominant_emotion
            ),
            metric_value: Some(race.controversy.controversy_score_100),
            related_topic_id: None,
            related_hub: Some(race.name.clone()),
        });
    }

    let mut peak: Option<(String, i64, String, usize)> = None;
    for series in &payload.temporal_series {
        for pt in &series.points {
            if peak.as_ref().map_or(true, |p| pt.count > p.3) {
                peak = Some((series.label.clone(), series.topic_id, pt.period.clone(), pt.count));
            }
        }
    }
    if let Some((label, tid, period, count)) = peak {
        if count > 0 {
            cards.push(InsightCard {
                id: "temporal_spike".into(),
                kind: "temporal_spike".into(),
                title: format!("Peak surge in {}: {}", period, label),
                summary: format!("Single-period peak of {} tweets occurred during {}.", count, period),
                metric_value: Some(count as f64),
                related_topic_id: Some(tid),
                related_hub: None,
            });
        }
    }

    if payload.metrics.num_topics > 0 {
        cards.push(InsightCard {
            id: "model_health".into(),
            kind: "model_health".into(),
            title: format!("{} consolidated F1 topics", payload.metrics.num_topics),
            summary: format!(
                "{:.1}% clustered | C_v coherence {:.3} | diversity {:.3}.",
                payload.metrics.clustering_success_pct,
                payload.metrics.average_coherence,
                payload.metrics.topic_diversity
            ),
            metric_value: Some(payload.metrics.average_coherence),
            related_topic_id: None,
            related_hub: None,
        });
    } else if !events.is_empty() {
        let mut emo_counts: HashMap<&str, usize> = HashMap::new();
        for ev in events {
            *emo_counts.entry(ev.dominant_emotion.as_str()).or_insert(0) += 1;
        }
        let top_emo = emo_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k)
            .unwrap_or("neutral");
        cards.push(InsightCard {
            id: "replay_volume".into(),
            kind: "model_health".into(),
            title: format!("{} replay events ready", events.len()),
            summary: format!("Dominant emotion across loaded stream: {}.", top_emo),
            metric_value: Some(events.len() as f64),
            related_topic_id: None,
            related_hub: None,
        });
    }

    payload.insights = cards;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event(favor: f64, neutral: f64, against: f64, antag: f64) -> ReplayPostEvent {
        ReplayPostEvent {
            prob_favor: favor,
            prob_neutral: neutral,
            prob_against: against,
            antagonism: antag,
            ..Default::default()
        }
    }

    #[test]
    fn controversy_separates_split_fight_from_consensus() {
        let mut split_owned = Vec::new();
        for i in 0..100 {
            if i % 2 == 0 {
                split_owned.push(sample_event(0.85, 0.10, 0.05, 0.80));
            } else {
                split_owned.push(sample_event(0.05, 0.10, 0.85, 0.80));
            }
        }
        let split_refs: Vec<&ReplayPostEvent> = split_owned.iter().collect();
        let split_metrics = compute_live_controversy(&split_refs, 10.0);
        assert!(split_metrics.controversy_index > 0.65);

        let calm_owned = vec![sample_event(0.90, 0.08, 0.02, 0.03); 100];
        let calm_refs: Vec<&ReplayPostEvent> = calm_owned.iter().collect();
        let calm_metrics = compute_live_controversy(&calm_refs, 10.0);
        assert!(calm_metrics.controversy_index < 0.10);
    }
}
