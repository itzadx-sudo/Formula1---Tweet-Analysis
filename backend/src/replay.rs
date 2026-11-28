use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, watch, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::insights::compute_live_controversy;
use crate::models::{
    LiveDriverPulse, LiveTopicCounter, ReplayControlAction, ReplayControlRequest,
    ReplayControlResponse, ReplayFrame, ReplayPostEvent, ReplayWindowStats,
};

/// Shared playback control state for an active SSE session.
#[derive(Debug, Clone)]
pub struct ReplayControlState {
    pub paused: bool,
    pub speed: f64,
    pub seek_to: Option<i64>,
    pub stopped: bool,
}

/// Stored metadata and watch channel for an interactive SSE replay session.
#[derive(Debug, Clone)]
pub struct ReplaySessionEntry {
    pub id: Uuid,
    pub events: Vec<ReplayPostEvent>,
    pub start_unix_ms: i64,
    pub end_unix_ms: i64,
    pub tick_ms: u64,
    pub window_secs: u64,
    pub tx: watch::Sender<ReplayControlState>,
}

/// Thread-safe registry of active live-replay sessions.
#[derive(Debug, Default)]
pub struct ReplayManager {
    sessions: RwLock<HashMap<Uuid, ReplaySessionEntry>>,
}

impl ReplayManager {
    /// Register a new controllable replay session over a sorted event slice.
    pub async fn create_session(
        &self,
        events: Vec<ReplayPostEvent>,
        speed: f64,
        tick_ms: u64,
        window_secs: u64,
        start_paused: bool,
    ) -> ReplaySessionEntry {
        let id = Uuid::new_v4();
        let start_unix_ms = events.first().map(|e| e.unix_ms).unwrap_or(0);
        let end_unix_ms = events.last().map(|e| e.unix_ms).unwrap_or(0);
        let state = ReplayControlState {
            paused: start_paused,
            speed: speed.clamp(0.25, 86_400.0),
            seek_to: None,
            stopped: false,
        };
        let (tx, _) = watch::channel(state);
        let entry = ReplaySessionEntry {
            id,
            events,
            start_unix_ms,
            end_unix_ms,
            tick_ms: tick_ms.clamp(100, 5_000),
            window_secs: window_secs.clamp(5, 86_400),
            tx,
        };
        self.sessions.write().await.insert(id, entry.clone());
        entry
    }

    /// Look up an existing replay session by UUID.
    pub async fn get_session(&self, id: Uuid) -> Option<ReplaySessionEntry> {
        self.sessions.read().await.get(&id).cloned()
    }

    /// Apply pause, resume, seek, speed, or stop controls to an active session.
    pub async fn control_session(
        &self,
        id: Uuid,
        req: ReplayControlRequest,
    ) -> Option<ReplayControlResponse> {
        let entry = self.get_session(id).await?;
        let mut next = entry.tx.borrow().clone();
        match req.action {
            ReplayControlAction::Pause => next.paused = true,
            ReplayControlAction::Resume => next.paused = false,
            ReplayControlAction::Seek => {
                if let Some(target_ms) = req.seek_unix_ms {
                    next.seek_to = Some(target_ms.clamp(entry.start_unix_ms, entry.end_unix_ms));
                }
            }
            ReplayControlAction::SetSpeed => {
                if let Some(spd) = req.speed {
                    next.speed = spd.clamp(0.25, 86_400.0);
                }
            }
            ReplayControlAction::Stop => next.stopped = true,
        }
        let _ = entry.tx.send(next.clone());
        if next.stopped {
            self.sessions.write().await.remove(&id);
        }
        Some(ReplayControlResponse {
            session_id: id,
            paused: next.paused,
            speed: next.speed,
            stopped: next.stopped,
            seek_to: next.seek_to,
        })
    }
}

fn format_unix_iso(unix_ms: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(unix_ms)
        .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".into())
}

/// Build a single replay frame at `cursor_unix_ms` with newly arrived posts and window stats.
pub fn build_replay_frame(
    events: &[ReplayPostEvent],
    frame_index: usize,
    prev_cursor_ms: Option<i64>,
    cursor_unix_ms: i64,
    start_unix_ms: i64,
    end_unix_ms: i64,
    window_secs: u64,
    speed: f64,
    paused: bool,
) -> ReplayFrame {
    let end_idx = events.partition_point(|e| e.unix_ms <= cursor_unix_ms);
    let prev_idx = match prev_cursor_ms {
        Some(p) if p < cursor_unix_ms => events.partition_point(|e| e.unix_ms <= p),
        Some(_) => end_idx,
        None => events.partition_point(|e| e.unix_ms < start_unix_ms),
    };

    let delta_slice = &events[prev_idx..end_idx];
    let cap_start = delta_slice.len().saturating_sub(50);
    let new_posts = delta_slice[cap_start..].to_vec();

    let win_start_ms = cursor_unix_ms.saturating_sub((window_secs as i64) * 1_000);
    let win_start_idx = events.partition_point(|e| e.unix_ms < win_start_ms);
    let win_slice = &events[win_start_idx..end_idx];
    let win_refs: Vec<&ReplayPostEvent> = win_slice.iter().collect();
    let live_cov = compute_live_controversy(&win_refs, 8.0);

    let mut emo_counts: HashMap<&str, usize> = HashMap::new();
    let mut topic_counts: HashMap<i64, (String, usize)> = HashMap::new();
    let mut driver_buckets: HashMap<String, Vec<&ReplayPostEvent>> = HashMap::new();

    for ev in &win_refs {
        *emo_counts.entry(ev.dominant_emotion.as_str()).or_insert(0) += 1;
        if ev.topic_id != -1 {
            topic_counts
                .entry(ev.topic_id)
                .and_modify(|(_, c)| *c += 1)
                .or_insert_with(|| (ev.topic_label.clone(), 1));
        }
        for drv in &ev.drivers {
            driver_buckets.entry(drv.clone()).or_default().push(ev);
        }
    }

    let dominant_emotion = emo_counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k.to_string())
        .unwrap_or_else(|| "neutral".into());

    let mut active_topics: Vec<LiveTopicCounter> = topic_counts
        .into_iter()
        .map(|(topic_id, (label, count))| LiveTopicCounter {
            topic_id,
            label,
            count,
        })
        .collect();
    active_topics.sort_by(|a, b| b.count.cmp(&a.count));
    active_topics.truncate(5);

    let mut active_drivers: Vec<LiveDriverPulse> = driver_buckets
        .into_iter()
        .map(|(driver, d_refs)| {
            let dc = compute_live_controversy(&d_refs, 5.0);
            LiveDriverPulse {
                driver,
                count: d_refs.len(),
                net_stance: dc.net_stance,
                controversy_100: dc.controversy_score_100,
            }
        })
        .collect();
    active_drivers.sort_by(|a, b| b.count.cmp(&a.count));
    active_drivers.truncate(5);

    let span = (end_unix_ms - start_unix_ms).max(1) as f64;
    let progress_pct = (((cursor_unix_ms - start_unix_ms) as f64 / span) * 100.0).clamp(0.0, 100.0);
    let window_mins = (window_secs as f64 / 60.0).max(0.05);
    let posts_per_min = ((win_refs.len() as f64 / window_mins) * 10.0).round() / 10.0;
    let spike_alert = live_cov.controversy_score_100 >= 65.0 && delta_slice.len() >= 3;

    ReplayFrame {
        frame_index,
        cursor_unix_ms,
        cursor_timestamp: format_unix_iso(cursor_unix_ms),
        progress_pct: (progress_pct * 10.0).round() / 10.0,
        speed,
        paused,
        cumulative_count: end_idx,
        total_events: events.len(),
        new_posts,
        window_stats: ReplayWindowStats {
            window_tweet_count: win_refs.len(),
            live_controversy_100: live_cov.controversy_score_100,
            polarization: live_cov.polarization,
            favor_pct: live_cov.favor_pct,
            against_pct: live_cov.against_pct,
            neutral_pct: live_cov.neutral_pct,
            dominant_emotion,
            posts_per_min,
            spike_alert,
            active_topics,
            active_drivers,
        },
    }
}

/// Precompute evenly spaced timeline scrubber frames across a sorted event slice.
pub fn build_timeline_frames(
    events: &[ReplayPostEvent],
    buckets: usize,
    window_secs: u64,
) -> Vec<ReplayFrame> {
    if events.is_empty() {
        return Vec::new();
    }
    let start_ms = events.first().map(|e| e.unix_ms).unwrap_or(0);
    let end_ms = events.last().map(|e| e.unix_ms).unwrap_or(start_ms);
    let n_buckets = buckets.clamp(1, 240);
    let span = (end_ms - start_ms).max(0);

    let mut frames = Vec::with_capacity(n_buckets);
    let mut prev_cursor: Option<i64> = None;
    for i in 0..n_buckets {
        let cursor = if n_buckets == 1 || span == 0 {
            end_ms
        } else {
            start_ms + (span * (i as i64 + 1)) / (n_buckets as i64)
        };
        frames.push(build_replay_frame(
            events,
            i,
            prev_cursor,
            cursor,
            start_ms,
            end_ms,
            window_secs,
            1.0,
            false,
        ));
        prev_cursor = Some(cursor);
    }
    frames
}

/// Spawn a live Server-Sent Events stream with optional interactive playback controls.
pub fn spawn_sse_replay(
    events: Vec<ReplayPostEvent>,
    start_unix_ms: i64,
    end_unix_ms: i64,
    tick_ms: u64,
    window_secs: u64,
    control_tx: watch::Sender<ReplayControlState>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let mut control_rx = control_tx.subscribe();

    tokio::spawn(async move {
        if events.is_empty() {
            let _ = tx
                .send(Ok(Event::default().event("complete").data("{\"total_events\":0}")))
                .await;
            return;
        }

        let mut cursor = start_unix_ms;
        let mut prev_cursor: Option<i64> = None;
        let mut frame_index: usize = 0;

        loop {
            let mut state = control_rx.borrow().clone();
            if state.stopped {
                break;
            }

            // Apply seek requests immediately even while paused.
            let mut seek_applied = false;
            if let Some(target_ms) = state.seek_to {
                cursor = target_ms.clamp(start_unix_ms, end_unix_ms);
                prev_cursor = Some(cursor.saturating_sub(1));
                seek_applied = true;
                let mut cleared = state.clone();
                cleared.seek_to = None;
                let _ = control_tx.send(cleared);
                state.seek_to = None;
            }

            if !state.paused || seek_applied || frame_index == 0 {
                let frame = build_replay_frame(
                    &events,
                    frame_index,
                    prev_cursor,
                    cursor,
                    start_unix_ms,
                    end_unix_ms,
                    window_secs,
                    state.speed,
                    state.paused,
                );
                if let Ok(json) = serde_json::to_string(&frame) {
                    if tx.send(Ok(Event::default().event("frame").data(json))).await.is_err() {
                        break;
                    }
                }
                frame_index += 1;
                prev_cursor = Some(cursor);

                if cursor >= end_unix_ms && !state.paused {
                    let done = format!("{{\"total_events\":{},\"frames\":{}}}", events.len(), frame_index);
                    let _ = tx.send(Ok(Event::default().event("complete").data(done))).await;
                    break;
                }
            }

            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(tick_ms)) => {
                    let latest = control_rx.borrow().clone();
                    if !latest.paused && !latest.stopped {
                        let advance_ms = ((tick_ms as f64) * latest.speed).round().max(1.0) as i64;
                        cursor = (cursor + advance_ms).min(end_unix_ms);
                    }
                }
                changed = control_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
            }
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_frames_cover_full_range_and_compute_window_stats() {
        let events = vec![
            ReplayPostEvent {
                event_id: 0,
                unix_ms: 1_000,
                topic_id: 0,
                topic_label: "Abu Dhabi 2021: fans call result robbery".into(),
                drivers: vec!["Lewis Hamilton".into()],
                prob_favor: 0.85,
                prob_neutral: 0.10,
                prob_against: 0.05,
                dominant_emotion: "anger".into(),
                antagonism: 0.70,
                ..Default::default()
            },
            ReplayPostEvent {
                event_id: 1,
                unix_ms: 5_000,
                topic_id: 0,
                topic_label: "Abu Dhabi 2021: fans call result robbery".into(),
                drivers: vec!["Max Verstappen".into()],
                prob_favor: 0.05,
                prob_neutral: 0.10,
                prob_against: 0.85,
                dominant_emotion: "anger".into(),
                antagonism: 0.75,
                ..Default::default()
            },
        ];

        let frames = build_timeline_frames(&events, 2, 60);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].cumulative_count, 2);
        assert!(!frames[1].window_stats.active_topics.is_empty());
    }
}
