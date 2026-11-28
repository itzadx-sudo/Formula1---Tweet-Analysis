use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Top-level pipeline quality and coverage metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetMetrics {
    pub total_tweets: usize,
    pub clustered_tweets: usize,
    pub outlier_tweets: usize,
    pub clustering_success_pct: f64,
    pub outlier_pct: f64,
    pub num_topics: usize,
    pub coherence_metric: String,
    pub average_coherence: f64,
    pub topic_diversity: f64,
    pub mean_controversy_index: Option<f64>,
}

/// Polarization, emotional heat, and sample-shrunk Controversy Index metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ControversyMetrics {
    pub controversy_index: f64,
    pub controversy_score_100: f64,
    pub polarization: f64,
    pub opinionated_share: f64,
    pub mean_antagonism: f64,
    pub favor_pct: f64,
    pub against_pct: f64,
    pub neutral_pct: f64,
    pub net_stance: f64,
}

/// Consolidated topic record produced by BERTopic and Gemini naming.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TopicSummary {
    pub topic_id: i64,
    pub rank: usize,
    pub label: String,
    pub keywords: Vec<String>,
    pub tweet_count: usize,
    pub share_pct: f64,
    pub coherence: Option<f64>,
    pub dominant_race: Option<String>,
    pub peak_period: Option<String>,
    pub technical_category: Option<String>,
    pub engagement_category: String,
    pub representative_tweets: Vec<String>,
    pub controversy: Option<ControversyMetrics>,
    pub dominant_emotion: Option<String>,
    pub emotion_distribution: HashMap<String, f64>,
}

/// Aggregated Driver, Team, or Race Hub summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HubSummary {
    pub hub_type: String,
    pub name: String,
    pub tweet_count: usize,
    pub share_pct: f64,
    pub controversy: ControversyMetrics,
    pub dominant_emotion: String,
    pub emotion_distribution: HashMap<String, f64>,
    pub top_topics: Vec<HubTopicRef>,
    pub representative_tweets: Vec<String>,
}

/// Topic reference embedded inside a Hub summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HubTopicRef {
    pub topic_id: i64,
    pub label: String,
    pub tweet_count: usize,
}

/// Collection of Driver, Team, and Race hubs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HubsBundle {
    pub drivers: Vec<HubSummary>,
    pub teams: Vec<HubSummary>,
    pub races: Vec<HubSummary>,
}

/// Computed insight card surfaced in the All Data + Insights mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InsightCard {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub summary: String,
    pub metric_value: Option<f64>,
    pub related_topic_id: Option<i64>,
    pub related_hub: Option<String>,
}

/// Leaderboard entry for the most controversial topics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ControversyLeaderItem {
    pub topic_id: i64,
    pub label: String,
    pub tweet_count: usize,
    pub dominant_emotion: Option<String>,
    pub controversy_index: f64,
    pub controversy_score_100: f64,
    pub polarization: f64,
    pub opinionated_share: f64,
    pub mean_antagonism: f64,
    pub favor_pct: f64,
    pub against_pct: f64,
    pub neutral_pct: f64,
    pub net_stance: f64,
}

/// Single time-bucket count inside a topic timeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TemporalPoint {
    pub period: String,
    pub count: usize,
}

/// Per-topic temporal volume series.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TemporalSeries {
    pub topic_id: i64,
    pub label: String,
    pub points: Vec<TemporalPoint>,
}

/// Complete analytics payload exported by `f1_topics` for Mode 1.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScreenPayload {
    pub metrics: DatasetMetrics,
    pub insights: Vec<InsightCard>,
    pub topics: Vec<TopicSummary>,
    pub controversy_leaderboard: Vec<ControversyLeaderItem>,
    pub hubs: HubsBundle,
    pub race_breakdown: HashMap<String, usize>,
    pub technical_breakdown: HashMap<String, usize>,
    pub engagement_breakdown: HashMap<String, usize>,
    pub temporal_series: Vec<TemporalSeries>,
}

/// Single chronological tweet event used in Mode 2 live replay.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplayPostEvent {
    pub event_id: usize,
    pub timestamp: String,
    pub unix_ms: i64,
    pub text: String,
    pub race: Option<String>,
    pub drivers: Vec<String>,
    pub teams: Vec<String>,
    pub topic_id: i64,
    pub topic_label: String,
    pub stance_label: String,
    pub prob_favor: f64,
    pub prob_neutral: f64,
    pub prob_against: f64,
    pub dominant_emotion: String,
    pub antagonism: f64,
}

/// Race weekend metadata returned by the replay race catalog endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRaceSummary {
    pub race: String,
    pub tweet_count: usize,
    pub start_timestamp: String,
    pub end_timestamp: String,
    pub start_unix_ms: i64,
    pub end_unix_ms: i64,
    pub peak_controversy_100: f64,
}

/// Active topic counter inside a rolling replay window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTopicCounter {
    pub topic_id: i64,
    pub label: String,
    pub count: usize,
}

/// Active driver pulse inside a rolling replay window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveDriverPulse {
    pub driver: String,
    pub count: usize,
    pub net_stance: f64,
    pub controversy_100: f64,
}

/// Rolling-window statistics emitted on every replay frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayWindowStats {
    pub window_tweet_count: usize,
    pub live_controversy_100: f64,
    pub polarization: f64,
    pub favor_pct: f64,
    pub against_pct: f64,
    pub neutral_pct: f64,
    pub dominant_emotion: String,
    pub posts_per_min: f64,
    pub spike_alert: bool,
    pub active_topics: Vec<LiveTopicCounter>,
    pub active_drivers: Vec<LiveDriverPulse>,
}

/// Single SSE or scrubber frame emitted during live replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub frame_index: usize,
    pub cursor_unix_ms: i64,
    pub cursor_timestamp: String,
    pub progress_pct: f64,
    pub speed: f64,
    pub paused: bool,
    pub cumulative_count: usize,
    pub total_events: usize,
    pub new_posts: Vec<ReplayPostEvent>,
    pub window_stats: ReplayWindowStats,
}

/// Mode discovery descriptor exposed at `GET /api/v1/modes`.
#[derive(Debug, Clone, Serialize)]
pub struct ModesResponse {
    pub modes: Vec<InterfaceModeDescriptor>,
    pub data_loaded: bool,
    pub total_topics: usize,
    pub total_replay_events: usize,
    pub available_races: Vec<String>,
}

/// Metadata describing one of the two supported backend interface modes.
#[derive(Debug, Clone, Serialize)]
pub struct InterfaceModeDescriptor {
    pub id: String,
    pub label: String,
    pub description: String,
    pub endpoints: Vec<String>,
}

/// Query parameters for `GET /api/v1/topics`.
#[derive(Debug, Deserialize)]
pub struct TopicsQuery {
    pub race: Option<String>,
    pub category: Option<String>,
    pub q: Option<String>,
    pub sort: Option<String>,
    pub limit: Option<usize>,
}

/// Query parameters for `GET /api/v1/hubs`.
#[derive(Debug, Deserialize)]
pub struct HubsQuery {
    pub kind: Option<String>,
    pub sort: Option<String>,
    pub limit: Option<usize>,
}

/// Query parameters for `GET /api/v1/insights`.
#[derive(Debug, Deserialize)]
pub struct InsightsQuery {
    pub kind: Option<String>,
}

/// Query parameters for `GET /api/v1/posts`.
#[derive(Debug, Deserialize)]
pub struct PostsQuery {
    pub race: Option<String>,
    pub topic_id: Option<i64>,
    pub driver: Option<String>,
    pub team: Option<String>,
    pub stance: Option<String>,
    pub emotion: Option<String>,
    pub q: Option<String>,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

/// Query parameters for stateless `GET /api/v1/replay/stream`.
#[derive(Debug, Deserialize)]
pub struct ReplayStreamQuery {
    pub race: Option<String>,
    pub topic_id: Option<i64>,
    pub from_unix_ms: Option<i64>,
    pub to_unix_ms: Option<i64>,
    pub speed: Option<f64>,
    pub tick_ms: Option<u64>,
    pub window_secs: Option<u64>,
}

/// Query parameters for stateless `GET /api/v1/replay/frames`.
#[derive(Debug, Deserialize)]
pub struct ReplayFramesQuery {
    pub race: Option<String>,
    pub topic_id: Option<i64>,
    pub from_unix_ms: Option<i64>,
    pub to_unix_ms: Option<i64>,
    pub buckets: Option<usize>,
    pub window_secs: Option<u64>,
}

/// Request body for `POST /api/v1/replay/sessions`.
#[derive(Debug, Deserialize)]
pub struct CreateReplaySessionRequest {
    pub race: Option<String>,
    pub topic_id: Option<i64>,
    pub from_unix_ms: Option<i64>,
    pub to_unix_ms: Option<i64>,
    pub speed: Option<f64>,
    pub tick_ms: Option<u64>,
    pub window_secs: Option<u64>,
    pub start_paused: Option<bool>,
}

/// Response returned after creating a controllable replay session.
#[derive(Debug, Serialize)]
pub struct CreateReplaySessionResponse {
    pub session_id: Uuid,
    pub total_events: usize,
    pub start_unix_ms: i64,
    pub end_unix_ms: i64,
    pub stream_url: String,
    pub control_url: String,
}

/// Playback commands accepted by `POST /api/v1/replay/sessions/:id/control`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayControlAction {
    Pause,
    Resume,
    Seek,
    SetSpeed,
    Stop,
}

/// Request body for controlling an active replay session.
#[derive(Debug, Deserialize)]
pub struct ReplayControlRequest {
    pub action: ReplayControlAction,
    pub seek_unix_ms: Option<i64>,
    pub speed: Option<f64>,
}

/// Acknowledgment payload returned after a replay control command.
#[derive(Debug, Serialize)]
pub struct ReplayControlResponse {
    pub session_id: Uuid,
    pub paused: bool,
    pub speed: f64,
    pub stopped: bool,
    pub seek_to: Option<i64>,
}
