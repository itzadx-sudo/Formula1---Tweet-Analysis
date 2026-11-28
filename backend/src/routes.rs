use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::{watch, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::error::ApiError;
use crate::models::{
    CreateReplaySessionRequest, CreateReplaySessionResponse, HubsQuery, InsightsQuery,
    InterfaceModeDescriptor, ModesResponse, PostsQuery, ReplayControlRequest,
    ReplayControlResponse, ReplayFramesQuery, ReplayStreamQuery, TopicsQuery,
};
use crate::replay::{
    build_timeline_frames, spawn_sse_replay, ReplayControlState, ReplayManager,
};
use crate::store::AnalyticsStore;

/// Shared application state injected into Axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub store: Arc<RwLock<AnalyticsStore>>,
    pub replay: Arc<ReplayManager>,
}

/// Build the full Axum router exposing Mode 1 (All Data + Insights) and Mode 2 (Live Replay).
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/modes", get(get_modes))
        .route("/api/v1/reload", post(reload_store))
        .route("/api/v1/overview", get(get_overview))
        .route("/api/v1/insights", get(get_insights))
        .route("/api/v1/topics", get(get_topics))
        .route("/api/v1/topics/:id", get(get_topic_by_id))
        .route("/api/v1/hubs", get(get_hubs))
        .route("/api/v1/hubs/:kind/:name", get(get_hub_detail))
        .route("/api/v1/controversy", get(get_controversy_leaderboard))
        .route("/api/v1/posts", get(get_posts))
        .route("/api/v1/replay/races", get(get_replay_races))
        .route("/api/v1/replay/frames", get(get_replay_frames))
        .route("/api/v1/replay/stream", get(stream_replay_stateless))
        .route("/api/v1/replay/sessions", post(create_replay_session))
        .route("/api/v1/replay/sessions/:id/stream", get(stream_replay_session))
        .route("/api/v1/replay/sessions/:id/control", post(control_replay_session))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let store = state.store.read().await;
    Json(json!({
        "status": "ok",
        "data_loaded": store.loaded,
        "data_dir": store.data_dir,
        "topics": store.payload.topics.len(),
        "replay_events": store.events.len()
    }))
}

async fn get_modes(State(state): State<AppState>) -> Json<ModesResponse> {
    let store = state.store.read().await;
    Json(ModesResponse {
        modes: vec![
            InterfaceModeDescriptor {
                id: "all_data".into(),
                label: "All Data + Insights".into(),
                description: "Full dataset metrics, insight cards, topic explorer, driver/team/race hubs, and controversy leaderboard.".into(),
                endpoints: vec![
                    "/api/v1/overview".into(),
                    "/api/v1/insights".into(),
                    "/api/v1/topics".into(),
                    "/api/v1/hubs".into(),
                    "/api/v1/controversy".into(),
                    "/api/v1/posts".into(),
                ],
            },
            InterfaceModeDescriptor {
                id: "live_replay".into(),
                label: "Live Replay".into(),
                description: "Chronological race/timeline replay over SSE with play, pause, seek, speed control, and rolling controversy.".into(),
                endpoints: vec![
                    "/api/v1/replay/races".into(),
                    "/api/v1/replay/frames".into(),
                    "/api/v1/replay/stream".into(),
                    "/api/v1/replay/sessions".into(),
                ],
            },
        ],
        data_loaded: store.loaded,
        total_topics: store.payload.topics.len(),
        total_replay_events: store.events.len(),
        available_races: store.available_races(),
    })
}

async fn reload_store(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let dir = { state.store.read().await.data_dir.clone() };
    let refreshed = AnalyticsStore::load_from_dir(&dir).map_err(ApiError::Internal)?;
    let summary = json!({
        "reloaded": true,
        "data_loaded": refreshed.loaded,
        "topics": refreshed.payload.topics.len(),
        "replay_events": refreshed.events.len()
    });
    *state.store.write().await = refreshed;
    Ok(Json(summary))
}

async fn get_overview(State(state): State<AppState>) -> Json<Value> {
    let store = state.store.read().await;
    Json(json!({
        "metrics": store.payload.metrics,
        "insights": store.payload.insights,
        "controversy_leaderboard": store.payload.controversy_leaderboard,
        "race_breakdown": store.payload.race_breakdown,
        "technical_breakdown": store.payload.technical_breakdown,
        "engagement_breakdown": store.payload.engagement_breakdown,
        "temporal_series": store.payload.temporal_series,
    }))
}

async fn get_insights(
    State(state): State<AppState>,
    Query(query): Query<InsightsQuery>,
) -> Json<Value> {
    let store = state.store.read().await;
    let items: Vec<_> = store
        .payload
        .insights
        .iter()
        .filter(|c| query.kind.as_deref().map_or(true, |k| c.kind.eq_ignore_ascii_case(k)))
        .cloned()
        .collect();
    Json(json!({ "count": items.len(), "insights": items }))
}

async fn get_topics(
    State(state): State<AppState>,
    Query(query): Query<TopicsQuery>,
) -> Json<Value> {
    let store = state.store.read().await;
    let mut items: Vec<_> = store
        .payload
        .topics
        .iter()
        .filter(|t| {
            if let Some(race) = &query.race {
                if !t.dominant_race.as_deref().map_or(false, |r| r.eq_ignore_ascii_case(race)) {
                    return false;
                }
            }
            if let Some(cat) = &query.category {
                let tech_match = t.technical_category.as_deref().map_or(false, |c| c.eq_ignore_ascii_case(cat));
                let eng_match = t.engagement_category.eq_ignore_ascii_case(cat);
                if !tech_match && !eng_match {
                    return false;
                }
            }
            if let Some(q) = &query.q {
                let needle = q.to_lowercase();
                let label_hit = t.label.to_lowercase().contains(&needle);
                let kw_hit = t.keywords.iter().any(|k| k.to_lowercase().contains(&needle));
                if !label_hit && !kw_hit {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect();

    match query.sort.as_deref() {
        Some("controversy") => items.sort_by(|a, b| {
            let ca = a.controversy.as_ref().map(|c| c.controversy_index).unwrap_or(0.0);
            let cb = b.controversy.as_ref().map(|c| c.controversy_index).unwrap_or(0.0);
            cb.total_cmp(&ca)
        }),
        Some("coherence") => items.sort_by(|a, b| {
            b.coherence.unwrap_or(0.0).total_cmp(&a.coherence.unwrap_or(0.0))
        }),
        _ => items.sort_by(|a, b| b.tweet_count.cmp(&a.tweet_count)),
    }

    if let Some(limit) = query.limit {
        items.truncate(limit);
    }
    Json(json!({ "count": items.len(), "topics": items }))
}

async fn get_topic_by_id(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Value>, ApiError> {
    let store = state.store.read().await;
    let topic = store
        .payload
        .topics
        .iter()
        .find(|t| t.topic_id == id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("Topic {id} not found")))?;

    let sample_posts: Vec<_> = store
        .events
        .iter()
        .filter(|e| e.topic_id == id)
        .take(25)
        .cloned()
        .collect();
    Ok(Json(json!({ "topic": topic, "sample_posts": sample_posts })))
}

async fn get_hubs(
    State(state): State<AppState>,
    Query(query): Query<HubsQuery>,
) -> Result<Json<Value>, ApiError> {
    let store = state.store.read().await;
    let sort_by = query.sort.as_deref();
    let limit = query.limit;

    let mut sort_slice = |mut list: Vec<crate::models::HubSummary>| {
        match sort_by {
            Some("controversy") => list.sort_by(|a, b| {
                b.controversy.controversy_index.total_cmp(&a.controversy.controversy_index)
            }),
            Some("net_stance") => list.sort_by(|a, b| {
                b.controversy.net_stance.total_cmp(&a.controversy.net_stance)
            }),
            _ => list.sort_by(|a, b| b.tweet_count.cmp(&a.tweet_count)),
        }
        if let Some(lim) = limit {
            list.truncate(lim);
        }
        list
    };

    match query.kind.as_deref() {
        Some("driver") | Some("drivers") => {
            let list = sort_slice(store.payload.hubs.drivers.clone());
            Ok(Json(json!({ "kind": "drivers", "count": list.len(), "hubs": list })))
        }
        Some("team") | Some("teams") => {
            let list = sort_slice(store.payload.hubs.teams.clone());
            Ok(Json(json!({ "kind": "teams", "count": list.len(), "hubs": list })))
        }
        Some("race") | Some("races") => {
            let list = sort_slice(store.payload.hubs.races.clone());
            Ok(Json(json!({ "kind": "races", "count": list.len(), "hubs": list })))
        }
        Some(other) => Err(ApiError::BadRequest(format!("Unknown hub kind: {other}"))),
        None => Ok(Json(json!({
            "drivers": sort_slice(store.payload.hubs.drivers.clone()),
            "teams": sort_slice(store.payload.hubs.teams.clone()),
            "races": sort_slice(store.payload.hubs.races.clone()),
        }))),
    }
}

async fn get_hub_detail(
    State(state): State<AppState>,
    Path((kind, name)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let store = state.store.read().await;
    let list = match kind.as_str() {
        "driver" | "drivers" => &store.payload.hubs.drivers,
        "team" | "teams" => &store.payload.hubs.teams,
        "race" | "races" => &store.payload.hubs.races,
        _ => return Err(ApiError::BadRequest(format!("Unknown hub kind: {kind}"))),
    };

    let hub = list
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(&name))
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("Hub {kind}/{name} not found")))?;

    let matching_posts: Vec<_> = store
        .events
        .iter()
        .filter(|e| match kind.as_str() {
            "driver" | "drivers" => e.drivers.iter().any(|d| d.eq_ignore_ascii_case(&hub.name)),
            "team" | "teams" => e.teams.iter().any(|t| t.eq_ignore_ascii_case(&hub.name)),
            _ => e.race.as_deref().map_or(false, |r| r.eq_ignore_ascii_case(&hub.name)),
        })
        .take(30)
        .cloned()
        .collect();

    Ok(Json(json!({ "hub": hub, "sample_posts": matching_posts })))
}

async fn get_controversy_leaderboard(State(state): State<AppState>) -> Json<Value> {
    let store = state.store.read().await;
    let mut hottest_drivers = store.payload.hubs.drivers.clone();
    hottest_drivers.sort_by(|a, b| {
        b.controversy.controversy_index.total_cmp(&a.controversy.controversy_index)
    });
    hottest_drivers.truncate(10);

    let mut hottest_races = store.payload.hubs.races.clone();
    hottest_races.sort_by(|a, b| {
        b.controversy.controversy_index.total_cmp(&a.controversy.controversy_index)
    });
    hottest_races.truncate(10);

    Json(json!({
        "topics": store.payload.controversy_leaderboard,
        "drivers": hottest_drivers,
        "races": hottest_races,
    }))
}

async fn get_posts(
    State(state): State<AppState>,
    Query(query): Query<PostsQuery>,
) -> Json<Value> {
    let store = state.store.read().await;
    let needle = query.q.as_ref().map(|s| s.to_lowercase());

    let matched: Vec<_> = store
        .events
        .iter()
        .filter(|ev| {
            if let Some(race) = &query.race {
                if !ev.race.as_deref().map_or(false, |r| r.eq_ignore_ascii_case(race)) {
                    return false;
                }
            }
            if let Some(tid) = query.topic_id {
                if ev.topic_id != tid {
                    return false;
                }
            }
            if let Some(drv) = &query.driver {
                if !ev.drivers.iter().any(|d| d.eq_ignore_ascii_case(drv)) {
                    return false;
                }
            }
            if let Some(tm) = &query.team {
                if !ev.teams.iter().any(|t| t.eq_ignore_ascii_case(tm)) {
                    return false;
                }
            }
            if let Some(st) = &query.stance {
                if !ev.stance_label.eq_ignore_ascii_case(st) {
                    return false;
                }
            }
            if let Some(emo) = &query.emotion {
                if !ev.dominant_emotion.eq_ignore_ascii_case(emo) {
                    return false;
                }
            }
            if let Some(q) = &needle {
                if !ev.text.to_lowercase().contains(q) {
                    return false;
                }
            }
            true
        })
        .collect();

    let total = matched.len();
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(50).min(200);
    let page: Vec<_> = matched.into_iter().skip(offset).take(limit).cloned().collect();

    Json(json!({
        "total": total,
        "offset": offset,
        "limit": limit,
        "posts": page,
    }))
}

async fn get_replay_races(State(state): State<AppState>) -> Json<Value> {
    let store = state.store.read().await;
    let races = store.race_catalog();
    Json(json!({ "count": races.len(), "races": races }))
}

async fn get_replay_frames(
    State(state): State<AppState>,
    Query(query): Query<ReplayFramesQuery>,
) -> Json<Value> {
    let store = state.store.read().await;
    let filtered = store.filter_events(
        query.race.as_deref(),
        query.topic_id,
        query.from_unix_ms,
        query.to_unix_ms,
    );
    let frames = build_timeline_frames(
        &filtered,
        query.buckets.unwrap_or(40),
        query.window_secs.unwrap_or(1800),
    );
    Json(json!({
        "total_events": filtered.len(),
        "frame_count": frames.len(),
        "frames": frames,
    }))
}

async fn stream_replay_stateless(
    State(state): State<AppState>,
    Query(query): Query<ReplayStreamQuery>,
) -> Sse<ReceiverStream<Result<Event, std::convert::Infallible>>> {
    let events = {
        let store = state.store.read().await;
        store.filter_events(
            query.race.as_deref(),
            query.topic_id,
            query.from_unix_ms,
            query.to_unix_ms,
        )
    };
    let start_ms = events.first().map(|e| e.unix_ms).unwrap_or(0);
    let end_ms = events.last().map(|e| e.unix_ms).unwrap_or(0);
    let (tx, _) = watch::channel(ReplayControlState {
        paused: false,
        speed: query.speed.unwrap_or(300.0).clamp(0.25, 86_400.0),
        seek_to: None,
        stopped: false,
    });
    spawn_sse_replay(
        events,
        start_ms,
        end_ms,
        query.tick_ms.unwrap_or(500).clamp(100, 5_000),
        query.window_secs.unwrap_or(1800).clamp(5, 86_400),
        tx,
    )
}

async fn create_replay_session(
    State(state): State<AppState>,
    Json(req): Json<CreateReplaySessionRequest>,
) -> Result<Json<CreateReplaySessionResponse>, ApiError> {
    let events = {
        let store = state.store.read().await;
        store.filter_events(
            req.race.as_deref(),
            req.topic_id,
            req.from_unix_ms,
            req.to_unix_ms,
        )
    };
    if events.is_empty() {
        return Err(ApiError::NotFound("No replay events match the requested filters".into()));
    }

    let entry = state
        .replay
        .create_session(
            events,
            req.speed.unwrap_or(300.0),
            req.tick_ms.unwrap_or(500),
            req.window_secs.unwrap_or(1800),
            req.start_paused.unwrap_or(false),
        )
        .await;

    Ok(Json(CreateReplaySessionResponse {
        session_id: entry.id,
        total_events: entry.events.len(),
        start_unix_ms: entry.start_unix_ms,
        end_unix_ms: entry.end_unix_ms,
        stream_url: format!("/api/v1/replay/sessions/{}/stream", entry.id),
        control_url: format!("/api/v1/replay/sessions/{}/control", entry.id),
    }))
}

async fn stream_replay_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Sse<ReceiverStream<Result<Event, std::convert::Infallible>>>, ApiError> {
    let entry = state
        .replay
        .get_session(id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Replay session {id} not found")))?;

    Ok(spawn_sse_replay(
        entry.events,
        entry.start_unix_ms,
        entry.end_unix_ms,
        entry.tick_ms,
        entry.window_secs,
        entry.tx,
    ))
}

async fn control_replay_session(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<ReplayControlRequest>,
) -> Result<Json<ReplayControlResponse>, ApiError> {
    state
        .replay
        .control_session(id, req)
        .await
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("Replay session {id} not found")))
}
