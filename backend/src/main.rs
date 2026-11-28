mod error;
mod insights;
mod models;
mod replay;
mod routes;
mod store;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use crate::replay::ReplayManager;
use crate::routes::{build_router, AppState};
use crate::store::AnalyticsStore;

fn resolve_data_dir() -> PathBuf {
    if let Ok(raw) = env::var("F1_DATA_DIR") {
        let p = PathBuf::from(raw.trim());
        if p.exists() {
            return p;
        }
    }
    let local = PathBuf::from("./outputs");
    if local.exists() {
        return local;
    }
    let parent = PathBuf::from("../outputs");
    if parent.exists() {
        return parent;
    }
    local
}

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    let _ = dotenvy::from_filename("../.env");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let data_dir = resolve_data_dir();
    let bind_addr = env::var("F1_API_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());

    let store = match AnalyticsStore::load_from_dir(&data_dir) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!("Starting with empty store ({err})");
            AnalyticsStore {
                data_dir: data_dir.clone(),
                loaded: false,
                payload: Default::default(),
                events: Vec::new(),
            }
        }
    };

    tracing::info!(
        "Loaded {} topics and {} replay events from {}",
        store.payload.topics.len(),
        store.events.len(),
        store.data_dir.display()
    );

    let state = AppState {
        store: Arc::new(RwLock::new(store)),
        replay: Arc::new(ReplayManager::default()),
    };

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind API listener");
    tracing::info!("F1 analytics API listening on http://{bind_addr}");

    axum::serve(listener, build_router(state))
        .await
        .expect("Server terminated with error");
}
