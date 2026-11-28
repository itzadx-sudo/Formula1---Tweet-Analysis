use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

use crate::insights::{compute_live_controversy, ensure_insights};
use crate::models::{ReplayPostEvent, ReplayRaceSummary, ScreenPayload};

/// In-memory analytics store backed by `topics_screen_payload.json` and `replay_events.jsonl`.
#[derive(Debug, Clone)]
pub struct AnalyticsStore {
    pub data_dir: PathBuf,
    pub loaded: bool,
    pub payload: ScreenPayload,
    pub events: Vec<ReplayPostEvent>,
}

impl AnalyticsStore {
    /// Load Mode 1 screen payload and Mode 2 chronological replay events from disk.
    pub fn load_from_dir(data_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = data_dir.as_ref().to_path_buf();
        let payload_path = dir.join("topics_screen_payload.json");
        let replay_path = dir.join("replay_events.jsonl");
        let csv_path = dir.join("tweet_topic_assignments.csv");

        let mut loaded = false;
        let mut payload = if payload_path.is_file() {
            let raw = fs::read_to_string(&payload_path)
                .map_err(|e| format!("Failed to read {}: {e}", payload_path.display()))?;
            loaded = true;
            serde_json::from_str::<ScreenPayload>(&raw)
                .map_err(|e| format!("Invalid JSON in {}: {e}", payload_path.display()))?
        } else {
            ScreenPayload::default()
        };

        let mut events = if replay_path.is_file() {
            loaded = true;
            parse_replay_jsonl(&replay_path)?
        } else if csv_path.is_file() {
            loaded = true;
            parse_assignments_csv(&csv_path)?
        } else {
            Vec::new()
        };

        events.sort_by_key(|ev| ev.unix_ms);
        for (idx, ev) in events.iter_mut().enumerate() {
            ev.event_id = idx;
        }

        ensure_insights(&mut payload, &events);

        Ok(Self {
            data_dir: dir,
            loaded,
            payload,
            events,
        })
    }

    /// Return unique sorted race names discovered across topics and replay events.
    pub fn available_races(&self) -> Vec<String> {
        let mut races: Vec<String> = self
            .payload
            .hubs
            .races
            .iter()
            .map(|h| h.name.clone())
            .collect();
        for ev in &self.events {
            if let Some(r) = &ev.race {
                if !races.iter().any(|x| x.eq_ignore_ascii_case(r)) {
                    races.push(r.clone());
                }
            }
        }
        races.sort();
        races
    }

    /// Filter chronological replay events by optional race, topic, and timestamp bounds.
    pub fn filter_events(
        &self,
        race: Option<&str>,
        topic_id: Option<i64>,
        from_unix_ms: Option<i64>,
        to_unix_ms: Option<i64>,
    ) -> Vec<ReplayPostEvent> {
        self.events
            .iter()
            .filter(|ev| {
                if let Some(r) = race {
                    if !ev.race.as_deref().map_or(false, |er| er.eq_ignore_ascii_case(r)) {
                        return false;
                    }
                }
                if let Some(tid) = topic_id {
                    if ev.topic_id != tid {
                        return false;
                    }
                }
                if let Some(lo) = from_unix_ms {
                    if ev.unix_ms < lo {
                        return false;
                    }
                }
                if let Some(hi) = to_unix_ms {
                    if ev.unix_ms > hi {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Summarize each race weekend available for live replay.
    pub fn race_catalog(&self) -> Vec<ReplayRaceSummary> {
        let mut grouped: HashMap<String, Vec<&ReplayPostEvent>> = HashMap::new();
        for ev in &self.events {
            if let Some(race) = &ev.race {
                grouped.entry(race.clone()).or_default().push(ev);
            }
        }

        let mut catalog: Vec<ReplayRaceSummary> = grouped
            .into_iter()
            .filter_map(|(race, slice)| {
                let first = slice.first()?;
                let last = slice.last()?;
                let peak = if slice.len() <= 30 {
                    compute_live_controversy(&slice, 10.0).controversy_score_100
                } else {
                    slice
                        .windows(30)
                        .step_by(10)
                        .map(|w| compute_live_controversy(w, 10.0).controversy_score_100)
                        .fold(0.0_f64, f64::max)
                };
                Some(ReplayRaceSummary {
                    race,
                    tweet_count: slice.len(),
                    start_timestamp: first.timestamp.clone(),
                    end_timestamp: last.timestamp.clone(),
                    start_unix_ms: first.unix_ms,
                    end_unix_ms: last.unix_ms,
                    peak_controversy_100: peak,
                })
            })
            .collect();

        catalog.sort_by(|a, b| b.tweet_count.cmp(&a.tweet_count));
        catalog
    }
}

fn parse_replay_jsonl(path: &Path) -> Result<Vec<ReplayPostEvent>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let mut events = Vec::new();
    for (line_no, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ev = serde_json::from_str::<ReplayPostEvent>(trimmed)
            .map_err(|e| format!("Invalid JSONL at {}:{}: {e}", path.display(), line_no + 1))?;
        events.push(ev);
    }
    Ok(events)
}

fn parse_py_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Vec::new();
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = &trimmed[1..trimmed.len() - 1];
        return inner
            .split(',')
            .map(|part| part.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    vec![trimmed.to_string()]
}

fn parse_timestamp_ms(raw: &str, fallback_ms: i64) -> (i64, String) {
    let s = raw.trim();
    if !s.is_empty() {
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            let utc = dt.with_timezone(&Utc);
            return (utc.timestamp_millis(), utc.to_rfc3339());
        }
        if let Ok(dt) = DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%#z") {
            let utc = dt.with_timezone(&Utc);
            return (utc.timestamp_millis(), utc.to_rfc3339());
        }
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            let utc = ndt.and_utc();
            return (utc.timestamp_millis(), utc.to_rfc3339());
        }
        if let Ok(nd) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            if let Some(ndt) = nd.and_hms_opt(12, 0, 0) {
                let utc = ndt.and_utc();
                return (utc.timestamp_millis(), utc.to_rfc3339());
            }
        }
    }
    let utc = DateTime::<Utc>::from_timestamp_millis(fallback_ms).unwrap_or_else(Utc::now);
    (fallback_ms, utc.to_rfc3339())
}

fn parse_assignments_csv(path: &Path) -> Result<Vec<ReplayPostEvent>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("Failed to read CSV headers: {e}"))?
        .clone();

    let col_idx = |name: &str| headers.iter().position(|h| h == name);
    let idx_disp = col_idx("display_text").or_else(|| col_idx("raw_text"));
    let idx_race = col_idx("inferred_race");
    let idx_drv = col_idx("mentioned_drivers");
    let idx_team = col_idx("mentioned_teams");
    let idx_date = col_idx("date_parsed");
    let idx_topic = col_idx("topic");
    let idx_tlabel = col_idx("topic_label");
    let idx_stance = col_idx("stance_label");
    let idx_fav = col_idx("prob_favor");
    let idx_neu = col_idx("prob_neutral");
    let idx_agn = col_idx("prob_against");
    let idx_emo = col_idx("dominant_emotion");
    let idx_ant = col_idx("antagonism");

    let base_ms: i64 = 1_639_310_400_000;
    let mut events = Vec::new();
    for (row_no, rec_res) in reader.records().enumerate() {
        let rec = rec_res.map_err(|e| format!("CSV parse error on row {row_no}: {e}"))?;
        let get = |idx: Option<usize>| idx.and_then(|i| rec.get(i)).unwrap_or("");
        let (unix_ms, timestamp) = parse_timestamp_ms(get(idx_date), base_ms + (row_no as i64) * 10_000);
        let race_raw = get(idx_race).trim();
        let race = if race_raw.is_empty() { None } else { Some(race_raw.to_string()) };

        events.push(ReplayPostEvent {
            event_id: row_no,
            timestamp,
            unix_ms,
            text: get(idx_disp).to_string(),
            race,
            drivers: parse_py_list(get(idx_drv)),
            teams: parse_py_list(get(idx_team)),
            topic_id: get(idx_topic).parse::<i64>().unwrap_or(-1),
            topic_label: {
                let s = get(idx_tlabel);
                if s.is_empty() { "Outliers / Unclustered".into() } else { s.to_string() }
            },
            stance_label: {
                let s = get(idx_stance);
                if s.is_empty() { "neutral".into() } else { s.to_string() }
            },
            prob_favor: get(idx_fav).parse::<f64>().unwrap_or(0.0),
            prob_neutral: get(idx_neu).parse::<f64>().unwrap_or(1.0),
            prob_against: get(idx_agn).parse::<f64>().unwrap_or(0.0),
            dominant_emotion: {
                let s = get(idx_emo);
                if s.is_empty() { "neutral".into() } else { s.to_string() }
            },
            antagonism: get(idx_ant).parse::<f64>().unwrap_or(0.0),
        });
    }
    Ok(events)
}
