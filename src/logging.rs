//! Session logging: one timestamped log file per run under `%TEMP%\imgvwr\`,
//! with startup cleanup (see plans/rewrite.md §19).

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use log::LevelFilter;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use time::OffsetDateTime;

const MAX_LOG_FILES: usize = 10;
const MAX_LOG_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// `%TEMP%\imgvwr\`.
pub fn log_dir() -> PathBuf {
    std::env::temp_dir().join("imgvwr")
}

/// Initialise the file + terminal loggers. Creates the log directory, prunes old
/// logs, then opens a fresh `imgvwr_YYYYMMDD_HHMMSS.log` for this session.
pub fn init() -> io::Result<()> {
    let dir = log_dir();
    fs::create_dir_all(&dir)?;
    cleanup(&dir);

    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let stamp = format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        now.year(),
        now.month() as u8,
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
    );
    let path = dir.join(format!("imgvwr_{stamp}.log"));

    let level = if cfg!(debug_assertions) {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    let file = File::create(&path)?;
    // Terminal logger is best-effort (no terminal under some launchers); fall
    // back to file-only if it cannot be constructed.
    let loggers: Vec<Box<dyn simplelog::SharedLogger>> = vec![
        TermLogger::new(level, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
        WriteLogger::new(level, Config::default(), file),
    ];
    let _ = CombinedLogger::init(loggers);

    log::info!("imgvwr logging initialised -> {}", path.display());
    Ok(())
}

/// Delete logs older than 7 days, then cap the directory at 10 files (oldest
/// first). All errors are non-fatal — a failed cleanup must never block startup.
fn cleanup(dir: &Path) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("log cleanup: cannot read {}: {e}", dir.display());
            return;
        }
    };

    let mut logs: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_log = path.extension().and_then(|s| s.to_str()) == Some("log")
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with("imgvwr_"));
        if !is_log {
            continue;
        }
        if let Ok(modified) = entry.metadata().and_then(|m| m.modified()) {
            logs.push((path, modified));
        }
    }

    let now = SystemTime::now();
    logs.retain(|(path, modified)| {
        let too_old = now
            .duration_since(*modified)
            .map(|age| age > MAX_LOG_AGE)
            .unwrap_or(false);
        if too_old {
            if let Err(e) = fs::remove_file(path) {
                log::warn!("log cleanup: cannot delete {}: {e}", path.display());
            }
            false
        } else {
            true
        }
    });

    if logs.len() > MAX_LOG_FILES {
        logs.sort_by_key(|(_, modified)| *modified); // oldest first
        let surplus = logs.len() - MAX_LOG_FILES;
        for (path, _) in logs.iter().take(surplus) {
            if let Err(e) = fs::remove_file(path) {
                log::warn!("log cleanup: cannot delete {}: {e}", path.display());
            }
        }
    }
}
