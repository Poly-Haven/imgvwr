//! Lightweight "is there a newer release?" check.
//!
//! Queries the GitHub REST API for the repository's latest release and compares
//! its tag with the running version. Networking goes through **WinHTTP** (already
//! available via `windows-sys`) so no HTTP/TLS crate is pulled in and the OS
//! certificate store is used. The check is best-effort: any failure (offline,
//! rate-limited, parse error) returns `None` and the app simply shows no update
//! link. The caller runs this on a background thread and throttles it to at most
//! once per day (see `App::maybe_check_for_update`).

/// The repository the build was published from (straight from `Cargo.toml`, so
/// the check can never drift from the manifest).
const REPO_URL: &str = env!("CARGO_PKG_REPOSITORY");

/// A release found on GitHub: its tag (e.g. `"v1.2.0"`) and its web page URL.
#[derive(Debug, Clone)]
pub struct LatestRelease {
    pub tag: String,
    pub url: String,
}

/// `owner/repo` extracted from the repository URL, or `None` if it isn't a
/// recognisable github.com URL.
fn repo_slug() -> Option<String> {
    let slug = REPO_URL
        .strip_prefix("https://github.com/")
        .or_else(|| REPO_URL.strip_prefix("http://github.com/"))?
        .trim_end_matches('/')
        .trim_end_matches(".git");
    if slug.is_empty() {
        None
    } else {
        Some(slug.to_string())
    }
}

/// Parse a `major.minor.patch` version (tolerating a leading `v` and any
/// `-pre`/`+build` suffix). Missing minor/patch default to 0.
fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let s = s.trim();
    let s = s
        .strip_prefix('v')
        .or_else(|| s.strip_prefix('V'))
        .unwrap_or(s);
    let mut it = s.split(['.', '-', '+', ' ']);
    let major = it.next()?.parse().ok()?;
    let minor = it.next().unwrap_or("0").parse().unwrap_or(0);
    let patch = it.next().unwrap_or("0").parse().unwrap_or(0);
    Some((major, minor, patch))
}

/// True if `remote` is a strictly newer semantic version than `local`. A version
/// that fails to parse is treated as "not newer" (never nags on a bad tag).
pub fn is_newer(remote: &str, local: &str) -> bool {
    match (parse_version(remote), parse_version(local)) {
        (Some(r), Some(l)) => r > l,
        _ => false,
    }
}

/// The running version (`CARGO_PKG_VERSION`).
pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Current wall-clock time as unix seconds (UTC), for throttling the daily check.
pub fn unix_now() -> i64 {
    time::OffsetDateTime::now_utc().unix_timestamp()
}

/// The web page for a release `tag` in this repository (matches GitHub's own
/// `html_url`, so the cached check can rebuild it without storing a second field).
pub fn release_url(tag: &str) -> String {
    format!("{REPO_URL}/releases/tag/{tag}")
}

/// Fetch the latest release from GitHub, or `None` on any error. Blocking — call
/// on a background thread.
pub fn fetch_latest_release() -> Option<LatestRelease> {
    let slug = repo_slug()?;
    let path = format!("/repos/{slug}/releases/latest");
    let body = http_get("api.github.com", &path)?;
    parse_release(&body)
}

/// Pull `tag_name` / `html_url` out of a GitHub release JSON payload.
fn parse_release(body: &str) -> Option<LatestRelease> {
    #[derive(serde::Deserialize)]
    struct Release {
        #[serde(default)]
        tag_name: String,
        #[serde(default)]
        html_url: String,
    }
    let r: Release = serde_json::from_str(body).ok()?;
    if r.tag_name.is_empty() {
        return None;
    }
    let url = if r.html_url.is_empty() {
        format!("{REPO_URL}/releases/latest")
    } else {
        r.html_url
    };
    Some(LatestRelease {
        tag: r.tag_name,
        url,
    })
}

/// HTTPS GET via WinHTTP; returns the response body as a UTF-8 string.
#[cfg(windows)]
fn http_get(host: &str, path: &str) -> Option<String> {
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Networking::WinHttp::{
        WinHttpCloseHandle, WinHttpConnect, WinHttpOpen, WinHttpOpenRequest,
        WinHttpQueryDataAvailable, WinHttpReadData, WinHttpReceiveResponse, WinHttpSendRequest,
        WinHttpSetTimeouts, WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_FLAG_SECURE,
    };

    /// Null-terminated UTF-16, for the wide-string WinHTTP APIs.
    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    // RAII close for the three HINTERNET handles (so every early return frees).
    struct Handle(*mut core::ffi::c_void);
    impl Drop for Handle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { WinHttpCloseHandle(self.0) };
            }
        }
    }

    unsafe {
        let agent = wide(concat!("imgvwr-update-check/", env!("CARGO_PKG_VERSION")));
        let session = Handle(WinHttpOpen(
            agent.as_ptr(),
            WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
            null(),
            null(),
            0,
        ));
        if session.0.is_null() {
            return None;
        }
        // Don't let a stalled network keep the thread alive: 5 s per phase.
        WinHttpSetTimeouts(session.0, 5000, 5000, 5000, 5000);

        let hostw = wide(host);
        let conn = Handle(WinHttpConnect(session.0, hostw.as_ptr(), 443, 0));
        if conn.0.is_null() {
            return None;
        }

        let pathw = wide(path);
        let verb = wide("GET");
        let req = Handle(WinHttpOpenRequest(
            conn.0,
            verb.as_ptr(),
            pathw.as_ptr(),
            null(),
            null(),
            null(),
            WINHTTP_FLAG_SECURE,
        ));
        if req.0.is_null() {
            return None;
        }

        // GitHub serves v3 JSON by default; the required User-Agent is the
        // WinHttpOpen agent above. Ask explicitly for the JSON media type anyway.
        let headers = wide("Accept: application/vnd.github+json\r\n");
        if WinHttpSendRequest(
            req.0,
            headers.as_ptr(),
            u32::MAX, // -1: headers is null-terminated, length computed for us
            null(),
            0,
            0,
            0,
        ) == 0
        {
            return None;
        }
        if WinHttpReceiveResponse(req.0, null_mut()) == 0 {
            return None;
        }

        let mut body: Vec<u8> = Vec::new();
        loop {
            let mut avail: u32 = 0;
            if WinHttpQueryDataAvailable(req.0, &mut avail) == 0 {
                return None;
            }
            if avail == 0 {
                break;
            }
            // Cap the total so a bogus/huge response can't exhaust memory.
            if body.len() + avail as usize > 4 * 1024 * 1024 {
                return None;
            }
            let mut chunk = vec![0u8; avail as usize];
            let mut read: u32 = 0;
            if WinHttpReadData(
                req.0,
                chunk.as_mut_ptr() as *mut core::ffi::c_void,
                avail,
                &mut read,
            ) == 0
            {
                return None;
            }
            if read == 0 {
                break;
            }
            chunk.truncate(read as usize);
            body.extend_from_slice(&chunk);
        }
        String::from_utf8(body).ok()
    }
}

#[cfg(not(windows))]
fn http_get(_host: &str, _path: &str) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_versions_compare() {
        assert!(is_newer("v1.2.0", "1.1.0"));
        assert!(is_newer("1.1.1", "1.1.0"));
        assert!(is_newer("2.0.0", "1.9.9"));
        assert!(!is_newer("1.1.0", "1.1.0"));
        assert!(!is_newer("1.0.0", "1.1.0"));
        // Pre-release / build suffixes parse to the base version.
        assert!(!is_newer("v1.1.0-rc1", "1.1.0"));
        // Garbage never nags.
        assert!(!is_newer("not-a-version", "1.1.0"));
    }

    #[test]
    fn parses_release_json() {
        let json = r#"{"tag_name":"v1.3.0","html_url":"https://example.com/r/v1.3.0","other":1}"#;
        let r = parse_release(json).unwrap();
        assert_eq!(r.tag, "v1.3.0");
        assert_eq!(r.url, "https://example.com/r/v1.3.0");
    }

    #[test]
    fn release_without_url_falls_back() {
        let json = r#"{"tag_name":"v1.3.0"}"#;
        let r = parse_release(json).unwrap();
        assert!(r.url.ends_with("/releases/latest"));
    }

    #[test]
    fn empty_tag_is_none() {
        assert!(parse_release(r#"{"tag_name":""}"#).is_none());
        assert!(parse_release("not json").is_none());
    }
}
