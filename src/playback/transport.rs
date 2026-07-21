//! The transport: the playhead, the wall clock that drives it, and the achieved
//! frame-rate measurement. See plans/image_sequence_playback.md §7.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::playback::cache::FrameCache;
use crate::playback::sequence::Sequence;

/// How often the directory is re-scanned while in playback mode, so a render
/// writing frames beside you extends the timeline live. One `read_dir`.
pub const RESCAN_INTERVAL: Duration = Duration::from_secs(2);

/// How long to wait before re-checking a frame that has not decoded yet. The
/// decode landing wakes the loop anyway; this is only a backstop so a stall can
/// never sit silently, and so the fps readout keeps falling while it lasts.
const STALL_RECHECK: Duration = Duration::from_millis(100);

/// How often the diagnostic heartbeat is logged while playing.
const LOG_INTERVAL: Duration = Duration::from_secs(5);

/// Advances kept for the achieved-rate readout — about a second of playback, so
/// the number is responsive without flickering.
const RATE_WINDOW: usize = 24;

/// How far short of the target the achieved rate may fall before the readout
/// turns red. Normal jitter must not flicker it.
const RATE_TOLERANCE: f32 = 0.03;

/// Frame rates offered in Settings.
pub const FRAME_RATES: [f32; 7] = [23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 60.0];

/// What the clock wants to happen this instant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tick {
    /// Nothing due yet.
    Idle,
    /// Display this frame now.
    Show(i64),
    /// The next frame is due but has not decoded yet. Playback waits for it —
    /// it never skips content — so this is what running slow looks like.
    Stalled,
}

/// A sequence being played: which frame is up, when the next one is due, and
/// how fast it is actually going.
pub struct Playback {
    pub seq: Sequence,
    pub cache: FrameCache,
    /// The playhead, as a real frame number.
    frame: i64,
    playing: bool,
    /// Frames per second the transport aims for.
    target_fps: f32,
    /// Wall-clock deadline for the next frame.
    next_frame_at: Instant,
    /// When recent frames were displayed, for the achieved-rate readout.
    recent: VecDeque<Instant>,
    /// True while waiting on a frame that has not decoded yet.
    stalled: bool,
    /// A fixed backstop instant to re-check a stall at. Fixed, not "now + a
    /// bit": the event loop only requests a redraw once a `WaitUntil` deadline
    /// has actually passed, so a deadline that moves every time it is asked for
    /// would never be reached.
    stall_recheck_at: Instant,
    next_rescan: Instant,
    next_log: Instant,
    /// False when the frames are too large to ring, so each one is a full
    /// texture re-upload. Playback still works; it is simply slower.
    pub ringed: bool,
    /// The frame actually on screen. It lags the playhead whenever a seek lands
    /// on a frame that has not decoded yet — the screen holds the old frame, and
    /// catches up the moment the new one arrives.
    pub shown: Option<i64>,
}

impl Playback {
    pub fn new(seq: Sequence, cache: FrameCache, frame: i64, target_fps: f32) -> Self {
        let now = Instant::now();
        Self {
            seq,
            cache,
            frame,
            playing: false,
            target_fps: target_fps.max(1.0),
            next_frame_at: now,
            recent: VecDeque::with_capacity(RATE_WINDOW),
            stalled: false,
            stall_recheck_at: now,
            next_rescan: now + RESCAN_INTERVAL,
            next_log: now + LOG_INTERVAL,
            ringed: false,
            shown: None,
        }
    }

    pub fn frame(&self) -> i64 {
        self.frame
    }

    pub fn playing(&self) -> bool {
        self.playing
    }

    pub fn target_fps(&self) -> f32 {
        self.target_fps
    }

    pub fn set_target_fps(&mut self, fps: f32) {
        self.target_fps = fps.max(1.0);
        self.recent.clear();
    }

    fn period(&self) -> Duration {
        Duration::from_secs_f32(1.0 / self.target_fps)
    }

    /// Start or stop the clock. Resuming holds the current frame for a full
    /// period rather than flipping immediately.
    pub fn set_playing(&mut self, playing: bool) {
        if self.playing == playing {
            return;
        }
        self.playing = playing;
        self.recent.clear();
        self.stalled = false;
        if playing {
            self.next_frame_at = Instant::now() + self.period();
        }
    }

    /// Jump the playhead (a click or drag on the timeline, or a frame step).
    /// Scrubbing while playing keeps playing from the new position.
    pub fn seek(&mut self, frame: i64) {
        let frame = frame.clamp(self.seq.first(), self.seq.last());
        if frame == self.frame {
            return;
        }
        self.frame = frame;
        self.recent.clear();
        self.stalled = false;
        if self.playing {
            self.next_frame_at = Instant::now() + self.period();
        }
    }

    /// Step one displayable frame in `dir` (±1), skipping holes and wrapping.
    pub fn step(&mut self, dir: i64) {
        let from = self.frame + dir.signum();
        if let Some(f) = self.seq.next_playable(from, dir.signum(), true) {
            self.seek(f);
        }
    }

    /// When the loop next needs to wake for playback: the frame deadline, or a
    /// backstop recheck while stalled. `None` when paused.
    pub fn deadline(&self) -> Option<Instant> {
        if !self.playing {
            return None;
        }
        Some(if self.stalled {
            self.stall_recheck_at
        } else {
            self.next_frame_at
        })
    }

    /// Advance the clock. Called once per rendered frame, from one place only.
    pub fn tick(&mut self, now: Instant) -> Tick {
        if !self.playing || now < self.next_frame_at {
            return Tick::Idle;
        }
        let Some(next) = self.seq.next_playable(self.frame + 1, 1, true) else {
            // Every slot is a hole. Nothing to play; hold where we are.
            self.playing = false;
            log::info!("playback stopped: no playable frames");
            return Tick::Idle;
        };
        if next == self.frame {
            // Only one playable frame so far — a render may still be writing the
            // rest. Hold it and keep checking at the target rate.
            self.next_frame_at = now + self.period();
            return Tick::Idle;
        }
        if !self.cache.contains(next) {
            // Never drop a frame: wait for it. The visible consequence is that
            // playback runs slow, which is what the red fps readout is for.
            // `next_frame_at` stays in the past, so the moment the frame lands
            // (the decode wakes the loop) it goes straight up.
            self.stalled = true;
            self.stall_recheck_at = now + STALL_RECHECK;
            return Tick::Stalled;
        }
        self.stalled = false;
        self.frame = next;
        // Never accumulate debt. Advancing from the *missed* deadline would make
        // playback sprint through frames to catch up after a stall, which both
        // looks wrong and contradicts "play every frame".
        let ideal = self.next_frame_at + self.period();
        self.next_frame_at = if ideal > now {
            ideal
        } else {
            now + self.period()
        };
        self.mark_shown(now);
        Tick::Show(next)
    }

    /// Record that a frame reached the screen, for the achieved-rate readout.
    pub fn mark_shown(&mut self, now: Instant) {
        if self.recent.len() >= RATE_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(now);
    }

    /// Achieved frames per second, a rolling average of recent advances.
    /// `None` until there is enough history to mean anything.
    pub fn achieved_fps(&self) -> Option<f32> {
        if self.recent.len() < 2 {
            return None;
        }
        let span = (*self.recent.back()?)
            .saturating_duration_since(*self.recent.front()?)
            .as_secs_f32();
        (span > 0.0).then(|| (self.recent.len() - 1) as f32 / span)
    }

    /// True when playback is meaningfully short of its target — what turns the
    /// fps readout red. A stall counts immediately; there is no point waiting
    /// for the rolling average to notice a frame that is not there.
    pub fn behind(&self) -> bool {
        if !self.playing {
            return false;
        }
        if self.stalled {
            return true;
        }
        self.achieved_fps()
            .is_some_and(|fps| fps < self.target_fps * (1.0 - RATE_TOLERANCE))
    }

    pub fn stalled(&self) -> bool {
        self.stalled
    }

    /// True about twice a rescan interval, for the diagnostic heartbeat. §5.3
    /// asks for the cache arithmetic to be *logged* rather than announced on
    /// screen — the cache bar and the fps readout are the user-facing signal.
    pub fn should_log(&mut self, now: Instant) -> bool {
        if !self.playing || now < self.next_log {
            return false;
        }
        self.next_log = now + LOG_INTERVAL;
        true
    }

    /// Re-scan the directory if the timer has come round. Returns the outcome so
    /// the caller can let refreshed frames be decoded again.
    pub fn maybe_rescan(
        &mut self,
        now: Instant,
    ) -> Option<crate::playback::sequence::RescanOutcome> {
        if now < self.next_rescan {
            return None;
        }
        self.next_rescan = now + RESCAN_INTERVAL;
        let outcome = self.seq.rescan();
        outcome.changed.then_some(outcome)
    }
}

/// The default target rate for this machine: **25 in PAL territories, 24
/// everywhere else**. A two-way split, because that is as much precision as the
/// locale actually gives us, and 24 is the right global default for the
/// film/VFX work imgvwr is used for. The Settings dropdown is the real answer
/// for anyone who needs 30 or 48.
pub fn default_target_fps() -> f32 {
    if user_region().is_some_and(|r| is_pal_region(&r)) {
        25.0
    } else {
        24.0
    }
}

/// The user's region as a two-letter code, from the OS locale (e.g. `en-GB` →
/// `GB`).
#[cfg(windows)]
fn user_region() -> Option<String> {
    use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;
    let mut buf = [0u16; 85]; // LOCALE_NAME_MAX_LENGTH
                              // SAFETY: `buf` is a writable u16 array of the length passed.
    let n = unsafe { GetUserDefaultLocaleName(buf.as_mut_ptr(), buf.len() as i32) };
    if n <= 1 {
        return None;
    }
    let name = String::from_utf16_lossy(&buf[..(n - 1) as usize]);
    name.rsplit('-')
        .find(|part| part.len() == 2 && part.chars().all(|c| c.is_ascii_alphabetic()))
        .map(|r| r.to_ascii_uppercase())
}

#[cfg(not(windows))]
fn user_region() -> Option<String> {
    None
}

/// The main 50 Hz (PAL / SECAM) territories. Deliberately a list of the common
/// ones rather than an exhaustive table — the split is a *default*, and anyone
/// it gets wrong changes it once in Settings.
fn is_pal_region(region: &str) -> bool {
    const PAL: &[&str] = &[
        // Western & Northern Europe
        "GB", "IE", "FR", "DE", "NL", "BE", "LU", "AT", "CH", "IT", "ES", "PT", "DK", "SE", "NO",
        "FI", "IS", // Central & Eastern Europe
        "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "SI", "RS", "BA", "MK", "AL", "ME", "EE",
        "LV", "LT", "UA", "BY", "RU", "MD", "CY", "MT", "TR",
        // Africa, the Middle East and South Asia
        "ZA", "NG", "KE", "GH", "TZ", "UG", "ZM", "ZW", "MA", "DZ", "TN", "EG", "IL", "SA", "AE",
        "QA", "KW", "JO", "LB", "IQ", "IR", "PK", "IN", "BD", "LK", "NP",
        // East & South-East Asia, Oceania
        "CN", "HK", "MO", "SG", "MY", "ID", "TH", "VN", "AU", "NZ",
        // South America (PAL-N / PAL-M territories that run at 25 or 50 Hz)
        "AR", "UY", "PY",
    ];
    PAL.contains(&region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A player over `present` (one flag per slot, frame numbers from 1) with
    /// every present frame already cached, running at 10 fps for round numbers.
    fn player(present: &[bool]) -> Playback {
        let seq = Sequence::for_test(1, present);
        let mut cache = FrameCache::new(1 << 30, 0, Arc::new(|| {}));
        for (i, &p) in present.iter().enumerate() {
            if p {
                cache.mark_resident_for_test(1 + i as i64);
            }
        }
        Playback::new(seq, cache, 1, 10.0)
    }

    /// Holes consume no time: the playhead moves straight past a gap in one
    /// frame period rather than sitting on it. A 50-frame hole must not look
    /// like a two-second freeze.
    #[test]
    fn holes_consume_no_time() {
        let mut pb = player(&[true, false, false, false, true]);
        pb.set_playing(true);
        let t0 = Instant::now();
        assert_eq!(pb.tick(t0), Tick::Idle, "not due yet");
        // One period later, the playhead is on frame 5 — not frame 2.
        let t1 = t0 + Duration::from_millis(101);
        assert_eq!(pb.tick(t1), Tick::Show(5));
        assert_eq!(pb.frame(), 5);
    }

    /// Never drop a frame: when the next one has not decoded, playback waits for
    /// it rather than skipping ahead.
    #[test]
    fn an_undecoded_frame_stalls_rather_than_being_skipped() {
        let mut pb = player(&[true, true, true]);
        // Frame 2 is not resident after all.
        pb.cache = FrameCache::new(1 << 30, 0, Arc::new(|| {}));
        pb.cache.mark_resident_for_test(1);
        pb.cache.mark_resident_for_test(3);
        pb.set_playing(true);

        let t = Instant::now() + Duration::from_millis(200);
        assert_eq!(pb.tick(t), Tick::Stalled);
        assert_eq!(pb.frame(), 1, "the playhead does not move past the gap");
        assert!(pb.behind(), "a stall reads as running short immediately");

        // Once it lands, playback picks up exactly where it stopped.
        pb.cache.mark_resident_for_test(2);
        assert_eq!(pb.tick(t), Tick::Show(2));
    }

    /// A stall must not be repaid: playback resumes at the target rate rather
    /// than sprinting through frames to catch up.
    #[test]
    fn a_stall_does_not_accumulate_debt() {
        let mut pb = player(&[true, true, true, true]);
        pb.set_playing(true);
        let t0 = Instant::now();
        // A full second late on a 10 fps clock — ten frames of "debt".
        let late = t0 + Duration::from_millis(1000);
        assert_eq!(pb.tick(late), Tick::Show(2));
        // The very next instant must NOT produce another frame.
        assert_eq!(pb.tick(late), Tick::Idle, "playback sprinted to catch up");
        assert_eq!(pb.tick(late + Duration::from_millis(99)), Tick::Idle);
        assert_eq!(pb.tick(late + Duration::from_millis(101)), Tick::Show(3));
    }

    /// On time, the deadline advances from the deadline — so the clock does not
    /// drift later by however long each frame took to arrive.
    #[test]
    fn an_on_time_clock_does_not_drift() {
        let mut pb = player(&[true, true, true, true, true]);
        pb.set_playing(true);
        let t0 = Instant::now();
        let mut shown = Vec::new();
        for i in 1..=4 {
            // Each tick arrives 2 ms after its deadline.
            let t = t0 + Duration::from_millis(100 * i + 2);
            if let Tick::Show(f) = pb.tick(t) {
                shown.push(f);
            }
        }
        assert_eq!(shown, vec![2, 3, 4, 5], "one frame per period, no drift");
    }

    /// Playback loops by default.
    #[test]
    fn playback_loops() {
        let mut pb = player(&[true, true]);
        pb.set_playing(true);
        let t0 = Instant::now();
        assert_eq!(pb.tick(t0 + Duration::from_millis(101)), Tick::Show(2));
        assert_eq!(pb.tick(t0 + Duration::from_millis(202)), Tick::Show(1));
    }

    /// Scrubbing while playing keeps playing from the new position.
    #[test]
    fn seeking_while_playing_keeps_playing() {
        let mut pb = player(&[true, true, true, true, true]);
        pb.set_playing(true);
        pb.seek(4);
        assert!(pb.playing());
        assert_eq!(pb.frame(), 4);
        let t = Instant::now() + Duration::from_millis(200);
        assert_eq!(pb.tick(t), Tick::Show(5));
    }

    /// Stepping skips holes in both directions and wraps at the ends.
    #[test]
    fn stepping_skips_holes() {
        let mut pb = player(&[true, false, false, true]);
        pb.step(1);
        assert_eq!(pb.frame(), 4);
        pb.step(1);
        assert_eq!(pb.frame(), 1, "wraps at the end");
        pb.step(-1);
        assert_eq!(pb.frame(), 4, "and at the start");
    }

    /// A paused player asks the event loop for nothing, so it can park on Wait
    /// instead of idling at the display rate.
    #[test]
    fn a_paused_player_schedules_nothing() {
        let mut pb = player(&[true, true]);
        assert!(pb.deadline().is_none());
        assert!(!pb.behind());
        pb.set_playing(true);
        assert!(pb.deadline().is_some());
    }

    /// The achieved rate tracks real advances, and turns red only when it falls
    /// meaningfully short — normal jitter must not flicker it.
    #[test]
    fn the_achieved_rate_tolerates_jitter_but_not_a_shortfall() {
        let mut pb = player(&[true, true]);
        pb.set_playing(true);
        assert_eq!(pb.achieved_fps(), None, "no history yet");

        let t0 = Instant::now();
        // Exactly on the 10 fps target, give or take a millisecond.
        for i in 0..12 {
            pb.mark_shown(t0 + Duration::from_micros(100_000 * i + (i % 3) * 500));
        }
        let fps = pb.achieved_fps().expect("history");
        assert!((fps - 10.0).abs() < 0.5, "achieved {fps}");
        assert!(!pb.behind(), "jitter must not turn the readout red");

        // Now genuinely half rate.
        pb.recent.clear();
        for i in 0..12 {
            pb.mark_shown(t0 + Duration::from_millis(200 * i));
        }
        assert!(pb.behind(), "half the target rate should read as short");
    }

    #[test]
    fn pal_split_is_a_two_way_default() {
        assert!(is_pal_region("GB"));
        assert!(is_pal_region("AU"));
        assert!(is_pal_region("DE"));
        assert!(!is_pal_region("US"), "NTSC territory");
        assert!(!is_pal_region("JP"));
        assert!(!is_pal_region("CA"));
        assert!(!is_pal_region("XX"));
        let fps = default_target_fps();
        assert!(fps == 24.0 || fps == 25.0, "unexpected default {fps}");
    }

    #[test]
    fn frame_rates_include_the_defaults() {
        assert!(FRAME_RATES.contains(&24.0));
        assert!(FRAME_RATES.contains(&25.0));
        assert!(FRAME_RATES.windows(2).all(|w| w[0] < w[1]), "sorted");
    }
}
