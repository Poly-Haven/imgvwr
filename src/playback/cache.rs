//! The decoded-frame cache and the worker pool that fills it.
//! See plans/image_sequence_playback.md §5.
//!
//! ## Why this is a priority function and not a queue
//!
//! Workers repeatedly ask *what is the most valuable uncached frame right now?*
//! rather than draining a work queue. Because the answer is recomputed from the
//! playhead rather than accumulated, scrubbing and looping need no special
//! handling at all: move the playhead and the next pull retargets. Dragging
//! across three hundred uncached frames cannot queue three hundred decodes,
//! because nothing was ever queued.
//!
//! ## Why it cannot thrash
//!
//! The cache is a window around the playhead — `lookahead` frames ahead plus a
//! short `tail` behind — and the window is at most `capacity` frames wide. Every
//! resident frame outside it is dropped, and the wish list is truncated so
//! `resident + in flight + wanted` never exceeds `capacity`. So when the window
//! is already full of cached frames the workers simply find nothing to do and
//! wait. A frame in the lookahead is never evicted to make room for one further
//! ahead — that would be decode work thrown away before it was ever displayed.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::image_loader::{load_image, DecodeIntent, ImageData, ReadProgress};
use crate::playback::sequence::{Sequence, SlotState};

/// The cache may never hold fewer than this many frames, whatever the configured
/// budget says. Playback needs the displayed frame plus one in flight; refusing
/// to play because a preference is set low is worse than playing slowly.
pub const MIN_CAPACITY_FRAMES: usize = 2;

/// Share of the cache kept *behind* the playhead so scrubbing back a little is
/// instant. Prior art brackets this tightly and all of it agrees on a small
/// tail: tlRender keeps 0.5 s behind, mrv2 5 s ahead / 0.5 s behind, xStudio
/// nothing at all behind.
const TAIL_FRACTION: f32 = 0.15;

/// The tail is a luxury paid out of slack: it is never allowed to eat into this
/// much lookahead. As the capacity approaches the floor the tail shrinks to
/// nothing, which is what "spend everything forward under pressure" means — a
/// tail you will never scrub into is just a smaller lookahead.
const MIN_LOOKAHEAD_FRAMES: usize = 24;

/// Default share of physical RAM the frame cache may use. Industry convention
/// brackets this: tlRender/DJV, xStudio and Blender all default to a flat 4 GB,
/// mrv2 to half of physical RAM, RV to 1 GB. A percentage is mrv2's model,
/// scaled down.
pub const DEFAULT_CACHE_PERCENT: u32 = 30;

/// A finished decode, tagged with the sequence generation it was issued under.
struct FrameResult {
    gen: u64,
    frame: i64,
    result: Result<ImageData, String>,
}

/// The work the pool is currently willing to do, in priority order. Replaced
/// wholesale on every reschedule — never appended to.
#[derive(Default)]
struct Wish {
    gen: u64,
    jobs: VecDeque<(i64, PathBuf)>,
    /// Frames a worker has taken and not yet reported.
    inflight: HashSet<i64>,
    shutdown: bool,
}

/// Persistent decode threads. They block on the condvar when there is nothing
/// worth doing, so an idle player costs nothing.
struct Pool {
    wish: Arc<(Mutex<Wish>, Condvar)>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(
        workers: usize,
        tx: Sender<FrameResult>,
        wake: Arc<dyn Fn() + Send + Sync>,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        let wish: Arc<(Mutex<Wish>, Condvar)> = Arc::default();
        let mut threads = Vec::with_capacity(workers);
        for i in 0..workers {
            let wish = wish.clone();
            let tx = tx.clone();
            let wake = wake.clone();
            let cancelled = cancelled.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("frame-decode-{i}"))
                .spawn(move || worker_loop(&wish, &tx, &*wake, &cancelled));
            match spawned {
                Ok(h) => threads.push(h),
                Err(e) => log::warn!("could not spawn frame-decode thread {i}: {e}"),
            }
        }
        Self { wish, threads }
    }

    /// Replace the wish list. Anything not yet taken is forgotten, which is how
    /// a scrub abandons work for frames the playhead has already left.
    fn publish(&self, gen: u64, jobs: VecDeque<(i64, PathBuf)>) {
        let (lock, cv) = &*self.wish;
        if let Ok(mut w) = lock.lock() {
            w.gen = gen;
            w.jobs = jobs;
        }
        cv.notify_all();
    }

    /// Frames currently being decoded.
    fn inflight(&self) -> HashSet<i64> {
        let (lock, _) = &*self.wish;
        lock.lock().map(|w| w.inflight.clone()).unwrap_or_default()
    }

    fn inflight_count(&self) -> usize {
        let (lock, _) = &*self.wish;
        lock.lock().map(|w| w.inflight.len()).unwrap_or(0)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let (lock, cv) = &*self.wish;
        if let Ok(mut w) = lock.lock() {
            w.shutdown = true;
            w.jobs.clear();
        }
        cv.notify_all();
        // Deliberately NOT joined. A decode cannot be interrupted, so joining
        // would make Stop wait out whatever frame happens to be in flight —
        // milliseconds for a 2K frame, seconds for a 24k one. Detaching is safe
        // because every worker holds its own `Arc`s: it finishes, finds the
        // channel's receiver gone, and exits. Its late result is dropped, and
        // `cancelled` stops it waking an event loop that may no longer exist.
        self.threads.clear();
    }
}

fn worker_loop(
    wish: &(Mutex<Wish>, Condvar),
    tx: &Sender<FrameResult>,
    wake: &(dyn Fn() + Send + Sync),
    cancelled: &AtomicBool,
) {
    let (lock, cv) = wish;
    loop {
        let job = {
            let Ok(guard) = lock.lock() else { return };
            let Ok(mut guard) = cv.wait_while(guard, |w| w.jobs.is_empty() && !w.shutdown) else {
                return;
            };
            if guard.shutdown {
                return;
            }
            match guard.jobs.pop_front() {
                Some((frame, path)) => {
                    guard.inflight.insert(frame);
                    Some((guard.gen, frame, path))
                }
                None => None,
            }
        };
        let Some((gen, frame, path)) = job else {
            continue;
        };
        // Throughput, not latency: the parallelism is already being spent across
        // frames, so this decode keeps to one thread (see `DecodeIntent`).
        let progress = Arc::new(ReadProgress::default());
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load_image(&path, &progress, DecodeIntent::Throughput)
        })) {
            Ok(Ok(data)) => Ok(data),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Err(_) => Err("decoder panicked".to_string()),
        };
        if let Ok(mut w) = lock.lock() {
            w.inflight.remove(&frame);
        }
        if tx.send(FrameResult { gen, frame, result }).is_err() {
            return;
        }
        // Wake the event loop so the result is collected promptly rather than
        // sitting in the channel until an unrelated event arrives.
        if !cancelled.load(Ordering::Relaxed) {
            wake();
        }
    }
}

/// Decoded frames in RAM, plus the pool filling them.
pub struct FrameCache {
    budget_bytes: u64,
    frames: HashMap<i64, Arc<ImageData>>,
    /// Bytes one decoded frame occupies. Zero until the first one lands — every
    /// frame of a sequence shares dimensions and pixel type, so one is enough.
    frame_bytes: u64,
    pool: Pool,
    rx: Receiver<FrameResult>,
    /// Bumped on entry, exit and rescan, so results for a superseded sequence
    /// are dropped rather than landing in the wrong cache. Same discipline as
    /// `App::load_gen`.
    gen: u64,
    /// Set on shutdown so a worker finishing after the event loop has gone does
    /// not try to wake it.
    cancelled: Arc<AtomicBool>,
    /// Frames whose decode failed at least once, so a permanently-broken frame
    /// is not retried every reschedule.
    failed: HashSet<i64>,
}

impl FrameCache {
    /// `wake` is called from a decode thread when a frame lands, to nudge the
    /// event loop (the app passes an `EventLoopProxy` send).
    pub fn new(budget_bytes: u64, workers: usize, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        log::info!(
            "frame cache: {} MB budget, {workers} decode workers",
            budget_bytes / (1024 * 1024)
        );
        Self {
            budget_bytes,
            frames: HashMap::new(),
            frame_bytes: 0,
            pool: Pool::new(workers, tx, wake, cancelled.clone()),
            rx,
            gen: 0,
            cancelled,
            failed: HashSet::new(),
        }
    }

    /// Abandon everything cached and in flight (a new sequence, or leaving
    /// playback). In-flight results are dropped when they arrive.
    pub fn reset(&mut self) {
        self.gen += 1;
        self.frames.clear();
        self.failed.clear();
        self.frame_bytes = 0;
        self.pool.publish(self.gen, VecDeque::new());
    }

    /// Stop waking the event loop — call before the loop exits.
    pub fn cancel_wakeups(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn set_budget(&mut self, bytes: u64) {
        self.budget_bytes = bytes;
    }

    /// Put an already-decoded frame straight into the cache. Entering playback
    /// uses this for the frame that is already on screen, so pressing `Space`
    /// never re-decodes what the viewer is already showing.
    pub fn seed(&mut self, frame: i64, data: Arc<ImageData>) {
        if self.frame_bytes == 0 {
            self.frame_bytes = frame_byte_size(&data);
        }
        self.frames.insert(frame, data);
    }

    pub fn get(&self, frame: i64) -> Option<Arc<ImageData>> {
        self.frames.get(&frame).cloned()
    }

    /// Borrow a resident frame rather than cloning its `Arc` — for the upload
    /// path, which only reads the pixels.
    pub fn peek(&self, frame: i64) -> Option<&ImageData> {
        self.frames.get(&frame).map(|d| d.as_ref())
    }

    pub fn contains(&self, frame: i64) -> bool {
        self.frames.contains_key(&frame)
    }

    pub fn resident_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.frame_bytes * self.frames.len() as u64
    }

    /// How many frames fit in the budget. Never below [`MIN_CAPACITY_FRAMES`]:
    /// refusing to play because a preference is set low is worse than playing
    /// slowly. Before the first frame lands nothing is known about frame size,
    /// so the floor is all we can promise.
    pub fn capacity_frames(&self) -> usize {
        if self.frame_bytes == 0 {
            return MIN_CAPACITY_FRAMES;
        }
        ((self.budget_bytes / self.frame_bytes) as usize).max(MIN_CAPACITY_FRAMES)
    }

    /// Collect finished decodes into the cache, updating `seq`'s slot states.
    /// Returns how many landed. Never blocks.
    pub fn poll(&mut self, seq: &mut Sequence) -> usize {
        let mut landed = 0;
        while let Ok(msg) = self.rx.try_recv() {
            if msg.gen != self.gen {
                continue;
            }
            match msg.result {
                Ok(data) => {
                    if self.frame_bytes == 0 {
                        self.frame_bytes = frame_byte_size(&data);
                        log::info!(
                            "sequence frame size {} MB -> cache holds {} frames",
                            self.frame_bytes / (1024 * 1024),
                            self.capacity_frames()
                        );
                    }
                    self.frames.insert(msg.frame, Arc::new(data));
                    seq.set_state(msg.frame, SlotState::Cached);
                    landed += 1;
                }
                Err(e) => {
                    log::warn!("frame {} failed to decode: {e}", msg.frame);
                    seq.set_state(msg.frame, SlotState::Corrupt);
                    self.failed.insert(msg.frame);
                }
            }
        }
        landed
    }

    /// A rescan found this frame's file rewritten: let it be tried again.
    pub fn forget_failure(&mut self, frame: i64) {
        self.failed.remove(&frame);
    }

    /// Recompute the cache window around `playhead`: drop everything outside it,
    /// then publish the nearest uncached frames ahead as the new wish list.
    pub fn schedule(&mut self, seq: &mut Sequence, playhead: i64, looping: bool) {
        let capacity = self.capacity_frames();
        let window = Window::new(seq, playhead, looping, capacity);

        // Evict outside the window. This alone bounds residency: the window is
        // at most `capacity` frames wide.
        self.frames.retain(|&f, _| window.rank(f).is_some());

        let inflight = self.pool.inflight();
        // Reflect reality in the slot states, so the cache bar and the transport
        // describe what is actually resident rather than what was wished for.
        for frame in seq.first()..=seq.last() {
            if !seq.has_file(frame) {
                continue;
            }
            let state = if self.frames.contains_key(&frame) {
                SlotState::Cached
            } else if inflight.contains(&frame) {
                SlotState::Pending
            } else if self.failed.contains(&frame) {
                SlotState::Corrupt
            } else {
                SlotState::Unknown
            };
            seq.set_state(frame, state);
        }

        // Room for new work, so `resident + in flight + wanted <= capacity`.
        let taken = self.frames.len() + inflight.len();
        let room = capacity.saturating_sub(taken);
        let mut jobs = VecDeque::with_capacity(room.min(1024));
        if room > 0 {
            for frame in window.ahead() {
                if jobs.len() >= room {
                    break;
                }
                if self.frames.contains_key(&frame)
                    || inflight.contains(&frame)
                    || self.failed.contains(&frame)
                {
                    continue;
                }
                if let Some(path) = seq.path(frame) {
                    jobs.push_back((frame, path));
                }
            }
        }
        self.pool.publish(self.gen, jobs);
    }

    /// Frames currently being decoded (diagnostics and the transport readout).
    pub fn inflight_count(&self) -> usize {
        self.pool.inflight_count()
    }

    /// Test hook: make a frame read as resident without decoding anything, so
    /// the transport's clock can be exercised without a filesystem.
    #[cfg(test)]
    pub(crate) fn mark_resident_for_test(&mut self, frame: i64) {
        use crate::image_loader::{PixelBuffer, CLIP_MAX_NORM};
        self.frames.insert(
            frame,
            Arc::new(ImageData {
                path: PathBuf::from("test"),
                width: 1,
                height: 1,
                channels: 4,
                dtype_name: "test".into(),
                compression: "test".into(),
                pixels: PixelBuffer::U8(vec![0; 4]),
                is_encoded_srgb: true,
                clip_max: CLIP_MAX_NORM,
                animation: None,
                camera: None,
            }),
        );
    }
}

/// The cache window: which frames are worth holding, and in what order they are
/// worth holding them.
struct Window {
    playhead: i64,
    first: i64,
    span: i64,
    looping: bool,
    lookahead: usize,
    tail: usize,
}

impl Window {
    fn new(seq: &Sequence, playhead: i64, looping: bool, capacity: usize) -> Self {
        let tail = tail_len(capacity);
        Self {
            playhead,
            first: seq.first(),
            span: seq.len() as i64,
            looping,
            lookahead: capacity - tail,
            tail,
        }
    }

    /// Priority rank: lower is more valuable to keep, `None` means outside the
    /// window entirely (evict it). Every lookahead frame outranks every tail
    /// frame, so the tail is what gives way first. The window holds at most
    /// `lookahead + tail == capacity` frames, which is what bounds residency.
    fn rank(&self, frame: i64) -> Option<usize> {
        let d = frame - self.playhead;
        if d >= 0 && (d as usize) < self.lookahead {
            return Some(d as usize);
        }
        if d < 0 && (-d as usize) <= self.tail {
            return Some(self.lookahead + (-d) as usize);
        }
        // Looping makes the ends adjacent, so the window wraps too.
        if self.looping && self.span > 0 {
            let fwd = d.rem_euclid(self.span);
            if (fwd as usize) < self.lookahead {
                return Some(fwd as usize);
            }
            let back = (self.span - fwd) as usize;
            if back <= self.tail {
                return Some(self.lookahead + back);
            }
        }
        None
    }

    /// Frames ahead of the playhead in playback order, nearest first: the order
    /// the scheduler wants them decoded in. Never longer than the lookahead, and
    /// never repeats a frame when the sequence is shorter than that.
    fn ahead(&self) -> impl Iterator<Item = i64> + '_ {
        let n = if self.looping {
            self.lookahead.min(self.span.max(0) as usize)
        } else {
            self.lookahead
        };
        (0..n as i64).filter_map(move |d| {
            let f = self.playhead + d;
            if self.looping {
                Some(self.first + (f - self.first).rem_euclid(self.span))
            } else {
                (f < self.first + self.span).then_some(f)
            }
        })
    }
}

/// How many frames of the cache to spend behind the playhead.
///
/// [`TAIL_FRACTION`] of the capacity when there is room for it, shrinking to
/// nothing as the capacity approaches [`MIN_LOOKAHEAD_FRAMES`] — under pressure
/// every frame kept behind is a frame not read ahead.
fn tail_len(capacity: usize) -> usize {
    let want = (capacity as f32 * TAIL_FRACTION) as usize;
    want.min(capacity.saturating_sub(MIN_LOOKAHEAD_FRAMES))
}

/// Decoded size of one frame in bytes.
fn frame_byte_size(data: &ImageData) -> u64 {
    let px = data.width as u64 * data.height as u64 * 4;
    match data.pixels {
        crate::image_loader::PixelBuffer::U8(_) => px,
        crate::image_loader::PixelBuffer::F32(_) => px * 4,
    }
}

/// Total physical RAM, for the percentage-of-memory cache budget. Falls back to
/// 8 GB when the OS will not say.
pub fn total_physical_memory() -> u64 {
    const FALLBACK: u64 = 8 * 1024 * 1024 * 1024;
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `status` is a correctly-sized, zeroed MEMORYSTATUSEX.
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 && status.ullTotalPhys > 0 {
            return status.ullTotalPhys;
        }
    }
    FALLBACK
}

/// Cache budget in bytes for a percentage-of-physical-RAM setting.
pub fn budget_from_percent(percent: u32) -> u64 {
    let total = total_physical_memory();
    total / 100 * u64::from(percent.clamp(1, 90))
}

/// How many decode threads to run.
///
/// Prior art spans a wide range — tlRender uses `cores/2`, xStudio 8 per source,
/// mrv2 a fixed 16, RV a strikingly conservative 1–4. Leave two cores for the
/// render thread and the OS, and clamp: past ~8 the win is likely
/// memory-bandwidth bound rather than core bound.
pub fn default_worker_count() -> usize {
    if let Some(n) = std::env::var("IMGVWR_PLAYBACK_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return n.clamp(1, 64);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .saturating_sub(2)
        .clamp(2, 12)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(playhead: i64, span: i64, looping: bool, capacity: usize) -> Window {
        let tail = tail_len(capacity);
        Window {
            playhead,
            first: 0,
            span,
            looping,
            lookahead: capacity - tail,
            tail,
        }
    }

    /// The tail is a luxury: generous when there is room, gone by the time the
    /// cache holds barely a second of playback.
    #[test]
    fn the_tail_shrinks_to_nothing_under_pressure() {
        assert_eq!(tail_len(1000), 150);
        assert_eq!(tail_len(100), 15);
        // Once the capacity is down near the minimum lookahead, everything goes
        // forward.
        assert_eq!(tail_len(MIN_LOOKAHEAD_FRAMES), 0);
        assert_eq!(tail_len(2), 0);
        assert_eq!(tail_len(0), 0);
    }

    /// Nearest-ahead is most valuable; the tail ranks after every lookahead
    /// frame, so it is what gives way first.
    #[test]
    fn rank_orders_ahead_before_behind() {
        let w = window(50, 1000, false, 100);
        assert_eq!(
            w.rank(50),
            Some(0),
            "the playhead itself is the most valuable"
        );
        assert!(w.rank(51).unwrap() < w.rank(60).unwrap());
        assert!(
            w.rank(85).unwrap() < w.rank(49).unwrap(),
            "ahead beats behind"
        );
        assert_eq!(w.rank(200), None, "beyond the lookahead: evict");
    }

    /// The window wraps at the loop point only when looping. Without it, the
    /// last frame of the sequence is far *ahead* of the first — not one behind —
    /// so it must not sneak into the tail by wrapping.
    #[test]
    fn the_window_wraps_only_when_looping() {
        let straight = window(0, 1000, false, 40);
        assert_eq!(
            straight.rank(999),
            None,
            "the far end is not 'one behind' without a loop"
        );
        let looped = window(0, 1000, true, 40);
        assert!(
            looped.rank(999).is_some(),
            "with a loop it really is one behind"
        );
        assert!(looped.rank(999).unwrap() > looped.rank(1).unwrap());
    }

    /// The tail keeps recently-played frames whether or not the sequence loops —
    /// scrubbing back a little should be instant either way.
    #[test]
    fn the_tail_holds_recently_played_frames() {
        let w = window(50, 1000, false, 100);
        assert!(w.rank(49).is_some(), "one frame back is in the tail");
        assert_eq!(w.rank(30), None, "well beyond the tail: evict");
    }

    /// Scrubbing to a completely different part of the timeline abandons the old
    /// window rather than keeping both.
    #[test]
    fn the_window_moves_with_the_playhead() {
        let a = window(10, 10_000, false, 50);
        let b = window(5_000, 10_000, false, 50);
        assert!(a.rank(20).is_some() && b.rank(20).is_none());
        assert!(b.rank(5_010).is_some() && a.rank(5_010).is_none());
    }

    /// The window never asks for more frames than the capacity, so residency is
    /// bounded by construction rather than by a separate eviction pass.
    #[test]
    fn ahead_never_exceeds_the_capacity() {
        for capacity in [2usize, 5, 24, 100, 1000] {
            let w = window(3, 500, true, capacity);
            let n = w.ahead().count();
            assert!(
                n <= capacity,
                "capacity {capacity} produced {n} lookahead frames"
            );
        }
    }

    /// A sequence shorter than the lookahead is fully covered, once, with no
    /// duplicate frames — otherwise a short loop would schedule the same frames
    /// over and over.
    #[test]
    fn a_short_looping_sequence_is_covered_exactly_once() {
        let w = window(2, 6, true, 100);
        let frames: Vec<i64> = w.ahead().collect();
        let mut sorted = frames.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), frames.len(), "no frame is scheduled twice");
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(frames[0], 2, "starting at the playhead");
    }

    #[test]
    fn the_budget_floor_holds_when_the_preference_is_absurd() {
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let mut cache = FrameCache::new(1, 1, wake);
        assert_eq!(cache.capacity_frames(), MIN_CAPACITY_FRAMES);
        cache.frame_bytes = 64 * 1024 * 1024;
        assert_eq!(
            cache.capacity_frames(),
            MIN_CAPACITY_FRAMES,
            "a one-byte budget still plays"
        );
        cache.set_budget(640 * 1024 * 1024);
        assert_eq!(cache.capacity_frames(), 10);
    }

    #[test]
    fn budget_percent_is_clamped() {
        let total = total_physical_memory();
        assert!(budget_from_percent(0) > 0, "0% would refuse to play");
        assert!(budget_from_percent(100) <= total * 90 / 100);
        assert!(budget_from_percent(30) >= total / 100 * 29);
    }
}
