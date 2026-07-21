//! End-to-end tests for the frame cache: real files, real decode threads.
//!
//! The scheduling arithmetic is unit-tested in `playback::cache`; what needs a
//! filesystem and a thread pool is that frames actually land, that the pool
//! throttles instead of thrashing when the budget is tiny, and that a scrub
//! abandons the old window instead of queueing work for it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use imgvwr::playback::{FrameCache, Sequence, SlotState};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("imgvwr_cache_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A scratch directory of `n` decodable frames, each a copy of a committed
/// fixture (4x4 RGBA8, so one frame is 64 bytes decoded).
fn scratch_dir(tag: &str, n: i64) -> TempDir {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_rgb.png");
    let dir = TempDir::new(tag);
    for f in 1..=n {
        std::fs::copy(&src, dir.path().join(format!("f{f:04}.png"))).expect("copy fixture");
    }
    dir
}

fn detect(dir: &TempDir) -> Sequence {
    Sequence::detect(&dir.path().join("f0001.png")).expect("should detect")
}

/// A scratch directory plus its detected sequence.
fn scratch_sequence(tag: &str, n: i64) -> (TempDir, Sequence) {
    let dir = scratch_dir(tag, n);
    let seq = detect(&dir);
    (dir, seq)
}

fn new_cache(budget_bytes: u64, workers: usize) -> FrameCache {
    FrameCache::new(budget_bytes, workers, Arc::new(|| {}))
}

/// Drive schedule/poll until `done` holds or the deadline passes. Returns
/// whether it converged — a failure here is a hang, not a wrong answer, so the
/// timeout is what makes it a test rather than a stall.
fn pump(
    cache: &mut FrameCache,
    seq: &mut Sequence,
    playhead: i64,
    looping: bool,
    mut done: impl FnMut(&FrameCache) -> bool,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        cache.poll(seq);
        cache.schedule(seq, playhead, looping);
        if done(cache) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

#[test]
fn frames_decode_and_land_in_the_cache() {
    let (_dir, mut seq) = scratch_sequence("land", 6);
    let mut cache = new_cache(64 * 1024 * 1024, 4);

    assert!(
        pump(&mut cache, &mut seq, 1, false, |c| c.resident_frames() == 6),
        "all six frames should decode"
    );
    for f in 1..=6 {
        assert!(cache.contains(f), "frame {f} missing from the cache");
        assert_eq!(seq.state(f), SlotState::Cached, "frame {f} state");
        let img = cache.get(f).expect("frame present");
        assert_eq!((img.width, img.height), (4, 4));
    }
}

/// The floor: a budget far too small to hold the sequence still plays. The pool
/// fills the window it *can* hold and then stops, rather than churning frames in
/// and out — and it never holds more than the capacity.
#[test]
fn a_tiny_budget_throttles_instead_of_thrashing() {
    let (_dir, mut seq) = scratch_sequence("tiny_budget", 40);
    // One byte of budget: the floor (displayed frame + one in flight) applies.
    let mut cache = new_cache(1, 4);

    assert!(
        pump(&mut cache, &mut seq, 1, true, |c| c.resident_frames()
            >= c.capacity_frames()),
        "the cache should fill to its floor capacity"
    );
    let cap = cache.capacity_frames();
    assert_eq!(
        cap, 2,
        "the floor is the displayed frame plus one in flight"
    );

    // Settle, then confirm it stays put instead of cycling.
    for _ in 0..50 {
        cache.poll(&mut seq);
        cache.schedule(&mut seq, 1, true);
        assert!(
            cache.resident_frames() <= cap,
            "residency {} exceeded the capacity {cap}",
            cache.resident_frames()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(cache.contains(1), "the displayed frame is always held");
}

/// Scrubbing to a distant frame abandons the old window rather than working
/// through it: the frames around the new playhead arrive, and the ones the
/// playhead left are dropped.
#[test]
fn scrubbing_retargets_the_window() {
    let (_dir, mut seq) = scratch_sequence("scrub", 60);
    // Room for ~8 of the 64-byte frames.
    let mut cache = new_cache(8 * 64, 4);

    assert!(
        pump(&mut cache, &mut seq, 1, false, |c| c.contains(1)),
        "the first window should fill"
    );
    let cap = cache.capacity_frames();
    assert_eq!(cap, 8);

    assert!(
        pump(&mut cache, &mut seq, 50, false, |c| c.contains(50)),
        "the new window should fill after the scrub"
    );
    assert!(
        cache.resident_frames() <= cap,
        "the old window was not dropped"
    );
    assert!(
        !cache.contains(1),
        "frames the playhead left should be evicted"
    );
}

/// The cache bar must show RAM residency, so a frame evicted when the playhead
/// moves on has to revert from cached to not-cached — even though the
/// reschedule only touches the working set rather than the whole timeline.
#[test]
fn an_evicted_frame_reverts_to_uncached_on_the_bar() {
    let (_dir, mut seq) = scratch_sequence("evict_state", 60);
    let mut cache = new_cache(8 * 64, 4); // room for ~8 frames

    assert!(
        pump(&mut cache, &mut seq, 1, false, |c| c.contains(1)),
        "the first window should fill"
    );
    assert_eq!(
        seq.state(1),
        SlotState::Cached,
        "frame 1 is cached at the start"
    );

    // Scrub far away; frame 1 leaves the window and is evicted.
    assert!(
        pump(&mut cache, &mut seq, 50, false, |c| c.contains(50)),
        "the new window should fill"
    );
    assert!(!cache.contains(1), "frame 1 was evicted");
    assert_ne!(
        seq.state(1),
        SlotState::Cached,
        "an evicted frame must not still read cached on the bar"
    );
    assert!(!seq.state(1).is_hole(), "and it is not a hole either");
}

/// A frame that cannot be decoded is marked corrupt, is not retried on every
/// reschedule, and does not stop the rest of the sequence caching.
#[test]
fn a_corrupt_frame_is_marked_and_skipped() {
    let dir = scratch_dir("corrupt", 5);
    // A half-written frame: the file exists, but nothing can decode it.
    std::fs::write(dir.path().join("f0003.png"), b"not a png").expect("truncate frame 3");
    let mut seq = detect(&dir);

    let mut cache = new_cache(64 * 1024 * 1024, 4);
    assert!(
        pump(&mut cache, &mut seq, 1, false, |c| c.resident_frames() == 4),
        "the four good frames should decode"
    );
    assert_eq!(seq.state(3), SlotState::Corrupt);
    assert!(!cache.contains(3));
    assert_eq!(
        seq.next_playable(3, 1, false),
        Some(4),
        "playback steps past it"
    );
}
