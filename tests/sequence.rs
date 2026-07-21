//! Filesystem-level tests for sequence detection and rescanning. The naming
//! rules themselves are unit-tested against synthetic listings in
//! `playback::sequence`; what needs real files is the directory scan, and the
//! live rescan that lets you leave the viewer open beside a running render.

use std::path::{Path, PathBuf};

use imgvwr::playback::{Sequence, SequenceError, SlotState};

/// A scratch directory that removes itself. Named from a counter plus the
/// process id so concurrently-running tests cannot collide.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("imgvwr_seq_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).expect("write temp file");
        p
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

fn present(seq: &Sequence) -> Vec<i64> {
    (seq.first()..=seq.last())
        .filter(|f| seq.has_file(*f))
        .collect()
}

#[test]
fn detects_a_sequence_on_disk_and_ignores_its_neighbours() {
    let dir = TempDir::new("detect");
    for n in 1..=4 {
        dir.write(&format!("shot_{n:04}.png"), b"x");
    }
    // Neighbours that must not be swept in: a different stem, a different
    // extension, and a name whose stem does not end in digits.
    dir.write("shot_0002.jpg", b"x");
    dir.write("other_0002.png", b"x");
    dir.write("shot_4k.png", b"x");

    let seq = Sequence::detect(&dir.path().join("shot_0002.png")).expect("should detect");
    assert_eq!(present(&seq), vec![1, 2, 3, 4]);
    assert_eq!(seq.dir(), dir.path());
    assert_eq!(seq.display_name(), "shot_####.png");
    assert_eq!(
        seq.path(3).unwrap().file_name().unwrap(),
        "shot_0003.png",
        "paths point at the real files"
    );
}

#[test]
fn a_file_with_no_frame_number_is_not_a_sequence() {
    let dir = TempDir::new("noframe");
    let p = dir.write("foobar_16k.png", b"x");
    assert_eq!(
        Sequence::detect(&p).err(),
        Some(SequenceError::NoFrameNumber)
    );
}

/// The point of rescanning: a render writing frames while you watch extends the
/// timeline live, and a frame that was half-written when we first tried it gets
/// picked up once it is rewritten.
#[test]
fn rescan_extends_the_timeline_and_retries_rewritten_frames() {
    let dir = TempDir::new("rescan");
    for n in 1..=3 {
        dir.write(&format!("f{n:04}.exr"), b"xx");
    }
    let mut seq = Sequence::detect(&dir.path().join("f0001.exr")).expect("should detect");
    assert_eq!(present(&seq), vec![1, 2, 3]);

    // Nothing changed on disk -> nothing to report.
    assert!(
        !seq.rescan().changed,
        "an unchanged directory is not a change"
    );

    // Frame 2 decoded as garbage — a half-written file.
    seq.set_state(2, SlotState::Corrupt);
    assert!(
        !seq.rescan().changed,
        "a corrupt frame stays corrupt until it changes"
    );
    assert_eq!(seq.state(2), SlotState::Corrupt);

    // The render finishes flushing frame 2 and writes two more frames.
    dir.write("f0002.exr", b"xxxxxx");
    dir.write("f0004.exr", b"xx");
    dir.write("f0005.exr", b"xx");
    assert!(seq.rescan().changed, "new frames are a change");
    assert_eq!(present(&seq), vec![1, 2, 3, 4, 5]);
    assert_eq!(
        seq.state(2),
        SlotState::Unknown,
        "a rewritten frame gets another chance"
    );
    assert_eq!(seq.last(), 5, "the timeline grew");
}

/// A frame that disappears becomes a hole rather than shortening the timeline —
/// the slot is still real, and the cache bar still has something to draw red.
#[test]
fn rescan_turns_a_deleted_frame_into_a_hole() {
    let dir = TempDir::new("deleted");
    for n in 1..=3 {
        dir.write(&format!("f{n:04}.png"), b"x");
    }
    let mut seq = Sequence::detect(&dir.path().join("f0001.png")).expect("should detect");
    seq.set_state(2, SlotState::Cached);

    std::fs::remove_file(dir.path().join("f0002.png")).unwrap();
    assert!(seq.rescan().changed);
    assert_eq!(seq.state(2), SlotState::Missing);
    assert!(!seq.has_file(2));
    assert_eq!((seq.first(), seq.last()), (1, 3), "the span is unchanged");
    assert_eq!(
        seq.next_playable(2, 1, false),
        Some(3),
        "playback steps past it"
    );
}

/// Frames written *before* the opened one extend the timeline backwards.
#[test]
fn rescan_extends_backwards() {
    let dir = TempDir::new("backwards");
    for n in 10..=12 {
        dir.write(&format!("f{n:04}.png"), b"x");
    }
    let mut seq = Sequence::detect(&dir.path().join("f0010.png")).expect("should detect");
    seq.set_state(11, SlotState::Cached);
    assert_eq!(seq.first(), 10);

    dir.write("f0008.png", b"x");
    assert!(seq.rescan().changed);
    assert_eq!((seq.first(), seq.last()), (8, 12));
    assert!(seq.has_file(8));
    assert_eq!(
        seq.state(11),
        SlotState::Cached,
        "known slots survive the re-base"
    );
    assert_eq!(seq.state(9), SlotState::Missing, "the new gap is a hole");
}
