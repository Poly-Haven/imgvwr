//! Frame-sequence detection and the slot model behind the timeline.
//!
//! Pure logic: no GPU, no UI, and the only I/O is one `read_dir` per scan. See
//! plans/image_sequence_playback.md §3.
//!
//! ## What counts as a sequence
//!
//! The filename stem must **end** in a run of decimal digits. `shot_0047.png`
//! is prefix `shot_`, frame 47; `foobar01234.png` is prefix `foobar`, frame 1234
//! — separators are optional, only the trailing digits matter. `foobar_16k.png`
//! ends in `k` and is not a sequence at all.
//!
//! Each frame number then has exactly **one** legal spelling ([`canonical_digits`]):
//! zero-padded to the sequence's width, or written out in full once it overflows
//! that width (so a 4-padded sequence legitimately reaches `10000`). A sibling
//! belongs to the sequence only if its digits are that spelling — which is what
//! keeps `foo1.png` and `foo0001.png` two sequences rather than one, and makes
//! collisions impossible.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Longest trailing digit run still read as a frame number. Borrowed from
/// tlRender/DJV: it stops unix timestamps (10 digits) and long hashes in
/// filenames from being mistaken for frame numbers. A run longer than this
/// means the file is not part of a sequence at all — truncating it to the last
/// nine digits would defeat the point.
pub const MAX_FRAME_DIGITS: usize = 9;

/// Hard cap on how many timeline slots one sequence may span. Holes are real
/// slots (that is what lets the cache bar draw them red), so frames 1001 and
/// 9999 with nothing between is a legitimate 8999-slot timeline. But the digit
/// cap allows frame numbers up to 999,999,999, and two files that far apart
/// would ask for gigabytes of slots — so the span is clamped around the frame
/// the user actually opened, and the clamp is logged.
const MAX_SPAN: usize = 1_000_000;

/// A filename split into a prefix, its trailing frame number, and an extension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameName<'a> {
    /// Everything before the trailing digits (may be empty).
    pub prefix: &'a str,
    /// The trailing digit run, exactly as written.
    pub digits: &'a str,
    /// That run parsed as a number.
    pub number: i64,
    /// Extension without the dot, exactly as written (may be empty).
    pub ext: &'a str,
}

/// Split `file_name` into prefix + trailing frame number + extension, or `None`
/// when the stem does not end in a usable digit run.
pub fn split_frame_name(file_name: &str) -> Option<FrameName<'_>> {
    // Split off the extension ourselves rather than via `Path`, so the borrow
    // lifetimes stay tied to the caller's string.
    let (stem, ext) = match file_name.rfind('.') {
        // A leading dot is a hidden file, not an extension.
        Some(i) if i > 0 => (&file_name[..i], &file_name[i + 1..]),
        _ => (file_name, ""),
    };
    let digit_start = stem
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    let digits = &stem[digit_start..];
    if digits.is_empty() || digits.len() > MAX_FRAME_DIGITS {
        return None;
    }
    Some(FrameName {
        prefix: &stem[..digit_start],
        digits,
        // Bounded by MAX_FRAME_DIGITS, so this cannot overflow i64.
        number: digits.parse().ok()?,
        ext,
    })
}

/// The one legal spelling of `number` in a sequence padded to `pad` digits:
/// zero-padded to that width, or written out in full once it no longer fits.
pub fn canonical_digits(number: i64, pad: usize) -> String {
    format!("{number:0pad$}")
}

/// What is known about one slot on the timeline.
///
/// `Missing` and `Corrupt` behave identically for playback and for the cache bar
/// — both are frames the playhead skips over and both draw red. They stay
/// distinct only so the log can say which it was.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SlotState {
    /// Never attempted.
    #[default]
    Unknown,
    /// A worker is decoding it.
    Pending,
    /// Decoded and resident in the frame cache.
    Cached,
    /// No file on disk for this frame number.
    Missing,
    /// A file exists but would not decode (often a half-written render).
    Corrupt,
}

impl SlotState {
    /// True when this slot can never produce a frame as things stand — the
    /// playhead moves straight past it and the cache bar draws it red.
    pub fn is_hole(self) -> bool {
        matches!(self, Self::Missing | Self::Corrupt)
    }
}

/// One timeline slot: the file backing it (if any) and what we know about it.
#[derive(Clone, Debug)]
struct Slot {
    /// File name within the sequence directory; `None` for a hole.
    name: Option<Box<str>>,
    /// File size when last seen. A change means the file was rewritten, which is
    /// how a frame that was half-written (and so decoded as `Corrupt`) gets
    /// another chance on the next rescan.
    len: u64,
    state: SlotState,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            name: None,
            len: 0,
            state: SlotState::Missing,
        }
    }
}

/// Why a file could not start playback.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SequenceError {
    /// The filename's stem does not end in a usable frame number.
    NoFrameNumber,
    /// Only one frame matches — there is nothing to play.
    SingleFrame,
    /// The parent directory could not be read.
    Io(String),
}

impl SequenceError {
    /// A one-line message for the toast.
    pub fn message(&self) -> String {
        match self {
            Self::NoFrameNumber => {
                "Not part of a sequence — the file name must end in a frame number".to_string()
            }
            Self::SingleFrame => "Only one frame in this sequence".to_string(),
            Self::Io(e) => format!("Could not scan the folder: {e}"),
        }
    }
}

/// What a [`Sequence::rescan`] found.
#[derive(Clone, Default, Debug)]
pub struct RescanOutcome {
    /// Anything at all differs from the previous scan.
    pub changed: bool,
    /// Frames whose file appeared or was rewritten. Their decode may have failed
    /// last time on a half-written file, so the caller should let them be tried
    /// again.
    pub refreshed: Vec<i64>,
}

/// A detected frame sequence: a contiguous span of frame numbers, each either
/// backed by a file or a hole.
#[derive(Debug)]
pub struct Sequence {
    dir: PathBuf,
    prefix: String,
    /// Extension as written on the opened file (case preserved for display; the
    /// scan compares extensions case-insensitively, as `is_supported` does).
    ext: String,
    pad: usize,
    /// Frame number of `slots[0]`.
    first: i64,
    slots: Vec<Slot>,
    /// Bumped whenever a slot's state actually changes. Lets the timeline's
    /// cache bar be rebuilt only when it would differ, instead of rescanning
    /// every slot on every rendered frame.
    state_epoch: u64,
}

impl Sequence {
    /// Detect the sequence `path` belongs to by scanning its parent directory.
    pub fn detect(path: &Path) -> Result<Self, SequenceError> {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or(SequenceError::NoFrameNumber)?;
        let opened = split_frame_name(file_name).ok_or(SequenceError::NoFrameNumber)?;
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let entries = read_dir_entries(&dir)?;
        Self::build(dir, file_name, &opened, &entries)
    }

    /// Re-scan the directory: pick up frames written since the last scan, extend
    /// the range, and reset any slot whose file changed size (a frame that was
    /// half-written when we first tried it decodes fine once the render finishes
    /// flushing it). Returns whether anything changed.
    ///
    /// Cheap enough to run on a slow timer while playing: one `read_dir`, and
    /// the sizes come along with the directory entries.
    pub fn rescan(&mut self) -> RescanOutcome {
        let mut out = RescanOutcome::default();
        let Ok(entries) = read_dir_entries(&self.dir) else {
            return out;
        };
        let found = self.match_entries(&entries);
        let Some((&lo, _)) = found.iter().next() else {
            return out;
        };
        let (&hi, _) = found.iter().next_back().expect("non-empty");
        // Keep every slot we already know about, so a frame that vanishes from
        // disk mid-playback becomes a hole rather than shortening the timeline.
        let new_first = lo.min(self.first);
        let new_last = hi.max(self.last());
        let (new_first, new_last) = clamp_span(new_first, new_last, self.first);

        if new_first != self.first || new_last != self.last() {
            let mut slots = vec![Slot::default(); (new_last - new_first + 1) as usize];
            for (i, slot) in self.slots.drain(..).enumerate() {
                let frame = self.first + i as i64;
                if (new_first..=new_last).contains(&frame) {
                    slots[(frame - new_first) as usize] = slot;
                }
            }
            self.slots = slots;
            self.first = new_first;
            self.state_epoch += 1;
            out.changed = true;
        }

        for (&frame, &(name, len)) in &found {
            let Some(i) = self.index_of(frame) else {
                continue;
            };
            let slot = &mut self.slots[i];
            // A rewritten (or newly appeared) file goes back to Unknown so the
            // scheduler will try it again.
            if slot.name.as_deref() != Some(name) || slot.len != len {
                slot.name = Some(name.into());
                slot.len = len;
                slot.state = SlotState::Unknown;
                self.state_epoch += 1;
                out.refreshed.push(frame);
                out.changed = true;
            }
        }
        // Files that disappeared become holes.
        let (first, mut epoch) = (self.first, self.state_epoch);
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let frame = first + i as i64;
            if slot.name.is_some() && !found.contains_key(&frame) {
                *slot = Slot::default();
                epoch += 1;
                out.changed = true;
            }
        }
        self.state_epoch = epoch;
        out
    }

    /// Build a sequence from a directory listing of `(file_name, size)` pairs.
    /// Split out from [`detect`](Self::detect) so every naming rule is testable
    /// without touching the filesystem.
    fn build(
        dir: PathBuf,
        opened_name: &str,
        opened: &FrameName,
        entries: &[(String, u64)],
    ) -> Result<Self, SequenceError> {
        let mut seq = Self {
            dir,
            prefix: opened.prefix.to_string(),
            ext: opened.ext.to_string(),
            pad: detect_pad(opened, entries),
            first: opened.number,
            slots: Vec::new(),
            state_epoch: 0,
        };
        let found = seq.match_entries(entries);
        // `match_entries` always accepts the opened file (it is canonical under
        // its own width, and `detect_pad` guarantees the width it picked keeps
        // it canonical), so the range always contains it.
        debug_assert!(found.contains_key(&opened.number));
        if found.len() < 2 {
            return Err(SequenceError::SingleFrame);
        }
        let (&lo, _) = found.iter().next().expect("non-empty");
        let (&hi, _) = found.iter().next_back().expect("non-empty");
        let (first, last) = clamp_span(lo, hi, opened.number);
        seq.first = first;
        seq.slots = vec![Slot::default(); (last - first + 1) as usize];
        for (&frame, &(name, len)) in &found {
            if let Some(i) = seq.index_of(frame) {
                seq.slots[i] = Slot {
                    name: Some(name.into()),
                    len,
                    state: SlotState::Unknown,
                };
            }
        }
        log::info!(
            "sequence {}[{}..{}] pad {} — {} frames, {} present ({})",
            seq.prefix,
            seq.first,
            seq.last(),
            seq.pad,
            seq.slots.len(),
            seq.present_count(),
            opened_name,
        );
        Ok(seq)
    }

    /// Directory entries that belong to this sequence, as `frame -> (name, size)`.
    fn match_entries<'a>(&self, entries: &'a [(String, u64)]) -> BTreeMap<i64, (&'a str, u64)> {
        let mut out = BTreeMap::new();
        for (name, len) in entries {
            let Some(f) = split_frame_name(name) else {
                continue;
            };
            if f.prefix != self.prefix || !f.ext.eq_ignore_ascii_case(&self.ext) {
                continue;
            }
            // Exactly one spelling per frame number, so no two files can claim
            // the same slot.
            if f.digits != canonical_digits(f.number, self.pad) {
                continue;
            }
            out.insert(f.number, (name.as_str(), *len));
        }
        out
    }

    /// Lowest frame number on the timeline.
    pub fn first(&self) -> i64 {
        self.first
    }

    /// Highest frame number on the timeline.
    pub fn last(&self) -> i64 {
        self.first + self.slots.len() as i64 - 1
    }

    /// Total slots, holes included.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Never true in practice — a sequence needs at least two frames to exist —
    /// but `len` deserves its counterpart.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Frames actually backed by a file.
    pub fn present_count(&self) -> usize {
        self.slots.iter().filter(|s| s.name.is_some()).count()
    }

    /// The directory this sequence lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A stable identity for this sequence, distinct from any single frame's
    /// path. Display rotation is remembered against this, so `↑`/`↓` keep
    /// working as the frame changes.
    pub fn identity(&self) -> PathBuf {
        self.dir.join(format!(
            "{}{}.{}",
            self.prefix,
            "#".repeat(self.pad.max(1)),
            self.ext
        ))
    }

    /// The frame number `path`'s file name spells, whether or not it is part of
    /// this sequence.
    pub fn frame_of(path: &Path) -> Option<i64> {
        let name = path.file_name()?.to_str()?;
        split_frame_name(name).map(|f| f.number)
    }

    /// Human-readable name, e.g. `shot_####.png`.
    pub fn display_name(&self) -> String {
        format!(
            "{}{}.{}",
            self.prefix,
            "#".repeat(self.pad.max(1)),
            self.ext
        )
    }

    fn index_of(&self, frame: i64) -> Option<usize> {
        (frame >= self.first && frame <= self.last()).then(|| (frame - self.first) as usize)
    }

    /// True when `frame` is on the timeline at all.
    pub fn contains(&self, frame: i64) -> bool {
        self.index_of(frame).is_some()
    }

    pub fn state(&self, frame: i64) -> SlotState {
        self.index_of(frame)
            .map(|i| self.slots[i].state)
            .unwrap_or(SlotState::Missing)
    }

    /// Record a slot's state. Ignored for a frame off the timeline, and for a
    /// hole — a slot with no file can only be `Missing`.
    pub fn set_state(&mut self, frame: i64, state: SlotState) {
        if let Some(i) = self.index_of(frame) {
            if self.slots[i].name.is_some() && self.slots[i].state != state {
                self.slots[i].state = state;
                self.state_epoch += 1;
            }
        }
    }

    /// A counter that changes exactly when some slot's state does.
    pub fn state_epoch(&self) -> u64 {
        self.state_epoch
    }

    /// Full path of the file backing `frame`, or `None` for a hole.
    pub fn path(&self, frame: i64) -> Option<PathBuf> {
        let i = self.index_of(frame)?;
        self.slots[i].name.as_ref().map(|n| self.dir.join(&**n))
    }

    /// True when `frame` has a file (whether or not it decodes).
    pub fn has_file(&self, frame: i64) -> bool {
        self.index_of(frame)
            .is_some_and(|i| self.slots[i].name.is_some())
    }

    /// Every slot's state, in frame order — what the cache bar draws.
    pub fn states(&self) -> impl Iterator<Item = SlotState> + '_ {
        self.slots.iter().map(|s| s.state)
    }

    /// The next frame at or after `from` (stepping by `step`, ±1) that could
    /// actually be displayed — holes consume no time, so the playhead moves
    /// straight past them. Wraps at the ends when `wrap` is set (looping).
    /// `None` when the whole timeline is holes.
    pub fn next_playable(&self, from: i64, step: i64, wrap: bool) -> Option<i64> {
        let n = self.slots.len() as i64;
        if n == 0 || step == 0 {
            return None;
        }
        let mut frame = from;
        for _ in 0..n {
            if frame < self.first || frame > self.last() {
                if !wrap {
                    return None;
                }
                frame = if step > 0 { self.first } else { self.last() };
            }
            if self.playable(frame) {
                return Some(frame);
            }
            frame += step;
        }
        None
    }

    /// Build a sequence directly from a presence map, for tests elsewhere in the
    /// crate that need a timeline without touching the filesystem.
    #[cfg(test)]
    pub(crate) fn for_test(first: i64, present: &[bool]) -> Self {
        Self {
            dir: PathBuf::from("/seq"),
            prefix: "f".into(),
            ext: "png".into(),
            pad: 4,
            first,
            slots: present
                .iter()
                .enumerate()
                .map(|(i, &p)| Slot {
                    name: p.then(|| format!("f{:04}.png", first + i as i64).into_boxed_str()),
                    len: 1,
                    state: if p {
                        SlotState::Unknown
                    } else {
                        SlotState::Missing
                    },
                })
                .collect(),
            state_epoch: 0,
        }
    }

    /// True when the slot has a file that has not (yet) failed to decode.
    fn playable(&self, frame: i64) -> bool {
        self.index_of(frame)
            .is_some_and(|i| self.slots[i].name.is_some() && self.slots[i].state.is_not_hole())
    }
}

impl SlotState {
    fn is_not_hole(self) -> bool {
        !self.is_hole()
    }
}

/// The sequence's field width.
///
/// Taken from the **narrowest** digit run in the directory, not from the file
/// the user happened to open: a 4-padded sequence opened at frame `1001` has no
/// leading zero to learn from, and an unpadded one opened at frame `10` would
/// otherwise be read as 2-padded and lose frames 1–9. Falls back to the opened
/// file's own width when that group width would exclude it — which is exactly
/// the `foo1.png` / `foo0001.png` case, where the two really are two sequences
/// and the user should get the one they opened.
fn detect_pad(opened: &FrameName, entries: &[(String, u64)]) -> usize {
    let narrowest = entries
        .iter()
        .filter_map(|(name, _)| split_frame_name(name))
        .filter(|f| f.prefix == opened.prefix && f.ext.eq_ignore_ascii_case(opened.ext))
        .map(|f| f.digits.len())
        .min()
        .unwrap_or(opened.digits.len());
    if opened.digits == canonical_digits(opened.number, narrowest) {
        narrowest
    } else {
        opened.digits.len()
    }
}

/// Clamp `[lo, hi]` to [`MAX_SPAN`] slots, keeping `around` inside.
fn clamp_span(lo: i64, hi: i64, around: i64) -> (i64, i64) {
    let span = (hi - lo + 1) as u64;
    if span <= MAX_SPAN as u64 {
        return (lo, hi);
    }
    let half = (MAX_SPAN / 2) as i64;
    let first = (around - half).max(lo);
    let last = (first + MAX_SPAN as i64 - 1).min(hi);
    log::warn!(
        "sequence spans {span} frames; showing {first}..{last} around the opened frame {around}"
    );
    (first, last)
}

/// `(file_name, size)` for every plain file in `dir`. Names that are not valid
/// UTF-8 are skipped — they cannot match a prefix taken from a `&str` anyway.
fn read_dir_entries(dir: &Path) -> Result<Vec<(String, u64)>, SequenceError> {
    let rd = std::fs::read_dir(dir).map_err(|e| SequenceError::Io(e.to_string()))?;
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let len = match entry.metadata() {
            Ok(m) if m.is_file() => m.len(),
            _ => continue,
        };
        out.push((name, len));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(n: &str) -> FrameName<'_> {
        split_frame_name(n).unwrap_or_else(|| panic!("{n} should split"))
    }

    #[test]
    fn splits_a_padded_frame_name() {
        let f = name("shot_0047.png");
        assert_eq!(f.prefix, "shot_");
        assert_eq!(f.digits, "0047");
        assert_eq!(f.number, 47);
        assert_eq!(f.ext, "png");
    }

    /// Separators are optional — only the trailing digits matter.
    #[test]
    fn separators_are_optional() {
        let f = name("foobar01234.png");
        assert_eq!(f.prefix, "foobar");
        assert_eq!(f.digits, "01234");
        assert_eq!(f.number, 1234);
    }

    /// The stem must END in digits: the case the plan calls out by name.
    #[test]
    fn rejects_a_stem_that_does_not_end_in_digits() {
        assert_eq!(split_frame_name("foobar_16k.png"), None);
        assert_eq!(split_frame_name("brown_photostudio_02_4k.exr"), None);
        assert_eq!(split_frame_name("plain.png"), None);
    }

    /// The nine-digit cap keeps unix timestamps and long hashes out. Truncating
    /// to the last nine would read a timestamp as a frame number, so a longer
    /// run means "not a sequence" rather than "a big frame number".
    #[test]
    fn rejects_digit_runs_past_the_cap() {
        assert!(split_frame_name("f999999999.png").is_some(), "9 is allowed");
        assert_eq!(split_frame_name("f1234567890.png"), None);
        assert_eq!(split_frame_name("image-1782489268279.png"), None);
    }

    #[test]
    fn handles_names_without_an_extension_and_hidden_files() {
        let f = name("frame007");
        assert_eq!((f.prefix, f.digits, f.ext), ("frame", "007", ""));
        // A leading dot is a hidden file, not an extension.
        assert_eq!(split_frame_name(".hidden"), None);
        let f = name(".hidden12");
        assert_eq!((f.prefix, f.digits, f.ext), (".hidden", "12", ""));
    }

    #[test]
    fn canonical_spelling_pads_then_overflows() {
        assert_eq!(canonical_digits(47, 4), "0047");
        assert_eq!(canonical_digits(9999, 4), "9999");
        // A 4-padded sequence legitimately overflows past 9999.
        assert_eq!(canonical_digits(10000, 4), "10000");
        assert_eq!(canonical_digits(1, 1), "1");
        assert_eq!(canonical_digits(0, 4), "0000");
    }

    /// Build a sequence from a synthetic listing.
    fn build(opened: &str, names: &[&str]) -> Result<Sequence, SequenceError> {
        let entries: Vec<(String, u64)> = names.iter().map(|n| ((*n).to_string(), 1)).collect();
        let f = split_frame_name(opened).ok_or(SequenceError::NoFrameNumber)?;
        Sequence::build(PathBuf::from("/seq"), opened, &f, &entries)
    }

    fn frames(seq: &Sequence) -> Vec<i64> {
        (seq.first()..=seq.last())
            .filter(|f| seq.has_file(*f))
            .collect()
    }

    #[test]
    fn groups_a_padded_sequence() {
        let seq = build(
            "shot_0002.png",
            &[
                "shot_0001.png",
                "shot_0002.png",
                "shot_0003.png",
                "other.png",
            ],
        )
        .unwrap();
        assert_eq!(frames(&seq), vec![1, 2, 3]);
        assert_eq!((seq.first(), seq.last(), seq.len()), (1, 3, 3));
    }

    /// A 4-padded sequence opened at frame 1001 — the standard VFX start frame,
    /// whose digits carry no leading zero to learn the padding from. Taking the
    /// width from the directory rather than the opened file is what makes this
    /// work.
    #[test]
    fn detects_padding_when_the_opened_frame_has_no_leading_zero() {
        let names: Vec<String> = (1001..=1005).map(|n| format!("sh.{n:04}.exr")).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let seq = build("sh.1001.exr", &refs).unwrap();
        assert_eq!(frames(&seq), vec![1001, 1002, 1003, 1004, 1005]);
    }

    /// An unpadded sequence opened at a two-digit frame must still find frames
    /// 1–9 (the mirror of the case above).
    #[test]
    fn unpadded_sequence_opened_at_a_wide_frame() {
        let names: Vec<String> = (1..=12).map(|n| format!("f{n}.png")).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let seq = build("f10.png", &refs).unwrap();
        assert_eq!(frames(&seq).len(), 12);
        assert_eq!((seq.first(), seq.last()), (1, 12));
    }

    /// A 4-padded sequence legitimately overflows past 9999.
    #[test]
    fn padding_overflow_is_part_of_the_same_sequence() {
        let seq = build(
            "f9998.png",
            &["f9998.png", "f9999.png", "f10000.png", "f10001.png"],
        )
        .unwrap();
        assert_eq!(frames(&seq), vec![9998, 9999, 10000, 10001]);
    }

    /// `foo1.png` and `foo0001.png` in one folder are two sequences, and you get
    /// the one you opened — whichever that is.
    #[test]
    fn differing_pad_widths_are_separate_sequences() {
        let all = &["foo1.png", "foo2.png", "foo0001.png", "foo0002.png"];
        let unpadded = build("foo1.png", all).unwrap();
        assert_eq!(frames(&unpadded), vec![1, 2]);
        assert_eq!(unpadded.path(1).unwrap().file_name().unwrap(), "foo1.png");

        let padded = build("foo0001.png", all).unwrap();
        assert_eq!(frames(&padded), vec![1, 2]);
        assert_eq!(padded.path(1).unwrap().file_name().unwrap(), "foo0001.png");
    }

    /// Extensions compare case-insensitively (as `is_supported` does), but a
    /// different extension is a different sequence.
    #[test]
    fn extension_matching() {
        let seq = build(
            "f001.png",
            &["f001.png", "f002.PNG", "f003.jpg", "g001.png"],
        )
        .unwrap();
        assert_eq!(frames(&seq), vec![1, 2]);
    }

    /// Holes are real slots on the timeline — that is what lets the cache bar
    /// draw them red — and they report as `Missing` with no file.
    #[test]
    fn holes_occupy_slots() {
        let seq = build("f001.png", &["f001.png", "f005.png"]).unwrap();
        assert_eq!((seq.first(), seq.last(), seq.len()), (1, 5, 5));
        assert_eq!(seq.present_count(), 2);
        for f in 2..=4 {
            assert_eq!(seq.state(f), SlotState::Missing);
            assert!(!seq.has_file(f));
            assert_eq!(seq.path(f), None);
        }
    }

    #[test]
    fn a_lone_frame_is_not_a_sequence() {
        assert_eq!(
            build("only0001.png", &["only0001.png", "unrelated.png"]).err(),
            Some(SequenceError::SingleFrame)
        );
    }

    /// Holes consume no time: the playhead moves straight to the next frame that
    /// exists, wrapping at the loop point.
    #[test]
    fn next_playable_skips_holes_and_wraps() {
        let seq = build("f001.png", &["f001.png", "f005.png", "f006.png"]).unwrap();
        assert_eq!(seq.next_playable(2, 1, false), Some(5));
        assert_eq!(seq.next_playable(5, 1, false), Some(5));
        assert_eq!(seq.next_playable(7, 1, false), None);
        assert_eq!(seq.next_playable(7, 1, true), Some(1));
        assert_eq!(seq.next_playable(4, -1, false), Some(1));
        assert_eq!(seq.next_playable(0, -1, true), Some(6));
    }

    /// A frame that failed to decode is a hole too, so playback steps past it
    /// rather than stalling on it.
    #[test]
    fn next_playable_skips_corrupt_frames() {
        let mut seq = build("f001.png", &["f001.png", "f002.png", "f003.png"]).unwrap();
        seq.set_state(2, SlotState::Corrupt);
        assert_eq!(seq.next_playable(2, 1, false), Some(3));
    }

    /// A hole can only ever be `Missing`; nothing can mark it cached.
    #[test]
    fn a_hole_cannot_be_given_another_state() {
        let mut seq = build("f001.png", &["f001.png", "f003.png"]).unwrap();
        seq.set_state(2, SlotState::Cached);
        assert_eq!(seq.state(2), SlotState::Missing);
        seq.set_state(3, SlotState::Cached);
        assert_eq!(seq.state(3), SlotState::Cached);
    }

    /// Two files can never claim the same slot: the canonical spelling is unique
    /// per frame number, so `f01.png` simply is not part of a 3-padded sequence.
    #[test]
    fn one_spelling_per_frame_number() {
        let seq = build("f001.png", &["f001.png", "f01.png", "f002.png"]).unwrap();
        assert_eq!(frames(&seq), vec![1, 2]);
        assert_eq!(seq.path(1).unwrap().file_name().unwrap(), "f001.png");
    }

    #[test]
    fn identity_is_shared_by_every_frame() {
        let a = build("f001.png", &["f001.png", "f002.png"]).unwrap();
        let b = build("f002.png", &["f001.png", "f002.png"]).unwrap();
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a.display_name(), "f###.png");
    }

    /// An absurd span is clamped around the frame the user opened rather than
    /// allocating slots for the whole range.
    #[test]
    fn an_absurd_span_is_clamped_around_the_opened_frame() {
        let seq = build("f000000001.png", &["f000000001.png", "f123456789.png"]);
        // 9 digits is within the cap, so both files parse; the span is not.
        let seq = seq.unwrap();
        assert!(seq.len() <= MAX_SPAN);
        assert!(seq.contains(1), "the opened frame stays on the timeline");
    }
}
