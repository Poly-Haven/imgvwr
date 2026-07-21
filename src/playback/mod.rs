//! Image-sequence playback: the frame model, the decode cache, and the
//! transport. See plans/image_sequence_playback.md.
//!
//! Everything here hangs off a single `Option` on `App`, so a session that never
//! presses `Space` pays nothing for it.

pub mod cache;
pub mod sequence;

pub use cache::FrameCache;
pub use sequence::{Sequence, SequenceError, SlotState};
