//! Where a player's frames come from.
//!
//! Two implementations, one player (plans/image_sequence_playback.md §3.4):
//! files on disk, where decoding is real work done by a pool; and frames already
//! in memory — the `Animation` inside an animated GIF / APNG / WebP — where
//! decoding is a no-op, every slot is cached the moment you press `Space`, and
//! the per-frame delays recorded in the file are the timing.
//!
//! Everything downstream — timeline, transport, scrubbing, texture ring, feature
//! degradation — is written against this and does not care which it is.

use std::sync::Arc;
use std::time::Duration;

use crate::image_loader::ImageData;
use crate::playback::cache::FrameCache;
use crate::playback::sequence::{Sequence, SlotState};

/// One frame's pixels, ready for upload.
pub enum FramePixels<'a> {
    /// A separately-decoded file frame.
    Image(&'a ImageData),
    /// One frame of an in-memory animation: the full canvas as interleaved
    /// RGBA8, exactly as the ring's texture wants it.
    Rgba8(&'a [u8]),
}

pub enum FrameSource {
    /// Files on disk, decoded on the worker pool into a byte-budgeted cache.
    /// Boxed because the cache is far larger than an `Arc`, and the in-memory
    /// case is the common small one.
    Files(Box<FrameCache>),
    /// The frames of the animated image itself. They are already resident, so
    /// there is no cache, no pool and no budget — pressing `Space` on a GIF
    /// costs nothing but the transport.
    Memory(Arc<ImageData>),
}

impl FrameSource {
    /// True when `frame` can be displayed right now.
    pub fn is_ready(&self, frame: i64) -> bool {
        match self {
            Self::Files(cache) => cache.contains(frame),
            Self::Memory(img) => anim_frame(img, frame).is_some(),
        }
    }

    /// The `ImageData` behind `frame` — for the colour picker, comparator slots
    /// and the frame kept when playback ends. For an animation every frame
    /// shares one image (the frame index selects within it), which is exactly
    /// what `ImageData::raw_pixel_at` already expects.
    pub fn image(&self, frame: i64) -> Option<Arc<ImageData>> {
        match self {
            Self::Files(cache) => cache.get(frame),
            Self::Memory(img) => anim_frame(img, frame).map(|_| img.clone()),
        }
    }

    /// A standalone still of `frame` — what you are left looking at when
    /// playback stops on it.
    ///
    /// For files that is just the decoded frame. For an animation the frames
    /// share one `ImageData` whose own buffer mirrors frame 0, so stopping on
    /// frame 47 has to lift frame 47's pixels out into an image of their own —
    /// otherwise Stop would silently jump back to the first frame. It copies one
    /// frame, once, at Stop; nothing per frame pays for it.
    pub fn still_of(&self, frame: i64) -> Option<Arc<ImageData>> {
        match self {
            Self::Files(cache) => cache.get(frame),
            Self::Memory(img) => {
                let f = anim_frame(img, frame)?;
                Some(Arc::new(ImageData {
                    path: img.path.clone(),
                    width: img.width,
                    height: img.height,
                    channels: img.channels,
                    dtype_name: img.dtype_name.clone(),
                    compression: img.compression.clone(),
                    pixels: crate::image_loader::PixelBuffer::U8(f.pixels.clone()),
                    is_encoded_srgb: img.is_encoded_srgb,
                    clip_max: img.clip_max,
                    // A still, not an animation: it must not restart the player.
                    animation: None,
                    camera: img.camera.clone(),
                }))
            }
        }
    }

    /// The pixels to upload for `frame`.
    pub fn pixels(&self, frame: i64) -> Option<FramePixels<'_>> {
        match self {
            Self::Files(cache) => {
                // The cache owns the `Arc`, so this borrow is of the cache, not
                // of a temporary — which is why `get` (cloning the `Arc`) and
                // this are separate methods.
                cache.peek(frame).map(FramePixels::Image)
            }
            Self::Memory(img) => anim_frame(img, frame).map(|f| FramePixels::Rgba8(&f.pixels)),
        }
    }

    /// How long `frame` should stay on screen, when the source dictates it.
    /// `None` means "use the configured target rate" — variable-rate GIFs keep
    /// their own per-frame timing, and the fps readout then reports whatever
    /// rate that works out to.
    pub fn frame_delay(&self, frame: i64) -> Option<Duration> {
        match self {
            Self::Files(_) => None,
            Self::Memory(img) => anim_frame(img, frame).map(|f| f.delay),
        }
    }

    /// Collect finished decodes. A no-op for in-memory frames.
    pub fn poll(&mut self, seq: &mut Sequence) -> usize {
        match self {
            Self::Files(cache) => cache.poll(seq),
            Self::Memory(_) => 0,
        }
    }

    /// Retarget the cache window. A no-op for in-memory frames — everything is
    /// already resident, so the cache bar is fully blue from the start.
    pub fn schedule(&mut self, seq: &mut Sequence, playhead: i64, looping: bool) {
        match self {
            Self::Files(cache) => cache.schedule(seq, playhead, looping),
            Self::Memory(_) => {}
        }
    }

    /// A rescan found this frame rewritten on disk: drop what we hold of it so
    /// it decodes afresh (in-memory animation frames never change on disk).
    pub fn forget(&mut self, frame: i64) {
        if let Self::Files(cache) = self {
            cache.forget(frame);
        }
    }

    /// Put an already-decoded frame straight in, so entering playback never
    /// re-decodes what is already on screen.
    pub fn seed(&mut self, frame: i64, data: Arc<ImageData>) {
        if let Self::Files(cache) = self {
            cache.seed(frame, data);
        }
    }

    /// Stop waking the event loop — call before it goes away.
    pub fn cancel_wakeups(&self) {
        if let Self::Files(cache) = self {
            cache.cancel_wakeups();
        }
    }

    /// `(resident, capacity, in flight)` for the diagnostic heartbeat.
    pub fn stats(&self, seq: &Sequence) -> (usize, usize, usize) {
        match self {
            Self::Files(cache) => (
                cache.resident_frames(),
                cache.capacity_frames(),
                cache.inflight_count(),
            ),
            Self::Memory(_) => (seq.len(), seq.len(), 0),
        }
    }
}

/// The `frame`-th animation frame of `img`, if it has one.
fn anim_frame(img: &ImageData, frame: i64) -> Option<&crate::image_loader::AnimFrame> {
    let idx = usize::try_from(frame).ok()?;
    img.animation.as_ref()?.frames.get(idx)
}

/// A timeline over an animated image's frames: every slot present, every slot
/// already cached, numbered from zero.
pub fn animation_sequence(img: &ImageData) -> Option<Sequence> {
    let count = img.animation.as_ref()?.frames.len();
    (count > 1).then(|| Sequence::for_animation(&img.path, count, SlotState::Cached))
}
