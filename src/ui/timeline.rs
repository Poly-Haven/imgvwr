//! The image-sequence transport row: play/pause, stop, the scrubbable timeline
//! with its cache bar, and the achieved-rate readout.
//! See plans/image_sequence_playback.md §9.
//!
//! Drawn inside the bottom panel's `egui::Area`, so it shares that panel's slide
//! animation, its single deferred background rect and its autohide. It is part
//! of the bottom bar, not a second panel.

use super::colors::{panel_bg, ACCENT, CHANNEL_R};
use super::{clickable, PlaybackInfo, UiAction};
use crate::playback::SlotState;

/// Height of the transport row (points).
pub const ROW_H: f32 = 36.0;
/// Band reserved above the track for the playhead's frame number, so it sits
/// inside the row rather than over the pixel ruler above it.
const LABEL_BAND: f32 = 14.0;
/// Transport button size.
const BTN: egui::Vec2 = egui::vec2(28.0, 22.0);
/// Thickness of the cache-residency strip along the bottom of the timeline.
const CACHE_BAR_H: f32 = 4.0;
/// Space reserved at the right for the fps readout, so it never collides with
/// the timeline as the window narrows (the same convention as the bottom
/// panel's Reset button gutter).
const FPS_GUTTER: f32 = 68.0;
/// Outer padding either side of the row.
const PAD: f32 = 11.0;
/// Shortest on-screen spacing between minor ticks (points), below which they are
/// suppressed — the rulers solve exactly this and the approach transfers.
const MIN_TICK_PX: f32 = 5.0;
/// Tick heights (points).
const TICK_MINOR: f32 = 4.0;
const TICK_MAJOR: f32 = 8.0;

fn tick_color() -> egui::Color32 {
    egui::Color32::from_gray(120)
}

fn track_color() -> egui::Color32 {
    egui::Color32::from_gray(58)
}

/// Cached frames. Deliberately not the accent colour — the cache bar is a
/// status readout, not a control.
const CACHE_BLUE: egui::Color32 = egui::Color32::from_rgb(70, 120, 200);

/// What one device-pixel column of the cache bar shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColumnFill {
    /// At least one frame in this column is missing or corrupt.
    Hole,
    /// Every frame here is either cached or being decoded, and at least one is
    /// cached.
    Cached,
    /// Nothing known about this column yet.
    Empty,
}

/// Resolve one column from the slot states it covers: **red beats blue beats
/// empty**.
///
/// This is the histogram lesson repeated. egui meshes go straight to the GPU
/// with no anti-aliasing, so a quad containing no pixel *centre* rasterises to
/// nothing. With 150 frames across 600 px each frame is 4 px wide and naive
/// drawing is fine — but a 10,000-frame sequence gives 0.06 px per frame, and an
/// isolated missing frame would silently vanish. That frame is precisely the
/// information the bar exists to convey, so it wins its whole column.
pub fn column_fill(states: &[SlotState]) -> ColumnFill {
    let mut fill = ColumnFill::Empty;
    for s in states {
        if s.is_hole() {
            return ColumnFill::Hole;
        }
        if *s == SlotState::Cached {
            fill = ColumnFill::Cached;
        }
    }
    fill
}

/// The half-open slot range column `i` of `cols` covers, over `n` slots. Every
/// slot lands in exactly one column, and every column covers at least one slot.
pub fn column_span(i: usize, cols: usize, n: usize) -> std::ops::Range<usize> {
    if cols == 0 || n == 0 {
        return 0..0;
    }
    let lo = i * n / cols;
    let hi = ((i + 1) * n / cols).max(lo + 1).min(n);
    lo..hi
}

/// The frame under `x` on a timeline drawn across `track`.
pub fn frame_at_x(x: f32, track: egui::Rect, first: i64, last: i64) -> i64 {
    let n = (last - first + 1).max(1);
    let t = ((x - track.left()) / track.width().max(1.0)).clamp(0.0, 1.0);
    // The last frame owns the final sliver, so dragging to the very end reaches
    // it rather than stopping one short.
    let idx = (t * n as f32).floor() as i64;
    first + idx.clamp(0, n - 1)
}

/// Where the centre of `frame` sits on a timeline drawn across `track`.
pub fn x_of_frame(frame: i64, track: egui::Rect, first: i64, last: i64) -> f32 {
    let n = (last - first + 1).max(1);
    let idx = (frame - first).clamp(0, n - 1) as f32;
    track.left() + track.width() * (idx + 0.5) / n as f32
}

/// Draw the transport row and emit its actions. Returns whether the pointer is
/// interacting with it (so the panel stays revealed).
pub fn transport_row(
    ui: &mut egui::Ui,
    pb: &PlaybackInfo,
    actions: &mut Vec<UiAction>,
    width: f32,
) -> bool {
    let (row, _) = ui.allocate_exact_size(egui::vec2(width, ROW_H), egui::Sense::hover());
    let mut hot = false;

    // The track sits below the label band; everything else lines up with it.
    let track_top = row.top() + LABEL_BAND;
    let track_bottom = row.bottom() - 4.0;
    let btn_y = (track_top + track_bottom) * 0.5;

    // ---- transport buttons -------------------------------------------------
    let play_rect =
        egui::Rect::from_center_size(egui::pos2(row.left() + PAD + BTN.x * 0.5, btn_y), BTN);
    let stop_rect = egui::Rect::from_center_size(
        egui::pos2(play_rect.right() + 4.0 + BTN.x * 0.5, btn_y),
        BTN,
    );
    let play_icon = if pb.playing {
        egui::include_image!("../../resources/icons/ui/pause-fill.svg")
    } else {
        egui::include_image!("../../resources/icons/ui/play-fill.svg")
    };
    if transport_button(ui, play_rect, play_icon, "play").clicked() {
        actions.push(UiAction::PlaybackToggle);
    }
    let stop_icon = egui::include_image!("../../resources/icons/ui/stop-fill.svg");
    if transport_button(ui, stop_rect, stop_icon, "stop").clicked() {
        actions.push(UiAction::PlaybackStop);
    }

    // ---- the timeline ------------------------------------------------------
    let track = egui::Rect::from_min_max(
        egui::pos2(stop_rect.right() + 14.0, track_top),
        egui::pos2(row.right() - PAD - FPS_GUTTER, track_bottom),
    );
    if track.width() > 8.0 {
        draw_track(ui, pb, track);
        hot |= scrub(ui, pb, actions, track);
    }

    // ---- achieved rate -----------------------------------------------------
    let fps = pb.achieved_fps.unwrap_or(pb.target_fps);
    let color = if pb.behind {
        CHANNEL_R
    } else {
        egui::Color32::from_gray(170)
    };
    ui.painter().text(
        egui::pos2(row.right() - PAD, btn_y),
        egui::Align2::RIGHT_CENTER,
        format!("{fps:.1} fps"),
        egui::FontId::proportional(12.0),
        color,
    );

    hot
}

fn transport_button(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    icon: egui::ImageSource<'static>,
    id: &str,
) -> egui::Response {
    let resp = ui.interact(rect, ui.id().with(id), egui::Sense::click());
    if resp.hovered() {
        ui.painter()
            .rect_filled(rect, 3.0, egui::Color32::from_white_alpha(30));
    }
    let icon_rect = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(13.0));
    egui::Image::new(icon)
        .tint(egui::Color32::from_gray(225))
        .paint_at(ui, icon_rect);
    clickable(resp)
}

/// Track, ticks, cache bar and playhead.
fn draw_track(ui: &mut egui::Ui, pb: &PlaybackInfo, track: egui::Rect) {
    let p = ui.painter();
    let n = (pb.last - pb.first + 1).max(1);
    let bar = egui::Rect::from_min_max(
        egui::pos2(track.left(), track.bottom() - CACHE_BAR_H),
        track.right_bottom(),
    );
    p.rect_filled(bar, 0.0, track_color());

    // Cache bar: one rect per *device pixel* column, each aligned to the device
    // pixel grid. This is the histogram lesson (CLAUDE.md, §9.3): a mesh quad
    // that covers no pixel *centre* rasterises to nothing, so a lone red frame
    // whose column is sub-pixel-wide or straddles a boundary would silently
    // vanish — precisely the information the bar exists to show. Working in
    // device space and snapping each column to `[k, k+1)` device pixels makes
    // every column exactly one pixel and grid-aligned, so nothing can drop out.
    let ppp = ui.ctx().pixels_per_point().max(0.5);
    let states = pb.states.as_slice();
    if !states.is_empty() {
        let d0 = (bar.left() * ppp).floor() as i64;
        let d1 = (bar.right() * ppp).ceil() as i64;
        let cols = (d1 - d0).max(1) as usize;
        let mut mesh = egui::Mesh::default();
        for i in 0..cols {
            let fill = match column_fill(&states[column_span(i, cols, states.len())]) {
                ColumnFill::Hole => CHANNEL_R,
                ColumnFill::Cached => CACHE_BLUE,
                ColumnFill::Empty => continue,
            };
            // Snap to the device-pixel grid, clamped to the bar in point space.
            let x0 = ((d0 + i as i64) as f32 / ppp).max(bar.left());
            let x1 = ((d0 + i as i64 + 1) as f32 / ppp).min(bar.right());
            if x1 <= x0 {
                continue;
            }
            mesh.add_colored_rect(
                egui::Rect::from_min_max(egui::pos2(x0, bar.top()), egui::pos2(x1, bar.bottom())),
                fill,
            );
        }
        p.add(egui::Shape::mesh(mesh));
    }

    // Ticks: minor every second of frames, major every minute, suppressed once
    // they would crowd together.
    let per_sec = pb.target_fps.max(1.0).round() as i64;
    let px_per_frame = track.width() / n as f32;
    let mut step = per_sec.max(1);
    while (step as f32) * px_per_frame < MIN_TICK_PX {
        step *= 2;
    }
    let major_every = (per_sec * 60).max(step);
    let tick_y = bar.top();
    let mut f = pb.first - pb.first.rem_euclid(step);
    while f <= pb.last {
        if f >= pb.first {
            let major = f.rem_euclid(major_every) == 0;
            let len = if major { TICK_MAJOR } else { TICK_MINOR };
            let x = x_of_frame(f, track, pb.first, pb.last).round();
            p.line_segment(
                [egui::pos2(x, tick_y), egui::pos2(x, tick_y - len)],
                egui::Stroke::new(1.0, tick_color()),
            );
        }
        f += step;
    }

    // Playhead, with its real frame number above it.
    let x = x_of_frame(pb.frame, track, pb.first, pb.last).round() + 0.5;
    p.line_segment(
        [egui::pos2(x, track.top()), egui::pos2(x, bar.bottom())],
        egui::Stroke::new(1.5, ACCENT),
    );
    let label = pb.frame.to_string();
    let font = egui::FontId::proportional(11.0);
    let galley = p.layout_no_wrap(label, font.clone(), egui::Color32::WHITE);
    // Keep the number inside the track when the playhead is near either end.
    let lx = (x - galley.size().x * 0.5).clamp(
        track.left(),
        (track.right() - galley.size().x).max(track.left()),
    );
    let pos = egui::pos2(lx, track.top() - galley.size().y - 1.0);
    p.rect_filled(
        egui::Rect::from_min_size(pos, galley.size()).expand2(egui::vec2(3.0, 0.0)),
        2.0,
        panel_bg(),
    );
    p.galley(pos, galley, egui::Color32::WHITE);
}

/// Click jumps, drag scrubs. Scrubbing while playing keeps playing from the new
/// position — there is no implicit pause.
fn scrub(
    ui: &mut egui::Ui,
    pb: &PlaybackInfo,
    actions: &mut Vec<UiAction>,
    track: egui::Rect,
) -> bool {
    // A generous vertical grab area: the track itself is only a few points tall.
    let zone = egui::Rect::from_min_max(
        egui::pos2(track.left(), track.top() - 6.0),
        egui::pos2(track.right(), track.bottom() + 4.0),
    );
    let resp = ui.interact(
        zone,
        ui.id().with("timeline"),
        egui::Sense::click_and_drag(),
    );
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
    // `interact_pointer_pos()` is where the pointer is NOW, not where it went
    // down. For deciding *what a gesture grabbed* that is the wrong position and
    // the press origin is what you want (the levels handles learned this the
    // hard way) — but the scrubber has no sub-elements to grab, and the playhead
    // is supposed to follow the pointer, so here the live position is exactly
    // right. Which widget the gesture belongs to is already egui's answer:
    // `dragged()` and `clicked()` are only true for the widget the press landed
    // on, and `press_origin` is cleared by the time a click is reported anyway.
    if resp.dragged() || resp.clicked() {
        if let Some(pos) = resp.interact_pointer_pos() {
            let frame = frame_at_x(pos.x, track, pb.first, pb.last);
            if frame != pb.frame {
                actions.push(UiAction::PlaybackSeek(frame));
            }
        }
    }
    // Reporting the drag (not just the hover) is what latches the panel open for
    // the whole gesture: the caller folds this into `pointer_over_panel`, so the
    // bar cannot slide away under a scrub that wanders off it. The levels drag
    // needs the same latch and gets it the same way.
    resp.contains_pointer() || resp.dragged()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(300.0, 12.0))
    }

    fn info(first: i64, last: i64, frame: i64, playing: bool) -> PlaybackInfo {
        PlaybackInfo {
            first,
            last,
            frame,
            playing,
            target_fps: 24.0,
            achieved_fps: None,
            behind: false,
            states: Default::default(),
        }
    }

    /// Drive a real pointer gesture through egui and return the actions it
    /// produced. The borderless window is invisible to computer-use tooling, so
    /// synthetic input through a bare `Context` is the only way to actually
    /// exercise a scrub — and the interaction bugs worth catching are all in
    /// *when* egui reports a drag and *where* it says the pointer was.
    fn gesture(pb: &PlaybackInfo, press_x: f32, release_x: f32, drag: bool) -> Vec<UiAction> {
        let ctx = egui::Context::default();
        let mut actions = Vec::new();
        let t = track();
        let y = t.center().y;

        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let press = egui::pos2(press_x, y);
        let release = egui::pos2(release_x, y);
        let half = egui::pos2((press_x + release_x) * 0.5, y);

        let mut frames: Vec<Vec<egui::Event>> = vec![
            // Two hover frames first: a press is hit-tested against the
            // PREVIOUS frame's widget rects, so a widget pressed the instant it
            // appears is never credited with it.
            vec![egui::Event::PointerMoved(press)],
            vec![egui::Event::PointerMoved(press)],
            vec![egui::Event::PointerMoved(press), button(press, true)],
        ];
        if drag {
            // The first move is what crosses egui's click threshold and turns
            // the press into a drag.
            frames.push(vec![egui::Event::PointerMoved(half)]);
            frames.push(vec![egui::Event::PointerMoved(release)]);
        }
        frames.push(vec![button(release, false)]);

        for (i, events) in frames.into_iter().enumerate() {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                time: Some(i as f64 * 0.1),
                events,
                ..Default::default()
            };
            let _ = ctx.run(raw, |ctx| {
                egui::Area::new(egui::Id::new("timeline_test"))
                    .fixed_pos(t.min)
                    .show(ctx, |ui| {
                        let _ = ui.allocate_exact_size(t.size(), egui::Sense::hover());
                        scrub(ui, pb, &mut actions, t);
                    });
            });
        }
        actions
    }

    fn last_seek(actions: &[UiAction]) -> Option<i64> {
        actions.iter().rev().find_map(|a| match a {
            UiAction::PlaybackSeek(f) => Some(*f),
            _ => None,
        })
    }

    /// A click on the timeline jumps the playhead to that frame.
    #[test]
    fn a_click_jumps_the_playhead() {
        let t = track();
        let pb = info(1001, 1150, 1001, false);
        let x = t.left() + t.width() * 0.5;
        let actions = gesture(&pb, x, x, false);
        assert_eq!(last_seek(&actions), Some(frame_at_x(x, t, 1001, 1150)));
        assert_ne!(last_seek(&actions), Some(1001), "the playhead moved");
    }

    /// A drag scrubs, ending on the frame under the pointer when it is released.
    #[test]
    fn a_drag_scrubs_to_where_it_ends() {
        let t = track();
        let pb = info(1001, 1150, 1001, true);
        let actions = gesture(&pb, t.left() + 20.0, t.right() - 20.0, true);
        assert!(
            actions.len() > 1,
            "a scrub should emit seeks as it moves, not just at the end"
        );
        assert_eq!(
            last_seek(&actions),
            Some(frame_at_x(t.right() - 20.0, t, 1001, 1150))
        );
    }

    /// Scrubbing while playing keeps playing: the transport emits seeks only,
    /// never a pause.
    #[test]
    fn scrubbing_does_not_pause() {
        let pb = info(1, 100, 50, true);
        let t = track();
        let actions = gesture(&pb, t.left() + 10.0, t.right() - 10.0, true);
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, UiAction::PlaybackSeek(_))),
            "a scrub must not toggle the transport"
        );
    }

    /// Dragging past either end clamps to the first / last frame rather than
    /// running off the timeline.
    #[test]
    fn dragging_past_the_ends_clamps() {
        let t = track();
        let pb = info(1001, 1150, 1075, true);
        assert_eq!(
            last_seek(&gesture(&pb, t.center().x, t.right() + 400.0, true)),
            Some(1150)
        );
        assert_eq!(
            last_seek(&gesture(&pb, t.center().x, t.left() - 400.0, true)),
            Some(1001)
        );
    }

    /// The timeline is indexed by real frame numbers, not 0..N.
    #[test]
    fn frame_mapping_uses_real_frame_numbers() {
        let t = track();
        assert_eq!(frame_at_x(t.left(), t, 1001, 1150), 1001);
        assert_eq!(frame_at_x(t.right(), t, 1001, 1150), 1150);
        assert_eq!(frame_at_x(t.left() - 500.0, t, 1001, 1150), 1001);
        assert_eq!(frame_at_x(t.right() + 500.0, t, 1001, 1150), 1150);
        // Round-tripping through the centre of a frame's band is exact.
        for f in [1001, 1002, 1075, 1149, 1150] {
            let x = x_of_frame(f, t, 1001, 1150);
            assert_eq!(frame_at_x(x, t, 1001, 1150), f, "frame {f}");
        }
    }

    /// A one-frame-wide timeline (degenerate but reachable while a render is
    /// still writing) must not divide by zero or drift off the track.
    #[test]
    fn a_single_slot_timeline_is_well_behaved() {
        let t = track();
        assert_eq!(frame_at_x(t.center().x, t, 7, 7), 7);
        let x = x_of_frame(7, t, 7, 7);
        assert!(t.contains(egui::pos2(x, t.center().y)));
    }

    /// Red beats blue beats empty, so a single missing frame in a long cached
    /// run is always visible however many frames share its column.
    #[test]
    fn a_lone_hole_wins_its_column() {
        let mut states = vec![SlotState::Cached; 500];
        states[321] = SlotState::Missing;
        assert_eq!(column_fill(&states[300..400]), ColumnFill::Hole);
        assert_eq!(column_fill(&states[0..100]), ColumnFill::Cached);
        assert_eq!(column_fill(&[]), ColumnFill::Empty);
        assert_eq!(
            column_fill(&[SlotState::Unknown, SlotState::Pending]),
            ColumnFill::Empty,
            "pending is not yet cached"
        );
        assert_eq!(
            column_fill(&[SlotState::Unknown, SlotState::Cached]),
            ColumnFill::Cached
        );
    }

    /// Every slot lands in exactly one column, and no column is empty — so
    /// nothing can fall between two columns and disappear.
    #[test]
    fn columns_partition_the_slots() {
        for (n, cols) in [(10usize, 3usize), (10_000, 617), (3, 900), (1, 1)] {
            let mut covered = vec![0u32; n];
            for i in 0..cols {
                let span = column_span(i, cols, n);
                assert!(!span.is_empty(), "column {i} of {cols} over {n} is empty");
                for s in span {
                    covered[s] += 1;
                }
            }
            if cols <= n {
                assert!(
                    covered.iter().all(|&c| c == 1),
                    "n={n} cols={cols}: slots covered {covered:?}"
                );
            } else {
                // More columns than slots: each slot appears at least once.
                assert!(covered.iter().all(|&c| c >= 1), "n={n} cols={cols}");
            }
        }
    }
}
