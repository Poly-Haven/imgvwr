//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::colors::{
    panel_bg, panel_bg_alpha, ACCENT, CHANNEL_B, CHANNEL_G, CHANNEL_R, GUIDE_ADD, GUIDE_HOVER,
    PANEL_ALPHA,
};
use super::{clickable, LevelsDrag, LevelsGrip, PanoProj, RulerInfo, UiAction, UiInputs, UiState};

/// Height of the borderless custom titlebar; the top strip is reserved for it so
/// the slot flags / metadata box never sit under the window controls.
const TITLEBAR_H: f32 = 30.0;
/// Allocation width of a titlebar control button and of the app icon. Shared by
/// the render and the overflow-fit math ([`titlebar_fit`]) so they can't desync —
/// if the two disagreed, the filename's reserved space would be wrong and it would
/// truncate even when a button was hidden to make room.
const TB_BTN_W: f32 = 34.0;
const TB_ICON_W: f32 = 26.0;

fn overlay_frame() -> egui::Frame {
    egui::Frame {
        fill: panel_bg(),
        inner_margin: egui::Margin::same(16),
        corner_radius: egui::CornerRadius::same(8),
        ..Default::default()
    }
}

pub fn build_overlays(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    if inputs.loading {
        loading(ctx, inputs);
    } else if let Some(error) = &inputs.error {
        error_panel(ctx, inputs, error, actions);
    } else if inputs.show_hint {
        hint(ctx);
    }

    // Comparator slot flags (always visible while any slot is saved).
    slot_flags(ctx, inputs, actions);

    state.pointer_over_metadata = if inputs.metadata_slide > 0.001 && !inputs.metadata.is_empty() {
        metadata_hud(ctx, inputs, state, actions)
    } else {
        state.view_menu_open = false;
        false
    };

    if let Some((text, alpha)) = &inputs.toast {
        toast(ctx, text, *alpha);
    }

    if inputs.show_help {
        help_dialog(ctx, inputs, actions);
    }

    if state.confirm_delete {
        delete_confirm_dialog(ctx, inputs, state, actions);
    }

    settings_dialog(ctx, inputs, state, actions);

    // Bottom-right navigation minimap (border + view box over the GL thumbnail).
    // Drawn before the rulers / bottom panel so those auto-hiding panels sit on
    // top where they overlap.
    minimap(ctx, inputs);

    // The left pixel ruler (drawn before the bottom panel so the panel covers the
    // overlapping bottom-left corner). The bottom ruler lives inside the panel.
    left_ruler(ctx, inputs, state, actions);

    // (Grabbing / moving / deleting existing guides is handled in the app's input
    // layer — see App::guide_at_cursor — so it has a single authority over
    // mouse-button routing vs panning and works in both 2D and panorama.)
    guide_tooltip(ctx, inputs, state);

    // Right-drag colour-pick tooltip. Drawn after everything else that could
    // auto-reveal (those are already suppressed by `App` while colour-picking,
    // but ordering it last keeps it on top of anything that was already up when
    // the drag started).
    color_pick_tooltip(ctx, inputs);

    // Auto-hiding bottom panel: the merged bottom ruler, the playback transport
    // (while playing), and the adjustment sliders.
    bottom_panel(ctx, inputs, state, actions);

    // Auto-hiding borderless titlebar, drawn last so its controls sit on top.
    titlebar(ctx, inputs, actions);
}

/// Ruler strip thickness (px).
const RULER_W: f32 = 21.0;
/// On-screen tick lengths (minor, major).
const TICK_MINOR: f32 = 7.0;
const TICK_MAJOR: f32 = 14.0;
/// Minimum on-screen spacing between minor ticks (px). Drives the dynamic step.
const MIN_TICK_PX: f32 = 6.0;

/// Tick colour: the old gray-210 darkened by 30%.
fn ruler_tick_stroke() -> egui::Stroke {
    egui::Stroke::new(1.0, egui::Color32::from_gray(147))
}

/// The smallest "nice" number (1, 2 or 5 × 10ⁿ) ≥ `raw`. Used to pick a ruler
/// step that keeps a comfortable on-screen spacing at any zoom.
fn nice_step(raw: f32) -> f32 {
    let raw = raw.max(f32::MIN_POSITIVE);
    let pow = 10f32.powf(raw.log10().floor());
    let m = raw / pow;
    let nice = if m <= 1.0 {
        1.0
    } else if m <= 2.0 {
        2.0
    } else if m <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * pow
}

/// Emit dynamic pixel-ruler ticks over the visible image-pixel range, choosing a
/// nice minor step so ticks stay ≥ [`MIN_TICK_PX`] apart on screen — so the
/// finest is one tick per image pixel when zoomed right in, and coarse (not
/// dense) when zoomed far out. `tick(image_px, is_major)` paints each; majors are
/// every 5th minor. The step floors at 1px (never finer than a pixel).
fn pixel_ticks(lo: f32, hi: f32, ppx: f32, extent: f32, mut tick: impl FnMut(f32, bool)) {
    let minor = nice_step(MIN_TICK_PX / ppx.max(1e-6)).max(1.0);
    let (a, b) = (lo.min(hi).max(0.0), hi.max(lo).min(extent));
    let k0 = (a / minor).floor() as i64;
    let k1 = (b / minor).ceil() as i64;
    if k1 - k0 > 5000 {
        return; // degenerate ppx guard
    }
    for k in k0..=k1 {
        let pos = k as f32 * minor;
        if pos < 0.0 || pos > extent {
            continue;
        }
        tick(pos, k % 5 == 0);
    }
}

/// Nice angle step (1/2/5/10/15/30/45/90/180°) ≥ `raw`, for the panorama rulers.
fn nice_degrees(raw: f32) -> f32 {
    const STEPS: [f32; 9] = [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 90.0, 180.0];
    STEPS.into_iter().find(|&s| s >= raw).unwrap_or(180.0)
}

/// Rotate a camera-space ray by the panorama yaw/pitch (mirrors the shader's
/// `rotation_yaw_pitch`), returning the world direction `(x, y, z)`.
fn pano_rotate(p: &PanoProj, rx: f32, ry: f32, rz: f32) -> (f32, f32, f32) {
    let (cy, sy) = (p.yaw.cos(), p.yaw.sin());
    let (cp, sp) = (p.pitch.cos(), p.pitch.sin());
    // Pitch about X — must match the shader's `mp` exactly (the transpose would
    // flip the latitude ruler's sign whenever the panorama is pitched) …
    let py = cp * ry + sp * rz;
    let pz = -sp * ry + cp * rz;
    (cy * rx - sy * pz, py, sy * rx + cy * pz) // … then yaw about Y
}

/// Emit panorama ruler ticks at degree marks. `angle_at(screen_coord)` gives the
/// longitude or latitude (degrees) under that screen position. Samples ~1/px,
/// unwraps the angle across the ±180° seam, picks a nice step keeping ticks ≥
/// [`MIN_TICK_PX`] apart, and interpolates each crossing's screen position.
/// `tick(screen_coord, is_major)`; majors are every 5th step.
fn degree_ticks(s0: f32, s1: f32, angle_at: impl Fn(f32) -> f32, mut tick: impl FnMut(f32, bool)) {
    let len = (s1 - s0).abs();
    if len < 2.0 {
        return;
    }
    let n = (len.ceil() as usize).clamp(8, 2048);
    let mut samples: Vec<(f32, f32)> = Vec::with_capacity(n + 1);
    let mut prev_raw = angle_at(s0);
    let mut acc = prev_raw;
    samples.push((s0, acc));
    for i in 1..=n {
        let s = s0 + (s1 - s0) * (i as f32 / n as f32);
        let raw = angle_at(s);
        let mut d = raw - prev_raw; // unwrap across the atan2 seam
        if d > 180.0 {
            d -= 360.0;
        } else if d < -180.0 {
            d += 360.0;
        }
        acc += d;
        prev_raw = raw;
        samples.push((s, acc));
    }
    let amin = samples
        .iter()
        .map(|&(_, a)| a)
        .fold(f32::INFINITY, f32::min);
    let amax = samples
        .iter()
        .map(|&(_, a)| a)
        .fold(f32::NEG_INFINITY, f32::max);
    let minor = nice_degrees((amax - amin).max(1e-4) / len * MIN_TICK_PX);
    if (amax - amin) / minor > 4000.0 {
        return;
    }
    for w in samples.windows(2) {
        let (sa, aa) = w[0];
        let (sb, ab) = w[1];
        let k0 = (aa.min(ab) / minor).ceil() as i64;
        let k1 = (aa.max(ab) / minor).floor() as i64;
        for k in k0..=k1 {
            let target = k as f32 * minor;
            let t = if (ab - aa).abs() < 1e-9 {
                0.0
            } else {
                (target - aa) / (ab - aa)
            };
            tick(sa + (sb - sa) * t, k % 5 == 0);
        }
    }
}

// Screen → image-coordinate mappings used by the rulers to spawn guides (these
// mirror the shader's UV mapping). A *vertical* guide sits at a constant image u
// (its on-screen line is vertical); a *horizontal* guide at a constant image v.

/// Image-u (0..1) under a pointer at screen x (2D).
fn x_to_guide_u(r: &RulerInfo, screen: egui::Rect, x: f32) -> f32 {
    0.5 + r.pan_u + ((x - screen.left()) / screen.width() - 0.5) * r.sx
}
/// Image-v (0..1) under a pointer at screen y (2D).
fn y_to_guide_v(r: &RulerInfo, screen: egui::Rect, y: f32) -> f32 {
    0.5 + r.pan_v - ((1.0 - (y - screen.top()) / screen.height()) - 0.5) * r.sy
}

/// Equirectangular image uv under a screen point in panorama mode (mirrors the
/// shader's pano projection), so a guide can be spawned at the cursor's
/// longitude/latitude.
fn pano_screen_uv(p: &PanoProj, screen: egui::Rect, pt: egui::Pos2) -> (f32, f32) {
    use std::f32::consts::{PI, TAU};
    let nx = (pt.x - screen.left()) / screen.width();
    let ny = (pt.y - screen.top()) / screen.height();
    let ndc_x = nx * 2.0 - 1.0;
    let ndc_y = (1.0 - ny) * 2.0 - 1.0; // GL y-up
    let rx = ndc_x * p.aspect * p.tan_half_fov;
    let ry = ndc_y * p.tan_half_fov;
    let inv = 1.0 / (rx * rx + ry * ry + 1.0).sqrt();
    let (wx, wy, wz) = pano_rotate(p, rx * inv, ry * inv, inv);
    let lon = wz.atan2(wx);
    let lat = wy.clamp(-1.0, 1.0).asin();
    (1.0 - (lon / TAU + 0.5), 0.5 - lat / PI)
}

/// Ctrl+drag guide-coordinate snapping for a NEW guide spawned out of a ruler:
/// nearest whole degree in panorama, nearest 10 displayed px in 2D. Mirrors
/// `App::snap_guide_coord` (same formulas, `RulerInfo` instead of App fields).
fn snap_guide_coord(coord: f32, horizontal: bool, r: &RulerInfo) -> f32 {
    if r.pano.is_some() {
        if horizontal {
            let lat = (0.5 - coord) * 180.0;
            0.5 - lat.round() / 180.0
        } else {
            (coord * 360.0).round() / 360.0
        }
    } else {
        let dim = if horizontal { r.img_h } else { r.img_w };
        if dim <= 0.0 {
            return coord;
        }
        (coord * dim / 10.0).round() * 10.0 / dim
    }
    .clamp(0.0, 1.0)
}

/// Spawn-and-drag a NEW guide pulled out of a ruler, shared by both rulers.
/// `horizontal` is the new guide's orientation; `guides_len` is the current guide
/// count (the index the spawned guide will occupy). The spawned index is captured
/// at drag-start into `state.guide_spawn`, so the rest of the gesture targets that
/// exact guide via the bounds-checked `MoveGuide`/`RemoveGuide` — robust to other
/// guides being appended mid-drag (e.g. the G key) and to the guide cap (at the
/// cap nothing is spawned and no existing guide is touched). Release past the
/// image edge / off-screen discards the new guide.
fn ruler_spawn_drag(
    ctx: &egui::Context,
    resp: &egui::Response,
    r: &RulerInfo,
    screen: egui::Rect,
    horizontal: bool,
    guides_len: usize,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    let coord_at = |pt: egui::Pos2| {
        // Pano: the spawned coord is the cursor's latitude (horizontal guide) or
        // longitude (vertical guide); 2D: the linear uv mapping.
        let raw = if let Some(p) = r.pano {
            let (u, v) = pano_screen_uv(&p, screen, pt);
            if horizontal {
                v
            } else {
                u
            }
        } else if horizontal {
            y_to_guide_v(r, screen, pt.y)
        } else {
            x_to_guide_u(r, screen, pt.x)
        };
        // Ctrl snaps a NEW guide the same way a drag on an existing one does (10
        // displayed px in 2D, a whole degree in pano) — see `App::snap_guide_coord`.
        if ctx.input(|i| i.modifiers.ctrl) {
            snap_guide_coord(raw, horizontal, r)
        } else {
            raw
        }
    };
    if resp.drag_started() {
        // Start every gesture from a clean slate, so a stale index left by an
        // earlier drag that was interrupted before its drag_stopped (e.g. the
        // ruler vanished on a P-toggle to panorama, or the window shrank) can't be
        // inherited. Only begin the gesture when a guide can actually be added; at
        // the cap guide_spawn stays None, so the drag is inert and never grabs a
        // pre-existing guide.
        state.guide_spawn = None;
        if guides_len < crate::renderer::MAX_GUIDES {
            if let Some(pt) = ctx.pointer_interact_pos() {
                state.guide_spawn = Some(guides_len);
                actions.push(UiAction::AddGuide {
                    coord: coord_at(pt),
                    horizontal,
                });
            }
        }
    }
    if resp.dragged() {
        if let (Some(idx), Some(pt)) = (state.guide_spawn, ctx.pointer_interact_pos()) {
            ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
            actions.push(UiAction::MoveGuide {
                index: idx,
                coord: coord_at(pt),
            });
        }
    }
    if resp.drag_stopped() {
        if let Some(idx) = state.guide_spawn.take() {
            if let Some(pt) = ctx.pointer_interact_pos() {
                let coord = coord_at(pt);
                if !(0.0..=1.0).contains(&coord) || !screen.contains(pt) {
                    actions.push(UiAction::RemoveGuide(idx));
                }
            }
        }
    }
}

/// Draw the navigation minimap's chrome: a subtle black border around the panel
/// and the white current-view outline. The thumbnail underneath is rendered by
/// the GL renderer (so it matches the main view's tone mapping), so this only
/// strokes lines on top — no interactive widget (clicks are routed in the app's
/// input layer, like the guides). Everything fades together via `info.alpha`.
fn minimap(ctx: &egui::Context, inputs: &UiInputs) {
    let Some(info) = &inputs.minimap else {
        return;
    };
    let a = info.alpha.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("imgvwr_minimap"),
    ));

    // Current-view region, clipped to the panel so a box larger than the image, or
    // a wrapped tile / panorama segment, can't spill past the edge. The interior is
    // a faint 5% white fill (a projected triangle mesh, so it's correct for any
    // shape — concave panorama bands, wrapped tiles) under a brighter outline.
    let clipped = painter.with_clip_rect(info.rect);
    let fill_color = egui::Color32::from_white_alpha((a * 13.0).round() as u8);
    let view_stroke = egui::Stroke::new(
        1.0,
        egui::Color32::from_white_alpha((a * 235.0).round() as u8),
    );
    if !info.view_fill.is_empty() {
        let mut mesh = egui::Mesh::default();
        for tri in &info.view_fill {
            let base = mesh.vertices.len() as u32;
            for p in tri {
                mesh.colored_vertex(*p, fill_color);
            }
            mesh.add_triangle(base, base + 1, base + 2);
        }
        clipped.add(egui::Shape::mesh(mesh));
    }
    for seg in &info.view_segments {
        if seg.len() >= 2 {
            clipped.add(egui::Shape::line(seg.clone(), view_stroke));
        }
    }

    // Subtle black border (unclipped so the full stroke shows on the panel edge).
    painter.rect_stroke(
        info.rect,
        egui::CornerRadius::same(2),
        egui::Stroke::new(
            1.0,
            egui::Color32::from_black_alpha((a * 180.0).round() as u8),
        ),
        egui::StrokeKind::Outside,
    );
}

/// The left pixel ruler (2D only). Reveals on its own slide — near the left edge
/// only — and stays while hovered (or mid spawn-drag) so a guide can be dragged
/// off it. Spans from just below the titlebar (when it's showing) to the bottom
/// edge; the bottom panel, drawn afterwards, covers the overlapping bottom-left
/// corner so there's no fixed panel-height assumption.
fn left_ruler(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    let Some(r) = inputs.ruler else {
        state.pointer_over_left_ruler = false;
        return;
    };
    let slide = inputs.left_ruler_slide.clamp(0.0, 1.0);
    if slide <= 0.001 {
        state.pointer_over_left_ruler = false;
        return;
    }
    let screen = ctx.screen_rect();
    let vh = screen.height();
    let ey = |py: f32| screen.top() + vh * (0.5 - (0.5 + r.pan_v - py / r.img_h) / r.sy);
    let uv_y = |sy: f32| 0.5 + r.pan_v - ((1.0 - (sy - screen.top()) / vh) - 0.5) * r.sy;
    let ppx_y = (vh / (r.img_h * r.sy)).abs();

    // Below the titlebar only while it's actually revealed (follows its slide).
    let top = screen.top() + TITLEBAR_H * inputs.titlebar_slide.clamp(0.0, 1.0);
    let base_x = screen.left() - (1.0 - slide) * (RULER_W + 12.0);
    let rect = egui::Rect::from_min_max(
        egui::pos2(base_x, top),
        egui::pos2(base_x + RULER_W, screen.bottom()),
    );
    let resp = egui::Area::new(egui::Id::new("imgvwr_ruler_left"))
        .fixed_pos(rect.min)
        .order(egui::Order::Middle)
        .constrain(false)
        .show(ctx, |ui| {
            let (_, resp) = ui.allocate_exact_size(rect.size(), egui::Sense::click_and_drag());
            let p = ui.painter();
            p.rect_filled(rect, 0.0, panel_bg());
            let stroke = ruler_tick_stroke();
            let draw = |y: f32, major: bool| {
                if y < rect.top() || y > rect.bottom() {
                    return;
                }
                let len = if major { TICK_MAJOR } else { TICK_MINOR };
                p.line_segment([egui::pos2(base_x, y), egui::pos2(base_x + len, y)], stroke);
            };
            if let Some(proj) = r.pano {
                // Latitude degrees along the centre column (ndc.x = 0).
                let angle_at = |sy: f32| -> f32 {
                    let ndc_y = (1.0 - (sy - screen.top()) / vh) * 2.0 - 1.0;
                    let ry = ndc_y * proj.tan_half_fov;
                    let inv = 1.0 / (ry * ry + 1.0).sqrt();
                    let (_, wy, _) = pano_rotate(&proj, 0.0, ry * inv, inv);
                    wy.clamp(-1.0, 1.0).asin().to_degrees()
                };
                degree_ticks(rect.top(), rect.bottom(), angle_at, draw);
            } else {
                let (py0, py1) = (uv_y(rect.top()) * r.img_h, uv_y(rect.bottom()) * r.img_h);
                pixel_ticks(py0, py1, ppx_y, r.img_h, |py, major| draw(ey(py), major));
            }
            resp
        });
    // Keep the ruler alive for the whole spawn-drag, even as the pointer leaves
    // the strip into the image (the drag stays routed to the strip's id).
    state.pointer_over_left_ruler = resp.response.contains_pointer() || resp.inner.dragged();
    // Drag a NEW *vertical* guide out of the (vertical) left ruler — Photoshop
    // style: it tracks the pointer's x as you pull it into the image. A plain
    // click does nothing. Works in 2D (pixel uv) and pano (longitude).
    ruler_spawn_drag(
        ctx,
        &resp.inner,
        &r,
        screen,
        false,
        inputs.guides.len(),
        state,
        actions,
    );
}

/// The bottom pixel ruler, drawn as the top strip of the bottom panel (so they
/// share one background rect — no gap). Ticks point up toward the image; dragging
/// a guide out of it (upward) creates a *horizontal* guide that tracks the
/// pointer. A plain click does nothing.
/// Returns whether the strip is currently being dragged (a guide is being pulled
/// out), so the caller can keep the bottom panel revealed for the whole gesture
/// (else it auto-hides when the pointer leaves the panel and kills the drag).
fn bottom_ruler_strip(
    ui: &mut egui::Ui,
    r: &RulerInfo,
    screen: egui::Rect,
    guides_len: usize,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) -> bool {
    let vw = screen.width();
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(vw, RULER_W), egui::Sense::click_and_drag());
    let base_y = rect.bottom();
    let p = ui.painter();
    let stroke = ruler_tick_stroke();
    let draw = |x: f32, major: bool| {
        let len = if major { TICK_MAJOR } else { TICK_MINOR };
        p.line_segment([egui::pos2(x, base_y), egui::pos2(x, base_y - len)], stroke);
    };
    if let Some(proj) = r.pano {
        // Longitude degrees along the centre row (ndc.y = 0).
        let angle_at = |sx: f32| -> f32 {
            let ndc_x = (sx - screen.left()) / vw * 2.0 - 1.0;
            let rx = ndc_x * proj.aspect * proj.tan_half_fov;
            let inv = 1.0 / (rx * rx + 1.0).sqrt();
            let (wx, _, wz) = pano_rotate(&proj, rx * inv, 0.0, inv);
            wz.atan2(wx).to_degrees()
        };
        degree_ticks(screen.left(), screen.right(), angle_at, draw);
    } else {
        let ex = |px: f32| screen.left() + vw * (0.5 + (px / r.img_w - 0.5 - r.pan_u) / r.sx);
        let uv_x = |sx: f32| 0.5 + r.pan_u + ((sx - screen.left()) / vw - 0.5) * r.sx;
        let ppx_x = (vw / (r.img_w * r.sx)).abs();
        let (px0, px1) = (
            uv_x(screen.left()) * r.img_w,
            uv_x(screen.right()) * r.img_w,
        );
        pixel_ticks(px0, px1, ppx_x, r.img_w, |px, major| draw(ex(px), major));
    }
    // Pull a NEW *horizontal* guide upward out of the bottom ruler (2D + pano).
    ruler_spawn_drag(ui.ctx(), &resp, r, screen, true, guides_len, state, actions);
    resp.dragged()
}

/// How far below the window edge the bottom panel parks when hidden (points).
/// Must exceed the panel's tallest form — pixel ruler (~21) + transport row
/// (~36) + a single-column stack of the four sliders (~4×24 + spacing) — or a
/// sliver of it would still show at slide 0 (and the last slide-out step would
/// pop). ~190 clears the ~179pt worst case with headroom.
const BOTTOM_PARK: f32 = 190.0;

/// Auto-hiding bottom panel (revealed by the cursor near the bottom edge). Sets
/// `state.pointer_over_panel` so the app keeps it up while hovered. Holds the
/// bottom pixel ruler, the playback transport row (only while playing back a
/// sequence), and the image-adjustment sliders.
fn bottom_panel(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    if inputs.bottom_slide <= 0.001 || !inputs.has_image {
        state.pointer_over_panel = false;
        return;
    }
    // A full-width bar anchored to the bottom edge, slid down off-screen at
    // slide 0 (constrain(false) lets it leave the viewport while animating). The
    // bottom ruler and the sliders are drawn into ONE area over a single shared
    // background rect, so there's never a gap between them — nor a wrong-sized one
    // after the sliders wrap and the window grows back.
    let slide = inputs.bottom_slide.clamp(0.0, 1.0);
    let screen = ctx.screen_rect();
    let resp = egui::Area::new(egui::Id::new("imgvwr_bottom"))
        .anchor(
            egui::Align2::LEFT_BOTTOM,
            egui::vec2(0.0, (1.0 - slide) * BOTTOM_PARK),
        )
        .constrain(false)
        .show(ctx, |ui| {
            ui.set_width(screen.width());
            // Background filled in once the content rect is known, so the ruler
            // strip and the sliders sit on exactly one rect.
            let bg = ui.painter().add(egui::Shape::Noop);
            let guides_len = inputs.guides.len();
            let content = ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                // Bottom ruler (2D only) flush at the top of the panel. Returns
                // whether a guide is being dragged out of it, to keep the panel up.
                let strip_dragged = if let Some(r) = &inputs.ruler {
                    bottom_ruler_strip(ui, r, screen, guides_len, state, actions)
                } else {
                    false
                };
                // Playback transport, between the ruler (which stays flush
                // against the image it measures) and the adjustment sliders.
                // Only present while playing back a sequence.
                let transport_hot = match &inputs.playback {
                    Some(pb) => {
                        super::timeline::transport_row(ui, pb, state, actions, screen.width())
                    }
                    None => false,
                };
                let sliders = egui::Frame::NONE
                    .inner_margin(egui::Margin::symmetric(12, 7))
                    .show(ui, |ui| {
                        // Manual row layout: egui's horizontal_wrapped won't wrap
                        // between nested groups, so chunk the atomic slider groups
                        // into rows by available width and centre each row. The
                        // Reset button is a separate bottom-right area (below), so
                        // `field` reserves room for it on the right.
                        const GROUP_W: f32 = 236.0;
                        let avail = (screen.width() - 24.0).max(GROUP_W);
                        ui.set_min_width(avail);
                        let field = (avail - 52.0).max(GROUP_W);
                        let per_row = ((field / GROUP_W).floor() as usize).clamp(1, 4);
                        ui.spacing_mut().item_spacing.y = 4.0;
                        ui.vertical(|ui| {
                            let mut i = 0usize;
                            while i < 4 {
                                let n = per_row.min(4 - i);
                                ui.horizontal(|ui| {
                                    let row_w = n as f32 * GROUP_W;
                                    ui.add_space(((field - row_w) * 0.5).max(0.0));
                                    for _ in 0..n {
                                        ui.allocate_ui(egui::vec2(GROUP_W, 24.0), |ui| match i {
                                            0 => adj_slider(
                                                ui,
                                                "Exposure",
                                                inputs.exposure,
                                                -16.0..=16.0,
                                                0.5,
                                                2,
                                                UiAction::SetExposure,
                                                actions,
                                            ),
                                            1 => adj_slider(
                                                ui,
                                                "Gamma",
                                                inputs.gamma,
                                                0.1..=4.0,
                                                0.1,
                                                2,
                                                UiAction::SetGamma,
                                                actions,
                                            ),
                                            2 => adj_slider(
                                                ui,
                                                "Clarity",
                                                inputs.clarity_amount,
                                                0.0..=10.0,
                                                0.5,
                                                2,
                                                UiAction::SetClarity,
                                                actions,
                                            ),
                                            _ => adj_slider(
                                                ui,
                                                "Radius",
                                                inputs.clarity_radius,
                                                8.0..=256.0,
                                                16.0,
                                                0,
                                                UiAction::SetClarityRadius,
                                                actions,
                                            ),
                                        });
                                        i += 1;
                                    }
                                });
                            }
                        });
                    });
                (strip_dragged || transport_hot, sliders.response.rect)
            });
            let content_rect = content.response.rect;
            let (strip_dragged, slider_rect) = content.inner;
            ui.painter()
                .set(bg, egui::Shape::rect_filled(content_rect, 0.0, panel_bg()));
            // Reset button, pinned to the panel's right edge and vertically aligned
            // with the slider rows. Drawn into the SAME area, on top of the
            // background, so its hit-test is unambiguous; placed independently of
            // the slider group widths (the rows reserve room for it on the right).
            (
                reset_button(ui, content_rect, slider_rect.center().y),
                strip_dragged,
            )
        });
    let (reset_resp, strip_dragged) = resp.inner;
    if reset_resp.clicked() {
        actions.push(UiAction::ResetAdjustments);
    }
    // Keep the panel up for the whole bottom-ruler spawn-drag, even once the
    // pointer leaves the panel into the image (mirrors the left ruler).
    state.pointer_over_panel = resp.response.contains_pointer() || strip_dragged;
}

/// A labelled slider with `−` / `+` step buttons, emitting `make(value)` on any
/// change. Used for the bottom-panel image adjustments.
#[allow(clippy::too_many_arguments)]
fn adj_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    decimals: usize,
    make: fn(f32) -> UiAction,
    actions: &mut Vec<UiAction>,
) {
    let (lo, hi) = (*range.start(), *range.end());
    // One atomic group: the label, ± buttons and slider never wrap apart. The
    // slider is given a FIXED width (add_sized) so the group's measured width
    // matches what it renders — otherwise the slider expands past the measured
    // size and the wrap never triggers (groups overflow instead of wrapping).
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(egui::Color32::from_gray(190)));
        if clickable(ui.small_button("−")).clicked() {
            actions.push(make((value - step).clamp(lo, hi)));
        }
        let mut v = value;
        if ui
            .add_sized(
                [86.0, 18.0],
                egui::Slider::new(&mut v, range).fixed_decimals(decimals),
            )
            .changed()
        {
            actions.push(make(v));
        }
        if clickable(ui.small_button("+")).clicked() {
            actions.push(make((value + step).clamp(lo, hi)));
        }
    });
}

/// The bottom-panel Reset button: a custom-drawn counter-clockwise circular
/// arrow (the thin SVG aliased badly at this size) that resets all image
/// adjustments (same as Ctrl+R). Sized to the adjustment row height and centred
/// vertically on the slider rows (`row_center_y`), pinned to the panel's right
/// edge.
fn reset_button(ui: &mut egui::Ui, area: egui::Rect, row_center_y: f32) -> egui::Response {
    let size = egui::vec2(24.0, 22.0);
    let pad = 11.0;
    let rect = egui::Rect::from_center_size(
        egui::pos2(area.right() - pad - size.x * 0.5, row_center_y),
        size,
    );
    let resp = ui.interact(rect, ui.id().with("reset_btn"), egui::Sense::click());
    // Style it exactly like the other adjustment-row buttons (the +/- steppers):
    // the same per-state widget visuals (fill, border, expansion, fg colour),
    // rather than the old bright white chip.
    let wv = *ui.style().interact(&resp);
    let r = rect.expand(wv.expansion);
    let painter = ui.painter();
    painter.rect(
        r,
        wv.corner_radius,
        wv.weak_bg_fill,
        wv.bg_stroke,
        egui::StrokeKind::Inside,
    );
    draw_reset_glyph(painter, r.center(), 6.3, wv.fg_stroke.color);
    clickable(resp).on_hover_text("Reset all adjustments (Ctrl+R)")
}

/// Draw a crisp counter-clockwise "reset" circular arrow centred at `c`, radius
/// `r` — an anti-aliased arc stroke plus a filled triangular arrowhead, so it
/// stays sharp at small sizes (unlike the hairline SVG).
fn draw_reset_glyph(painter: &egui::Painter, c: egui::Pos2, r: f32, color: egui::Color32) {
    use std::f32::consts::PI;
    let stroke = egui::Stroke::new(1.7, color);
    // ~295° arc, swept counter-clockwise (decreasing angle in egui's y-down
    // space); the open wedge (top-right) carries the arrowhead.
    let (a0, a1) = (1.42 * PI, -0.22 * PI);
    let n = 40;
    let pts: Vec<egui::Pos2> = (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * (i as f32 / n as f32);
            egui::pos2(c.x + r * a.cos(), c.y + r * a.sin())
        })
        .collect();
    painter.add(egui::Shape::line(pts.clone(), stroke));
    // Arrowhead at the arc end, pointing along its travel direction.
    let tip = pts[n];
    let fwd = (pts[n] - pts[n - 1]).normalized();
    let perp = egui::vec2(-fwd.y, fwd.x);
    let (hl, hw) = (r * 1.05, r * 0.72);
    let apex = tip + fwd * (hl * 0.45);
    let base = tip - fwd * (hl * 0.55);
    painter.add(egui::Shape::convex_polygon(
        vec![apex, base + perp * hw, base - perp * hw],
        color,
        egui::Stroke::NONE,
    ));
}

/// Fixed content width of the settings dialog (px). The app grows the OS window
/// to comfortably fit this when Settings opens, so controls never wrap.
const SETTINGS_WIDTH: f32 = 440.0;

/// A bold section heading with an underline, used to group the settings dialog.
fn settings_heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(text)
            .strong()
            .color(egui::Color32::from_gray(205))
            .size(13.0),
    );
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(4.0);
}

/// The settings dialog (opened from the toolbar gear). Organised into labelled
/// sections; each row is a left-aligned label and a right-aligned control via a
/// two-column grid, scrollable so it never overflows the window. The app grows
/// the window to [`SETTINGS_WIDTH`]+ when this opens (see `App::sync_settings_window`).
fn settings_dialog(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    if !state.show_settings {
        return;
    }
    let max_h = (ctx.screen_rect().height() - 80.0).max(200.0);
    egui::Area::new(egui::Id::new("imgvwr_settings"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                ui.set_width(SETTINGS_WIDTH);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Settings")
                            .strong()
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if clickable(ui.button("Close")).clicked() {
                            state.show_settings = false;
                            state.confirm_default = false;
                        }
                    });
                });
                // Version line, with a lime-green "update available" link appended
                // when the daily check found a newer release (see App::poll_update).
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.label(
                        egui::RichText::new(concat!("imgvwr v", env!("CARGO_PKG_VERSION")))
                            .color(egui::Color32::from_gray(150))
                            .size(12.0),
                    );
                    if let Some((ver, url)) = &inputs.available_update {
                        ui.label(
                            egui::RichText::new("·")
                                .color(egui::Color32::from_gray(110))
                                .size(12.0),
                        );
                        // Bright lime, to draw the eye to the available update.
                        let lime = egui::Color32::from_rgb(140, 240, 80);
                        clickable(
                            ui.hyperlink_to(
                                egui::RichText::new(format!("{ver} available"))
                                    .color(lime)
                                    .size(12.0),
                                url,
                            ),
                        )
                        .on_hover_text(url);
                    }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false, true])
                    .max_height(max_h)
                    .show(ui, |ui| {
                        ui.set_width(SETTINGS_WIDTH);

                        // ── Display & view ──────────────────────────────────
                        settings_heading(ui, "Display & view");
                        egui::Grid::new("set_display")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .min_col_width(170.0)
                            .show(ui, |ui| {
                                if inputs.ocio_available {
                                    ui.label("Default view transform");
                                    default_view_combo(ui, inputs, actions);
                                    ui.end_row();
                                }
                                ui.label("Open on display");
                                startup_display_combo(ui, inputs, actions);
                                ui.end_row();
                            });

                        // ── Appearance ──────────────────────────────────────
                        settings_heading(ui, "Appearance");
                        egui::Grid::new("set_appearance")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .min_col_width(170.0)
                            .show(ui, |ui| {
                                ui.label("Window corner radius");
                                let mut radius = inputs.corner_radius;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut radius)
                                            .range(0..=40)
                                            .suffix(" px"),
                                    )
                                    .changed()
                                {
                                    actions.push(UiAction::SetCornerRadius(radius));
                                }
                                ui.end_row();

                                ui.label("Background colour");
                                let mut bg = inputs.background_color;
                                if ui.color_edit_button_srgb(&mut bg).changed() {
                                    actions.push(UiAction::SetBackgroundColor(bg));
                                }
                                ui.end_row();

                                ui.label("Guide colour");
                                let mut gc = inputs.guide_color;
                                if ui.color_edit_button_srgb(&mut gc).changed() {
                                    actions.push(UiAction::SetGuideColor(gc));
                                }
                                ui.end_row();

                                ui.label("Titlebar");
                                let mut pin = inputs.pin_titlebar;
                                if ui
                                    .checkbox(&mut pin, "Always show")
                                    .on_hover_text(
                                        "Keep the titlebar permanently revealed instead of \
                                         auto-hiding at the top edge.",
                                    )
                                    .changed()
                                {
                                    actions.push(UiAction::SetPinTitlebar(pin));
                                }
                                ui.end_row();
                            });

                        // ── On open (per-image-type exposure) ───────────────
                        settings_heading(ui, "On open");
                        let mut auto = inputs.auto_exposure;
                        if ui
                            .checkbox(&mut auto, "Auto-expose HDR panoramas")
                            .changed()
                        {
                            actions.push(UiAction::SetAutoExposure(auto));
                        }
                        ui.add_space(4.0);
                        let mut raw_auto = inputs.raw_auto_exposure;
                        if ui
                            .checkbox(&mut raw_auto, "Auto-expose RAW photos")
                            .changed()
                        {
                            actions.push(UiAction::SetRawAutoExposure(raw_auto));
                        }

                        // ── Review tools ────────────────────────────────────
                        settings_heading(ui, "Review tools");
                        egui::Grid::new("set_review")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .min_col_width(170.0)
                            .show(ui, |ui| {
                                ui.label("Clipping overlay margin");
                                let mut pct = inputs.clip_margin * 100.0;
                                if ui
                                    .add(
                                        egui::Slider::new(&mut pct, 0.0..=5.0)
                                            .suffix("%")
                                            .fixed_decimals(1),
                                    )
                                    .changed()
                                {
                                    actions.push(UiAction::SetClipMargin(pct / 100.0));
                                }
                                ui.end_row();
                            });

                        // ── Playback ────────────────────────────────────────
                        settings_heading(ui, "Playback");
                        egui::Grid::new("set_playback")
                            .num_columns(2)
                            .spacing([16.0, 10.0])
                            .min_col_width(170.0)
                            .show(ui, |ui| {
                                ui.label("Frame cache").on_hover_text(
                                    "Share of system memory the sequence frame cache may \
                                         use. A sequence that does not fit still plays — it \
                                         simply decodes on demand, and the timeline's cache bar \
                                         shows how much is held.",
                                );
                                let mut pct = inputs.playback_cache_percent;
                                if ui
                                    .add(
                                        egui::Slider::new(&mut pct, 5..=80)
                                            .suffix("% of RAM")
                                            .custom_formatter(|v, _| {
                                                let gb = inputs.total_memory_gb * v as f32 / 100.0;
                                                format!("{v:.0}%  ({gb:.1} GB)")
                                            }),
                                    )
                                    .changed()
                                {
                                    actions.push(UiAction::SetPlaybackCachePercent(pct));
                                }
                                ui.end_row();
                            });

                        // ── Performance ─────────────────────────────────────
                        settings_heading(ui, "Performance");
                        let mut half = inputs.half_float_textures;
                        if ui
                            .checkbox(&mut half, "Store 32-bit float as 16-bit (less GPU memory)")
                            .on_hover_text(
                                "Halves the VRAM used by HDR / EXR / float images, \
                                 with a small precision trade-off. Helps very large \
                                 images fit; applies to images opened afterwards.",
                            )
                            .changed()
                        {
                            actions.push(UiAction::SetHalfFloat(half));
                        }

                        // ── System ──────────────────────────────────────────
                        settings_heading(ui, "System");
                        if state.confirm_default {
                            ui.label(
                                "Make imgvwr the default viewer for all supported image types?",
                            );
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if clickable(ui.button("Confirm")).clicked() {
                                    actions.push(UiAction::SetDefaultApp);
                                    state.confirm_default = false;
                                }
                                if clickable(ui.button("Cancel")).clicked() {
                                    state.confirm_default = false;
                                }
                            });
                        } else if clickable(ui.button("⭐  Set as default viewer")).clicked() {
                            state.confirm_default = true;
                        }
                        ui.add_space(6.0);
                    });
            });
        });
}

/// The Default View Transform dropdown — lists the views available on the active
/// (or first) OCIO display; the stored name is matched case-insensitively.
fn default_view_combo(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    let display = inputs
        .active
        .as_ref()
        .map(|(d, _)| d.clone())
        .or_else(|| inputs.displays().into_iter().next());
    let views = display
        .as_ref()
        .map(|d| inputs.views_for(d))
        .unwrap_or_default();
    egui::ComboBox::from_id_salt("imgvwr_default_view")
        .selected_text(&inputs.default_view_transform)
        .width(200.0)
        .show_ui(ui, |ui| {
            for v in &views {
                let selected = v.eq_ignore_ascii_case(&inputs.default_view_transform);
                if ui.selectable_label(selected, v).clicked() {
                    actions.push(UiAction::SetDefaultView(v.clone()));
                }
            }
        });
}

/// The "Open on display" startup-monitor dropdown.
fn startup_display_combo(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    let current = match &inputs.startup_display {
        None => "Remember last used".to_string(),
        Some(name) => inputs
            .monitors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| l.clone())
            .unwrap_or_else(|| name.clone()),
    };
    egui::ComboBox::from_id_salt("imgvwr_startup_display")
        .selected_text(current)
        .width(200.0)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(inputs.startup_display.is_none(), "Remember last used")
                .clicked()
            {
                actions.push(UiAction::SetStartupDisplay(None));
            }
            for (name, label) in &inputs.monitors {
                let selected = inputs.startup_display.as_deref() == Some(name);
                if ui.selectable_label(selected, label).clicked() {
                    actions.push(UiAction::SetStartupDisplay(Some(name.clone())));
                }
            }
        });
}

/// Which optional titlebar elements are shown at the current width (close and the
/// filename are always present, so they're not listed here).
struct TitlebarFit {
    minimize: bool,
    settings: bool,
    maximize: bool,
    app_icon: bool,
    open: bool,
}

/// Decide which optional titlebar elements fit, given the window width and the
/// (accurately measured) filename width. Elements are dropped lowest-priority
/// first; the full priority, highest kept the longest → lowest dropped first, is:
/// close, filename, minimize, settings, maximize, app-icon, open. Close is always
/// shown and the filename is elastic (it truncates with an ellipsis), and the
/// filename outranks every button — so a long name pushes the buttons out before
/// it begins to shrink. Strict priority: a lower-priority element is never shown
/// while a higher-priority one is hidden.
fn titlebar_fit(width: f32, name_text_w: f32) -> TitlebarFit {
    const NAME_PAD: f32 = 12.0; // 6px left inset + a little gap before the next item
                                // Space left after reserving close + the filename's full width. Each button is
                                // then granted in priority order only if it still fits, and once one doesn't,
                                // every lower-priority one is dropped too (the `&&` chain).
    let mut rem = width - TB_BTN_W - (name_text_w + NAME_PAD);
    let minimize = rem >= TB_BTN_W;
    if minimize {
        rem -= TB_BTN_W;
    }
    let settings = minimize && rem >= TB_BTN_W;
    if settings {
        rem -= TB_BTN_W;
    }
    let maximize = settings && rem >= TB_BTN_W;
    if maximize {
        rem -= TB_BTN_W;
    }
    let app_icon = maximize && rem >= TB_ICON_W;
    if app_icon {
        rem -= TB_ICON_W;
    }
    let open = app_icon && rem >= TB_BTN_W;
    TitlebarFit {
        minimize,
        settings,
        maximize,
        app_icon,
        open,
    }
}

/// The auto-hiding borderless titlebar: a drag strip showing the filename, plus
/// minimize / maximize / close controls. Opacity is `inputs.titlebar_slide`
/// (eased by the cursor entering/leaving the window). Dragging the strip moves
/// the window (OS loop → Aero Snap); double-clicking it toggles fullscreen.
/// Elements are selectively hidden by priority when the window is too narrow for
/// all of them (see [`titlebar_fit`]).
fn titlebar(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    let slide = inputs.titlebar_slide.clamp(0.0, 1.0);
    if slide <= 0.001 {
        return;
    }
    // Full opacity: the reveal is a slide down from the top edge, not a fade.
    let a = 1.0_f32;
    let alpha = |c: u8| (c as f32 * a) as u8;
    let bg = panel_bg_alpha(PANEL_ALPHA);
    let fg = egui::Color32::from_rgba_unmultiplied(220, 220, 220, 255);

    let screen = ctx.screen_rect();
    let y = screen.top() - (1.0 - slide) * TITLEBAR_H;

    // Filename (the drag-strip label), measured with the exact font it is drawn in
    // so the overflow logic below is pixel-accurate.
    let name = if inputs.title.is_empty() {
        "imgvwr"
    } else {
        inputs.title.as_str()
    };
    let name_font = egui::FontId::proportional(13.0);
    let name_text_w = ctx.fonts(|f| {
        f.layout_no_wrap(name.to_owned(), name_font.clone(), fg)
            .size()
            .x
    });

    // Selectively hide elements when the window is too narrow for all of them.
    let TitlebarFit {
        minimize: show_minimize,
        settings: show_settings,
        maximize: show_maximize,
        app_icon: show_app_icon,
        open: show_open,
    } = titlebar_fit(screen.width(), name_text_w);

    egui::Area::new(egui::Id::new("imgvwr_titlebar"))
        .fixed_pos(egui::pos2(screen.left(), y))
        .constrain(false)
        .show(ctx, |ui| {
            ui.set_width(screen.width());
            egui::Frame::NONE.fill(bg).show(ui, |ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(screen.width(), TITLEBAR_H),
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        // Pack elements edge-to-edge: the overflow reservation above
                        // assumes exact button widths, so any inter-widget spacing
                        // would eat into the filename's reserved space and truncate
                        // it even when a button was hidden to make room. (Inherited
                        // by the nested left-to-right group below.)
                        ui.spacing_mut().item_spacing.x = 0.0;
                        // Window controls (laid out right-to-left): close, maximize,
                        // min, then settings — Bootstrap SVG icons. All but close are
                        // dropped first when space is tight (see the priority above).
                        if titlebar_button(
                            ui,
                            egui::include_image!("../../resources/icons/ui/x-lg.svg"),
                            14.0,
                            a,
                            true,
                        )
                        .clicked()
                        {
                            actions.push(UiAction::Close);
                        }
                        if show_maximize {
                            let max_icon = if inputs.is_maximized {
                                egui::include_image!("../../resources/icons/ui/window-stack.svg")
                            } else {
                                egui::include_image!("../../resources/icons/ui/square.svg")
                            };
                            if titlebar_button(ui, max_icon, 13.0, a, false).clicked() {
                                actions.push(UiAction::ToggleMaximize);
                            }
                        }
                        if show_minimize
                            && titlebar_button(
                                ui,
                                egui::include_image!("../../resources/icons/ui/dash-lg.svg"),
                                16.0,
                                a,
                                false,
                            )
                            .clicked()
                        {
                            actions.push(UiAction::Minimize);
                        }
                        if show_settings
                            && titlebar_button(
                                ui,
                                // Filled gear: renders bolder/crisper at this size
                                // than the thin-stroked outline gear (like Open).
                                egui::include_image!("../../resources/icons/ui/gear-fill.svg"),
                                15.0,
                                a,
                                false,
                            )
                            .clicked()
                        {
                            actions.push(UiAction::OpenSettings);
                        }
                        // The remaining strip, laid out left-to-right: app icon, an
                        // icon-only Open button, then the filename over a drag region
                        // (move / double-click fullscreen).
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            if show_app_icon {
                                if let Some(icon) = &inputs.icon {
                                    let (icon_rect, _) = ui.allocate_exact_size(
                                        egui::vec2(TB_ICON_W, TITLEBAR_H),
                                        egui::Sense::hover(),
                                    );
                                    let sz = 18.0;
                                    let img_rect = egui::Rect::from_center_size(
                                        icon_rect.center() + egui::vec2(2.0, 0.0),
                                        egui::vec2(sz, sz),
                                    );
                                    ui.painter().image(
                                        icon.id(),
                                        img_rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::from_white_alpha(alpha(255)),
                                    );
                                }
                            }
                            if show_open
                                && titlebar_button(
                                    ui,
                                    // Filled folder: renders bold/crisp at this size,
                                    // unlike the thin-stroked outline open-folder icon.
                                    egui::include_image!(
                                        "../../resources/icons/ui/folder-fill.svg"
                                    ),
                                    15.0,
                                    a,
                                    false,
                                )
                                .on_hover_text("Open file…")
                                .clicked()
                            {
                                actions.push(UiAction::OpenFile);
                            }
                            // Filename over the remaining drag strip, truncated with
                            // an ellipsis to the space that's actually left (so it
                            // never overruns the window controls).
                            let rect = ui.available_rect_before_wrap();
                            let drag = ui.interact(
                                rect,
                                ui.id().with("titlebar_drag"),
                                egui::Sense::click_and_drag(),
                            );
                            if drag.drag_started() {
                                actions.push(UiAction::DragWindow);
                            }
                            if drag.double_clicked() {
                                actions.push(UiAction::ToggleFullscreen);
                            }
                            let mut job = egui::text::LayoutJob::single_section(
                                name.to_owned(),
                                egui::TextFormat {
                                    font_id: name_font.clone(),
                                    color: fg,
                                    ..Default::default()
                                },
                            );
                            job.wrap = egui::text::TextWrapping {
                                max_width: (rect.width() - 6.0).max(0.0),
                                max_rows: 1,
                                break_anywhere: true,
                                overflow_character: Some('…'),
                            };
                            let galley = ui.fonts(|f| f.layout_job(job));
                            let pos = egui::pos2(
                                rect.left() + 6.0,
                                rect.center().y - galley.size().y * 0.5,
                            );
                            ui.painter().galley(pos, galley, fg);
                        });
                    },
                );
            });
        });
}

/// A single titlebar control with a Bootstrap SVG icon (`icon_px` square) and a
/// hover highlight (red for Close).
fn titlebar_button(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'static>,
    icon_px: f32,
    a: f32,
    danger: bool,
) -> egui::Response {
    let btn = egui::vec2(TB_BTN_W, ui.available_height());
    let (rect, resp) = ui.allocate_exact_size(btn, egui::Sense::click());
    if resp.hovered() {
        let hover = if danger {
            egui::Color32::from_rgba_unmultiplied(232, 17, 35, (255.0 * a) as u8)
        } else {
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, (38.0 * a) as u8)
        };
        ui.painter().rect_filled(rect, 0.0, hover);
    }
    let fg = egui::Color32::from_rgba_unmultiplied(230, 230, 230, (255.0 * a) as u8);
    let icon_rect = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(icon_px));
    egui::Image::new(icon).tint(fg).paint_at(ui, icon_rect);
    super::clickable(resp)
}

/// Small numbered flags hanging from the right edge for saved comparator slots,
/// stacked and vertically centred on that edge; the active (currently-viewed)
/// slot is filled with the accent colour. Each flag is clickable (recall the
/// slot) and shows its filename on hover.
fn slot_flags(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    if inputs.slot_labels.iter().all(|s| s.is_none()) {
        return;
    }
    egui::Area::new(egui::Id::new("imgvwr_slots"))
        // Centred vertically on the right edge (clear of the top-right metadata
        // box, which can then sit closer to the edge). constrain(false) so it sits
        // flush against the edge rather than being nudged inward by egui.
        .anchor(egui::Align2::RIGHT_CENTER, egui::Vec2::ZERO)
        .constrain(false)
        .show(ctx, |ui| {
            // Left-aligned column (Align::Min) so the Area hugs its content. A
            // right-aligned (Align::Max) column instead claims the *full* available
            // width to find a right edge to align against — ballooning this
            // RIGHT_CENTER Area across the whole window, whose interact rect then
            // swallows every click in that horizontal band (the dead-zone bug).
            // To still pin the narrow "1" flag flush to the edge, give every flag
            // the same width: size each cell to the widest digit and centre it.
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 3.0);
                let cell_w = (1..=9)
                    .map(|n| {
                        ui.fonts(|f| {
                            f.layout_no_wrap(
                                n.to_string(),
                                egui::FontId::proportional(13.0),
                                egui::Color32::WHITE,
                            )
                            .size()
                            .x
                        })
                    })
                    .fold(0.0_f32, f32::max);
                for (i, label) in inputs.slot_labels.iter().enumerate() {
                    let Some(label) = label else {
                        continue;
                    };
                    // Highlight the displayed slot AND (when diffing) the slot
                    // being compared against, so both ends of a diff stand out.
                    let active = inputs.active_slot == Some(i) || inputs.diff_slot == Some(i);
                    let (fill, fg) = if active {
                        (ACCENT, egui::Color32::WHITE)
                    } else {
                        (panel_bg(), egui::Color32::from_gray(200))
                    };
                    // Square right corners so the flag reads as hanging from the
                    // right edge.
                    let inner = egui::Frame {
                        fill,
                        inner_margin: egui::Margin::symmetric(7, 3),
                        corner_radius: egui::CornerRadius {
                            nw: 3,
                            ne: 0,
                            sw: 3,
                            se: 0,
                        },
                        ..Default::default()
                    }
                    .show(ui, |ui| {
                        ui.set_min_width(cell_w);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new((i + 1).to_string())
                                    .color(fg)
                                    .size(13.0)
                                    .strong(),
                            );
                        });
                    });
                    let resp = clickable(inner.response.interact(egui::Sense::click()))
                        .on_hover_text(label);
                    if resp.clicked() {
                        actions.push(UiAction::RecallSlot(i));
                    }
                }
            });
        });
}

/// Width of the loading box / progress bar (30% wider than the original 190).
const LOADING_W: f32 = 247.0;

fn loading(ctx: &egui::Context, inputs: &UiInputs) {
    // Keep the bar animating (the barber-pole scrolls every frame).
    ctx.request_repaint();
    let name = inputs.loading_name.as_deref().unwrap_or("Loading");
    let right = match inputs.progress {
        Some(p) => format!("{}%", (p * 100.0).round() as i32),
        None => "…".to_string(),
    };
    let text = |s: String| {
        egui::RichText::new(s)
            .color(egui::Color32::WHITE)
            .size(13.0)
    };
    egui::Area::new(egui::Id::new("imgvwr_loading"))
        // Centred along the bottom edge.
        .anchor(egui::Align2::CENTER_BOTTOM, egui::Vec2::new(0.0, -12.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame {
                fill: panel_bg(),
                inner_margin: egui::Margin::symmetric(10, 8),
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            }
            .show(ui, |ui| {
                ui.set_width(LOADING_W);
                // Filename on the left, percentage flush to the right edge (the
                // percentage is reserved first so a long name can't push it off).
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(text(right));
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            ui.add(egui::Label::new(text(name.to_string())).truncate());
                        });
                    });
                });
                ui.add_space(4.0);
                progress_bar(ui, inputs.progress);
            });
        });
}

/// Colours cycled by the progress bar's diagonal "barber-pole" bands.
const BARBER_COLORS: [egui::Color32; 4] = [
    egui::Color32::from_rgb(190, 111, 255),
    egui::Color32::from_rgb(65, 187, 217),
    egui::Color32::from_rgb(243, 130, 55),
    egui::Color32::from_rgb(161, 208, 77),
];

/// A progress bar filled with scrolling diagonal "barber-pole" bands cycling
/// [`BARBER_COLORS`]. `Some(p)` fills `p` of the width; `None` fills the whole bar
/// to signal indeterminate work.
fn progress_bar(ui: &mut egui::Ui, value: Option<f32>) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(LOADING_W, 12.0), egui::Sense::hover());
    let painter = ui.painter().with_clip_rect(rect);
    let radius = egui::CornerRadius::same(3);
    // Track.
    painter.rect_filled(
        rect,
        radius,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 28),
    );
    // Filled portion: the whole bar when indeterminate, else the value fraction.
    // The accent base under the bands keeps the rounded corners reading cleanly.
    let mut fill = rect;
    if let Some(p) = value {
        fill.set_width(rect.width() * p.clamp(0.0, 1.0));
    }
    painter.rect_filled(fill, radius, ACCENT);
    // Abutting diagonal colour bands tiling the filled portion, scrolling with
    // time so the colours travel along the bar. Clipped to the fill.
    if fill.width() > 0.5 {
        let sp = painter.with_clip_rect(fill);
        let h = rect.height();
        const BAND: f32 = 13.0;
        const SPEED: f32 = 64.0;
        // Negative time → bands travel right-to-left (against the bar's own
        // left-to-right fill), which reads as faster. rem_euclid/floor below
        // stay continuous for negative `t`.
        let t = -(ui.input(|i| i.time) as f32) * SPEED;
        let scroll = (t / BAND).floor() as i64;
        let phase = t.rem_euclid(BAND);
        let mut x = fill.left() - h - BAND + phase;
        let mut k = 0i64;
        while x < fill.right() + BAND {
            let c = BARBER_COLORS[((k - scroll).rem_euclid(4)) as usize];
            let pts = vec![
                egui::pos2(x, fill.bottom()),
                egui::pos2(x + h, fill.top()),
                egui::pos2(x + h + BAND, fill.top()),
                egui::pos2(x + BAND, fill.bottom()),
            ];
            sp.add(egui::Shape::convex_polygon(pts, c, egui::Stroke::NONE));
            x += BAND;
            k += 1;
        }
    }
}

fn error_panel(ctx: &egui::Context, inputs: &UiInputs, error: &str, actions: &mut Vec<UiAction>) {
    egui::Area::new(egui::Id::new("imgvwr_error"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: egui::Color32::from_rgba_unmultiplied(60, 20, 20, 235),
                inner_margin: egui::Margin::same(16),
                corner_radius: egui::CornerRadius::same(8),
                ..Default::default()
            };
            frame.show(ui, |ui| {
                ui.set_max_width(560.0);
                ui.vertical_centered(|ui| {
                    let title = match &inputs.loading_name {
                        Some(name) => format!("Failed to open {name}"),
                        None => "Failed to open file".to_string(),
                    };
                    ui.label(
                        egui::RichText::new(title)
                            .strong()
                            .color(egui::Color32::from_rgb(255, 180, 180)),
                    );
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(error).color(egui::Color32::from_gray(220)));
                    ui.add_space(10.0);
                    if clickable(ui.button("Dismiss")).clicked() {
                        actions.push(UiAction::DismissError);
                    }
                });
            });
        });
}

/// The Delete-key "delete this file?" confirmation — same centred-panel shape as
/// [`error_panel`], with an explicit Delete / Cancel pair rather than a single
/// dismiss (a destructive action shouldn't be a single accidental keystroke away).
fn delete_confirm_dialog(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    egui::Area::new(egui::Id::new("imgvwr_delete_confirm"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: egui::Color32::from_rgba_unmultiplied(60, 20, 20, 235),
                inner_margin: egui::Margin::same(16),
                corner_radius: egui::CornerRadius::same(8),
                ..Default::default()
            };
            frame.show(ui, |ui| {
                ui.set_max_width(420.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Delete this file?")
                            .strong()
                            .color(egui::Color32::from_rgb(255, 180, 180)),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(&inputs.title).color(egui::Color32::from_gray(220)),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("This permanently removes it from disk.")
                            .size(12.0)
                            .color(egui::Color32::from_gray(160)),
                    );
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        // `horizontal` reports its available (not shrunk-to-content)
                        // width to the enclosing `vertical_centered`, so without an
                        // explicit nudge the pair hugs the left edge instead of
                        // sitting under the centred text above it.
                        ui.add_space(((ui.available_width() - 130.0) / 2.0).max(0.0));
                        if clickable(ui.button("Delete")).clicked() {
                            actions.push(UiAction::DeleteCurrentFile);
                            state.confirm_delete = false;
                        }
                        if clickable(ui.button("Cancel")).clicked() {
                            state.confirm_delete = false;
                        }
                    });
                });
            });
        });
}

fn hint(ctx: &egui::Context) {
    egui::Area::new(egui::Id::new("imgvwr_hint"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .interactable(false)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("Drop an image here, or press O to open one")
                    .color(egui::Color32::WHITE)
                    .size(18.0),
            );
        });
}

/// Draw the F2 metadata box (top-right, below the slot flags). Returns whether
/// it should stay revealed — the pointer is over it, or its View dropdown menu is
/// open (so navigating into the menu doesn't dismiss the box and close the menu).
fn metadata_hud(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) -> bool {
    // Tucked close to the right edge (the slot flags now sit at the vertical
    // middle, so the box no longer has to clear a top-right flag column).
    // Slides in from the right edge: at slide 0 it's pushed fully off-screen.
    let slide = inputs.metadata_slide.clamp(0.0, 1.0);
    let off_x = -8.0 + (1.0 - slide) * 360.0;
    let mut view_menu_open = false;
    let resp = egui::Area::new(egui::Id::new("imgvwr_metadata"))
        .anchor(
            egui::Align2::RIGHT_TOP,
            egui::Vec2::new(off_x, TITLEBAR_H + 4.0),
        )
        .constrain(false)
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: panel_bg(),
                inner_margin: egui::Margin::same(8),
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            };
            frame.show(ui, |ui| {
                // Fixed content width, so the box no longer grows and shrinks
                // with whatever the current file's metadata happens to say — and
                // so the histogram's one-point-per-bin geometry always fits.
                ui.set_width(HIST_ROW_W);
                egui::Grid::new("imgvwr_metadata_grid")
                    .num_columns(2)
                    // Pin the key column, so the value column's width — and so
                    // where the text wraps — doesn't shift as metadata changes.
                    .min_col_width(88.0)
                    .max_col_width(HIST_ROW_W - 100.0)
                    .spacing([12.0, 2.0])
                    .show(ui, |ui| {
                        for (key, value) in &inputs.metadata {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(key).color(egui::Color32::from_gray(150)),
                                )
                                .selectable(true),
                            );
                            // Wrapped, not truncated: the box is a fixed width
                            // now, and a long filename is exactly the value you'd
                            // want to read in full (and to be able to select).
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(value).color(egui::Color32::WHITE),
                                )
                                .selectable(true)
                                .wrap(),
                            );
                            ui.end_row();
                        }
                        // Colour management: a Display picker (only when the config
                        // exposes more than one display) and a View picker whose
                        // contents follow the active display — two single-level
                        // dropdowns, no rightward sub-menus. Falls back to a plain
                        // label when OCIO isn't available.
                        let meta_label = |ui: &mut egui::Ui, text: &str| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(text).color(egui::Color32::from_gray(150)),
                                )
                                .selectable(false),
                            );
                        };
                        if inputs.has_image {
                            if !inputs.ocio_available {
                                meta_label(ui, "View");
                                ui.label(
                                    egui::RichText::new("gamma 2.2").color(egui::Color32::WHITE),
                                );
                                ui.end_row();
                            } else {
                                if inputs.displays().len() > 1 {
                                    meta_label(ui, "Display");
                                    view_menu_open |= display_dropdown(ui, inputs, actions);
                                    ui.end_row();
                                }
                                meta_label(ui, "View");
                                view_menu_open |= view_dropdown(ui, inputs, actions);
                                ui.end_row();
                            }
                        }
                        // Channels: a clickable colour box per channel that
                        // isolates it as greyscale (click again to show all).
                        if inputs.channel_count > 0 {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new("Channels")
                                        .color(egui::Color32::from_gray(150)),
                                )
                                .selectable(false),
                            );
                            ui.horizontal(|ui| {
                                for (label, color, idx) in channel_boxes(inputs.channel_count) {
                                    channel_box(ui, label, color, idx, inputs, actions);
                                }
                            });
                            ui.end_row();
                        }
                        // Guides: list each active guide with a remove button.
                        if !inputs.guides.is_empty() {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new("Guides")
                                        .color(egui::Color32::from_gray(150)),
                                )
                                .selectable(false),
                            );
                            ui.vertical(|ui| {
                                let (iw, ih) = inputs.image_size;
                                let pano = inputs.ruler.and_then(|r| r.pano).is_some();
                                // `V 425px` (vertical → x pixel) / `H 312px`
                                // (horizontal → y pixel), plus degrees in panorama
                                // mode; remove button on the right. Sorted
                                // alphabetically (H before V, then by pixel), keeping
                                // each row's original guide index for removal.
                                let mut rows: Vec<(usize, &str, i64, String)> = inputs
                                    .guides
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &g)| {
                                        let (axis, dim) =
                                            if g[1] >= 0.5 { ("H", ih) } else { ("V", iw) };
                                        let px = (g[0] * dim as f32).round() as i64;
                                        (i, axis, px, guide_readout(g, inputs.image_size, pano))
                                    })
                                    .collect();
                                rows.sort_by(|a, b| a.1.cmp(b.1).then(a.2.cmp(&b.2)));
                                for (i, _, _, text) in rows {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(text)
                                                .color(egui::Color32::WHITE)
                                                .size(12.0),
                                        );
                                        if clickable(ui.small_button("×")).clicked() {
                                            actions.push(UiAction::RemoveGuide(i));
                                        }
                                    });
                                }
                            });
                            ui.end_row();
                        }
                    });
                // Full width under the grid rather than inside it — a grid cell
                // would pin the graph to the narrow value column.
                if inputs.has_image {
                    histogram_plot(ui, inputs, state, actions);
                }
            });
        });
    state.view_menu_open = view_menu_open;
    // A levels drag routinely wanders off the handle strip and out of the
    // box; without this the box would slide away mid-gesture, taking the graph
    // being aimed at with it.
    resp.response.contains_pointer() || view_menu_open || state.levels_drag.is_some()
}

/// A guide's angle in panorama mode, from its stored image uv coord — the direct
/// inverse of the shader's equirect uv mapping (`u = 1 - (lon/TAU + 0.5)`,
/// `v = 0.5 - lat/PI`, see `direction_to_equirect_uv`). Horizontal guides
/// (constant v) read latitude: 0° at centre, +90° top, -90° bottom. Vertical
/// guides (constant u) read longitude: 0..360° left→right.
fn guide_degrees(g: [f32; 2]) -> f32 {
    if g[1] >= 0.5 {
        (0.5 - g[0]) * 180.0
    } else {
        g[0].rem_euclid(1.0) * 360.0
    }
}

/// A guide's readout text — `H 312px` / `V 425px`, plus degrees in panorama mode —
/// shared by the F2 box's guide list and the cursor tooltip.
fn guide_readout(g: [f32; 2], image_size: (u32, u32), pano: bool) -> String {
    let (iw, ih) = image_size;
    let (axis, dim) = if g[1] >= 0.5 { ("H", ih) } else { ("V", iw) };
    let px = (g[0] * dim as f32).round() as i64;
    if pano {
        format!("{axis} {px}px  {:.2}°", guide_degrees(g))
    } else {
        format!("{axis} {px}px")
    }
}

/// A colour-coded box hovering just above-right of the cursor while a guide is
/// hovered/grabbed (blue, [`GUIDE_HOVER`]) or being dragged out of a ruler (green,
/// [`GUIDE_ADD`]), showing its pixel (and, in panorama, degree) coordinate — so the
/// value is visible without opening the F2 metadata box.
fn guide_tooltip(ctx: &egui::Context, inputs: &UiInputs, state: &UiState) {
    let (idx, color) = if let Some(idx) = state.guide_spawn {
        (idx, GUIDE_ADD)
    } else if let Some(idx) = state.hovered_guide {
        (idx, GUIDE_HOVER)
    } else {
        return;
    };
    let (Some(&g), Some(pos)) = (inputs.guides.get(idx), ctx.pointer_hover_pos()) else {
        return;
    };
    let pano = inputs.ruler.and_then(|r| r.pano).is_some();
    let text = guide_readout(g, inputs.image_size, pano);

    const FONT_SIZE: f32 = 13.0;
    let text_color = egui::Color32::from_gray(20);
    let font = egui::FontId::proportional(FONT_SIZE);
    let galley = ctx.fonts(|f| f.layout_no_wrap(text.clone(), font, text_color));
    let size = galley.size() + egui::vec2(16.0, 8.0);
    // Hover to the top-right of the cursor.
    let tip_pos = egui::pos2(pos.x + 16.0, pos.y - size.y - 16.0);

    egui::Area::new(egui::Id::new("imgvwr_guide_tooltip"))
        .fixed_pos(tip_pos)
        .order(egui::Order::Tooltip)
        .interactable(false)
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: color,
                inner_margin: egui::Margin::symmetric(8, 4),
                corner_radius: egui::CornerRadius::same(3),
                ..Default::default()
            };
            frame.show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(text).color(text_color).size(FONT_SIZE))
                        .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
        });
}

/// The right-drag colour-pick tooltip: a header bar filled with the inspected
/// pixel's colour, its coordinates (plus degrees in panorama), and its Linear /
/// Display RGB values. Positioned to the right of the cursor, flipped to the left
/// near the right edge and clamped fully on-screen near the top/bottom — but
/// always with a horizontal gap from the cursor, so it can never cover the pixel
/// being inspected regardless of how it's clamped vertically.
fn color_pick_tooltip(ctx: &egui::Context, inputs: &UiInputs) {
    let Some(cp) = inputs.color_pick else {
        return;
    };
    let Some(pos) = ctx.pointer_hover_pos() else {
        return;
    };

    const MARGIN: f32 = 16.0;
    const SWATCH_H: f32 = 16.0;
    const PAD: f32 = 8.0;
    const ROW_GAP: f32 = 3.0;
    // Monospace (so the digit columns line up) at 80% of the guide tooltip's size.
    const FONT_SIZE: f32 = 13.0 * 0.8;
    let font = egui::FontId::monospace(FONT_SIZE);
    let value_color = egui::Color32::WHITE;
    let label_color = egui::Color32::from_gray(150);

    let coord_line = match cp.degrees {
        Some((lon, lat)) => format!("X {}px {lon:.2}°   Y {}px {lat:.2}°", cp.x, cp.y),
        None => format!("X {}px   Y {}px", cp.x, cp.y),
    };
    // Plain text for width measurement only — matches what `channel_row` (below)
    // actually renders, one channel set per `inputs.channel_count` (1=L, 2=LA,
    // 3=RGB, 4=RGBA), same grouping as the F2 box's `channel_boxes`.
    let channel_line = |c: [f32; 4]| match inputs.channel_count {
        1 => format!("L:{:.3}", c[0]),
        2 => format!("L:{:.3} A:{:.3}", c[0], c[3]),
        3 => format!("R:{:.3} G:{:.3} B:{:.3}", c[0], c[1], c[2]),
        _ => format!("R:{:.3} G:{:.3} B:{:.3} A:{:.3}", c[0], c[1], c[2], c[3]),
    };
    let linear_line = channel_line(cp.linear);
    let display_line = channel_line(cp.display);
    let rows = [
        coord_line.as_str(),
        "Linear:",
        linear_line.as_str(),
        "Display:",
        display_line.as_str(),
    ];

    // Measure before laying out: the box has to be positioned (and possibly
    // flipped/clamped) before `Area::show` allocates its actual content.
    let widest = rows
        .iter()
        .map(|t| {
            ctx.fonts(|f| f.layout_no_wrap((*t).to_owned(), font.clone(), value_color))
                .size()
                .x
        })
        .fold(0.0_f32, f32::max);
    let box_w = widest + PAD * 2.0;
    let row_h = ctx.fonts(|f| f.row_height(&font));
    let box_h =
        SWATCH_H + PAD * 2.0 + row_h * rows.len() as f32 + ROW_GAP * (rows.len() - 1) as f32;

    let screen = ctx.screen_rect();
    let left = if pos.x + MARGIN + box_w <= screen.right() {
        pos.x + MARGIN
    } else {
        pos.x - MARGIN - box_w
    };
    let top =
        (pos.y - box_h / 2.0).clamp(screen.top(), (screen.bottom() - box_h).max(screen.top()));
    let tip_pos = egui::pos2(
        left.clamp(screen.left(), (screen.right() - box_w).max(screen.left())),
        top,
    );

    egui::Area::new(egui::Id::new("imgvwr_color_pick_tooltip"))
        .fixed_pos(tip_pos)
        .order(egui::Order::Tooltip)
        .interactable(false)
        .show(ctx, |ui| {
            let outer = egui::Frame {
                fill: panel_bg(),
                inner_margin: egui::Margin::ZERO,
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            };
            outer.show(ui, |ui| {
                ui.set_width(box_w);
                ui.spacing_mut().item_spacing.y = 0.0;
                let swatch = egui::Color32::from_rgb(
                    (cp.display[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (cp.display[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (cp.display[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(box_w, SWATCH_H), egui::Sense::hover());
                ui.painter().rect_filled(
                    rect,
                    egui::CornerRadius {
                        nw: 4,
                        ne: 4,
                        sw: 0,
                        se: 0,
                    },
                    swatch,
                );
                let text_frame = egui::Frame {
                    inner_margin: egui::Margin::symmetric(PAD as i8, PAD as i8),
                    ..Default::default()
                };
                // Each row as colour-coded spans (matching the F2 box's channel
                // swatches: R/G/B, plus L/A in a neutral grey) rather than one
                // uniformly-coloured string. A trailing space baked into each span's
                // text (instead of egui inter-widget spacing) keeps the monospace
                // columns aligned exactly like a single string would.
                let la_color = egui::Color32::from_gray(190);
                let channel_row = |ui: &mut egui::Ui, c: [f32; 4]| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        let span = |ui: &mut egui::Ui, text: String, color: egui::Color32| {
                            ui.label(egui::RichText::new(text).color(color).font(font.clone()));
                        };
                        match inputs.channel_count {
                            1 => span(ui, format!("L:{:.3}", c[0]), la_color),
                            2 => {
                                span(ui, format!("L:{:.3} ", c[0]), la_color);
                                span(ui, format!("A:{:.3}", c[3]), la_color);
                            }
                            3 => {
                                span(ui, format!("R:{:.3} ", c[0]), CHANNEL_R);
                                span(ui, format!("G:{:.3} ", c[1]), CHANNEL_G);
                                span(ui, format!("B:{:.3}", c[2]), CHANNEL_B);
                            }
                            _ => {
                                span(ui, format!("R:{:.3} ", c[0]), CHANNEL_R);
                                span(ui, format!("G:{:.3} ", c[1]), CHANNEL_G);
                                span(ui, format!("B:{:.3} ", c[2]), CHANNEL_B);
                                span(ui, format!("A:{:.3}", c[3]), la_color);
                            }
                        }
                    });
                };
                text_frame.show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = ROW_GAP;
                    // `.font(font.clone())` (not `.size()`, which leaves the family at
                    // the default proportional one) — the width measured above for
                    // `box_w` used the monospace metrics, so rendering must match or
                    // the box ends up wider than the text actually needs.
                    ui.label(
                        egui::RichText::new(coord_line)
                            .color(value_color)
                            .font(font.clone()),
                    );
                    ui.label(
                        egui::RichText::new("Linear:")
                            .color(label_color)
                            .font(font.clone()),
                    );
                    channel_row(ui, cp.linear);
                    ui.label(
                        egui::RichText::new("Display:")
                            .color(label_color)
                            .font(font.clone()),
                    );
                    channel_row(ui, cp.display);
                });
            });
        });
}

/// Height of the histogram plot area, in points.
const HIST_H: f32 = 68.0;
/// Width of the plot area: exactly one point per bin, so a bin is one column and
/// no resampling happens between the data and the picture. 256 is not arbitrary —
/// it's 8-bit display quantisation, one bin per output code, so there is nothing
/// finer to resolve and an 8-bit image maps to it exactly.
const HIST_PLOT_W: f32 = crate::renderer::HISTOGRAM_BINS as f32;
/// The plot's allocation: the plot itself, a 1pt inset each side, and the 1pt
/// over-range spike with its separating gap.
const HIST_ROW_W: f32 = HIST_PLOT_W + 4.0;
/// Coverage each channel's filled area contributes. The three are composited
/// *additively* — see [`hist_band_color`] — so red over green reads yellow and
/// all three read white, and the sum saturates rather than wrapping.
const HIST_CHANNEL_ALPHA: f32 = 0.9;

/// The plot's own backing plate: black at 10%, just enough to seat the graph
/// against the box without becoming a slab of black over the image.
fn hist_plate() -> egui::Color32 {
    egui::Color32::from_black_alpha(26)
}

/// The colour for a band covered by `channels`, as a *premultiplied* egui colour.
///
/// egui blends premultiplied, so a colour whose RGB is the sum of the covered
/// channels' contributions and whose alpha is the sum of their coverages
/// composites to exactly `Σ(αᵢ·cᵢ) + (1 − Σαᵢ)·backdrop`. That is additive
/// blending: red + green land on yellow, all three on white, and no channel is
/// privileged by draw order the way sequential alpha compositing would.
fn hist_band_color(channels: &[egui::Color32]) -> egui::Color32 {
    let mut rgb = [0.0f32; 3];
    for c in channels {
        for (dst, src) in rgb.iter_mut().zip([c.r(), c.g(), c.b()]) {
            *dst += src as f32 / 255.0 * HIST_CHANNEL_ALPHA;
        }
    }
    let a = HIST_CHANNEL_ALPHA * channels.len() as f32;
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgba_premultiplied(q(rgb[0]), q(rgb[1]), q(rgb[2]), q(a))
}

/// Height of the levels handle strip below the plot.
const LEVELS_STRIP_H: f32 = 13.0;
/// How far (in points) either side of a levels handle still grabs the handle
/// rather than sliding the pair. Wider than the 8pt triangle, since an 8pt
/// target is a fiddly thing to hit exactly; past that band — *including* the
/// space outside the pair — a drag slides both (see [`LevelsGrip::at`]).
const LEVELS_GRAB: f32 = 7.0;

/// The histogram of the displayed image, its scale selector, and the levels
/// handles beneath it. Drawn under the metadata grid at a fixed [`HIST_ROW_W`] —
/// the box is sized around the graph, not the other way round, so the two line
/// up. Greyscale images draw a single neutral curve, matching how
/// [`channel_boxes`] collapses L / LA.
///
/// The plate and the handles are drawn whether or not a measurement has landed
/// yet. A measurement takes a frame or two to come back (and never arrives at
/// all on a driver without compute support), and letting the widget vanish in
/// the meantime would both pop the box's height on every image change and take
/// the levels control away with it — the handles are useful regardless.
fn histogram_plot(
    ui: &mut egui::Ui,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    // Header: label on the left, the three scale selectors against the right
    // edge of the plot. Laid out right-to-left inside a rect of exactly the
    // plot's width — NOT by padding out `available_width`, which inside an
    // auto-sizing Area is however wide the box already is and so feeds straight
    // back into the box's width, ballooning it a little further every frame.
    ui.allocate_ui_with_layout(
        egui::vec2(HIST_ROW_W, 16.0),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            // Right-to-left, so the first item added is the rightmost: the eye
            // toggle sits alone on the right, then a gap, then the three scale
            // selectors grouped tightly together (they're one control; the eye
            // is a different one and shouldn't read as a fourth option).
            hist_viewport_button(ui, inputs, actions);
            ui.add_space(6.0);
            ui.spacing_mut().item_spacing.x = 2.0;
            // Reversed, so they read L / Sq / Log left-to-right.
            for scale in crate::prefs::HistogramScale::ALL.iter().rev() {
                hist_scale_button(ui, inputs, actions, *scale);
            }
            // The label goes in whatever space is left, laid out left-to-right so
            // it sits at the plot's left edge like the grid labels above it —
            // adding it directly to the right-to-left layout would park it right
            // next to the buttons instead.
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new("Histogram").color(egui::Color32::from_gray(150)),
                    )
                    .selectable(false),
                );
            });
        },
    );

    // Both rows are allocated up front so the interaction can span them, and so
    // the drawing below knows which element is live before it paints anything.
    let (rect, _) = ui.allocate_exact_size(egui::vec2(HIST_ROW_W, HIST_H), egui::Sense::hover());
    let (strip_row, _) =
        ui.allocate_exact_size(egui::vec2(HIST_ROW_W, LEVELS_STRIP_H), egui::Sense::hover());

    // One column per bin, plus a final 1px column for the over-range spike:
    // samples the screen can only show as full white. Dropping exposure pulls
    // them back into range and the spike shrinks away.
    let bins = crate::renderer::HISTOGRAM_BINS;
    // With a channel isolated, the pass measures that channel alone as the shader
    // displays it — greyscale, alpha included — so it plots as a single white
    // area. An L / LA image is the same story in neutral grey, matching how
    // `channel_boxes` labels it. Both leave all three measured channels
    // identical, so only the first is drawn; plotting all three would just stack
    // the same curve on itself.
    let single = inputs.isolate_channel.is_some() || inputs.channel_count <= 2;
    let palette: [egui::Color32; 3] = if inputs.isolate_channel.is_some() {
        [egui::Color32::WHITE; 3]
    } else if single {
        [egui::Color32::from_gray(190); 3]
    } else {
        [CHANNEL_R, CHANNEL_G, CHANNEL_B]
    };
    let active = if single { 1 } else { 3 };

    // Reserve the rightmost pixel for the spike so it can never be mistaken for
    // the top bin, and inset by a hairline so the fills don't touch the plate's
    // rounded corners.
    let spike_w = 1.0;
    let plot = egui::Rect::from_min_max(
        rect.min + egui::vec2(1.0, 1.0),
        rect.max - egui::vec2(spike_w + 2.0, 1.0),
    );
    // One column per bin, one point wide: the plot is exactly `HIST_PLOT_W` =
    // `BINS` points across, which is the entire reason that constant is what it
    // is. Keeping a column at least a whole device pixel is load-bearing, not
    // tidiness — `Shape::mesh` goes straight to the GPU with no anti-aliasing,
    // and a quad narrower than a pixel that falls between two pixel centres
    // rasterises to nothing at all. A broad distribution hides that (its
    // neighbours fill in), but an isolated spike just vanishes, and isolated
    // spikes are exactly what this graph is worth reading for.
    let col_w = plot.width() / bins as f32;

    // The strip shares the plot's x axis; the interaction spans both, so the
    // graph's own vertical edges are the handles' grab zones.
    let strip = egui::Rect::from_min_size(
        egui::pos2(plot.left(), strip_row.top()),
        egui::vec2(plot.width(), strip_row.height()),
    );
    let live = levels_interaction(ui, inputs, state, actions, rect.union(strip_row), plot);

    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(3), hist_plate());

    if let Some(hist) = &inputs.histogram {
        let peak = hist.peak();
        let mut mesh = egui::Mesh::default();
        let mut heights = [0.0f32; 3];
        for b in 0..bins {
            let x0 = plot.left() + b as f32 * col_w;
            for (c, h) in heights[..active].iter_mut().enumerate() {
                *h = inputs.histogram_scale.normalise(hist.bins[c][b], peak);
            }
            hist_column(
                &mut mesh,
                x0,
                x0 + col_w,
                plot,
                &heights[..active],
                &palette,
            );
        }
        // The spike shares the bins' normalisation, so it reads on the same
        // scale; an over-range population larger than the tallest bin saturates.
        //
        // Unless *everything* is over-range — raise the exposure far enough and
        // it will be. Then no ordinary bin holds anything, `peak` is 0, and
        // normalising the spike against it would scale it to nothing, blanking
        // the graph at the exact moment it has the most to say. Fall back to the
        // spike's own height so it reads full.
        let spike_peak = if peak == 0 {
            hist.over.iter().copied().max().unwrap_or(0)
        } else {
            peak
        };
        let mut spike = [0.0f32; 3];
        for (c, s) in spike[..active].iter_mut().enumerate() {
            *s = inputs
                .histogram_scale
                .normalise(hist.over[c], spike_peak)
                .min(1.0);
        }
        // Snapped to whole pixels for the same reason as the columns above: a
        // 1px bar straddling a pixel boundary can rasterise to nothing, and this
        // is the one bar that has no neighbours to cover for it.
        let spike_x = (plot.right() + 1.0).round();
        hist_column(
            &mut mesh,
            spike_x,
            spike_x + spike_w,
            plot,
            &spike[..active],
            &palette,
        );
        painter.add(egui::Shape::mesh(mesh));
    }

    // Where the two levels cuts fall on the graph. These aren't just annotation —
    // each line *is* its handle's grab zone, so it's drawn whenever that handle
    // is live even when parked at an end with nothing cut off yet, otherwise the
    // thing you're about to drag would be invisible.
    let (black, white) = inputs.levels;
    for (grip, t) in [(LevelsGrip::Black, black), (LevelsGrip::White, white)] {
        let is_live = live == Some(grip);
        if !is_live && (t <= 0.0 || t >= 1.0) {
            continue;
        }
        let x = plot.left() + t * plot.width();
        painter.line_segment(
            [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
            if is_live {
                egui::Stroke::new(1.0, ACCENT)
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(60))
            },
        );
    }

    draw_levels_strip(painter, inputs.levels, strip, live);
}

/// The levels interaction: one zone covering the graph *and* the handle strip
/// beneath it. Returns which element the pointer is on or has grabbed, so the
/// caller can highlight it.
///
/// The graph is part of the control, not just a picture of the data. Each
/// handle's grab zone is the full-height vertical line at its position — so you
/// pull the graph's own edges around — and everything else in the box slides
/// both levels together. That last part is what makes the pair reachable once
/// they're butted up against each other and the strip between them has vanished:
/// the entire rest of the graph is the "move both" target.
///
/// Dragging remaps the display range; the graph itself deliberately doesn't
/// follow (see `Renderer::update_histogram`).
fn levels_interaction(
    ui: &mut egui::Ui,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
    zone: egui::Rect,
    plot: egui::Rect,
) -> Option<LevelsGrip> {
    let (black, white) = inputs.levels;
    let resp = ui.interact(
        zone,
        ui.id().with("levels_zone"),
        egui::Sense::click_and_drag(),
    );
    let x_of = |t: f32| plot.left() + t.clamp(0.0, 1.0) * plot.width();
    let t_of = |x: f32| ((x - plot.left()) / plot.width().max(1.0)).clamp(0.0, 1.0);

    // `pointer_interact_pos` rather than the response's hover position: a drag
    // routinely leaves the zone, and losing the pointer there would strand the
    // handle (the same reason `ruler_spawn_drag` reads it this way).
    let ctx = ui.ctx().clone();
    let pointer = ctx.pointer_interact_pos();
    let pointer_t = pointer.map(|p| t_of(p.x));

    // Which element a given x belongs to. Hit-tested by hand rather than with
    // overlapping widgets, so the priority is explicit — and deliberately
    // *exhaustive*: every point in the zone belongs to something.
    //
    // Except when the range already spans everything, where "slide both" has
    // nowhere to go. It then grabs nothing and shows nothing, rather than giving
    // most of the graph a grab cursor that leads to a drag doing precisely
    // nothing. Only the full span counts: with one handle pulled in there is
    // room to slide toward the other side, so it stays live.
    let both_inert = black <= 0.0 && white >= 1.0;
    let grip_at = |x: f32| {
        let grip = LevelsGrip::at(x, x_of(black), x_of(white), LEVELS_GRAB);
        (!(both_inert && grip == LevelsGrip::Both)).then_some(grip)
    };
    let hot = resp
        .contains_pointer()
        .then(|| pointer.and_then(|p| grip_at(p.x)))
        .flatten();

    // The grip is decided once, from where the pointer went *down*, and held for
    // the rest of the gesture.
    //
    // It must be `press_origin`, and specifically not `interact_pointer_pos`:
    // `drag_started` doesn't fire on the press, it fires once the pointer has
    // passed egui's click threshold, several points later — and despite its
    // name, `interact_pointer_pos` is `pointer.interact_pos()`, the pointer's
    // position *now*, not where it went down. Deciding the grip from it grabbed
    // whatever the pointer had already moved onto, so pressing a triangle and
    // dragging inward silently took the bar instead. `press_origin` is the one
    // that actually means "where did this drag start".
    if resp.drag_started() {
        if let Some((press, grip)) = ctx
            .input(|i| i.pointer.press_origin())
            .and_then(|p| grip_at(p.x).map(|g| (p, g)))
        {
            state.levels_drag = Some(LevelsDrag {
                grip,
                grab_t: t_of(press.x),
                start: (black, white),
            });
        }
    }
    if let (Some(drag), Some(t)) = (state.levels_drag, pointer_t) {
        // `drag_stopped` included so the release frame's final position lands —
        // without it the last few pixels of a fast drag are dropped.
        if resp.dragged() || resp.drag_stopped() {
            let (black, white) = drag.resolve(t, crate::app::LEVELS_MIN_GAP);
            actions.push(UiAction::SetLevels { black, white });
        }
    }

    // What the indicators should describe. Once the button is down the answer is
    // fixed by where it went down — including through the window *before* egui
    // calls it a drag, which is a real gap: the click threshold is 6pt and the
    // grab radius is 7, so pulling a handle inward leaves its zone before the
    // drag starts. Following the live pointer there made the bar light up and the
    // cursor change for a frame or two, advertising a grab that was never going
    // to happen.
    let pressed_grip = resp
        .is_pointer_button_down_on()
        .then(|| ctx.input(|i| i.pointer.press_origin()))
        .flatten()
        .and_then(|p| grip_at(p.x));
    let live = state.levels_drag.map(|d| d.grip).or(pressed_grip).or(hot);
    if let Some(grip) = live {
        let dragging = state.levels_drag.is_some();
        ctx.set_cursor_icon(match (grip, dragging) {
            (LevelsGrip::Both, true) => egui::CursorIcon::Grabbing,
            (LevelsGrip::Both, false) => egui::CursorIcon::Grab,
            _ => egui::CursorIcon::ResizeHorizontal,
        });
    }
    if !resp.dragged() && !resp.is_pointer_button_down_on() {
        state.levels_drag = None;
    }
    // Double-click resets whatever was under the pointer: a handle goes back to
    // its own end of the range, the bar between them clears both.
    if resp.double_clicked() {
        if let Some(grip) = resp.interact_pointer_pos().and_then(|p| grip_at(p.x)) {
            let (black, white) = match grip {
                LevelsGrip::Black => (0.0, white),
                LevelsGrip::White => (black, 1.0),
                LevelsGrip::Both => (0.0, 1.0),
            };
            actions.push(UiAction::SetLevels { black, white });
        }
    }

    if let Some(grip) = live {
        resp.on_hover_text(match grip {
            LevelsGrip::Black => {
                "Display black point — drag the line or the handle; double-click to reset it"
            }
            LevelsGrip::White => {
                "Display white point — drag the line or the handle; double-click to reset it"
            }
            LevelsGrip::Both => "Drag anywhere else to slide both; double-click to reset both",
        });
    }
    live
}

/// The handle strip under the plot: the surviving slice of the range as a bar,
/// with a triangle at each end. `live` is the element the pointer is on or has
/// grabbed, which is the only one that highlights — lighting all of them says
/// nothing about what a drag would actually move.
fn draw_levels_strip(
    painter: &egui::Painter,
    levels: (f32, f32),
    strip: egui::Rect,
    live: Option<LevelsGrip>,
) {
    let (black, white) = levels;
    let x_of = |t: f32| strip.left() + t.clamp(0.0, 1.0) * strip.width();
    let bar_y = strip.top() + 1.0;
    painter.line_segment(
        [
            egui::pos2(x_of(black), bar_y),
            egui::pos2(x_of(white), bar_y),
        ],
        if live == Some(LevelsGrip::Both) {
            egui::Stroke::new(3.0, ACCENT)
        } else {
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(70))
        },
    );
    for (grip, t, fill) in [
        (LevelsGrip::Black, black, egui::Color32::from_gray(40)),
        (LevelsGrip::White, white, egui::Color32::from_gray(225)),
    ] {
        let x = x_of(t);
        let half = 4.0;
        let tri = vec![
            egui::pos2(x, bar_y),
            egui::pos2(x - half, strip.bottom() - 1.0),
            egui::pos2(x + half, strip.bottom() - 1.0),
        ];
        let stroke = if live == Some(grip) {
            egui::Stroke::new(1.5, ACCENT)
        } else {
            egui::Stroke::new(1.0, egui::Color32::from_gray(120))
        };
        painter.add(egui::Shape::convex_polygon(tri, fill, stroke));
    }
}

/// Emit one column of the area chart: the covered channels split the column into
/// stacked bands, each painted with the additive colour of the channels that
/// reach it, so overlaps mix without any per-shape blend mode.
fn hist_column(
    mesh: &mut egui::Mesh,
    x0: f32,
    x1: f32,
    plot: egui::Rect,
    heights: &[f32],
    palette: &[egui::Color32; 3],
) {
    // Walk the distinct heights bottom-up; between two consecutive ones exactly
    // the channels that are still taller than the lower bound are covered.
    //
    // Three channels at most, so both working sets are fixed three-slot arrays.
    // This runs once per bin on every frame the box is up — a heap allocation
    // per column, let alone per band, is not the shape that work should have.
    let mut bands = [0.0f32; 3];
    let mut n = 0;
    for &h in heights {
        if h > 0.0 {
            bands[n] = h;
            n += 1;
        }
    }
    let bands = &mut bands[..n];
    bands.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut low = 0.0f32;
    for &high in bands.iter() {
        // Ascending, so a repeat of the height just drawn would be a band of no
        // height at all — which is what the dedup is for.
        if high <= low {
            continue;
        }
        // At least one channel always reaches `high`: that is where it came from.
        let mut covered = [egui::Color32::TRANSPARENT; 3];
        let mut n_covered = 0;
        for (c, h) in heights.iter().enumerate() {
            if *h >= high {
                covered[n_covered] = palette[c];
                n_covered += 1;
            }
        }
        let color = hist_band_color(&covered[..n_covered]);
        let band = egui::Rect::from_min_max(
            egui::pos2(x0, plot.bottom() - high * plot.height()),
            egui::pos2(x1, plot.bottom() - low * plot.height()),
        );
        let idx = mesh.vertices.len() as u32;
        for p in [
            band.left_top(),
            band.right_top(),
            band.right_bottom(),
            band.left_bottom(),
        ] {
            mesh.colored_vertex(p, color);
        }
        mesh.add_triangle(idx, idx + 1, idx + 2);
        mesh.add_triangle(idx, idx + 2, idx + 3);
        low = high;
    }
}

/// The viewport-sampling toggle: measure only the visible region instead of the
/// whole image, so zooming in gives a reading of just that area.
fn hist_viewport_button(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    let on = inputs.histogram_viewport;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(18.0, 14.0), egui::Sense::click());
    let radius = egui::CornerRadius::same(2);
    if on {
        ui.painter().rect_filled(rect, radius, ACCENT);
    } else if resp.hovered() {
        ui.painter()
            .rect_filled(rect, radius, egui::Color32::from_white_alpha(24));
    }
    // Square, so the 16×16 glyph isn't stretched by the wider button.
    let icon = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(11.0));
    egui::Image::new(egui::include_image!("../../resources/icons/ui/eye.svg"))
        .tint(if on {
            egui::Color32::from_gray(20)
        } else {
            egui::Color32::from_gray(170)
        })
        .paint_at(ui, icon);
    if clickable(resp)
        .on_hover_text(if on {
            "Sampling the visible region — click for the whole image"
        } else {
            "Sample only the visible region (zoom in for a focused histogram)"
        })
        .clicked()
    {
        actions.push(UiAction::SetHistogramViewport(!on));
    }
}

/// Allocation width of one scale selector: its label plus a little padding.
fn hist_scale_button_width(ui: &egui::Ui, scale: crate::prefs::HistogramScale) -> f32 {
    let galley = ui.painter().layout_no_wrap(
        scale.label().to_string(),
        egui::FontId::proportional(10.0),
        egui::Color32::WHITE,
    );
    galley.size().x + 8.0
}

/// One mini scale selector. Accent-filled when active, matching how the channel
/// boxes mark the isolated channel.
fn hist_scale_button(
    ui: &mut egui::Ui,
    inputs: &UiInputs,
    actions: &mut Vec<UiAction>,
    scale: crate::prefs::HistogramScale,
) {
    let active = inputs.histogram_scale == scale;
    let w = hist_scale_button_width(ui, scale);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 14.0), egui::Sense::click());
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(2);
    if active {
        painter.rect_filled(rect, radius, ACCENT);
    } else if resp.hovered() {
        painter.rect_filled(rect, radius, egui::Color32::from_white_alpha(24));
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        scale.label(),
        egui::FontId::proportional(10.0),
        if active {
            egui::Color32::from_gray(20)
        } else {
            egui::Color32::from_gray(170)
        },
    );
    let hint = match scale {
        crate::prefs::HistogramScale::Linear => "Linear scale",
        crate::prefs::HistogramScale::Sqrt => "Square-root scale",
        crate::prefs::HistogramScale::Log => "Logarithmic scale",
    };
    if clickable(resp).on_hover_text(hint).clicked() {
        actions.push(UiAction::SetHistogramScale(scale));
    }
}

/// The `(label, colour, channel-index)` boxes to show for a channel count.
fn channel_boxes(count: u8) -> Vec<(&'static str, egui::Color32, u8)> {
    let a = egui::Color32::from_gray(190);
    match count {
        1 => vec![("L", a, 0)],
        2 => vec![("L", a, 0), ("A", a, 3)],
        3 => vec![
            ("R", CHANNEL_R, 0),
            ("G", CHANNEL_G, 1),
            ("B", CHANNEL_B, 2),
        ],
        _ => vec![
            ("R", CHANNEL_R, 0),
            ("G", CHANNEL_G, 1),
            ("B", CHANNEL_B, 2),
            ("A", a, 3),
        ],
    }
}

/// One channel-isolation box. Accent border when active; click toggles.
fn channel_box(
    ui: &mut egui::Ui,
    label: &str,
    color: egui::Color32,
    idx: u8,
    inputs: &UiInputs,
    actions: &mut Vec<UiAction>,
) {
    let active = inputs.isolate_channel == Some(idx);
    // While one channel is isolated the others fade right back, so the active one
    // reads at a glance rather than only by its outline.
    let muted = inputs.isolate_channel.is_some() && !active;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(3);
    let fill = if muted {
        egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 51)
    } else {
        color
    };
    painter.rect_filled(rect, radius, fill);
    if active {
        painter.rect_stroke(
            rect,
            radius,
            egui::Stroke::new(2.0, ACCENT),
            egui::StrokeKind::Outside,
        );
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0),
        // Dark-on-colour reads well on a full swatch, but vanishes on a 20% one
        // over the dark panel — the muted label goes light instead.
        if muted {
            egui::Color32::from_gray(150)
        } else {
            egui::Color32::from_gray(20)
        },
    );
    let resp = clickable(resp).on_hover_text(format!("Isolate {label} channel"));
    if resp.clicked() {
        let next = if active { None } else { Some(idx) };
        actions.push(UiAction::SetChannelIsolate(next));
    }
}

/// The View-transform dropdown for the metadata box: the current display's views
/// inline, plus a `Display ›` sub-menu whose entries open each display's views.
/// Returns whether the menu (or one of its sub-menus) is currently open, so the
/// caller can keep the metadata box revealed while the user navigates it.
/// The shared dropdown trigger: an SVG chevron-down icon before the current
/// value. Used by both the Display and View pickers (no nested sub-menus — they
/// opened rightward off the screen edge; the two are single-level instead).
fn chevron_button(current: &str) -> egui::Button<'static> {
    let chevron = egui::Image::new(egui::include_image!(
        "../../resources/icons/ui/chevron-down.svg"
    ))
    .tint(egui::Color32::WHITE)
    .fit_to_exact_size(egui::vec2(10.0, 10.0));
    egui::Button::image_and_text(
        chevron,
        egui::RichText::new(current.to_owned()).color(egui::Color32::WHITE),
    )
}

/// The OCIO display the View picker lists views for: the active display, else
/// the first one in the config.
fn active_display(inputs: &UiInputs) -> String {
    inputs
        .active
        .as_ref()
        .map(|(d, _)| d.clone())
        .or_else(|| inputs.displays().first().cloned())
        .unwrap_or_default()
}

/// Single-level Display picker. Selecting a display keeps the current view if it
/// is valid for that display, else falls back to "Standard", else the first
/// view. Returns whether its menu is open (to keep the metadata box revealed).
fn display_dropdown(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) -> bool {
    let current = active_display(inputs);
    let current_view = inputs.active.as_ref().map(|(_, v)| v.clone());
    let bar_id = ui.id();
    egui::menu::menu_custom_button(ui, chevron_button(&current), |ui| {
        for display in inputs.displays() {
            if ui.selectable_label(display == current, &display).clicked() {
                let views = inputs.views_for(&display);
                // Keep the current view if it's valid for the new display, else
                // revert to Standard (then Raw, then the first view) — matching the
                // case-insensitive Standard→Raw convention used elsewhere (the T
                // toggle, select_standard_view, the SetView handler).
                let view = current_view
                    .clone()
                    .filter(|v| views.contains(v))
                    .or_else(|| {
                        views
                            .iter()
                            .find(|v| v.eq_ignore_ascii_case("standard"))
                            .cloned()
                    })
                    .or_else(|| {
                        views
                            .iter()
                            .find(|v| v.eq_ignore_ascii_case("raw"))
                            .cloned()
                    })
                    .or_else(|| views.first().cloned())
                    .unwrap_or_default();
                actions.push(UiAction::SetView { display, view });
                ui.close_menu();
            }
        }
    });
    egui::menu::BarState::load(ui.ctx(), bar_id).is_some()
}

/// Single-level View picker, listing the views available for the active display
/// (so its contents follow the Display picker). Returns whether its menu is open.
fn view_dropdown(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) -> bool {
    let display = active_display(inputs);
    let current_view = inputs
        .active
        .as_ref()
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let bar_id = ui.id();
    egui::menu::menu_custom_button(ui, chevron_button(&current_view), |ui| {
        for view in inputs.views_for(&display) {
            if ui.selectable_label(view == current_view, &view).clicked() {
                actions.push(UiAction::SetView {
                    display: display.clone(),
                    view,
                });
                ui.close_menu();
            }
        }
    });
    egui::menu::BarState::load(ui.ctx(), bar_id).is_some()
}

/// Transient bottom-right HUD; `alpha` fades it out (see `App::toast_render`).
///
/// The box is positioned with an explicit `fixed_pos` computed from the
/// pre-measured text size, rather than an anchored auto-sized `Area`. An anchored
/// area lags one frame behind size changes, which made the toast visibly jump
/// each time its content changed; fixed positioning removes that flicker.
fn toast(ctx: &egui::Context, text: &str, alpha: f32) {
    let a = alpha.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let bg = (PANEL_ALPHA as f32 * a) as u8;
    let fg = (255.0 * a) as u8;
    const FONT_SIZE: f32 = 16.0;
    let font = egui::FontId::proportional(FONT_SIZE);
    let galley = ctx.fonts(|f| f.layout_no_wrap(text.to_owned(), font, egui::Color32::WHITE));
    // Frame inner margin is symmetric(10, 6), so add (20, 12) to the text size.
    let size = galley.size() + egui::vec2(20.0, 12.0);
    let screen = ctx.screen_rect();
    let pos = egui::pos2(
        screen.right() - 12.0 - size.x,
        screen.bottom() - 12.0 - size.y,
    );
    egui::Area::new(egui::Id::new("imgvwr_toast"))
        .fixed_pos(pos)
        .interactable(false)
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: panel_bg_alpha(bg),
                inner_margin: egui::Margin::symmetric(10, 6),
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            };
            frame.show(ui, |ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(text)
                            .color(egui::Color32::from_rgba_unmultiplied(fg, fg, fg, fg))
                            .size(FONT_SIZE),
                    )
                    .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
        });
}

/// Centred hotkey reference (toggled with H, dismissed with H/Esc/Close). Sections
/// are laid out in columns that wrap to as many rows as the window allows; if even
/// wrapping can't fit them all, the least-important trailing sections are dropped
/// and a "Show more" button (which goes fullscreen) appears. Sections are listed
/// most-important first, so dropping from the end keeps the essentials.
fn help_dialog(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    type Section = (&'static str, &'static [(&'static str, &'static str)]);
    const SECTIONS: &[Section] = &[
        (
            "View & zoom",
            &[
                ("Drag (L/M)", "Pan / look around"),
                ("Wheel", "Zoom (2D) / FOV (pano)"),
                ("Shift / Ctrl + wheel", "Pan horizontally / vertically"),
                ("Numpad 1–9", "Zoom 2^(N-1)× (Ctrl = out)"),
                ("Home / R", "Reset view & window"),
                ("Backspace", "Centre + fit window (keep zoom)"),
                ("↑ / ↓", "Rotate 90° (remembered)"),
                ("P", "Toggle 2D / panorama"),
                ("W", "Toggle 2D tiled wrap"),
                ("I", "Pin nearest / bilinear (auto: nearest > 200%)"),
                ("M", "Navigation minimap"),
                ("B", "Cycle background backdrop"),
            ],
        ),
        (
            "Tone & files",
            &[
                (", / .", "Exposure −/+"),
                ("Ctrl + , / .", "Gamma −/+"),
                ("[ / ]", "Clarity radius −/+"),
                ("; / '", "Clarity strength −/+"),
                ("Ctrl + R", "Reset all adjustments"),
                ("Ctrl + Z / Y", "Undo / redo edits"),
                ("T", "Standard / last view transform"),
                ("O", "Open file…"),
                ("Ctrl + C", "Copy the window to the clipboard"),
                ("Delete", "Delete the file (confirms first)"),
                ("← / →", "Previous / next image"),
                ("F2", "Metadata overlay"),
            ],
        ),
        (
            "Playback",
            &[
                ("Space", "Play a sequence / animation, then pause"),
                ("Right-click", "Play / pause"),
                ("Esc / ⏹", "Stop (stays on the current frame)"),
                ("← / →", "Step one frame"),
                ("Click timeline", "Jump to that frame"),
                ("Drag timeline", "Scrub (keeps playing)"),
                ("Blue / red bar", "Frames cached / missing"),
                ("fps readout", "Click to pick the rate (or add one)"),
                ("fps in red", "Running short of the target rate"),
            ],
        ),
        (
            "Window",
            &[
                ("Alt + drag · titlebar", "Move window"),
                ("Window edges", "Drag to resize"),
                ("Alt + right-drag", "Resize (by third)"),
                ("Alt + scroll", "Grow / shrink window"),
                ("A", "Always on top"),
                ("F / F11 / dbl-click", "Toggle fullscreen"),
                ("Esc / Q", "Exit fullscreen / quit"),
            ],
        ),
        (
            "Inspect",
            &[
                ("S", "Sharpness (original res)"),
                ("C", "Clipping overlay (per channel)"),
                ("Alt + middle-drag", "Squash / stretch image"),
                ("G", "Show / hide guides (adds the first one)"),
                ("Shift + G", "Add guide level (½, ¼ … 1/32)"),
                ("Ctrl + G", "Remove guide level"),
                ("Pull from a ruler", "Drag out a guide"),
                ("Drag / right-click guide", "Move / delete it"),
                ("Ctrl + drag guide", "Snap to 10px / 1°"),
                ("Right-click + hold-drag", "Pick pixel colour values"),
                ("Channel boxes (F2)", "Isolate R/G/B/A"),
                (
                    "Histogram (F2)",
                    "L/Sq/Log scale; eye = visible region only",
                ),
                ("Levels (F2)", "Drag a graph edge; elsewhere slides both"),
            ],
        ),
        (
            "Compare",
            &[
                ("Ctrl + 1–9", "Save image / sequence to a slot"),
                ("1–9 (top row)", "Recall slot (sequences resume playing)"),
                ("Alt + 1–9", "Difference vs that slot"),
                ("L", "Lock zoom/pan across images"),
                ("H", "This help"),
            ],
        ),
    ];
    // Layout constants (points). Column gap between sections; vertical gap between
    // wrapped rows; the chrome heights reserved when deciding how many sections fit.
    const COL_GAP: f32 = 22.0;
    const ROW_GAP: f32 = 14.0;
    const SCREEN_MARGIN: f32 = 24.0; // keep the dialog off the very window edges
    const FRAME_PAD: f32 = 16.0; // overlay_frame inner margin
    const TITLE_H: f32 = 56.0; // title label + version/link subtitle + spacing
    const FOOTER_H: f32 = 44.0; // spacing + button row

    // App version + repository link, shown as a subtitle under the title. The repo
    // URL comes straight from Cargo.toml so the link can never drift from the
    // manifest; the visible label just drops the scheme.
    const VERSION_LINE: &str = concat!("imgvwr v", env!("CARGO_PKG_VERSION"));
    const REPO_URL: &str = env!("CARGO_PKG_REPOSITORY");
    let repo_label = REPO_URL
        .trim_start_matches("https://")
        .trim_start_matches("http://");

    // Measure with the real body font so the column count and the omission
    // decision match what egui will actually lay out.
    let body = ctx
        .style()
        .text_styles
        .get(&egui::TextStyle::Body)
        .cloned()
        .unwrap_or(egui::FontId::proportional(14.0));
    let text_w = |s: &str| {
        ctx.fonts(|f| {
            f.layout_no_wrap(s.to_string(), body.clone(), egui::Color32::WHITE)
                .size()
                .x
        })
    };
    let line_h = ctx.fonts(|f| {
        f.layout_no_wrap("Mg".to_string(), body.clone(), egui::Color32::WHITE)
            .size()
            .y
    });
    // Widest single section (its key column + 10px grid gap + action column), so a
    // row of `cols` such sections is guaranteed to fit `cols * sec_w` of width.
    let sec_widths: Vec<f32> = SECTIONS
        .iter()
        .map(|(_, keys)| {
            let kw = keys.iter().map(|(k, _)| text_w(k)).fold(0.0, f32::max);
            let aw = keys.iter().map(|(_, a)| text_w(a)).fold(0.0, f32::max);
            kw + 10.0 + aw
        })
        .collect();
    let sec_w = sec_widths.iter().copied().fold(1.0, f32::max);
    let title_w = ctx.fonts(|f| {
        f.layout_no_wrap(
            "Keyboard & mouse".to_string(),
            egui::FontId::proportional(18.0),
            egui::Color32::WHITE,
        )
        .size()
        .x
    });
    let row_h = line_h + 3.0; // grid row + spacing
    let header_h = line_h + 8.0; // section header + the 4px space under it
    let section_h = |keys: &[(&str, &str)]| header_h + keys.len() as f32 * row_h;

    let screen = ctx.screen_rect();
    let avail_w = (screen.width() - 2.0 * (SCREEN_MARGIN + FRAME_PAD)).max(sec_w);
    let avail_h = screen.height() - 2.0 * (SCREEN_MARGIN + FRAME_PAD) - TITLE_H - FOOTER_H;
    let cols =
        (((avail_w + COL_GAP) / (sec_w + COL_GAP)).floor() as usize).clamp(1, SECTIONS.len());

    // Packed height of the first `keep` sections in rows of `cols` (each row as
    // tall as its tallest section).
    let packed_h = |keep: usize| {
        let mut h = 0.0;
        let mut i = 0;
        while i < keep {
            let end = (i + cols).min(keep);
            let rh = SECTIONS[i..end]
                .iter()
                .map(|(_, k)| section_h(k))
                .fold(0.0, f32::max);
            h += rh;
            if end < keep {
                h += ROW_GAP;
            }
            i = end;
        }
        h
    };
    let mut keep = SECTIONS.len();
    while keep > 1 && packed_h(keep) > avail_h {
        keep -= 1;
    }
    // Even one section (the tallest, always-kept first one) can be taller than a
    // very short window; in that case the body scrolls so the title and footer
    // stay on-screen and clickable.
    let overflow = packed_h(keep) > avail_h;
    // Only offer "Show more" → fullscreen when it could actually help (not already
    // fullscreen); otherwise there's nothing more room to reveal.
    let truncated = (keep < SECTIONS.len() || overflow) && !inputs.is_fullscreen;

    // Width of a row of `n` sections starting at `start` (sum of their measured
    // widths + the gaps), and the fixed dialog content width every part centres in.
    let row_w = |start: usize, end: usize| -> f32 {
        let n = end - start;
        sec_widths[start..end].iter().sum::<f32>() + n.saturating_sub(1) as f32 * COL_GAP
    };
    // The version/link subtitle must fit on one line too, so it widens the dialog
    // when the sections alone would be narrower than it. Measure at the SAME 13px
    // the labels render at (not the 14px Body that `text_w` uses), so the width
    // mirrors the render exactly: [version] 6px [·] 6px [link] (the item_spacing
    // set below). A measure≠render mismatch would shove the subtitle off-centre.
    let sub_w = |s: &str| {
        ctx.fonts(|f| {
            f.layout_no_wrap(
                s.to_string(),
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            )
            .size()
            .x
        })
    };
    let subtitle_w = sub_w(VERSION_LINE) + 6.0 + sub_w("·") + 6.0 + sub_w(repo_label);
    let mut content_w = title_w.max(subtitle_w);
    for row_start in (0..keep).step_by(cols) {
        content_w = content_w.max(row_w(row_start, (row_start + cols).min(keep)));
    }
    // Approximate footer width (button text + chrome) so it can be centred too.
    const BTN_CHROME: f32 = 22.0;
    let buttons_w = text_w("Close")
        + BTN_CHROME
        + if truncated {
            text_w("Show more") + BTN_CHROME + 8.0
        } else {
            0.0
        };
    content_w = content_w.max(buttons_w);

    egui::Area::new(egui::Id::new("imgvwr_help"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                // Fix the content width (bounded — content_w ≤ avail_w) so the
                // title, every section row, and the footer all centre against the
                // same reference (egui won't centre a full-width horizontal group
                // on its own). set_width, not set_min_width, so an auto-sizing Area
                // can't let the body expand past the window.
                ui.set_width(content_w);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Keyboard & mouse")
                            .strong()
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                    );
                });
                ui.add_space(2.0);
                // App version + clickable repository link (opens in the browser via
                // egui's hyperlink handling). The label drops the URL scheme.
                // Centred with manual padding — a nested horizontal group isn't
                // reliably centred by vertical_centered (the footer does the same).
                ui.horizontal(|ui| {
                    ui.add_space((content_w - subtitle_w).max(0.0) * 0.5);
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.label(
                        egui::RichText::new(VERSION_LINE)
                            .color(egui::Color32::from_gray(150))
                            .size(13.0),
                    );
                    ui.label(
                        egui::RichText::new("·")
                            .color(egui::Color32::from_gray(110))
                            .size(13.0),
                    );
                    clickable(
                        ui.hyperlink_to(egui::RichText::new(repo_label).size(13.0), REPO_URL),
                    )
                    .on_hover_text(REPO_URL);
                });
                ui.add_space(10.0);
                // Cap the body height so a too-short window scrolls the sections
                // instead of pushing the footer (Close / Show more) off-screen.
                // auto_shrink horizontal stays TRUE so the scroll viewport hugs the
                // content_w-wide rows rather than ballooning in the unbounded Area.
                egui::ScrollArea::vertical()
                    .max_height(avail_h.max(48.0))
                    .auto_shrink([true, true])
                    .show(ui, |ui| {
                        ui.set_width(content_w);
                        for row_start in (0..keep).step_by(cols) {
                            if row_start > 0 {
                                ui.add_space(ROW_GAP);
                            }
                            let row_end = (row_start + cols).min(keep);
                            ui.horizontal_top(|ui| {
                                let pad = (content_w - row_w(row_start, row_end)).max(0.0) * 0.5;
                                ui.add_space(pad);
                                for (j, (title, keys)) in
                                    SECTIONS[row_start..row_end].iter().enumerate()
                                {
                                    if j > 0 {
                                        ui.add_space(COL_GAP);
                                    }
                                    ui.vertical(|ui| {
                                        ui.label(
                                            egui::RichText::new(*title).strong().color(ACCENT),
                                        );
                                        ui.add_space(4.0);
                                        egui::Grid::new(("imgvwr_help", *title))
                                            .num_columns(2)
                                            .spacing([10.0, 3.0])
                                            .show(ui, |ui| {
                                                for (key, action) in *keys {
                                                    ui.label(egui::RichText::new(*key).color(
                                                        egui::Color32::from_rgb(180, 200, 255),
                                                    ));
                                                    ui.label(
                                                        egui::RichText::new(*action)
                                                            .color(egui::Color32::from_gray(220)),
                                                    );
                                                    ui.end_row();
                                                }
                                            });
                                    });
                                }
                            });
                        }
                    });
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.add_space((content_w - buttons_w).max(0.0) * 0.5);
                    if truncated
                        && clickable(ui.button("Show more"))
                            .on_hover_text("Go fullscreen to show every shortcut")
                            .clicked()
                    {
                        actions.push(UiAction::ToggleFullscreen);
                    }
                    if clickable(ui.button("Close")).clicked() {
                        actions.push(UiAction::CloseHelp);
                    }
                });
            });
        });
}

#[cfg(test)]
mod tests {
    use super::{
        levels_interaction, titlebar_fit, TitlebarFit, HIST_H, HIST_PLOT_W, HIST_ROW_W,
        LEVELS_GRAB, LEVELS_STRIP_H,
    };
    use crate::ui::{UiAction, UiInputs, UiState};

    /// Where the test places the strip. Any fixed position works; the assertions
    /// derive the handle coordinates from it.
    const ORIGIN: egui::Pos2 = egui::pos2(40.0, 60.0);

    /// Drive a real pointer gesture through egui and return the actions it
    /// produced.
    ///
    /// This exists because the levels handles kept misbehaving in ways that
    /// couldn't be caught by testing the arithmetic: the bugs were all in *when*
    /// egui reports an interaction and *where* it says the pointer was. imgvwr's
    /// borderless window can't be driven by the computer-use tooling, so the only
    /// way to actually exercise a drag is to feed egui synthetic input.
    ///
    /// The gesture is press → move → move → release across four frames, matching
    /// how a real drag arrives; egui only decides a press is a drag once the
    /// pointer has travelled past its click threshold, which is the exact
    /// behaviour under test.
    fn drag_levels(levels: (f32, f32), press_x: f32, release_x: f32) -> Vec<UiAction> {
        // Default to a y inside the *graph*: the graph's own area is part of the
        // control, so that's the interesting place to press.
        drag_levels_at(levels, press_x, release_x, ORIGIN.y + 20.0)
    }

    /// [`drag_levels`] at an explicit y, for reaching the strip below the graph.
    fn drag_levels_at(
        levels: (f32, f32),
        press_x: f32,
        release_x: f32,
        press_y: f32,
    ) -> Vec<UiAction> {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let inputs = UiInputs {
            levels,
            ..Default::default()
        };
        let plot = egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_PLOT_W, HIST_H));
        // The zone spans the graph and the strip beneath it — the graph's own
        // area is now part of the control.
        let zone =
            egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_ROW_W, HIST_H + LEVELS_STRIP_H));
        let y = press_y;
        let mut actions = Vec::new();

        let mut frame = |time: f64, events: Vec<egui::Event>, actions: &mut Vec<UiAction>| {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                time: Some(time),
                events,
                ..Default::default()
            };
            let _ = ctx.run(raw, |ctx| {
                egui::Area::new(egui::Id::new("levels_test"))
                    .fixed_pos(ORIGIN)
                    .show(ctx, |ui| {
                        let _ = ui.allocate_exact_size(zone.size(), egui::Sense::hover());
                        levels_interaction(ui, &inputs, &mut state, actions, zone, plot);
                    });
            });
        };

        let press = egui::pos2(press_x, y);
        let release = egui::pos2(release_x, y);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };

        // Two frames of hover before the press: egui hit-tests a press against
        // the *previous* frame's widget rects, so a widget pressed the instant it
        // first appears is never credited with the press.
        let half = egui::pos2((press_x + release_x) * 0.5, y);
        frame(0.0, vec![egui::Event::PointerMoved(press)], &mut actions);
        frame(0.1, vec![egui::Event::PointerMoved(press)], &mut actions);
        frame(
            0.2,
            vec![egui::Event::PointerMoved(press), button(press, true)],
            &mut actions,
        );
        // The first move is what crosses egui's click threshold and turns the
        // press into a drag — and so the frame where the grip gets decided, by
        // which point the pointer is already well away from where it went down.
        frame(0.3, vec![egui::Event::PointerMoved(half)], &mut actions);
        frame(0.4, vec![egui::Event::PointerMoved(release)], &mut actions);
        frame(0.5, vec![button(release, false)], &mut actions);
        actions
    }

    /// Press on a handle and creep the pointer inward by less than egui's click
    /// threshold, returning the cursor egui asked for on each of those frames.
    ///
    /// This window is real and awkward: the click threshold is 6pt but the grab
    /// radius is 7, so pulling a handle inward leaves its zone *before* the drag
    /// officially starts. Following the live pointer there made the bar highlight
    /// and the cursor flick to Grab for a frame or two, advertising a grab that
    /// was never going to happen.
    fn cursors_during_pre_drag_creep(levels: (f32, f32), press_x: f32) -> Vec<egui::CursorIcon> {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let inputs = UiInputs {
            levels,
            ..Default::default()
        };
        let plot = egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_PLOT_W, HIST_H));
        // The zone spans the graph and the strip beneath it — the graph's own
        // area is now part of the control.
        let zone =
            egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_ROW_W, HIST_H + LEVELS_STRIP_H));
        let y = ORIGIN.y + 4.0;
        let mut actions = Vec::new();
        let mut cursors = Vec::new();
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let press = egui::pos2(press_x, y);
        let mut seq = vec![
            vec![egui::Event::PointerMoved(press)],
            vec![egui::Event::PointerMoved(press)],
            vec![egui::Event::PointerMoved(press), button(press, true)],
        ];
        // 1.5pt a frame, three frames: 4.5pt total, comfortably under the 6pt
        // click threshold, but past the 7pt grab radius when pressed near a
        // handle's inner edge.
        for i in 1..=3 {
            seq.push(vec![egui::Event::PointerMoved(egui::pos2(
                press_x + 1.5 * i as f32,
                y,
            ))]);
        }
        for (f, events) in seq.into_iter().enumerate() {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                time: Some(f as f64 * 0.02),
                events,
                ..Default::default()
            };
            let out = ctx.run(raw, |ctx| {
                egui::Area::new(egui::Id::new("levels_creep"))
                    .fixed_pos(ORIGIN)
                    .show(ctx, |ui| {
                        let _ = ui.allocate_exact_size(zone.size(), egui::Sense::hover());
                        levels_interaction(ui, &inputs, &mut state, &mut actions, zone, plot);
                    });
            });
            // Only the frames from the press onward matter.
            if f >= 2 {
                cursors.push(out.platform_output.cursor_icon);
            }
        }
        cursors
    }

    /// Once the button is down, the indicators must describe what the press
    /// grabbed and stop following the pointer.
    #[test]
    fn pre_drag_creep_does_not_flicker_to_the_bar() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        // Pressed on the handle's inner edge, which is the worst case: only 4pt
        // of travel leaves the grab radius, well under the click threshold.
        let cursors = cursors_during_pre_drag_creep((black, white), bx + 3.0);
        assert!(
            !cursors.is_empty(),
            "the harness produced no frames to check"
        );
        assert!(
            cursors
                .iter()
                .all(|c| *c == egui::CursorIcon::ResizeHorizontal),
            "cursor flickered off the handle during the pre-drag window: {cursors:?}"
        );
    }

    /// Drive two quick clicks at the same spot — a real double-click, including
    /// egui's timing requirement — and return the actions.
    fn double_click_levels(levels: (f32, f32), x: f32) -> Vec<UiAction> {
        let ctx = egui::Context::default();
        let mut state = UiState::default();
        let inputs = UiInputs {
            levels,
            ..Default::default()
        };
        let plot = egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_PLOT_W, HIST_H));
        // The zone spans the graph and the strip beneath it — the graph's own
        // area is now part of the control.
        let zone =
            egui::Rect::from_min_size(ORIGIN, egui::vec2(HIST_ROW_W, HIST_H + LEVELS_STRIP_H));
        let at = egui::pos2(x, ORIGIN.y + 4.0);
        let mut actions = Vec::new();
        let button = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let seq = [
            vec![egui::Event::PointerMoved(at)],
            vec![egui::Event::PointerMoved(at)],
            vec![button(true)],
            vec![button(false)],
            vec![button(true)],
            vec![button(false)],
        ];
        for (f, events) in seq.into_iter().enumerate() {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                // Well inside egui's double-click window.
                time: Some(f as f64 * 0.03),
                events,
                ..Default::default()
            };
            let _ = ctx.run(raw, |ctx| {
                egui::Area::new(egui::Id::new("levels_dbl"))
                    .fixed_pos(ORIGIN)
                    .show(ctx, |ui| {
                        let _ = ui.allocate_exact_size(zone.size(), egui::Sense::hover());
                        levels_interaction(ui, &inputs, &mut state, &mut actions, zone, plot);
                    });
            });
        }
        actions
    }

    /// Double-clicking a handle returns that handle to its own end of the range
    /// and leaves the other alone; double-clicking the bar clears both.
    #[test]
    fn double_click_resets_what_is_under_the_pointer() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        let wx = ORIGIN.x + white * HIST_PLOT_W;
        let mid = ORIGIN.x + 0.5 * HIST_PLOT_W;

        let acts = double_click_levels((black, white), bx);
        assert_eq!(final_levels(&acts), Some((0.0, white)), "black handle");

        let acts = double_click_levels((black, white), wx);
        assert_eq!(final_levels(&acts), Some((black, 1.0)), "white handle");

        let acts = double_click_levels((black, white), mid);
        assert_eq!(final_levels(&acts), Some((0.0, 1.0)), "the bar");
    }

    /// The last levels the gesture asked for, as `(black, white)`.
    fn final_levels(actions: &[UiAction]) -> Option<(f32, f32)> {
        actions.iter().rev().find_map(|a| match a {
            UiAction::SetLevels { black, white } => Some((*black, *white)),
            _ => None,
        })
    }

    /// Pressing just *inside* a handle — between it and the bar — must drag that
    /// handle, not the bar.
    ///
    /// This is the regression that survived two attempted fixes. egui's
    /// `drag_started` doesn't fire on the press but once the pointer has passed
    /// its click threshold, and `Response::interact_pointer_pos` returns the
    /// pointer's position *at that moment*, not where it went down — so deciding
    /// the grip from it grabbed whatever the pointer had already moved onto.
    /// Dragging inward off a handle therefore grabbed the bar.
    #[test]
    fn press_inside_a_handle_drags_that_handle() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        let wx = ORIGIN.x + white * HIST_PLOT_W;

        // Black handle, pressed on its inner edge, dragged inward (toward the bar).
        let acts = drag_levels((black, white), bx + 3.0, bx + 60.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nb > black + 0.1, "black should have moved, got {nb}");
        assert!(
            (nw - white).abs() < 1e-4,
            "white must not move when dragging the black handle, got {nw}"
        );

        // White handle, pressed on its inner edge, dragged inward.
        let acts = drag_levels((black, white), wx - 3.0, wx - 60.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nw < white - 0.1, "white should have moved, got {nw}");
        assert!(
            (nb - black).abs() < 1e-4,
            "black must not move when dragging the white handle, got {nb}"
        );
    }

    /// The outward direction, which happened to work already — kept so a fix for
    /// the inward case can't regress it.
    #[test]
    fn press_outside_a_handle_drags_that_handle() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        let acts = drag_levels((black, white), bx - 3.0, bx - 40.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nb < black - 0.1, "black should have moved down, got {nb}");
        assert!((nw - white).abs() < 1e-4, "white must not move, got {nw}");
    }

    /// Dragging the graph's own vertical edge is dragging that handle — the
    /// pattern the whole design rests on. Pressed high up in the graph, nowhere
    /// near the triangles.
    #[test]
    fn dragging_the_graph_edge_drags_that_handle() {
        let (black, white) = (0.25f32, 0.75f32);
        let wx = ORIGIN.x + white * HIST_PLOT_W;
        let acts = drag_levels_at((black, white), wx, wx - 50.0, ORIGIN.y + 6.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nw < white - 0.1, "white should have moved, got {nw}");
        assert!((nb - black).abs() < 1e-4, "black must not move, got {nb}");
    }

    /// The strip below the graph still drives the same handles.
    #[test]
    fn the_strip_below_the_graph_still_works() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        let strip_y = ORIGIN.y + HIST_H + LEVELS_STRIP_H * 0.5;
        let acts = drag_levels_at((black, white), bx, bx + 40.0, strip_y);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nb > black + 0.1, "black should have moved, got {nb}");
        assert!((nw - white).abs() < 1e-4, "white must not move, got {nw}");
    }

    /// Dragging *outside* the pair slides both, rather than grabbing the nearest
    /// handle as it used to.
    #[test]
    fn dragging_outside_the_pair_slides_both() {
        let (black, white) = (0.4f32, 0.6f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        let acts = drag_levels((black, white), bx - 40.0, bx - 10.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nb > black + 0.05, "black should have slid, got {nb}");
        assert!(
            ((nw - nb) - (white - black)).abs() < 1e-4,
            "the range width must be preserved, got {}",
            nw - nb
        );
    }

    /// With the range spanning everything, "slide both" has nowhere to go, so it
    /// must not advertise itself: no grab cursor, no highlight, no drag. That
    /// would otherwise be most of the graph offering an interaction that then
    /// does nothing at all.
    #[test]
    fn slide_both_is_inert_at_the_default_range() {
        let mid = ORIGIN.x + 0.5 * HIST_PLOT_W;
        let acts = drag_levels((0.0, 1.0), mid, mid + 40.0);
        assert_eq!(
            final_levels(&acts),
            None,
            "a drag across the full-range graph should do nothing"
        );
        let cursors = cursors_during_pre_drag_creep((0.0, 1.0), mid);
        assert!(
            cursors.iter().all(|c| *c != egui::CursorIcon::Grab),
            "the grab cursor was offered with nothing to slide: {cursors:?}"
        );
    }

    /// …but it comes back the moment either handle is pulled in, since there is
    /// then room to slide toward the other side.
    #[test]
    fn slide_both_wakes_up_once_a_handle_moves() {
        let mid = ORIGIN.x + 0.5 * HIST_PLOT_W;
        // Only the white point pulled in: sliding right is still possible.
        let acts = drag_levels((0.0, 0.8), mid, mid + 30.0);
        let (nb, nw) = final_levels(&acts).expect("the pair should still be slidable");
        assert!(nb > 0.0, "black should have slid in, got {nb}");
        assert!(
            ((nw - nb) - 0.8).abs() < 1e-4,
            "width preserved, got {}",
            nw - nb
        );
    }

    /// Handles butted together are still movable as a pair — the case that had no
    /// answer before, since the bar between them had shrunk to nothing.
    #[test]
    fn butted_handles_can_still_be_slid() {
        let (black, white) = (0.5f32, 0.505f32);
        let acts = drag_levels(
            (black, white),
            ORIGIN.x + 0.2 * HIST_PLOT_W,
            ORIGIN.x + 60.0,
        );
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(
            (nb - black).abs() > 0.02,
            "the butted pair should have moved, got {nb}"
        );
        assert!(
            ((nw - nb) - (white - black)).abs() < 1e-4,
            "the (tiny) range width must be preserved, got {}",
            nw - nb
        );
    }

    /// Pressing the bar between the handles slides both, keeping their distance.
    #[test]
    fn press_the_bar_slides_both() {
        let (black, white) = (0.25f32, 0.75f32);
        let mid = ORIGIN.x + 0.5 * HIST_PLOT_W;
        let acts = drag_levels((black, white), mid, mid + 30.0);
        let (nb, nw) = final_levels(&acts).expect("the drag must emit a levels change");
        assert!(nb > black + 0.05, "black should have slid up, got {nb}");
        assert!(
            ((nw - nb) - (white - black)).abs() < 1e-4,
            "the range width must be preserved, got {}",
            nw - nb
        );
    }

    /// A press anywhere within the grab radius belongs to the handle, in either
    /// direction — the whole width of the target, not just the half that worked.
    #[test]
    fn the_whole_grab_radius_drags_the_handle() {
        let (black, white) = (0.25f32, 0.75f32);
        let bx = ORIGIN.x + black * HIST_PLOT_W;
        for offset in [-LEVELS_GRAB + 1.0, -2.0, 0.0, 2.0, LEVELS_GRAB - 1.0] {
            let acts = drag_levels((black, white), bx + offset, bx + offset + 40.0);
            let (_, nw) = final_levels(&acts)
                .unwrap_or_else(|| panic!("no levels change for offset {offset}"));
            assert!(
                (nw - white).abs() < 1e-4,
                "offset {offset} grabbed something other than the black handle (white -> {nw})"
            );
        }
    }

    impl TitlebarFit {
        /// Count of optional elements shown (0..=5), for monotonicity checks.
        fn count(&self) -> u32 {
            [
                self.minimize,
                self.settings,
                self.maximize,
                self.app_icon,
                self.open,
            ]
            .iter()
            .filter(|&&b| b)
            .count() as u32
        }
    }

    #[test]
    fn titlebar_fit_respects_strict_priority() {
        // Across a sweep of widths and filename widths, a lower-priority element is
        // never shown while a higher-priority one is hidden (minimize ≥ settings ≥
        // maximize ≥ app-icon ≥ open).
        for name_w in [0.0_f32, 40.0, 120.0, 300.0, 600.0] {
            for w in (140..1400).step_by(7) {
                let f = titlebar_fit(w as f32, name_w);
                assert!(f.minimize || !f.settings, "settings without minimize");
                assert!(f.settings || !f.maximize, "maximize without settings");
                assert!(f.maximize || !f.app_icon, "app_icon without maximize");
                assert!(f.app_icon || !f.open, "open without app_icon");
            }
        }
    }

    #[test]
    fn titlebar_fit_is_monotonic_in_width() {
        // For a fixed filename, growing the window only ever reveals elements — it
        // never hides one that was shown at a narrower width (no flicker / churn).
        for name_w in [40.0_f32, 200.0, 450.0] {
            let mut prev = 0;
            for w in (140..1400).step_by(3) {
                let c = titlebar_fit(w as f32, name_w).count();
                assert!(c >= prev, "element hidden as width grew at {w}px");
                prev = c;
            }
        }
    }

    #[test]
    fn titlebar_fit_endpoints() {
        // A very long name leaves room for nothing but close + the (truncated) name.
        let none = titlebar_fit(400.0, 600.0);
        assert_eq!(none.count(), 0);
        // A short name at a roomy width shows every element.
        let all = titlebar_fit(400.0, 50.0);
        assert!(all.minimize && all.settings && all.maximize && all.app_icon && all.open);
        // A short name at the 170px minimum still fits the top few (close, name,
        // minimize, settings) without overlap, but not all of them.
        let tiny = titlebar_fit(170.0, 45.0);
        assert!(tiny.minimize && tiny.settings);
        assert!(!tiny.app_icon && !tiny.open);
    }

    /// Build a stack of mock flags (like `slot_flags`) inside a RIGHT_CENTER
    /// Area and return the resulting Area rect (laid out against a 1000x800
    /// screen). Runs two frames so the area geometry settles.
    fn flag_area_rect(layout: egui::Layout, cell_w: Option<f32>) -> egui::Rect {
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1000.0, 800.0),
            )),
            ..Default::default()
        };
        let mut rect = egui::Rect::NOTHING;
        for _ in 0..2 {
            let _ = ctx.run(input(), |ctx| {
                let resp = egui::Area::new(egui::Id::new("test_slots"))
                    .anchor(egui::Align2::RIGHT_CENTER, egui::Vec2::ZERO)
                    .constrain(false)
                    .show(ctx, |ui| {
                        ui.with_layout(layout, |ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(0.0, 3.0);
                            for n in 1..=3 {
                                egui::Frame {
                                    fill: egui::Color32::DARK_GRAY,
                                    inner_margin: egui::Margin::symmetric(7, 3),
                                    ..Default::default()
                                }
                                .show(ui, |ui| {
                                    if let Some(w) = cell_w {
                                        ui.set_min_width(w);
                                        ui.vertical_centered(|ui| {
                                            ui.label(
                                                egui::RichText::new(n.to_string())
                                                    .size(13.0)
                                                    .strong(),
                                            );
                                        });
                                    } else {
                                        ui.label(
                                            egui::RichText::new(n.to_string()).size(13.0).strong(),
                                        );
                                    }
                                });
                            }
                        });
                    });
                rect = resp.response.rect;
            });
        }
        rect
    }

    #[test]
    fn slot_flag_area_does_not_span_full_width() {
        // Root cause: a top_down(Align::Max) column expands to the full available
        // width to have a right edge to align against, so the RIGHT_CENTER Area's
        // interact rect spans the whole window and egui swallows clicks to the left
        // of the flags. Align::Min + uniform-width cells keeps it tight.
        let max = flag_area_rect(egui::Layout::top_down(egui::Align::Max), None);
        let min_uniform = flag_area_rect(egui::Layout::top_down(egui::Align::Min), Some(11.0));
        eprintln!(
            "Align::Max width={} (screen 1000), Align::Min+uniform width={}",
            max.width(),
            min_uniform.width()
        );
        // The buggy layout balloons to (nearly) the full 1000px screen width.
        assert!(
            max.width() > 500.0,
            "expected Align::Max to span the screen, got {}",
            max.width()
        );
        // The fix stays tight to the flags (a single digit + 2*7 margin ~= 25px).
        assert!(
            min_uniform.width() < 60.0,
            "expected the fixed layout to hug the flags, got {}",
            min_uniform.width()
        );
        // And it must still hang flush against the right screen edge.
        assert!(
            (min_uniform.max.x - 1000.0).abs() < 0.5,
            "flags should be flush right, right edge at {}",
            min_uniform.max.x
        );
    }
}
