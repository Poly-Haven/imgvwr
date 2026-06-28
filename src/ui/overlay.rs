//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::colors::{panel_bg, panel_bg_alpha, ACCENT, PANEL_ALPHA};
use super::{clickable, PanoProj, RulerInfo, UiAction, UiInputs, UiState};

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

    // Auto-hiding bottom panel (tone sliders + the merged bottom ruler).
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
        if let Some(p) = r.pano {
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

/// Auto-hiding bottom panel of image-adjustment sliders (revealed by the cursor
/// near the bottom edge). Sets `state.pointer_over_panel` so the app keeps it up
/// while hovered. Currently holds the Exposure and Gamma sliders.
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
            egui::vec2(0.0, (1.0 - slide) * 96.0),
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
                (strip_dragged, sliders.response.rect)
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
    let name_text_w =
        ctx.fonts(|f| f.layout_no_wrap(name.to_owned(), name_font.clone(), fg).size().x);

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
                egui::Grid::new("imgvwr_metadata_grid")
                    .num_columns(2)
                    .spacing([12.0, 2.0])
                    .show(ui, |ui| {
                        for (key, value) in &inputs.metadata {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(key).color(egui::Color32::from_gray(150)),
                                )
                                .selectable(true),
                            );
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(value).color(egui::Color32::WHITE),
                                )
                                .selectable(true),
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
                                // `V 425px` (vertical → x pixel) / `H 312px`
                                // (horizontal → y pixel), remove button on the right.
                                // Sorted alphabetically (H before V, then by pixel),
                                // keeping each row's original guide index for removal.
                                let mut rows: Vec<(usize, &str, i64)> = inputs
                                    .guides
                                    .iter()
                                    .enumerate()
                                    .map(|(i, g)| {
                                        let (axis, dim) =
                                            if g[1] >= 0.5 { ("H", ih) } else { ("V", iw) };
                                        (i, axis, (g[0] * dim as f32).round() as i64)
                                    })
                                    .collect();
                                rows.sort_by(|a, b| a.1.cmp(b.1).then(a.2.cmp(&b.2)));
                                for (i, axis, px) in rows {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{axis} {px}px"))
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
            });
        });
    state.view_menu_open = view_menu_open;
    resp.response.contains_pointer() || view_menu_open
}

/// The `(label, colour, channel-index)` boxes to show for a channel count.
fn channel_boxes(count: u8) -> Vec<(&'static str, egui::Color32, u8)> {
    let r = egui::Color32::from_rgb(220, 80, 80);
    let g = egui::Color32::from_rgb(90, 195, 90);
    let b = egui::Color32::from_rgb(95, 145, 240);
    let a = egui::Color32::from_gray(190);
    match count {
        1 => vec![("L", a, 0)],
        2 => vec![("L", a, 0), ("A", a, 3)],
        3 => vec![("R", r, 0), ("G", g, 1), ("B", b, 2)],
        _ => vec![("R", r, 0), ("G", g, 1), ("B", b, 2), ("A", a, 3)],
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
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(3);
    painter.rect_filled(rect, radius, color);
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
        egui::Color32::from_gray(20),
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
            if ui
                .selectable_label(display == current, &display)
                .clicked()
            {
                let views = inputs.views_for(&display);
                // Keep the current view if it's valid for the new display, else
                // revert to Standard (then Raw, then the first view) — matching the
                // case-insensitive Standard→Raw convention used elsewhere (the T
                // toggle, select_standard_view, the SetView handler).
                let view = current_view
                    .clone()
                    .filter(|v| views.contains(v))
                    .or_else(|| views.iter().find(|v| v.eq_ignore_ascii_case("standard")).cloned())
                    .or_else(|| views.iter().find(|v| v.eq_ignore_ascii_case("raw")).cloned())
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
                ("← / →", "Previous / next image"),
                ("Space", "Pause / play animation (GIF/WebP/APNG)"),
                ("F2", "Metadata overlay"),
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
                ("G", "Add guide level (½, ¼ … 1/32)"),
                ("Pull from a ruler", "Drag out a guide"),
                ("Drag / right-click guide", "Move / delete it"),
                ("Channel boxes (F2)", "Isolate R/G/B/A"),
            ],
        ),
        (
            "Compare",
            &[
                ("Ctrl + 1–9", "Save to comparator slot"),
                ("1–9 (top row)", "Recall slot (again = back)"),
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
            f.layout_no_wrap(s.to_string(), egui::FontId::proportional(13.0), egui::Color32::WHITE)
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
    use super::{titlebar_fit, TitlebarFit};

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
        let min_uniform =
            flag_area_rect(egui::Layout::top_down(egui::Align::Min), Some(11.0));
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
