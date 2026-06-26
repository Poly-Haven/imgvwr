//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::colors::{panel_bg, panel_bg_alpha, ACCENT, PANEL_ALPHA};
use super::{clickable, RulerInfo, UiAction, UiInputs, UiState};

/// Height of the borderless custom titlebar; the top strip is reserved for it so
/// the slot flags / metadata box never sit under the window controls.
const TITLEBAR_H: f32 = 30.0;

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
        help_dialog(ctx, actions);
    }

    settings_dialog(ctx, inputs, state, actions);

    // The left pixel ruler (drawn before the bottom panel so the panel covers the
    // overlapping bottom-left corner). The bottom ruler lives inside the panel.
    left_ruler(ctx, inputs, state, actions);

    // Interactive grab/move/delete strips for existing guides (lowest order, so
    // they only catch clicks that land on a guide line).
    guides_layer(ctx, inputs, state, actions);

    // Auto-hiding bottom panel (tone sliders + the merged bottom ruler).
    bottom_panel(ctx, inputs, state, actions);

    // Auto-hiding borderless titlebar, drawn last so its controls sit on top.
    titlebar(ctx, inputs, actions);
}

/// Ruler strip thickness (px). 3px shorter than the original 24 (see ticks).
const RULER_W: f32 = 21.0;
/// Tick `(image-pixel interval, on-screen length, coarser interval to skip)`. A
/// level is skipped once its ticks would be closer than ~3px on screen.
const RULER_LEVELS: [(f32, f32, f32); 4] = [
    (100.0, 17.0, 0.0),
    (50.0, 10.0, 100.0),
    (10.0, 6.0, 50.0),
    (5.0, 3.0, 10.0),
];

/// Tick colour: the old gray-210 darkened by 30%.
fn ruler_tick_stroke() -> egui::Stroke {
    egui::Stroke::new(1.0, egui::Color32::from_gray(147))
}

// Screen ↔ image-coordinate mappings shared by the rulers and the interactive
// guide layer (these mirror the shader's 2D UV mapping exactly). A *vertical*
// guide sits at a constant image u (its on-screen line is vertical); a
// *horizontal* guide at a constant image v.

/// On-screen x of a vertical guide at image-u `coord` (0..1).
fn guide_u_to_x(r: &RulerInfo, screen: egui::Rect, coord: f32) -> f32 {
    screen.left() + screen.width() * (0.5 + (coord - 0.5 - r.pan_u) / r.sx)
}
/// Image-u (0..1) under a pointer at screen x.
fn x_to_guide_u(r: &RulerInfo, screen: egui::Rect, x: f32) -> f32 {
    0.5 + r.pan_u + ((x - screen.left()) / screen.width() - 0.5) * r.sx
}
/// On-screen y of a horizontal guide at image-v `coord` (0..1).
fn guide_v_to_y(r: &RulerInfo, screen: egui::Rect, coord: f32) -> f32 {
    screen.top() + screen.height() * (0.5 - (0.5 + r.pan_v - coord) / r.sy)
}
/// Image-v (0..1) under a pointer at screen y.
fn y_to_guide_v(r: &RulerInfo, screen: egui::Rect, y: f32) -> f32 {
    0.5 + r.pan_v - ((1.0 - (y - screen.top()) / screen.height()) - 0.5) * r.sy
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
        if horizontal {
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
            let (py0, py1) = (uv_y(rect.top()) * r.img_h, uv_y(rect.bottom()) * r.img_h);
            for (interval, len, coarser) in RULER_LEVELS {
                if interval * ppx_y < 3.0 {
                    continue;
                }
                for k in (py0.min(py1) / interval).floor() as i64
                    ..=(py0.max(py1) / interval).ceil() as i64
                {
                    let pos = k as f32 * interval;
                    if pos < 0.0 || pos > r.img_h || (coarser > 0.0 && (pos % coarser).abs() < 0.5)
                    {
                        continue;
                    }
                    let y = ey(pos);
                    if y < rect.top() || y > rect.bottom() {
                        continue;
                    }
                    p.line_segment([egui::pos2(base_x, y), egui::pos2(base_x + len, y)], stroke);
                }
            }
            resp
        });
    // Keep the ruler alive for the whole spawn-drag, even as the pointer leaves
    // the strip into the image (the drag stays routed to the strip's id).
    state.pointer_over_left_ruler = resp.response.contains_pointer() || resp.inner.dragged();
    // Drag a NEW *vertical* guide out of the (vertical) left ruler — Photoshop
    // style: it tracks the pointer's x as you pull it into the image. A plain
    // click does nothing (no drag → no guide).
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
    let ex = |px: f32| screen.left() + vw * (0.5 + (px / r.img_w - 0.5 - r.pan_u) / r.sx);
    let uv_x = |sx: f32| 0.5 + r.pan_u + ((sx - screen.left()) / vw - 0.5) * r.sx;
    let ppx_x = (vw / (r.img_w * r.sx)).abs();

    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(vw, RULER_W), egui::Sense::click_and_drag());
    let base_y = rect.bottom();
    let p = ui.painter();
    let stroke = ruler_tick_stroke();
    let (px0, px1) = (
        uv_x(screen.left()) * r.img_w,
        uv_x(screen.right()) * r.img_w,
    );
    for (interval, len, coarser) in RULER_LEVELS {
        if interval * ppx_x < 3.0 {
            continue;
        }
        for k in (px0.min(px1) / interval).floor() as i64..=(px0.max(px1) / interval).ceil() as i64
        {
            let pos = k as f32 * interval;
            if pos < 0.0 || pos > r.img_w || (coarser > 0.0 && (pos % coarser).abs() < 0.5) {
                continue;
            }
            let x = ex(pos);
            p.line_segment([egui::pos2(x, base_y), egui::pos2(x, base_y - len)], stroke);
        }
    }
    // Pull a NEW *horizontal* guide upward out of the (horizontal) bottom ruler.
    ruler_spawn_drag(ui.ctx(), &resp, r, screen, true, guides_len, state, actions);
    resp.dragged()
}

/// Interactive layer for grabbing / moving / deleting existing guides directly
/// on the image (2D only — it uses the ruler mapping). Each guide gets a thin,
/// full-length hit strip along its on-screen line at the LOWEST egui order, so a
/// click away from any guide still falls through to the image (pan), while a
/// click near a line grabs it. Hover → grab cursor + mark the guide for the
/// renderer's hover colour; drag → move it; right-click or drag-off → delete.
fn guides_layer(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    state.hovered_guide = None;
    let Some(r) = inputs.ruler else {
        return;
    };
    if inputs.guides.is_empty() {
        return;
    }
    let screen = ctx.screen_rect();
    const HALF: f32 = 4.0; // hit half-thickness around the line (px)
    egui::Area::new(egui::Id::new("imgvwr_guides"))
        .order(egui::Order::Background)
        .fixed_pos(screen.min)
        .constrain(false)
        .show(ctx, |ui| {
            for (i, g) in inputs.guides.iter().enumerate() {
                let horizontal = g[1] >= 0.5;
                let rect = if horizontal {
                    let y = guide_v_to_y(&r, screen, g[0]);
                    egui::Rect::from_min_max(
                        egui::pos2(screen.left(), y - HALF),
                        egui::pos2(screen.right(), y + HALF),
                    )
                } else {
                    let x = guide_u_to_x(&r, screen, g[0]);
                    egui::Rect::from_min_max(
                        egui::pos2(x - HALF, screen.top()),
                        egui::pos2(x + HALF, screen.bottom()),
                    )
                };
                let id = ui.id().with(("guide", i));
                let resp = ui.interact(rect, id, egui::Sense::click_and_drag());
                if resp.hovered() || resp.dragged() {
                    state.hovered_guide = Some(i);
                    ctx.set_cursor_icon(if resp.dragged() {
                        egui::CursorIcon::Grabbing
                    } else {
                        egui::CursorIcon::Grab
                    });
                }
                // Right-click removes.
                if resp.secondary_clicked() {
                    actions.push(UiAction::RemoveGuide(i));
                    continue;
                }
                // Map the pointer to this guide's axis coordinate (for move/drop).
                let coord = ctx.pointer_interact_pos().map(|pt| {
                    if horizontal {
                        y_to_guide_v(&r, screen, pt.y)
                    } else {
                        x_to_guide_u(&r, screen, pt.x)
                    }
                });
                if resp.dragged() {
                    if let Some(coord) = coord {
                        actions.push(UiAction::MoveGuide { index: i, coord });
                    }
                }
                if resp.drag_stopped() {
                    // Released past the image edge / off-screen → discard it.
                    let off = match (coord, ctx.pointer_interact_pos()) {
                        (Some(c), Some(pt)) => !(0.0..=1.0).contains(&c) || !screen.contains(pt),
                        _ => false,
                    };
                    if off {
                        actions.push(UiAction::RemoveGuide(i));
                    }
                }
            }
        });
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
                egui::Frame::NONE
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
                strip_dragged
            });
            let content_rect = content.response.rect;
            let strip_dragged = content.inner;
            ui.painter()
                .set(bg, egui::Shape::rect_filled(content_rect, 0.0, panel_bg()));
            // Reset button, pinned to the panel's bottom-right corner — drawn into
            // the SAME area, on top of the background, so its hit-test is
            // unambiguous. (A separate overlapping area had its clicks swallowed
            // by the panel and rendered faint behind the panel background.) Its
            // placement is the area's right edge, independent of the slider group
            // widths (the slider rows already reserve room for it on the right).
            (reset_button(ui, content_rect), strip_dragged)
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

/// The bottom-panel Reset button: an icon that resets all image adjustments
/// (same as Ctrl+R), drawn at the bottom-right corner of the panel `area` rect.
/// Carries a permanent subtle chip so it reads as a button (not faint icon).
fn reset_button(ui: &mut egui::Ui, area: egui::Rect) -> egui::Response {
    let size = egui::vec2(35.0, 30.0);
    let pad = 8.0;
    let rect = egui::Rect::from_min_size(
        egui::pos2(area.right() - pad - size.x, area.bottom() - pad - size.y),
        size,
    );
    let resp = ui.interact(rect, ui.id().with("reset_btn"), egui::Sense::click());
    let chip = if resp.hovered() { 52 } else { 26 };
    ui.painter()
        .rect_filled(rect, 5.0, egui::Color32::from_white_alpha(chip));
    let icon = egui::Image::new(egui::include_image!(
        "../../resources/icons/ui/arrow-counterclockwise.svg"
    ))
    .tint(egui::Color32::from_gray(235));
    icon.paint_at(
        ui,
        egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(19.0)),
    );
    clickable(resp).on_hover_text("Reset all adjustments (Ctrl+R)")
}

/// The settings dialog (opened from the toolbar): startup-display picker and the
/// "set as default viewer" action (with a confirmation step).
fn settings_dialog(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    if !state.show_settings {
        return;
    }
    egui::Area::new(egui::Id::new("imgvwr_settings"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                ui.set_min_width(340.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Settings")
                            .strong()
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                    );
                });
                ui.add_space(14.0);

                // Startup display.
                ui.horizontal(|ui| {
                    ui.label("Open on display:");
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
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    inputs.startup_display.is_none(),
                                    "Remember last used",
                                )
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
                });
                ui.add_space(14.0);

                // Window corner radius (live).
                ui.horizontal(|ui| {
                    ui.label("Window corner radius:");
                    let mut radius = inputs.corner_radius;
                    let resp = ui.add(
                        egui::DragValue::new(&mut radius)
                            .range(0..=40)
                            .suffix(" px"),
                    );
                    if resp.changed() {
                        actions.push(UiAction::SetCornerRadius(radius));
                    }
                });
                ui.add_space(14.0);

                // Background colour (behind transparent images).
                ui.horizontal(|ui| {
                    ui.label("Background colour:");
                    let mut col = inputs.background_color;
                    if ui.color_edit_button_srgb(&mut col).changed() {
                        actions.push(UiAction::SetBackgroundColor(col));
                    }
                });
                ui.add_space(14.0);

                // Guide-line colour.
                ui.horizontal(|ui| {
                    ui.label("Guide colour:");
                    let mut col = inputs.guide_color;
                    if ui.color_edit_button_srgb(&mut col).changed() {
                        actions.push(UiAction::SetGuideColor(col));
                    }
                });
                ui.add_space(14.0);

                // Auto-exposure for HDR panoramas on load.
                let mut auto = inputs.auto_exposure;
                if ui
                    .checkbox(&mut auto, "Auto-expose HDR panoramas on open")
                    .changed()
                {
                    actions.push(UiAction::SetAutoExposure(auto));
                }
                ui.add_space(14.0);

                // Set as default viewer, with a confirmation step.
                if state.confirm_default {
                    ui.label("Make imgvwr the default viewer for all supported image types?");
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

                ui.add_space(14.0);
                ui.vertical_centered(|ui| {
                    if clickable(ui.button("Close")).clicked() {
                        state.show_settings = false;
                        state.confirm_default = false;
                    }
                });
            });
        });
}

/// The auto-hiding borderless titlebar: a drag strip showing the filename, plus
/// minimize / maximize / close controls. Opacity is `inputs.titlebar_slide`
/// (eased by the cursor entering/leaving the window). Dragging the strip moves
/// the window (OS loop → Aero Snap); double-clicking it toggles fullscreen.
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
                        // Window controls (laid out right-to-left): close, maximize,
                        // min, then settings — Bootstrap SVG icons.
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
                        let max_icon = if inputs.is_maximized {
                            egui::include_image!("../../resources/icons/ui/window-stack.svg")
                        } else {
                            egui::include_image!("../../resources/icons/ui/square.svg")
                        };
                        if titlebar_button(ui, max_icon, 13.0, a, false).clicked() {
                            actions.push(UiAction::ToggleMaximize);
                        }
                        if titlebar_button(
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
                        if titlebar_button(
                            ui,
                            // Filled gear: renders bolder/crisper at this size than
                            // the thin-stroked outline gear (same as the Open icon).
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
                            if let Some(icon) = &inputs.icon {
                                let (icon_rect, _) = ui.allocate_exact_size(
                                    egui::vec2(26.0, TITLEBAR_H),
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
                            if titlebar_button(
                                ui,
                                // Filled folder: renders bold/crisp at this size,
                                // unlike the thin-stroked outline open-folder icon.
                                egui::include_image!("../../resources/icons/ui/folder-fill.svg"),
                                15.0,
                                a,
                                false,
                            )
                            .on_hover_text("Open file…")
                            .clicked()
                            {
                                actions.push(UiAction::OpenFile);
                            }
                            // Filename over the remaining drag strip.
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
                            let name = if inputs.title.is_empty() {
                                "imgvwr"
                            } else {
                                inputs.title.as_str()
                            };
                            ui.painter().text(
                                rect.left_center() + egui::vec2(6.0, 0.0),
                                egui::Align2::LEFT_CENTER,
                                name,
                                egui::FontId::proportional(13.0),
                                fg,
                            );
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
    let btn = egui::vec2(34.0, ui.available_height());
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
        // box, which can then sit closer to the edge).
        .anchor(egui::Align2::RIGHT_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 3.0);
                for (i, label) in inputs.slot_labels.iter().enumerate() {
                    let Some(label) = label else {
                        continue;
                    };
                    let active = inputs.active_slot == Some(i);
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
                        ui.label(
                            egui::RichText::new((i + 1).to_string())
                                .color(fg)
                                .size(13.0)
                                .strong(),
                        );
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
    let text = |s: String| egui::RichText::new(s).color(egui::Color32::WHITE).size(13.0);
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
    egui::Color32::from_rgb(243, 130, 55),
    egui::Color32::from_rgb(65, 187, 217),
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
        const SPEED: f32 = 32.0;
        let t = ui.input(|i| i.time) as f32 * SPEED;
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
                        // View transform: a dropdown of views, with a Display
                        // sub-sub-menu, when OCIO is available.
                        if inputs.has_image {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new("View")
                                        .color(egui::Color32::from_gray(150)),
                                )
                                .selectable(false),
                            );
                            view_menu_open = view_dropdown(ui, inputs, actions);
                            ui.end_row();
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
                                for (i, g) in inputs.guides.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        // `V 425px` (vertical → x pixel) / `H 312px`
                                        // (horizontal → y pixel), remove button at
                                        // the right.
                                        let horizontal = g[1] >= 0.5;
                                        let (axis, dim) =
                                            if horizontal { ("H", ih) } else { ("V", iw) };
                                        let px = (g[0] * dim as f32).round() as i64;
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
fn view_dropdown(ui: &mut egui::Ui, inputs: &UiInputs, actions: &mut Vec<UiAction>) -> bool {
    let current = inputs
        .active
        .as_ref()
        .map(|(d, v)| format!("{d}/{v}"))
        .unwrap_or_else(|| "gamma 2.2".to_string());
    if !inputs.ocio_available {
        ui.label(egui::RichText::new(current).color(egui::Color32::WHITE));
        return false;
    }
    let active_display = inputs
        .active
        .as_ref()
        .map(|(d, _)| d.clone())
        .or_else(|| inputs.displays().first().cloned())
        .unwrap_or_default();
    // The trigger uses an SVG chevron icon (not a font glyph) before the current
    // view name. The nested "Display" sub-menus get egui's own ▸ arrow, so they
    // carry no manual one (that produced a double arrow).
    let chevron = egui::Image::new(egui::include_image!(
        "../../resources/icons/ui/chevron-down.svg"
    ))
    .tint(egui::Color32::WHITE)
    .fit_to_exact_size(egui::vec2(10.0, 10.0));
    let button = egui::Button::image_and_text(
        chevron,
        egui::RichText::new(current).color(egui::Color32::WHITE),
    );
    // The menu state lives in egui's `BarState`, keyed by this ui's id.
    let bar_id = ui.id();
    egui::menu::menu_custom_button(ui, button, |ui| {
        for view in inputs.views_for(&active_display) {
            let is_active = inputs
                .active
                .as_ref()
                .is_some_and(|(d, v)| d == &active_display && v == &view);
            if ui.selectable_label(is_active, &view).clicked() {
                actions.push(UiAction::SetView {
                    display: active_display.clone(),
                    view: view.clone(),
                });
                ui.close_menu();
            }
        }
        ui.separator();
        ui.menu_button("Display", |ui| {
            for display in inputs.displays() {
                ui.menu_button(display.clone(), |ui| {
                    for view in inputs.views_for(&display) {
                        if ui.selectable_label(false, &view).clicked() {
                            actions.push(UiAction::SetView {
                                display: display.clone(),
                                view: view.clone(),
                            });
                            ui.close_menu();
                        }
                    }
                });
            }
        });
    });
    // Whether the View menu (or a sub-menu) is open. The menu_custom_button
    // return's `.inner` is NOT a reliable signal — egui only surfaces it on the
    // frame the menu *closes*, so it reads false the whole time the menu is open.
    // The authoritative state is egui's stored BarState for this ui's id.
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

/// Centred hotkey reference (toggled with H, dismissed with H/Esc/Close), laid
/// out as side-by-side sections to keep it short vertically.
fn help_dialog(ctx: &egui::Context, actions: &mut Vec<UiAction>) {
    type Section = (&'static str, &'static [(&'static str, &'static str)]);
    const SECTIONS: &[Section] = &[
        (
            "View & zoom",
            &[
                ("Drag (L/M)", "Pan / look around"),
                ("Wheel", "Zoom (2D) / FOV (pano)"),
                ("Shift / Ctrl + wheel", "Pan horizontally / vertically"),
                ("Numpad 1–9", "Zoom 2^(N-1)× (Ctrl = out)"),
                ("Home / Backspace / R", "Reset view & window"),
                ("P", "Toggle 2D / panorama"),
                ("W", "Toggle 2D tiled wrap"),
                ("I", "Nearest / bilinear filtering"),
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
                ("T", "Standard / last view transform"),
                ("O", "Open file…"),
                ("← / →", "Previous / next image"),
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
                ("Alt + middle-drag", "Squash / stretch image"),
                ("G", "Add guide (½, ¼ … to 1/32)"),
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
    egui::Area::new(egui::Id::new("imgvwr_help"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Keyboard & mouse")
                            .strong()
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                    );
                });
                ui.add_space(10.0);
                ui.horizontal_top(|ui| {
                    for (i, (title, keys)) in SECTIONS.iter().enumerate() {
                        if i > 0 {
                            ui.add_space(22.0);
                        }
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new(*title).strong().color(ACCENT));
                            ui.add_space(4.0);
                            egui::Grid::new(("imgvwr_help", *title))
                                .num_columns(2)
                                .spacing([10.0, 3.0])
                                .show(ui, |ui| {
                                    for (key, action) in *keys {
                                        ui.label(
                                            egui::RichText::new(*key)
                                                .color(egui::Color32::from_rgb(180, 200, 255)),
                                        );
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
                ui.add_space(12.0);
                ui.vertical_centered(|ui| {
                    if clickable(ui.button("Close")).clicked() {
                        actions.push(UiAction::CloseHelp);
                    }
                });
            });
        });
}
