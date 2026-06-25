//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::colors::{panel_bg, panel_bg_alpha, ACCENT, PANEL_ALPHA};
use super::{clickable, UiAction, UiInputs, UiState};

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

    state.pointer_over_metadata = if inputs.show_metadata && !inputs.metadata.is_empty() {
        metadata_hud(ctx, inputs)
    } else {
        false
    };

    if let Some((text, alpha)) = &inputs.toast {
        toast(ctx, text, *alpha);
    }

    if inputs.show_help {
        help_dialog(ctx, actions);
    }

    // Auto-hiding borderless titlebar, drawn last so its controls sit on top.
    titlebar(ctx, inputs, actions);
}

/// The auto-hiding borderless titlebar: a drag strip showing the filename, plus
/// minimize / maximize / close controls. Opacity is `inputs.titlebar_alpha`
/// (eased by the cursor entering/leaving the window). Dragging the strip moves
/// the window (OS loop → Aero Snap); double-clicking it toggles fullscreen.
fn titlebar(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    let a = inputs.titlebar_alpha.clamp(0.0, 1.0);
    if a <= 0.01 {
        return;
    }
    let alpha = |c: u8| (c as f32 * a) as u8;
    let bg = panel_bg_alpha(alpha(PANEL_ALPHA));
    let fg = egui::Color32::from_rgba_unmultiplied(220, 220, 220, alpha(255));

    egui::TopBottomPanel::top("imgvwr_titlebar")
        .exact_height(TITLEBAR_H)
        // No separator line — its 1px stroke fades on a different schedule.
        .show_separator_line(false)
        .frame(egui::Frame::NONE.fill(bg))
        .show(ctx, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Window controls (laid out right-to-left): close, maximize, min.
                if titlebar_button(ui, "×", a, true).clicked() {
                    actions.push(UiAction::Close);
                }
                let max_glyph = if inputs.is_maximized { "❐" } else { "□" };
                if titlebar_button(ui, max_glyph, a, false).clicked() {
                    actions.push(UiAction::ToggleMaximize);
                }
                if titlebar_button(ui, "—", a, false).clicked() {
                    actions.push(UiAction::Minimize);
                }
                // The remaining strip is the drag region (move / double-click
                // fullscreen) and carries the filename.
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
                    rect.left_center() + egui::vec2(10.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    name,
                    egui::FontId::proportional(13.0),
                    fg,
                );
            });
        });
}

/// A single titlebar control glyph with a hover highlight (red for Close).
fn titlebar_button(ui: &mut egui::Ui, glyph: &str, a: f32, danger: bool) -> egui::Response {
    let size = egui::vec2(34.0, ui.available_height());
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.hovered() {
        let hover = if danger {
            egui::Color32::from_rgba_unmultiplied(232, 17, 35, (255.0 * a) as u8)
        } else {
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, (38.0 * a) as u8)
        };
        ui.painter().rect_filled(rect, 0.0, hover);
    }
    let fg = egui::Color32::from_rgba_unmultiplied(230, 230, 230, (255.0 * a) as u8);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        glyph,
        egui::FontId::proportional(15.0),
        fg,
    );
    super::clickable(resp)
}

/// Small numbered flags hanging from the top-right edge for saved comparator
/// slots; the active (currently-viewed) slot is filled with the accent colour.
/// Each flag is clickable (recall the slot) and shows its filename on hover.
fn slot_flags(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    if inputs.slot_labels.iter().all(|s| s.is_none()) {
        return;
    }
    egui::Area::new(egui::Id::new("imgvwr_slots"))
        // Hang from just below the reserved titlebar strip.
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-10.0, TITLEBAR_H))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(3.0, 0.0);
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
                    // Square top corners so the flag reads as hanging from the edge.
                    let inner = egui::Frame {
                        fill,
                        inner_margin: egui::Margin::symmetric(7, 3),
                        corner_radius: egui::CornerRadius {
                            nw: 0,
                            ne: 0,
                            sw: 3,
                            se: 3,
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

fn loading(ctx: &egui::Context, inputs: &UiInputs) {
    // Keep the bar animating (determinate fill, or barber-pole when indeterminate).
    ctx.request_repaint();
    let name = inputs.loading_name.as_deref().unwrap_or("Loading");
    let label = match inputs.progress {
        Some(p) => format!("{name}  {}%", (p * 100.0).round() as i32),
        None => format!("{name}…"),
    };
    egui::Area::new(egui::Id::new("imgvwr_loading"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::Vec2::new(12.0, -12.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame {
                fill: panel_bg(),
                inner_margin: egui::Margin::symmetric(10, 8),
                corner_radius: egui::CornerRadius::same(4),
                ..Default::default()
            }
            .show(ui, |ui| {
                ui.set_width(190.0);
                ui.label(
                    egui::RichText::new(label)
                        .color(egui::Color32::WHITE)
                        .size(13.0),
                );
                ui.add_space(4.0);
                progress_bar(ui, inputs.progress);
            });
        });
}

/// A progress bar in the accent colour. `Some(p)` fills `p`; `None` shows a full
/// bar with animated diagonal "barber-pole" stripes to signal ongoing work.
fn progress_bar(ui: &mut egui::Ui, value: Option<f32>) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(190.0, 12.0), egui::Sense::hover());
    let painter = ui.painter().with_clip_rect(rect);
    let radius = egui::CornerRadius::same(3);
    painter.rect_filled(
        rect,
        radius,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 28),
    );
    match value {
        Some(p) => {
            let mut fill = rect;
            fill.set_width(rect.width() * p.clamp(0.0, 1.0));
            painter.rect_filled(fill, radius, ACCENT);
        }
        None => {
            painter.rect_filled(rect, radius, ACCENT);
            // Scrolling diagonal stripes (a translucent lighter overlay).
            let h = rect.height();
            let stripe = 9.0;
            let period = stripe * 2.0;
            let phase = (ui.input(|i| i.time) as f32 * 36.0) % period;
            let mut x = rect.left() - h - period + phase;
            let light = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 55);
            while x < rect.right() + period {
                let pts = vec![
                    egui::pos2(x, rect.bottom()),
                    egui::pos2(x + h, rect.top()),
                    egui::pos2(x + h + stripe, rect.top()),
                    egui::pos2(x + stripe, rect.bottom()),
                ];
                painter.add(egui::Shape::convex_polygon(pts, light, egui::Stroke::NONE));
                x += period;
            }
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
                egui::RichText::new("Move cursor to left edge to open toolbar")
                    .color(egui::Color32::WHITE)
                    .size(18.0),
            );
        });
}

/// Draw the F2 metadata box (top-right, below the slot flags). Returns whether
/// the pointer is over it, so the caller can keep it visible while hovered and
/// suppress image panning over it (the values are selectable text).
fn metadata_hud(ctx: &egui::Context, inputs: &UiInputs) -> bool {
    // Below the slot-flag row so the two never overlap.
    let resp = egui::Area::new(egui::Id::new("imgvwr_metadata"))
        .anchor(
            egui::Align2::RIGHT_TOP,
            egui::Vec2::new(-10.0, 40.0 + TITLEBAR_H),
        )
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
                    });
            });
        });
    resp.response.contains_pointer()
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

/// Centred hotkey reference (toggled with H, dismissed with H/Esc/Close).
fn help_dialog(ctx: &egui::Context, actions: &mut Vec<UiAction>) {
    const KEYS: &[(&str, &str)] = &[
        ("Drag (L/M mouse)", "Pan (2D) / look around (pano)"),
        ("Alt + drag", "Move window (also titlebar / 2D-fit body)"),
        ("Window edges", "Drag to resize"),
        ("Mouse wheel", "Zoom (2D) / FOV (pano)"),
        ("Shift + wheel", "Pan horizontally"),
        ("Ctrl + wheel", "Pan vertically"),
        (", / .", "Exposure −/+"),
        ("Ctrl + , / .", "Gamma −/+"),
        ("Ctrl + R", "Reset exposure & gamma"),
        ("Numpad 1–9", "Zoom in 2^(N-1)× (Ctrl = zoom out)"),
        ("← / →", "Previous / next image in folder"),
        ("Ctrl + 1–9", "Save to comparator slot"),
        ("1–9 (top row)", "Recall slot (again = toggle back)"),
        ("L", "Lock zoom/pan across images"),
        ("P", "Toggle 2D / panorama"),
        ("W", "Toggle 2D tiled wrap"),
        ("I", "Toggle nearest / bilinear filtering"),
        ("T", "Toggle Standard / last view transform"),
        ("O", "Open file…"),
        ("F2", "Metadata overlay"),
        ("Home / Backspace", "Reset view (fit)"),
        ("F / F11 / dbl-click", "Toggle fullscreen"),
        ("H", "This help"),
        ("Esc / Q", "Exit fullscreen / quit"),
    ];
    egui::Area::new(egui::Id::new("imgvwr_help"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                ui.set_max_width(420.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Keyboard & mouse")
                            .strong()
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                    );
                });
                ui.add_space(8.0);
                egui::Grid::new("imgvwr_help_grid")
                    .num_columns(2)
                    .spacing([18.0, 4.0])
                    .show(ui, |ui| {
                        for (key, action) in KEYS {
                            ui.label(
                                egui::RichText::new(*key)
                                    .color(egui::Color32::from_rgb(180, 200, 255)),
                            );
                            ui.label(
                                egui::RichText::new(*action).color(egui::Color32::from_gray(220)),
                            );
                            ui.end_row();
                        }
                    });
                ui.add_space(10.0);
                ui.vertical_centered(|ui| {
                    if clickable(ui.button("Close")).clicked() {
                        actions.push(UiAction::CloseHelp);
                    }
                });
            });
        });
}
