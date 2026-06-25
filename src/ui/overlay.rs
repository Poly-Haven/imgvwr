//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::{UiAction, UiInputs, UiState};

/// Accent colour for the active comparator slot flag.
const ACCENT: egui::Color32 = egui::Color32::from_rgb(190, 111, 255);

fn overlay_frame() -> egui::Frame {
    egui::Frame {
        fill: egui::Color32::from_rgba_unmultiplied(20, 20, 20, 220),
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
}

/// Small numbered flags hanging from the top-right edge for saved comparator
/// slots; the active (currently-viewed) slot is filled with the accent colour.
/// Each flag is clickable (recall the slot) and shows its filename on hover.
fn slot_flags(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    if inputs.slot_labels.iter().all(|s| s.is_none()) {
        return;
    }
    egui::Area::new(egui::Id::new("imgvwr_slots"))
        // Touch the top edge of the screen (y = 0).
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-10.0, 0.0))
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
                        (
                            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 170),
                            egui::Color32::from_gray(200),
                        )
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
                    let resp = inner
                        .response
                        .interact(egui::Sense::click())
                        .on_hover_text(label);
                    if resp.clicked() {
                        actions.push(UiAction::RecallSlot(i));
                    }
                }
            });
        });
}

fn loading(ctx: &egui::Context, inputs: &UiInputs) {
    // Keep the bar (determinate fill or indeterminate sweep) animating.
    ctx.request_repaint();
    let determinate = inputs.progress.is_some();
    let value = match inputs.progress {
        Some(p) => p.clamp(0.0, 1.0),
        // Indeterminate (decoding): a repeating left-to-right sweep.
        None => {
            let t = ctx.input(|i| i.time);
            (t % 1.2) as f32 / 1.2
        }
    };
    let name = inputs.loading_name.as_deref().unwrap_or("Loading");
    let label = match inputs.progress {
        Some(p) => format!("{name}  {}%", (p * 100.0).round() as i32),
        None => format!("{name}…"),
    };
    egui::Area::new(egui::Id::new("imgvwr_loading"))
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::Vec2::new(-12.0, -12.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame {
                fill: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200),
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
                ui.add(
                    egui::ProgressBar::new(value)
                        .desired_width(190.0)
                        .animate(!determinate),
                );
            });
        });
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
                    if ui.button("Dismiss").clicked() {
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
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-10.0, 40.0))
        .show(ctx, |ui| {
            let frame = egui::Frame {
                fill: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
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
    let bg = (180.0 * a) as u8;
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
                fill: egui::Color32::from_rgba_unmultiplied(0, 0, 0, bg),
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
        ("Home", "Reset view (fit)"),
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
                    if ui.button("Close").clicked() {
                        actions.push(UiAction::CloseHelp);
                    }
                });
            });
        });
}
