//! Centred overlays drawn over the scene: loading spinner, error panel, the
//! "no image" hint, and the F2 metadata HUD (see plans/rewrite.md §12.3, §12.5,
//! §12.6).

use super::{UiAction, UiInputs};

fn overlay_frame() -> egui::Frame {
    egui::Frame {
        fill: egui::Color32::from_rgba_unmultiplied(20, 20, 20, 220),
        inner_margin: egui::Margin::same(16),
        corner_radius: egui::CornerRadius::same(8),
        ..Default::default()
    }
}

pub fn build_overlays(ctx: &egui::Context, inputs: &UiInputs, actions: &mut Vec<UiAction>) {
    if inputs.loading {
        loading(ctx, inputs);
    } else if let Some(error) = &inputs.error {
        error_panel(ctx, inputs, error, actions);
    } else if inputs.show_hint {
        hint(ctx);
    }

    if inputs.show_metadata && !inputs.metadata.is_empty() {
        metadata_hud(ctx, inputs);
    }
}

fn loading(ctx: &egui::Context, inputs: &UiInputs) {
    // Keep animating the spinner while a load is in flight.
    ctx.request_repaint();
    egui::Area::new(egui::Id::new("imgvwr_loading"))
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .interactable(false)
        .show(ctx, |ui| {
            overlay_frame().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add(egui::Spinner::new().size(36.0));
                    ui.add_space(8.0);
                    let label = match &inputs.loading_name {
                        Some(name) => format!("Opening {name}…"),
                        None => "Opening file…".to_string(),
                    };
                    ui.label(egui::RichText::new(label).color(egui::Color32::WHITE));
                });
            });
        });
}

fn error_panel(
    ctx: &egui::Context,
    inputs: &UiInputs,
    error: &str,
    actions: &mut Vec<UiAction>,
) {
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

fn metadata_hud(ctx: &egui::Context, inputs: &UiInputs) {
    egui::Area::new(egui::Id::new("imgvwr_metadata"))
        .anchor(egui::Align2::LEFT_TOP, egui::Vec2::new(10.0, 10.0))
        .interactable(false)
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
                            ui.label(
                                egui::RichText::new(key)
                                    .color(egui::Color32::from_gray(150))
                                    .small(),
                            );
                            ui.label(
                                egui::RichText::new(value)
                                    .color(egui::Color32::WHITE)
                                    .small(),
                            );
                            ui.end_row();
                        }
                    });
            });
        });
}
