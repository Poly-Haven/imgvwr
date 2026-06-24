//! Left-edge hover toolbar (see plans/rewrite.md §12.1, §12.2).

use super::{UiAction, UiInputs, UiState};

const PANEL_WIDTH: f32 = 200.0;

fn panel_frame() -> egui::Frame {
    egui::Frame {
        fill: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 170),
        inner_margin: egui::Margin::same(6),
        ..Default::default()
    }
}

/// Build the toolbar and its submenus, pushing any chosen actions into `actions`.
pub fn build_toolbar(
    ctx: &egui::Context,
    inputs: &UiInputs,
    state: &mut UiState,
    actions: &mut Vec<UiAction>,
) {
    if !inputs.toolbar_visible {
        state.show_view_submenu = false;
        state.show_display_submenu = false;
        state.pointer_over_panel = false;
        return;
    }

    // Root column.
    egui::SidePanel::left("imgvwr_toolbar")
        .resizable(false)
        .exact_width(PANEL_WIDTH)
        .frame(panel_frame())
        .show(ctx, |ui| {
            ui.add_space(4.0);
            if ui.button("📂  Open file…").clicked() {
                actions.push(UiAction::OpenFile);
            }
            if ui
                .add_enabled(inputs.has_image, egui::Button::new("🔁  Reload"))
                .clicked()
            {
                actions.push(UiAction::Reload);
            }

            let vt = ui.add_enabled(
                inputs.ocio_available,
                egui::Button::new("🎨  View transform ›"),
            );
            if vt.clicked() {
                state.show_view_submenu = !state.show_view_submenu;
                if state.show_view_submenu && state.browse_display.is_none() {
                    state.browse_display =
                        inputs.active.as_ref().map(|(d, _)| d.clone());
                }
            }

            ui.separator();
            ui.label(
                egui::RichText::new(if inputs.file_info.is_empty() {
                    "No image".to_string()
                } else {
                    inputs.file_info.clone()
                })
                .small()
                .color(egui::Color32::from_gray(180)),
            );
        });

    // View-transform submenu (second column).
    if state.show_view_submenu {
        let browse = state
            .browse_display
            .clone()
            .or_else(|| inputs.active.as_ref().map(|(d, _)| d.clone()))
            .or_else(|| inputs.displays().first().cloned());

        egui::SidePanel::left("imgvwr_view_submenu")
            .resizable(false)
            .exact_width(PANEL_WIDTH)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.add_space(4.0);
                let display = browse.clone().unwrap_or_default();
                if ui
                    .button(format!("Display: {display} ›"))
                    .clicked()
                {
                    state.show_display_submenu = !state.show_display_submenu;
                }
                if let Some((_, active_view)) = &inputs.active {
                    ui.label(
                        egui::RichText::new(format!("View: {active_view}"))
                            .color(egui::Color32::from_gray(160)),
                    );
                }
                ui.separator();
                for view in inputs.views_for(&display) {
                    let is_active = inputs
                        .active
                        .as_ref()
                        .is_some_and(|(d, v)| d == &display && v == &view);
                    let label = if is_active {
                        egui::RichText::new(format!("● {view}")).strong()
                    } else {
                        egui::RichText::new(format!("   {view}"))
                    };
                    if ui.button(label).clicked() {
                        actions.push(UiAction::SetView {
                            display: display.clone(),
                            view: view.clone(),
                        });
                        state.show_view_submenu = false;
                        state.show_display_submenu = false;
                    }
                }
            });
    }

    // Display submenu (third column).
    if state.show_view_submenu && state.show_display_submenu {
        egui::SidePanel::left("imgvwr_display_submenu")
            .resizable(false)
            .exact_width(PANEL_WIDTH)
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Display").color(egui::Color32::from_gray(160)));
                ui.separator();
                for display in inputs.displays() {
                    if ui.button(&display).clicked() {
                        state.browse_display = Some(display.clone());
                        state.show_display_submenu = false;
                    }
                }
            });
    }

    state.pointer_over_panel = ctx.is_pointer_over_area();
}
