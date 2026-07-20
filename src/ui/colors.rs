//! Shared UI colours, centralised here for easy adjustment.

use egui::Color32;

/// Base RGB for every chrome background.
const PANEL_RGB: (u8, u8, u8) = (20, 20, 20);
/// Default chrome background opacity (75%).
pub const PANEL_ALPHA: u8 = 191;

/// Standard translucent dark background for all chrome — toolbar, titlebar,
/// overlays, metadata box, toast, slot flags: `rgb(20, 20, 20)` at 75% opacity.
pub fn panel_bg() -> Color32 {
    panel_bg_alpha(PANEL_ALPHA)
}

/// [`panel_bg`] at a custom alpha, for the elements that fade (toast, titlebar).
pub fn panel_bg_alpha(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(PANEL_RGB.0, PANEL_RGB.1, PANEL_RGB.2, alpha)
}

/// Accent colour (active comparator slot flag, progress bar).
pub const ACCENT: Color32 = Color32::from_rgb(190, 111, 255);

/// R/G/B channel colours — the metadata box's channel-isolation swatches, and
/// anywhere else a value wants to read as "this is the red/green/blue channel"
/// (e.g. the colour-pick tooltip's Linear/Display rows).
pub const CHANNEL_R: Color32 = Color32::from_rgb(220, 80, 80);
pub const CHANNEL_G: Color32 = Color32::from_rgb(90, 195, 90);
pub const CHANNEL_B: Color32 = Color32::from_rgb(95, 145, 240);

/// Guide tooltip fill while hovering (or grab-dragging) an existing guide.
/// Matches [`CHANNEL_B`].
pub const GUIDE_HOVER: Color32 = CHANNEL_B;
/// Guide tooltip fill while dragging a new guide out of a ruler. Matches
/// [`CHANNEL_G`].
pub const GUIDE_ADD: Color32 = CHANNEL_G;
