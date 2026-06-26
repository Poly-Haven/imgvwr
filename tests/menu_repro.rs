//! Regression guard for the View-transform menu keep-open logic.
//!
//! egui's `menu_custom_button` returns an `InnerResponse` whose `.inner` is
//! `Some` ONLY on the frame the menu *closes* — it reads `None` the entire time
//! the menu is open. So `resp.inner.is_some()` is the wrong "is the menu open?"
//! signal (it made the metadata box hide and dismiss the menu when the pointer
//! moved into the popup). The authoritative state is egui's stored `BarState`,
//! keyed by the ui id. This test pins that behaviour down.

use egui::{pos2, vec2, Button, Context, Event, Modifiers, PointerButton, RawInput, Rect};

/// Render an `Area` with a `menu_custom_button` and report
/// `(bar_state_open, inner_is_some)`.
fn frame(ctx: &Context, events: Vec<Event>) -> (bool, bool) {
    let input = RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 400.0))),
        events,
        ..Default::default()
    };
    let mut out = (false, false);
    let _ = ctx.run(input, |ctx| {
        egui::Area::new("box".into())
            .fixed_pos(pos2(100.0, 50.0))
            .show(ctx, |ui| {
                let bar_id = ui.id();
                let resp = egui::menu::menu_custom_button(ui, Button::new("View"), |ui| {
                    let _ = ui.button("AgX");
                    let _ = ui.button("Filmic");
                });
                let bar_open = egui::menu::BarState::load(ui.ctx(), bar_id).is_some();
                out = (bar_open, resp.inner.is_some());
            });
    });
    out
}

fn click(pos: egui::Pos2) -> [Event; 2] {
    let m = Modifiers::default();
    [
        Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed: true,
            modifiers: m,
        },
        Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: m,
        },
    ]
}

#[test]
fn barstate_tracks_menu_open_when_pointer_enters_popup() {
    let ctx = Context::default();
    let btn = pos2(110.0, 60.0);

    frame(&ctx, vec![]); // warmup (egui's first pass is a sizing pass)
    frame(&ctx, vec![Event::PointerMoved(btn)]); // hover the button
    let [press, release] = click(btn);
    let on_click = frame(&ctx, vec![Event::PointerMoved(btn), press, release]);
    let opened = frame(&ctx, vec![]);

    // Move the pointer down into the popup (below the button) and settle.
    let m1 = frame(&ctx, vec![Event::PointerMoved(pos2(110.0, 92.0))]);
    let m2 = frame(&ctx, vec![Event::PointerMoved(pos2(112.0, 100.0))]);
    let m3 = frame(&ctx, vec![]);

    // BarState reports the menu open from the click onward, including while the
    // pointer is in the popup — even though `inner.is_some()` is false throughout.
    for (label, (bar_open, inner_some)) in [
        ("on_click", on_click),
        ("opened", opened),
        ("into_popup_1", m1),
        ("into_popup_2", m2),
        ("settle", m3),
    ] {
        assert!(
            bar_open,
            "BarState should report the menu open at `{label}`"
        );
        assert!(
            !inner_some,
            "menu_custom_button .inner should be None while open at `{label}`",
        );
    }
}
