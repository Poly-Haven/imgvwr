//! Camera state and controller (see plans/rewrite.md §10).
//!
//! The camera is an enum, not a single struct with overloaded fields. Panorama
//! and 2D are genuinely different state with different ranges and semantics;
//! conflating them (reusing yaw/pitch as pan) is a known bug magnet, especially
//! across the P-key transition. We keep them distinct and convert explicitly.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use glam::{Vec2, Vec3};

pub const MIN_PITCH_RAD: f32 = -FRAC_PI_2;
pub const MAX_PITCH_RAD: f32 = FRAC_PI_2;
pub const MIN_FOV_DEG: f32 = 0.5;
/// Rectilinear maximum, set on entering panorama mode.
pub const PANORAMA_MAX_FOV_DEG: f32 = 140.0;
pub const FLAT_MAX_FOV_DEG: f32 = 170.0;
/// Default zoom when an image first opens in 2D mode (90° => tan_half_fov 1.0).
pub const FLAT_FIT_FOV_DEG: f32 = 90.0;
/// Default field of view when an image first opens in panorama mode.
pub const DEFAULT_PANO_FOV_DEG: f32 = 100.0;

/// Time constant (seconds) for the exponential easing of zoom/FOV/pan toward
/// their target. Smaller = snappier. Frame-rate independent (see `animate`).
const EASE_TAU: f32 = 0.035;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Camera {
    /// Equirectangular look-around.
    Pano {
        yaw_rad: f32,
        pitch_rad: f32,
        fov_deg: f32,
    },
    /// 2D pan / zoom. `pan` is the screen-centre UV offset (added to 0.5);
    /// `zoom` is a scale factor (1.0 = fit, larger = zoomed in).
    Flat { pan: Vec2, zoom: f32 },
}

impl Camera {
    /// Uniform values for the shader (`u_yaw`, `u_pitch`, `u_tan_half_fov`).
    pub fn yaw(&self) -> f32 {
        match self {
            Camera::Pano { yaw_rad, .. } => *yaw_rad,
            // 2D pan_u = u_yaw / 2π  =>  u_yaw = pan.x * 2π.
            Camera::Flat { pan, .. } => pan.x * TAU,
        }
    }

    pub fn pitch(&self) -> f32 {
        match self {
            Camera::Pano { pitch_rad, .. } => *pitch_rad,
            // 2D pan_v = -u_pitch / π  =>  u_pitch = -pan.y * π.
            Camera::Flat { pan, .. } => -pan.y * PI,
        }
    }

    /// `u_tan_half_fov` (panorama: tan of half FOV; 2D: inverse zoom).
    pub fn tan_half_fov(&self) -> f32 {
        match self {
            Camera::Pano { fov_deg, .. } => (fov_deg.to_radians() * 0.5).tan(),
            Camera::Flat { zoom, .. } => 1.0 / zoom.max(1e-4),
        }
    }

    pub fn half_fov_radians(&self) -> f32 {
        self.tan_half_fov().atan()
    }

    /// 0 = equirectangular panorama, 1 = 2D pan/zoom.
    pub fn projection_mode(&self) -> i32 {
        match self {
            Camera::Pano { .. } => 0,
            Camera::Flat { .. } => 1,
        }
    }

    /// The screen-centre image UV for the current look direction / pan.
    pub fn center_uv(&self) -> Vec2 {
        match self {
            Camera::Pano {
                yaw_rad, pitch_rad, ..
            } => pano_center_uv(*yaw_rad, *pitch_rad),
            Camera::Flat { pan, .. } => Vec2::new((0.5 + pan.x).rem_euclid(1.0), 0.5 + pan.y),
        }
    }
}

/// The controller owns the rendered camera plus the `target` it eases toward.
///
/// Zoom (2D), FOV (panorama) and 2D pan animate: setters that should ease
/// (wheel, numpad, Home) write `target`, and [`animate`](Self::animate) moves
/// `camera` toward it each frame. Direct manipulation (drag look / drag pan) and
/// the panorama look angle are instant — they write `camera` and `target`
/// together, and `animate` snaps the look angle rather than easing it.
#[derive(Clone, Debug)]
pub struct CameraController {
    /// The rendered camera (eased toward `target`).
    pub camera: Camera,
    /// Destination the eased fields move toward.
    target: Camera,
}

impl CameraController {
    /// Build the camera for a freshly-loaded image (settled: camera == target).
    pub fn for_image(equirectangular: bool) -> Self {
        let camera = if equirectangular {
            Camera::Pano {
                yaw_rad: 0.0,
                pitch_rad: 0.0,
                fov_deg: DEFAULT_PANO_FOV_DEG,
            }
        } else {
            Camera::Flat {
                pan: Vec2::ZERO,
                zoom: 1.0,
            }
        };
        Self {
            camera,
            target: camera,
        }
    }

    pub fn is_panorama(&self) -> bool {
        matches!(self.camera, Camera::Pano { .. })
    }

    /// The target the eased fields are heading toward (for the zoom/FOV toast).
    pub fn target(&self) -> Camera {
        self.target
    }

    /// Target 2D zoom, if in flat mode (used by the window-follow).
    pub fn target_zoom(&self) -> Option<f32> {
        match self.target {
            Camera::Flat { zoom, .. } => Some(zoom),
            Camera::Pano { .. } => None,
        }
    }

    /// Finish any in-progress easing immediately (e.g. across an image load, so
    /// the new image doesn't animate in from the previous view).
    pub fn snap(&mut self) {
        self.camera = self.target;
    }

    /// Freeze the target at the current camera, so no easing kicks in after the
    /// rendered camera was set directly (e.g. headless debug overrides).
    pub fn settle(&mut self) {
        self.target = self.camera;
    }

    /// Set the absolute 2D zoom on both the rendered camera and its target
    /// (instant, no easing) — for comparator swaps that must not animate. No-op
    /// on `Pano`.
    pub fn set_zoom_now(&mut self, scale: f32) {
        let z = scale.max(flat_zoom_min());
        for cam in [&mut self.camera, &mut self.target] {
            if let Camera::Flat { zoom, .. } = cam {
                *zoom = z;
            }
        }
    }

    /// Snap the 2D view to centred-and-fit (pan 0, zoom 1) — instant. No-op on
    /// `Pano`. Used when entering 2D mode in a normal window so the image fills
    /// the (re-framed) window with no black canvas, rather than carrying a look
    /// direction across from panorama mode.
    pub fn center_flat_now(&mut self) {
        for cam in [&mut self.camera, &mut self.target] {
            if let Camera::Flat { pan, zoom } = cam {
                *pan = Vec2::ZERO;
                *zoom = 1.0;
            }
        }
    }

    /// Rotate the panorama look (radians) — instant (drag). No-op on `Flat`.
    pub fn rotate(&mut self, dyaw_rad: f32, dpitch_rad: f32) {
        for cam in [&mut self.camera, &mut self.target] {
            if let Camera::Pano {
                yaw_rad, pitch_rad, ..
            } = cam
            {
                *yaw_rad += dyaw_rad;
                *pitch_rad = (*pitch_rad + dpitch_rad).clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
            }
        }
    }

    /// Snap the panorama look to an absolute yaw/pitch — instant (Home). No-op
    /// on `Flat`.
    pub fn snap_look(&mut self, yaw_rad: f32, pitch_rad: f32) {
        for cam in [&mut self.camera, &mut self.target] {
            if let Camera::Pano {
                yaw_rad: y,
                pitch_rad: p,
                ..
            } = cam
            {
                *y = yaw_rad;
                *p = pitch_rad.clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
            }
        }
    }

    /// Pan in 2D by a UV delta — instant (drag). No-op on `Pano`.
    pub fn pan(&mut self, d_uv: Vec2) {
        for cam in [&mut self.camera, &mut self.target] {
            if let Camera::Flat { pan, .. } = cam {
                *pan += d_uv;
            }
        }
    }

    /// Pan in 2D by a UV delta — eased (wheel pan). No-op on `Pano`.
    pub fn pan_target(&mut self, d_uv: Vec2) {
        if let Camera::Flat { pan, .. } = &mut self.target {
            *pan += d_uv;
        }
    }

    /// Set the absolute 2D pan target — eased (Home). No-op on `Pano`.
    pub fn set_pan_target(&mut self, pan: Vec2) {
        if let Camera::Flat { pan: p, .. } = &mut self.target {
            *p = pan;
        }
    }

    /// Adjust panorama FOV by `delta_deg` — eased. No-op on `Flat`.
    pub fn adjust_fov(&mut self, delta_deg: f32) {
        if let Camera::Pano { fov_deg, .. } = &mut self.target {
            *fov_deg = (*fov_deg + delta_deg).clamp(MIN_FOV_DEG, PANORAMA_MAX_FOV_DEG);
        }
    }

    /// Multiply the 2D zoom target by `factor` — eased. No-op on `Pano`.
    /// Zoom-in is uncapped; only the zoom-out (minimum) bound is enforced.
    pub fn adjust_zoom(&mut self, factor: f32) {
        if let Camera::Flat { zoom, .. } = &mut self.target {
            *zoom = (*zoom * factor).max(flat_zoom_min());
        }
    }

    /// Set the absolute 2D zoom target (zoom-in uncapped) — eased. No-op on
    /// `Pano`. Used by the numpad exact-zoom keys and the window-follow.
    pub fn set_zoom(&mut self, scale: f32) {
        if let Camera::Flat { zoom, .. } = &mut self.target {
            *zoom = scale.max(flat_zoom_min());
        }
    }

    /// Set the absolute panorama FOV target — eased. No-op on `Flat`.
    pub fn set_fov(&mut self, fov: f32) {
        if let Camera::Pano { fov_deg, .. } = &mut self.target {
            *fov_deg = fov.clamp(MIN_FOV_DEG, PANORAMA_MAX_FOV_DEG);
        }
    }

    /// Switch projection mode, preserving the screen-centre pixel and zoom level
    /// across the transition (§10). Instant: both `camera` and `target` convert.
    pub fn set_mode(&mut self, panorama: bool) {
        if panorama == self.is_panorama() {
            return;
        }
        self.camera = switched(self.camera, panorama);
        self.target = switched(self.target, panorama);
    }

    /// Ease the rendered camera toward `target` by `dt` seconds. Returns true
    /// while still moving, so the event loop keeps scheduling frames (and stops
    /// — returning to `ControlFlow::Wait` — once settled). Frame-rate
    /// independent: the per-frame fraction is `1 - exp(-dt/EASE_TAU)`.
    pub fn animate(&mut self, dt: f32) -> bool {
        let k = 1.0 - (-dt / EASE_TAU).exp();
        match (&mut self.camera, self.target) {
            (Camera::Flat { pan, zoom }, Camera::Flat { pan: tp, zoom: tz }) => {
                // Zoom eases in log space so it feels uniform across magnitudes.
                let nz = (zoom.ln() + (tz.ln() - zoom.ln()) * k).exp();
                let np = *pan + (tp - *pan) * k;
                let zoom_settled = (nz - tz).abs() <= tz * 1e-3;
                let pan_settled = (tp - np).length() <= 1e-4;
                *zoom = if zoom_settled { tz } else { nz };
                *pan = if pan_settled { tp } else { np };
                !(zoom_settled && pan_settled)
            }
            (
                Camera::Pano {
                    yaw_rad,
                    pitch_rad,
                    fov_deg,
                },
                Camera::Pano {
                    yaw_rad: ty,
                    pitch_rad: tpi,
                    fov_deg: tf,
                },
            ) => {
                // The look angle is instant (already kept equal); only FOV eases.
                *yaw_rad = ty;
                *pitch_rad = tpi;
                let nf = *fov_deg + (tf - *fov_deg) * k;
                let settled = (nf - tf).abs() <= 1e-2;
                *fov_deg = if settled { tf } else { nf };
                !settled
            }
            // Mode mismatch never happens (set_mode converts both); nothing to do.
            _ => false,
        }
    }
}

/// Convert a camera to the other projection mode, preserving the screen-centre
/// pixel and the zoom/FOV equivalence (§10).
fn switched(cam: Camera, panorama: bool) -> Camera {
    let uv = cam.center_uv();
    if panorama {
        let (yaw_rad, pitch_rad) = uv_to_yaw_pitch(uv);
        let fov_deg = match cam {
            Camera::Flat { zoom, .. } => {
                zoom_to_fov_deg(zoom).clamp(MIN_FOV_DEG, PANORAMA_MAX_FOV_DEG)
            }
            _ => DEFAULT_PANO_FOV_DEG,
        };
        Camera::Pano {
            yaw_rad,
            pitch_rad,
            fov_deg,
        }
    } else {
        let pan = Vec2::new(uv.x - 0.5, uv.y - 0.5);
        let zoom = match cam {
            Camera::Pano { fov_deg, .. } => fov_to_zoom(fov_deg),
            _ => 1.0,
        };
        Camera::Flat { pan, zoom }
    }
}

/// Forward look direction for `(yaw, pitch)` (matches the shader's
/// `rotation_yaw_pitch * (0,0,1)`).
fn forward_dir(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        yaw.sin() * pitch.cos(),
        -pitch.sin(),
        yaw.cos() * pitch.cos(),
    )
}

/// Equirectangular UV of a direction (matches the shader exactly).
fn direction_to_equirect_uv(dir: Vec3) -> Vec2 {
    let lon = dir.z.atan2(dir.x);
    let lat = dir.y.clamp(-1.0, 1.0).asin();
    let u = 1.0 - (lon / TAU + 0.5);
    let v = 0.5 - lat / PI;
    Vec2::new(u.rem_euclid(1.0), v)
}

/// Screen-centre UV for a panorama look direction.
fn pano_center_uv(yaw: f32, pitch: f32) -> Vec2 {
    direction_to_equirect_uv(forward_dir(yaw, pitch))
}

/// Inverse of `pano_center_uv`: the yaw/pitch that centres `uv`.
fn uv_to_yaw_pitch(uv: Vec2) -> (f32, f32) {
    let yaw = TAU * uv.x - FRAC_PI_2;
    let pitch = (PI * (uv.y - 0.5)).clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
    (yaw, pitch)
}

/// 2D zoom (scale) <-> equivalent FOV. inv_zoom = tan(fov/2) = 1/zoom.
fn zoom_to_fov_deg(zoom: f32) -> f32 {
    (1.0 / zoom.max(1e-4)).atan().to_degrees() * 2.0
}

fn fov_to_zoom(fov_deg: f32) -> f32 {
    1.0 / (fov_deg.to_radians() * 0.5).tan().max(1e-4)
}

/// 2D zoom-out bound (the most zoomed-out / widest equivalent FOV). Zoom-in is
/// intentionally uncapped.
fn flat_zoom_min() -> f32 {
    fov_to_zoom(FLAT_MAX_FOV_DEG)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn uv_approx(a: Vec2, b: Vec2, eps: f32) -> bool {
        // Compare with horizontal wrap (u is periodic).
        let du = (a.x - b.x).rem_euclid(1.0);
        let du = du.min(1.0 - du);
        du <= eps && approx(a.y, b.y, eps)
    }

    #[test]
    fn pano_to_flat_preserves_center_pixel() {
        let cases = [
            (0.0_f32, 0.0_f32),
            (0.7, 0.3),
            (-1.2, -0.4),
            (2.5, 0.9),
            (PI, -1.0),
        ];
        for (yaw, pitch) in cases {
            let mut c = CameraController {
                camera: Camera::Pano {
                    yaw_rad: yaw,
                    pitch_rad: pitch,
                    fov_deg: 100.0,
                },
                target: Camera::Flat {
                    pan: Vec2::ZERO,
                    zoom: 1.0,
                },
            };
            let before = c.camera.center_uv();
            c.set_mode(false); // -> Flat
            let mid = c.camera.center_uv();
            assert!(
                uv_approx(before, mid, 1e-4),
                "Flat center {mid:?} != {before:?}"
            );
            c.set_mode(true); // -> Pano
            let after = c.camera.center_uv();
            assert!(
                uv_approx(before, after, 1e-4),
                "round-trip {after:?} != {before:?}"
            );
        }
    }

    #[test]
    fn flat_to_pano_round_trip_preserves_pan() {
        let pans = [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.2, -0.1),
            Vec2::new(-0.3, 0.25),
        ];
        for pan in pans {
            let mut c = CameraController {
                camera: Camera::Flat { pan, zoom: 1.5 },
                target: Camera::Flat {
                    pan: Vec2::ZERO,
                    zoom: 1.0,
                },
            };
            let before = c.camera.center_uv();
            c.set_mode(true); // -> Pano
            c.set_mode(false); // -> Flat
            let after = c.camera.center_uv();
            assert!(
                uv_approx(before, after, 1e-4),
                "pan round-trip {after:?} != {before:?}"
            );
        }
    }

    #[test]
    fn yaw_pitch_uv_inverse() {
        for (yaw, pitch) in [(0.0_f32, 0.0_f32), (1.0, 0.5), (-2.0, -0.8)] {
            let uv = pano_center_uv(yaw, pitch);
            let (y2, p2) = uv_to_yaw_pitch(uv);
            // Yaw is periodic; compare wrapped.
            let dyaw = (yaw - y2).rem_euclid(TAU);
            let dyaw = dyaw.min(TAU - dyaw);
            assert!(dyaw <= 1e-3, "yaw {yaw} != {y2}");
            assert!(approx(pitch, p2, 1e-3), "pitch {pitch} != {p2}");
        }
    }

    #[test]
    fn zoom_fov_inverse() {
        for fov in [30.0_f32, 60.0, 90.0, 120.0] {
            let z = fov_to_zoom(fov);
            assert!(approx(zoom_to_fov_deg(z), fov, 1e-3));
        }
        // 90° FOV == fit zoom 1.0.
        assert!(approx(fov_to_zoom(FLAT_FIT_FOV_DEG), 1.0, 1e-4));
    }

    #[test]
    fn pitch_is_clamped() {
        let mut c = CameraController::for_image(true);
        c.rotate(0.0, 100.0);
        if let Camera::Pano { pitch_rad, .. } = c.camera {
            assert!(pitch_rad <= MAX_PITCH_RAD + 1e-6);
        } else {
            panic!("expected pano");
        }
    }
}
