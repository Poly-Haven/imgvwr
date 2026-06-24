//! Camera state and controller (see plans/rewrite.md §10).
//!
//! The camera is an enum, not a single struct with overloaded fields. Panorama
//! and 2-D are genuinely different state with different ranges and semantics;
//! conflating them (reusing yaw/pitch as pan) is a known bug magnet, especially
//! across the P-key transition. We keep them distinct and convert explicitly.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use glam::{Vec2, Vec3};

pub const MIN_PITCH_RAD: f32 = -FRAC_PI_2;
pub const MAX_PITCH_RAD: f32 = FRAC_PI_2;
pub const MIN_FOV_DEG: f32 = 5.0;
/// Rectilinear maximum, set on entering panorama mode.
pub const PANORAMA_MAX_FOV_DEG: f32 = 140.0;
pub const FLAT_MAX_FOV_DEG: f32 = 170.0;
/// Default zoom when an image first opens in 2-D mode (90° => tan_half_fov 1.0).
pub const FLAT_FIT_FOV_DEG: f32 = 90.0;
/// Default field of view when an image first opens in panorama mode.
pub const DEFAULT_PANO_FOV_DEG: f32 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Camera {
    /// Equirectangular look-around.
    Pano {
        yaw_rad: f32,
        pitch_rad: f32,
        fov_deg: f32,
    },
    /// 2-D pan / zoom. `pan` is the screen-centre UV offset (added to 0.5);
    /// `zoom` is a scale factor (1.0 = fit, larger = zoomed in).
    Flat { pan: Vec2, zoom: f32 },
}

impl Camera {
    /// Uniform values for the shader (`u_yaw`, `u_pitch`, `u_tan_half_fov`).
    pub fn yaw(&self) -> f32 {
        match self {
            Camera::Pano { yaw_rad, .. } => *yaw_rad,
            // 2-D pan_u = u_yaw / 2π  =>  u_yaw = pan.x * 2π.
            Camera::Flat { pan, .. } => pan.x * TAU,
        }
    }

    pub fn pitch(&self) -> f32 {
        match self {
            Camera::Pano { pitch_rad, .. } => *pitch_rad,
            // 2-D pan_v = -u_pitch / π  =>  u_pitch = -pan.y * π.
            Camera::Flat { pan, .. } => -pan.y * PI,
        }
    }

    /// `u_tan_half_fov` (panorama: tan of half FOV; 2-D: inverse zoom).
    pub fn tan_half_fov(&self) -> f32 {
        match self {
            Camera::Pano { fov_deg, .. } => (fov_deg.to_radians() * 0.5).tan(),
            Camera::Flat { zoom, .. } => 1.0 / zoom.max(1e-4),
        }
    }

    pub fn half_fov_radians(&self) -> f32 {
        self.tan_half_fov().atan()
    }

    /// 0 = equirectangular panorama, 1 = 2-D pan/zoom.
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

/// The controller owns the live camera plus the post-load reset state (Home).
#[derive(Clone, Debug)]
pub struct CameraController {
    pub camera: Camera,
    reset: Camera,
}

impl CameraController {
    /// Build the camera for a freshly-loaded image, capturing the reset state.
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
            reset: camera,
        }
    }

    pub fn is_panorama(&self) -> bool {
        matches!(self.camera, Camera::Pano { .. })
    }

    /// Restore the camera captured just after the last load (Home key).
    pub fn reset_view(&mut self) {
        self.camera = self.reset;
    }

    /// Rotate in panorama mode (radians). No-op on `Flat`.
    pub fn rotate(&mut self, dyaw_rad: f32, dpitch_rad: f32) {
        if let Camera::Pano {
            yaw_rad, pitch_rad, ..
        } = &mut self.camera
        {
            *yaw_rad += dyaw_rad;
            *pitch_rad = (*pitch_rad + dpitch_rad).clamp(MIN_PITCH_RAD, MAX_PITCH_RAD);
        }
    }

    /// Pan in 2-D mode by a UV delta. No-op on `Pano`.
    pub fn pan(&mut self, d_uv: Vec2) {
        if let Camera::Flat { pan, .. } = &mut self.camera {
            *pan += d_uv;
        }
    }

    /// Adjust panorama FOV by `delta_deg` (clamped). No-op on `Flat`.
    pub fn adjust_fov(&mut self, delta_deg: f32) {
        if let Camera::Pano { fov_deg, .. } = &mut self.camera {
            *fov_deg = (*fov_deg + delta_deg).clamp(MIN_FOV_DEG, PANORAMA_MAX_FOV_DEG);
        }
    }

    /// Multiply 2-D zoom by `factor` (clamped to the FOV range). No-op on `Pano`.
    pub fn adjust_zoom(&mut self, factor: f32) {
        if let Camera::Flat { zoom, .. } = &mut self.camera {
            let min_zoom = fov_to_zoom(FLAT_MAX_FOV_DEG);
            let max_zoom = fov_to_zoom(MIN_FOV_DEG);
            *zoom = (*zoom * factor).clamp(min_zoom, max_zoom);
        }
    }

    /// Switch projection mode, preserving the screen-centre pixel and zoom level
    /// across the transition (§10).
    pub fn set_mode(&mut self, panorama: bool) {
        if panorama == self.is_panorama() {
            return;
        }
        let uv = self.camera.center_uv();
        self.camera = if panorama {
            let (yaw_rad, pitch_rad) = uv_to_yaw_pitch(uv);
            let fov_deg = match self.camera {
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
            let zoom = match self.camera {
                Camera::Pano { fov_deg, .. } => fov_to_zoom(fov_deg),
                _ => 1.0,
            };
            Camera::Flat { pan, zoom }
        };
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

/// 2-D zoom (scale) <-> equivalent FOV. inv_zoom = tan(fov/2) = 1/zoom.
fn zoom_to_fov_deg(zoom: f32) -> f32 {
    (1.0 / zoom.max(1e-4)).atan().to_degrees() * 2.0
}

fn fov_to_zoom(fov_deg: f32) -> f32 {
    1.0 / (fov_deg.to_radians() * 0.5).tan().max(1e-4)
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
                reset: Camera::Flat {
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
                reset: Camera::Flat {
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
