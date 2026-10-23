//! The Lab-stage kernels of negpy/features/lab/logic.py, in the float32 order NumPy evaluates
//! them, around the bit-exact OpenCV filters of `filter` and `morph`.

use rayon::prelude::*;

use crate::color::{self, LabSpace};
use crate::filter::{self, Border};

// Sharpen constants of logic.py; lab.wgsl mirrors them.
const GATE_LO: f32 = 0.25;
const GATE_HI: f64 = 0.33;
const OVERSHOOT_LIGHT: f32 = 1.0;
const OVERSHOOT_DARK: f32 = 2.0;
const MASK_T_HI: f64 = 10.0;
const SHADOW_FLOOR: f64 = 1.0 / 3.0;
const SHADOW_L_HI: f32 = 35.0;

/// `_smoothstep(e0, e1, x)`: the edges are Python floats, so e1 - e0 is a float64 difference
/// rounded to float32 once.
#[inline(always)]
fn smoothstep(e0: f64, e1: f64, x: f32) -> f32 {
    let t = clip((x - e0 as f32) / (e1 - e0) as f32, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `sharpen_shadow_gain(l)`.
#[inline(always)]
fn shadow_gain(l: f32) -> f32 {
    let f = SHADOW_FLOOR as f32;
    f + (1.0 - f) * smoothstep(0.0, SHADOW_L_HI as f64, l)
}

/// `_edge_mask(l, masking)`: central differences on the edge-padded plane, `np.hypot` (glibc's
/// float64 formula), a 3x3 replicate-border box mean, then the smoothstep over 0.5t..t.
fn edge_mask(l: &[f32], (h, w): (usize, usize), masking: f64) -> Option<Vec<f32>> {
    let mut hyp = vec![0f32; h * w];
    hyp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let (up, down) = (y.saturating_sub(1), (y + 1).min(h - 1));
        let row = &l[y * w..][..w];
        for (x, o) in out.iter_mut().enumerate() {
            let gx = (row[(x + 1).min(w - 1)] - row[x.saturating_sub(1)]) * 0.5;
            let gy = (l[down * w + x] - l[up * w + x]) * 0.5;
            let (gx, gy) = (gx as f64, gy as f64);
            *o = (gx * gx + gy * gy).sqrt() as f32;
        }
    });
    let mut grad = vec![0f32; h * w];
    filter::box_filter(&hyp, &mut grad, (h, w, 1), (3, 3), true, Border::Replicate)?;
    let t = MASK_T_HI * masking;
    grad.par_iter_mut()
        .for_each(|g| *g = smoothstep(0.5 * t, t, *g));
    Some(grad)
}

/// `_usm_sharpening_block` of an HxWx3 linear image: the L* channel's soft-gated unsharp mask
/// with `k` as the blur taps, clamped to the local 3x3 range, then clipped to 0..1 in RGB.
/// `gain` is `np.float32(amount * 2.5)`. None where a filter does not cover the shape.
pub fn usm_sharpen(
    src: &[f32],
    (h, w): (usize, usize),
    (gain, masking): (f32, f64),
    k: &[f32],
    (to_lab, to_rgb): (LabSpace, LabSpace),
    dst: &mut [f32],
) -> Option<()> {
    let n = h * w;
    let mut lab = vec![0f32; n * 3];
    color::rgb_to_lab(src, &mut lab, to_lab);
    let l: Vec<f32> = lab.par_chunks(3).map(|p| p[0]).collect();
    let mut blur = vec![0f32; n];
    filter::sep_filter2d(&l, &mut blur, (h, w, 1), (k, k), Border::Reflect101)?;
    let edge = if masking > 0.0 {
        Some(edge_mask(&l, (h, w), masking)?)
    } else {
        None
    };
    // The 3x3 erode and dilate (replicated edges) run inline, a row at a time.
    lab.par_chunks_mut(w * 3).enumerate().for_each(|(y, out)| {
        let rows = [y.saturating_sub(1), y, (y + 1).min(h - 1)].map(|r| &l[r * w..][..w]);
        let col = |x: usize, f: fn(f32, f32) -> f32| f(f(rows[0][x], rows[1][x]), rows[2][x]);
        let (mut lo_col, mut hi_col) = (vec![0f32; w], vec![0f32; w]);
        for x in 0..w {
            lo_col[x] = col(x, f32_lo);
            hi_col[x] = col(x, f32_hi);
        }
        for (x, p) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let lo = f32_lo(f32_lo(lo_col[xl], lo_col[x]), lo_col[xr]);
            let hi = f32_hi(f32_hi(hi_col[xl], hi_col[x]), hi_col[xr]);
            let i = y * w + x;
            let (l0, d) = (l[i], l[i] - blur[i]);
            let mut g = gain * smoothstep(GATE_LO as f64, GATE_HI, d.abs()) * shadow_gain(l0);
            if let Some(e) = &edge {
                g *= e[i];
            }
            let v = clip(l0 + d * g, lo - OVERSHOOT_DARK, hi + OVERSHOOT_LIGHT);
            p[0] = clip(v, 0.0, 100.0);
        }
    });
    color::lab_to_rgb(&lab, dst, to_rgb);
    dst.par_iter_mut().for_each(|v| *v = clip(*v, 0.0, 1.0));
    Some(())
}

/// `cv2.erode`'s float32 minimum.
fn f32_lo(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// `cv2.dilate`'s float32 maximum.
fn f32_hi(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// `np.clip(v, lo, hi)`, NumPy's `min(max(v, lo), hi)` with `max(a, b) = a > b ? a : b`
/// unless a is NaN: a NaN in any argument comes through, and clip(-0.0, 0.0, ..) is +0.0.
#[inline(always)]
fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}
