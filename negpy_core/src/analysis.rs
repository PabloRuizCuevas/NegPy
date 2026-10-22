//! Ports of the frame analysis in negpy/features/geometry/logic.py and exposure/normalization.py.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::stats::percentiles;

const LUMA_R: f64 = 0.2126;
const LUMA_G: f64 = 0.7152;
const LUMA_B: f64 = 0.0722;

/// `kernel.image.logic.get_luminance` of an HxWx3 float32 image: Numba's fastmath kernel,
/// f64 with fused multiply-adds, then float32.
pub fn luminance(rgb: &[f32], dst: &mut [f32]) {
    dst.par_chunks_mut(4096)
        .zip(rgb.par_chunks(4096 * 3))
        .for_each(|(d, p)| luminance_chunk(p, d));
}

/// `mul_add` is one instruction only where the CPU has FMA, so this compiles twice.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn luminance_chunk(rgb: &[f32], dst: &mut [f32]) {
    for (d, p) in dst.iter_mut().zip(rgb.as_chunks::<3>().0) {
        let t = LUMA_R * p[0] as f64;
        let t = LUMA_G.mul_add(p[1] as f64, t);
        *d = LUMA_B.mul_add(p[2] as f64, t) as f32;
    }
}

/// `_normalize_to_uint8` for a float32 image: the 1st and 99th percentiles of its finite
/// values stretched to 0..255, NumPy's float32 arithmetic (NEP 50) and truncating cast.
pub fn normalize_to_uint8(img: &[f32], dst: &mut [u8]) {
    let valid: Vec<f32> = img.iter().copied().filter(|v| v.is_finite()).collect();
    let Some(p) = percentiles(&valid, &[1.0, 99.0]) else {
        dst.fill(0);
        return;
    };
    let low = p[0] as f64;
    let high = if p[1] as f64 <= low {
        low + 1.0
    } else {
        p[1] as f64
    };
    let (low32, scale32) = (low as f32, (255.0 / (high - low)) as f32);
    dst.par_iter_mut().zip(img.par_iter()).for_each(|(d, &v)| {
        let s = ((v - low32) * scale32).clamp(0.0, 255.0);
        // NumPy's float32 to uint8 cast truncates, NaN to 0 on x86; Rust's `as` does the same.
        *d = s as u8;
    });
}

/// `_detection_luma` of a float32 luminance plane: divided by its 99.5th percentile and
/// clipped to 0..2, in float32.
pub fn detection_luma(lum: &[f32], dst: &mut [f32]) -> Option<()> {
    // np.percentile is NaN when any value is, and Python's max(nan, 1e-6) keeps the NaN.
    let p = percentiles(lum, &[99.5])?[0] as f64;
    let anchor = if lum.iter().any(|v| v.is_nan()) {
        f32::NAN
    } else {
        p.max(1e-6) as f32
    };
    dst.par_iter_mut().zip(lum.par_iter()).for_each(|(d, &v)| {
        *d = (v / anchor).clamp(0.0, 2.0);
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_spans_0_to_255() {
        let img: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let mut out = vec![0u8; img.len()];
        normalize_to_uint8(&img, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[999], 255);
    }

    #[test]
    fn all_nan_normalizes_to_zero() {
        let img = vec![f32::NAN; 10];
        let mut out = vec![7u8; 10];
        normalize_to_uint8(&img, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }
}
