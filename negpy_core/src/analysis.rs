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

const BASE_BLOCK: usize = 8;
const CLIP_LEVEL: f32 = 0.98;
const FLAT_CV: f32 = 0.03;
const MIN_BASE_BLOCKS: usize = 5;

/// The first half of `_base_blocks`: per-channel means of the 8x8 blocks of `det` (HxWx3) that
/// lie in `band`, hold no clipped pixel and are flat, in block order; None with fewer than
/// five. NumPy's order: each block sums row by row, the 8 rows outer, from +0.
fn flat_block_means(det: &[f32], band: &[u8], h: usize, w: usize) -> Option<Vec<[f32; 3]>> {
    let b = BASE_BLOCK;
    let (hb, wb) = (h / b, w / b);
    let rows: Vec<Vec<[f32; 3]>> = (0..hb)
        .into_par_iter()
        .map(|by| {
            let mut out = Vec::new();
            for bx in 0..wb {
                let px = |i1: usize, i3: usize| {
                    let o = ((by * b + i1) * w + bx * b + i3) * 3;
                    [det[o], det[o + 1], det[o + 2]]
                };
                let mut sum = [0f32; 3];
                let mut top = f32::NEG_INFINITY;
                let mut count = 0u32;
                for i1 in 0..b {
                    for i3 in 0..b {
                        let p = px(i1, i3);
                        for c in 0..3 {
                            sum[c] += p[c];
                            // np.max propagates NaN, so a block with one is never unclipped.
                            top = if p[c].is_nan() || top.is_nan() {
                                f32::NAN
                            } else {
                                top.max(p[c])
                            };
                        }
                        count += (band[(by * b + i1) * w + bx * b + i3] != 0) as u32;
                    }
                }
                let n = (b * b) as f32;
                let mean = sum.map(|v| v / n);
                let mut sq = [0f32; 3];
                for i1 in 0..b {
                    for i3 in 0..b {
                        let p = px(i1, i3);
                        for c in 0..3 {
                            let d = p[c] - mean[c];
                            sq[c] += d * d;
                        }
                    }
                }
                let std = sq.map(|v| (v / n).sqrt());
                let lum = ((mean[0] + mean[1]) + mean[2]) / 3.0;
                let cv = ((std[0] + std[1]) + std[2]) / 3.0 / lum.max(1e-6);
                let inside = count as f64 / (b * b) as f64 >= 0.9;
                if inside && top < CLIP_LEVEL && lum > 1e-4 && cv < FLAT_CV {
                    out.push(mean);
                }
            }
            out
        })
        .collect();
    let flat: Vec<[f32; 3]> = rows.into_iter().flatten().collect();
    (flat.len() >= MIN_BASE_BLOCKS).then_some(flat)
}

const GROUP_GAP: f32 = 0.04;

/// `_base_blocks`: the flat blocks grouped by log brightness, brightest first, split where two
/// neighbours differ by more than `GROUP_GAP` decades; the largest group of at least
/// max(5, 2%) blocks. log10 is libm's (see LEDGER.md); ties sort by block index.
pub fn base_blocks(det: &[f32], band: &[u8], h: usize, w: usize) -> Option<Vec<[f32; 3]>> {
    let cand = flat_block_means(det, band, h, w)?;
    let log_lum: Vec<f32> = cand
        .iter()
        .map(|m| libm::log10f((((m[0] + m[1]) + m[2]) / 3.0).max(1e-6)))
        .collect();
    let mut order: Vec<usize> = (0..cand.len()).collect();
    order.sort_by(|&a, &b| log_lum[a].total_cmp(&log_lum[b]));
    order.reverse();
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut start = 0;
    for i in 1..=order.len() {
        if i == order.len() || log_lum[order[i - 1]] - log_lum[order[i]] > GROUP_GAP {
            groups.push(&order[start..i]);
            start = i;
        }
    }
    let min_n = MIN_BASE_BLOCKS.max((0.02 * order.len() as f64) as usize);
    let mut best: Option<&[usize]> = None;
    for g in groups.into_iter().filter(|g| g.len() >= min_n) {
        if best.is_none_or(|b| g.len() > b.len()) {
            best = Some(g);
        }
    }
    best.map(|g| g.iter().map(|&i| cand[i]).collect())
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
