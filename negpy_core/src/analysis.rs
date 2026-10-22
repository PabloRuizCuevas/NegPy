//! Ports of the frame analysis in negpy/features/geometry/logic.py and exposure/normalization.py.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::stats::{np_percentiles, percentiles};

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
    // Python's max(nan, 1e-6) keeps the NaN.
    let p = np_percentiles(lum, &[99.5])?[0] as f64;
    let anchor = if p.is_nan() {
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

/// A copy of `img[y1:y2, x1:x2]` of an HxW plane, row by row.
fn crop(img: &[f32], w: usize, (y1, y2, x1, x2): (usize, usize, usize, usize)) -> Vec<f32> {
    (y1..y2)
        .flat_map(|y| img[y * w + x1..y * w + x2].iter().copied())
        .collect()
}

/// Python's round() of a float: half to even.
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `_find_rebate_level(lum, film_roi, bed)`: the brightest uniform rebate plateau on a side of
/// the film box, (level, spread), when two opposite sides show one.
pub fn find_rebate_level(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
    bed: Option<f64>,
) -> Option<(f64, f64)> {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return None;
    }
    let bed = match bed {
        Some(b) => b,
        None => np_percentiles(lum, &[99.0])?[0] as f64,
    };
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let box_median = np_percentiles(&boxed, &[50.0])?[0] as f64;
    let ring = py_round(0.04 * bh.min(bw) as f64).max(3) as usize;
    let ring_h = ring.min(bh);
    let ring_w = ring.min(bw);
    // top, bottom, left, right: NumPy slices of the box, in its row-major order.
    let sides = [
        crop(&boxed, bw, (0, ring_h, 0, bw)),
        crop(&boxed, bw, (bh - ring_h, bh, 0, bw)),
        crop(&boxed, bw, (0, bh, 0, ring_w)),
        crop(&boxed, bw, (0, bh, bw - ring_w, bw)),
    ];
    // NEP 50: the Python float threshold joins the float32 comparison as float32.
    let cut = (bed - 0.05) as f32;
    let mut qualifying: [Option<(f64, f64)>; 4] = [None; 4];
    for (slot, strip) in qualifying.iter_mut().zip(&sides) {
        let vals: Vec<f32> = strip.iter().copied().filter(|&v| v < cut).collect();
        if (vals.len() as f64) < 0.25 * strip.len() as f64 {
            continue;
        }
        let Some(p) = percentiles(&vals, &[80.0, 20.0, 60.0]) else {
            continue;
        };
        let spread = (p[0] - p[1]) as f64;
        if spread > 0.10 {
            continue;
        }
        let p60 = p[2] as f64;
        if p60 < box_median + 0.10 {
            continue;
        }
        *slot = Some((p60, spread));
    }
    let [top, bottom, left, right] = qualifying;
    if !((top.is_some() && bottom.is_some()) || (left.is_some() && right.is_some())) {
        return None;
    }
    // max() keeps the first of equal levels, in the order the sides were checked.
    qualifying
        .into_iter()
        .flatten()
        .fold(None, |best: Option<(f64, f64)>, q| match best {
            Some(b) if q.0 <= b.0 => Some(b),
            _ => Some(q),
        })
}

// Camera-scan holder mask against a bed normalized to 1.0. Dense negative bottoms out near
// 0.05, so the cut sits below that. The mask is not always this clean, which is why the walk
// bridges instead of classifying every pixel.
const EDGE_MASK_LEVEL: f32 = 0.03;
// Clear film base / bare bed. _detection_luma anchors P99.5 at 1.0, so the bed is a stable
// landmark; picture only reaches here when the frame carries no base at all.
const EDGE_BASE_LEVEL: f32 = 0.95;
// Borders that no absolute level finds, read against the picture on the same line. A rebate
// on a thin negative sits above the picture but below the bed, because the frame's own
// highlights set the P99.5 anchor. A holder edge sits below the picture but above the mask
// cut, shading from near-black to mid-grey along the side. Both must end in a step, which is
// what separates them from a bright subject or a deep shadow touching the frame edge.
const EDGE_REBATE_MARGIN: f32 = 0.10;
// The dark side runs a wider margin: dense negative crowds the bottom of the range, so a
// shadow sits nearer the picture median than a highlight does.
const EDGE_SHADE_MARGIN: f32 = 0.16;
// Mirrors geometry/logic.py's _EDGE_REBATE_STEP while it has Python users.
const EDGE_REBATE_STEP: f32 = 0.18;
const EDGE_REBATE_REF: usize = 60;
// The step carries most of the drop from the outer level to the picture. A boundary does that
// and a subject shading toward the edge does not. Replaces the fixed EDGE_REBATE_STEP at
// depth, where a dense frame compresses every level.
const EDGE_STEP_SHARE: f32 = 0.45;
const EDGE_STEP_FLOOR: f32 = 0.06;

/// `np.median` of a column of float32 values: the middle one, or the float32 mean of the two
/// middle ones; NaN when any value is NaN.
fn np_median(col: &mut [f32]) -> f32 {
    if col.iter().any(|v| v.is_nan()) {
        return f32::NAN;
    }
    let n = col.len();
    let (_, &mut hi, _) = col.select_nth_unstable_by(n / 2, f32::total_cmp);
    if n % 2 == 1 {
        return hi;
    }
    let lo = col[..n / 2]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    (lo + hi) / 2.0
}

/// np.max with NaN propagation.
fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `_walk_border_depth(lines, limit, bridge, relative_step)`: `lines` is L x N (steps inward x
/// lines), the result one depth per line.
pub fn walk_border_depth(
    lines: &[f32],
    rows: usize,
    n: usize,
    limit: usize,
    bridge: i64,
    relative_step: bool,
) -> Option<Vec<i64>> {
    if rows < limit + 3 || lines.len() != rows * n {
        return None;
    }
    let at = |r: usize, c: usize| lines[r * n + c];
    let ref_end = (limit + EDGE_REBATE_REF).min(rows);
    let picture: Vec<f32> = (0..n)
        .map(|c| np_median(&mut (limit..ref_end).map(|r| at(r, c)).collect::<Vec<_>>()))
        .collect();

    let mut depth = vec![0i64; n];
    let mut last = vec![-1i64; n];
    let mut soft = vec![false; n];
    let mut alive = vec![true; n];
    let mut firm_depth = vec![0i64; n];
    let mut firm_last = vec![-1i64; n];
    let mut firm_alive = vec![true; n];
    for i in 0..=limit {
        let ii = i as i64;
        for c in 0..n {
            alive[c] &= ii - last[c] <= bridge;
            firm_alive[c] &= ii - firm_last[c] <= bridge;
        }
        if !alive.iter().any(|&a| a) && !firm_alive.iter().any(|&a| a) {
            break;
        }
        for c in 0..n {
            let v = at(i, c);
            let firm = v < EDGE_MASK_LEVEL || v > EDGE_BASE_LEVEL;
            let border =
                firm || v > picture[c] + EDGE_REBATE_MARGIN || v < picture[c] - EDGE_SHADE_MARGIN;
            if alive[c] && border {
                last[c] = ii;
                depth[c] = ii + 1;
                soft[c] = !firm;
            }
            if firm_alive[c] && firm {
                firm_last[c] = ii;
                firm_depth[c] = ii + 1;
            }
        }
    }

    let out = (0..n)
        .map(|c| {
            // span[j] = |lines[j + 3] - lines[j]|, masked to -1 past the walk's depth.
            let masked: Vec<f32> = (0..limit)
                .map(|j| {
                    if (j as i64) < depth[c] {
                        (at(j + 3, c) - at(j, c)).abs()
                    } else {
                        -1.0
                    }
                })
                .collect();
            let peak = masked.iter().copied().fold(masked[0], np_max);
            let near = 0.9f32 * peak;
            let peak_at = masked.iter().position(|&m| m >= near).unwrap_or(0) as i64;
            let resolved = (peak_at + 3).clamp(0, limit as i64);
            let gate = if relative_step {
                let outer = np_median(&mut [at(0, c), at(1, c), at(2, c)]);
                let g = EDGE_STEP_SHARE * (outer - picture[c]).abs();
                if g.is_nan() {
                    g
                } else {
                    g.max(EDGE_STEP_FLOOR)
                }
            } else {
                EDGE_REBATE_STEP
            };
            let d = if soft[c] {
                if peak >= gate { resolved } else { 0 }
            } else {
                depth[c]
            };
            d.max(firm_depth[c])
        })
        .collect();
    Some(out)
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
