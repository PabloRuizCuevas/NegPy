//! Ports of the frame analysis in negpy/features/geometry/logic.py and exposure/normalization.py.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::stats::{np_percentiles, pairwise_sum_f32, percentiles};

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
    if col.is_empty() || col.iter().any(|v| v.is_nan()) {
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

/// `_walk_border_depth(lines, limit, bridge, relative_step)`: per line, the depth of the
/// non-image border walking inward from the frame edge. `lines` is L x N (steps inward x
/// lines). A camera scan puts two tiers outside the picture, either of which may be absent: a
/// holder mask at the very edge, then a film-base sliver. Neither is separable by level alone,
/// but the tiers are adjacent, so the walk carries across a short gap and stops at the last
/// border evidence. A run that ended on a relative test takes its boundary from the strongest
/// step inside it, and that step must clear a gate (fixed, or a share of the run's own drop).
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

// Gap the walk carries across, covering the mid-gray ramp between mask and base.
const EDGE_BRIDGE: i64 = 8;
const EDGE_TRIM_CAP: f64 = 0.03;
const EDGE_CORE: f64 = 0.70;
const EDGE_QUANTILE: f64 = 80.0;
const EDGE_MIN_SUPPORT: f64 = 0.45;
// Second, deeper walk. EDGE_TRIM_CAP is the depth a border may reach on trust alone. Past it
// a border must prove itself, because that much of a side is worth real picture. A wide
// holder margin, a sprocket rail or a neighbouring frame all run several times deeper than
// the cap, and clamping there leaves a band the eye reads as an uncropped edge.
const EDGE_DEEP_CAP: f64 = 0.10;
// Wider than EDGE_BRIDGE: a border this thick is layered (mask, ramp, base, sometimes a second
// mask segment) and the mid-gray runs between the tiers are longer at this depth.
const EDGE_DEEP_BRIDGE: i64 = 12;
// Proof one: the step ends a straight edge. A film boundary is a cut line, so the per-line
// depths fit a sloped line to within a fraction of the allowance. Picture content resolved by
// the same rule scatters several times wider.
const EDGE_DEEP_STRAIGHT: f64 = 0.025;
// Proof two: the walk stopped on its own. Reaching the allowance means no boundary was found
// inside it, and the depth is then an artefact of where the search was cut off.
const EDGE_DEEP_FULL: f64 = 0.92;
// Proof three is EDGE_STEP_SHARE. Proof four: the outermost pixels are one material. Mask,
// bed and film base each hold their level along the whole side, so the outer level barely
// moves from line to line, while picture reaching the frame edge moves with the subject.
const EDGE_DEEP_OUTER_SPREAD: f64 = 0.15;

/// `np.percentile` of float64 data (or integers, which NumPy interpolates in float64).
fn np_percentile_f64(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let n = v.len();
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let t = virtual_index - lo as f64;
    v.sort_by(f64::total_cmp);
    let (a, b) = (v[lo], v[hi]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.median` of float64 data.
fn np_median_f64(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `_pool_border_depth`: the 80th percentile of the line depths, capped at `limit`, or 0 when
/// fewer than 45% of the lines show a border.
fn pool_border_depth(depth: &[i64], limit: i64) -> i64 {
    let live = depth.iter().filter(|&&d| d > 0).count();
    if (live as f64 / depth.len() as f64) < EDGE_MIN_SUPPORT {
        return 0;
    }
    let as_f: Vec<f64> = depth.iter().map(|&d| d as f64).collect();
    (limit as f64).min(np_percentile_f64(&as_f, EDGE_QUANTILE)) as i64
}

/// `_border_edge_scatter`: the median residual of the live depths around a Theil-Sen line.
fn border_edge_scatter(depth: &[i64]) -> f64 {
    let index: Vec<f64> = (0..depth.len())
        .filter(|&i| depth[i] > 0)
        .map(|i| i as f64)
        .collect();
    if index.len() < 8 {
        return f64::INFINITY;
    }
    let value: Vec<f64> = depth
        .iter()
        .filter(|&&d| d > 0)
        .map(|&d| d as f64)
        .collect();
    let step = (index.len() / 48).max(1);
    let a: Vec<f64> = index.iter().step_by(step).copied().collect();
    let b: Vec<f64> = value.iter().step_by(step).copied().collect();
    let mut slopes = Vec::new();
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            let da = a[i] - a[j];
            if da != 0.0 {
                slopes.push((b[i] - b[j]) / da);
            }
        }
    }
    let slope = if slopes.is_empty() {
        0.0
    } else {
        np_median_f64(&slopes)
    };
    let offsets: Vec<f64> = value
        .iter()
        .zip(&index)
        .map(|(v, i)| v - slope * i)
        .collect();
    let intercept = np_median_f64(&offsets);
    let residuals: Vec<f64> = value
        .iter()
        .zip(&index)
        .map(|(v, i)| (v - (slope * i + intercept)).abs())
        .collect();
    np_median_f64(&residuals)
}

#[derive(Clone, Copy)]
enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

/// `_EDGE_ORIENT[side](box)`, cropped to the core columns: lines run inward from the edge
/// (rows), one per position along it (columns).
fn edge_lines(boxed: &[f32], bh: usize, bw: usize, side: Side) -> (Vec<f32>, usize, usize) {
    let (rows, n) = match side {
        Side::Top | Side::Bottom => (bh, bw),
        Side::Left | Side::Right => (bw, bh),
    };
    let at = |r: usize, c: usize| match side {
        Side::Top => boxed[r * bw + c],
        Side::Bottom => boxed[(bh - 1 - r) * bw + c],
        Side::Left => boxed[c * bw + r],
        Side::Right => boxed[c * bw + (bw - 1 - r)],
    };
    let margin = py_round((1.0 - EDGE_CORE) * 0.5 * n as f64).max(0) as usize;
    let (c0, c1) = if n >= 2 * margin + 8 {
        (margin, n - margin)
    } else {
        (0, n)
    };
    let cols = c1 - c0;
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in c0..c1 {
            out.push(at(r, c));
        }
    }
    (out, rows, cols)
}

/// `_edge_border_depth(box, side, limit, deep_limit)`: the border depth on one side, per line
/// (these frames are slightly tilted) and pooled. A first walk to `limit` is believed; a second
/// to `deep_limit` is adopted only where it goes deeper and passes all four proofs.
fn edge_border_depth(
    boxed: &[f32],
    bh: usize,
    bw: usize,
    side: Side,
    limit: i64,
    deep_limit: i64,
) -> i64 {
    let (lines, rows, cols) = edge_lines(boxed, bh, bw, side);
    if limit <= 0 || (rows as i64) < limit + 8 || cols < 8 {
        return 0;
    }
    let Some(walk) = walk_border_depth(&lines, rows, cols, limit as usize, EDGE_BRIDGE, false)
    else {
        return 0;
    };
    let near = pool_border_depth(&walk, limit);
    if deep_limit <= limit || (rows as i64) < deep_limit + 8 {
        return near;
    }
    let Some(deep) = walk_border_depth(
        &lines,
        rows,
        cols,
        deep_limit as usize,
        EDGE_DEEP_BRIDGE,
        true,
    ) else {
        return near;
    };
    let far = pool_border_depth(&deep, deep_limit);
    if far <= near || far as f64 >= EDGE_DEEP_FULL * deep_limit as f64 {
        return near;
    }
    if border_edge_scatter(&deep) > EDGE_DEEP_STRAIGHT * deep_limit as f64 {
        return near;
    }
    let mut outer: Vec<f32> = (0..cols)
        .filter(|&c| deep[c] > 0)
        .map(|c| np_median(&mut [lines[c], lines[cols + c], lines[2 * cols + c]]))
        .collect();
    // np.median of nothing is NaN, and a NaN spread passes no comparison.
    if outer.is_empty() {
        return far;
    }
    let mid = np_median(&mut outer.clone());
    let mut spread: Vec<f32> = outer.iter_mut().map(|v| (*v - mid).abs()).collect();
    if np_median(&mut spread) as f64 > EDGE_DEEP_OUTER_SPREAD {
        return near;
    }
    far
}

/// `measure_film_edges(lum, film_roi)`: per side (top, bottom, left, right) the border depth as
/// a fraction of that side.
pub fn measure_film_edges(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> [f64; 4] {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return [0.0; 4];
    }
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let ly = py_round(EDGE_TRIM_CAP * bh as f64);
    let lx = py_round(EDGE_TRIM_CAP * bw as f64);
    let dy = py_round(EDGE_DEEP_CAP * bh as f64);
    let dx = py_round(EDGE_DEEP_CAP * bw as f64);
    [
        edge_border_depth(&boxed, bh, bw, Side::Top, ly, dy) as f64 / bh as f64,
        edge_border_depth(&boxed, bh, bw, Side::Bottom, ly, dy) as f64 / bh as f64,
        edge_border_depth(&boxed, bh, bw, Side::Left, lx, dx) as f64 / bw as f64,
        edge_border_depth(&boxed, bh, bw, Side::Right, lx, dx) as f64 / bw as f64,
    ]
}

/// `_trim_film_edges(lum, film_roi)`: the box inset past the border on each side, or the box
/// itself when that would take half a side.
pub fn trim_film_edges(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (i64, i64, i64, i64),
) -> (i64, i64, i64, i64) {
    let (y1, y2, x1, x2) = roi;
    let (height, width) = (y2 - y1, x2 - x1);
    let clamp = |v: i64| v.max(0) as usize;
    let m = measure_film_edges(lum, h, w, (clamp(y1), clamp(y2), clamp(x1), clamp(x2)));
    let out = (
        y1 + py_round(m[0] * height as f64),
        y2 - py_round(m[1] * height as f64),
        x1 + py_round(m[2] * width as f64),
        x2 - py_round(m[3] * width as f64),
    );
    if ((out.1 - out.0) as f64) < 0.5 * height as f64
        || ((out.3 - out.2) as f64) < 0.5 * width as f64
    {
        return roi;
    }
    out
}

// Walk cap per side, as a fraction of that side. Real rebate plus bed slop stays well under
// it, so reaching the cap means the profile never came back down and bright picture content
// touches that edge. Those sides report NaN and pooling ignores them.
const BORDER_WALK_CAP: f64 = 0.06;
const BORDER_MIN_SEPARATION: f64 = 0.15;

/// `measure_film_border(lum, film_roi)`: per side (top, bottom, left, right) the bright
/// border's thickness as a fraction of that side; 0.0 for none, NaN for no clean transition.
pub fn measure_film_border(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (usize, usize, usize, usize),
) -> [f64; 4] {
    let (y1, y2, x1, x2) = (roi.0.min(h), roi.1.min(h), roi.2.min(w), roi.3.min(w));
    let (bh, bw) = (y2.saturating_sub(y1), x2.saturating_sub(x1));
    if bh < 16 || bw < 16 {
        return [f64::NAN; 4];
    }
    let boxed = crop(lum, w, (y1, y2, x1, x2));
    let Some(level) = np_percentiles(&boxed, &[40.0]) else {
        return [f64::NAN; 4];
    };
    let image_level = level[0] as f64;
    // mean(axis=1): a pairwise sum per row; mean(axis=0): sequential down each column.
    let rows: Vec<f32> = boxed
        .chunks_exact(bw)
        .map(|r| pairwise_sum_f32(r) / bw as f32)
        .collect();
    let mut cols = vec![0f32; bw];
    for r in boxed.chunks_exact(bw) {
        for (c, &v) in cols.iter_mut().zip(r) {
            *c += v;
        }
    }
    cols.iter_mut().for_each(|c| *c /= bh as f32);

    let side = |profile: &[f32], reversed: bool| -> f64 {
        let length = profile.len();
        let walk: Vec<f32> = if reversed {
            profile.iter().rev().copied().collect()
        } else {
            profile.to_vec()
        };
        let outer_n = py_round(0.03 * length as f64).max(2) as usize;
        let Some(b) = np_percentiles(&walk[..outer_n.min(length)], &[90.0]) else {
            return f64::NAN;
        };
        let base = b[0] as f64;
        if base - image_level < BORDER_MIN_SEPARATION {
            return 0.0;
        }
        // NEP 50: the Python float threshold joins float32 comparisons as float32.
        let threshold = (0.5 * (base + image_level)) as f32;
        let cap = ((BORDER_WALK_CAP * length as f64) as usize)
            .max(3)
            .min(length);
        let head = &walk[..cap];
        // np.argmax: the first maximum, or the first NaN.
        let peak = head.iter().position(|v| v.is_nan()).unwrap_or_else(|| {
            head.iter()
                .enumerate()
                .fold(0, |best, (i, &v)| if v > head[best] { i } else { best })
        });
        if head[peak] <= threshold {
            return 0.0;
        }
        match head[peak..].iter().position(|&v| v <= threshold) {
            Some(k) => (peak + k) as f64 / length as f64,
            None => f64::NAN,
        }
    };
    [
        side(&rows, false),
        side(&rows, true),
        side(&cols, false),
        side(&cols, true),
    ]
}

/// `_roi_from_measured_border(lum, film_roi)`: the box inset by its measured border, when an
/// opposite pair agrees.
pub fn roi_from_measured_border(
    lum: &[f32],
    h: usize,
    w: usize,
    roi: (i64, i64, i64, i64),
) -> Option<(i64, i64, i64, i64)> {
    let (y1, y2, x1, x2) = roi;
    let clamp = |v: i64| v.max(0) as usize;
    let m = measure_film_border(lum, h, w, (clamp(y1), clamp(y2), clamp(x1), clamp(x2)));
    let (height, width) = ((y2 - y1) as f64, (x2 - x1) as f64);
    let opposite = [1, 0, 3, 2];
    let mut insets = [0f64; 4];
    for i in 0..4 {
        let (mut own, other) = (m[i], m[opposite[i]]);
        if !own.is_finite() {
            own = other;
        }
        if !own.is_finite() {
            continue;
        }
        if own > 0.0 && other.is_finite() && other > 0.0 {
            own = own.min(2.0 * other);
        }
        insets[i] = own;
    }
    if !((insets[0] > 0.0 && insets[1] > 0.0) || (insets[2] > 0.0 && insets[3] > 0.0)) {
        return None;
    }
    let out = (
        y1 + py_round(insets[0] * height),
        y2 - py_round(insets[1] * height),
        x1 + py_round(insets[2] * width),
        x2 - py_round(insets[3] * width),
    );
    (out.1 > out.0 && out.3 > out.2).then_some(out)
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
