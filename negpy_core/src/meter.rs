//! The textured-cell meters of negpy/features/exposure/normalization.py: anchor, textural
//! range, shadow reach and highlight hold, read off the block-median grid.

use crate::stats::{np_percentiles, np_percentiles_f64, pairwise_sum_f32};

const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// One channel of `(grid - floor) / span`. A float64 operand (a NumPy float64 scalar) makes
/// NumPy compute that step in float64; a Python float joins the float32 math (NEP 50).
#[derive(Clone, Copy)]
pub struct ChannelNorm {
    pub floor: f64,
    pub floor_wide: bool,
    pub span: f64,
    pub span_wide: bool,
}

impl ChannelNorm {
    fn apply(&self, g: f32) -> f32 {
        if self.floor_wide || self.span_wide {
            let d = if self.floor_wide {
                g as f64 - self.floor
            } else {
                (g - self.floor as f32) as f64
            };
            (d / self.span) as f32
        } else {
            (g - self.floor as f32) / self.span as f32
        }
    }
}

/// The float32 luma of an HxWx3 grid, each channel normalized first when `norm` is given.
pub fn luma(grid: &[f32], norm: Option<&[ChannelNorm; 3]>) -> Vec<f32> {
    grid.as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let [r, g, b] = match norm {
                Some(n) => [n[0].apply(p[0]), n[1].apply(p[1]), n[2].apply(p[2])],
                None => *p,
            };
            LUMA_R * r + LUMA_G * g + LUMA_B * b
        })
        .collect()
}

/// The activity gate: sectors of 2x2 blocks of `block` x `block` cells vote when their four
/// block means span more than `gate`.
#[derive(Clone, Copy)]
pub struct Activity {
    pub block: usize,
    pub gate: f32,
    pub min_fraction: f64,
}

/// `_textured_cells(lum)` of an HxW luma plane: the cells of the voting sectors in row order,
/// or every cell when fewer than `min_fraction` of the sectors vote.
pub fn textured_cells(lum: Vec<f32>, h: usize, w: usize, a: Activity) -> Vec<f32> {
    let b = a.block;
    let (hs, ws) = ((h / (2 * b)) * 2 * b, (w / (2 * b)) * 2 * b);
    if hs == 0 || ws == 0 {
        return lum;
    }
    let (bh, bw) = (hs / b, ws / b);
    // NumPy's block mean: each row of the block summed pairwise, the rows added in order.
    let mut means = vec![0f32; bh * bw];
    for (i, m) in means.iter_mut().enumerate() {
        let (by, bx) = (i / bw, i % bw);
        let mut sum = 0f32;
        for y in by * b..(by + 1) * b {
            sum += pairwise_sum_f32(&lum[y * w + bx * b..y * w + (bx + 1) * b]);
        }
        *m = sum / (b * b) as f32;
    }
    let (sh, sw) = (bh / 2, bw / 2);
    let active: Vec<bool> = (0..sh * sw)
        .map(|i| {
            let (sy, sx) = (i / sw, i % sw);
            let quad = [
                means[2 * sy * bw + 2 * sx],
                means[2 * sy * bw + 2 * sx + 1],
                means[(2 * sy + 1) * bw + 2 * sx],
                means[(2 * sy + 1) * bw + 2 * sx + 1],
            ];
            let (lo, hi) = quad
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    (np_min(l, v), np_max(h, v))
                });
            hi - lo > a.gate
        })
        .collect();
    let voting = active.iter().filter(|&&v| v).count();
    if (voting as f64 / active.len() as f64) < a.min_fraction {
        return lum;
    }
    let mut out = Vec::with_capacity(voting * 4 * b * b);
    for y in 0..hs {
        let sy = y / (2 * b);
        for x in 0..ws {
            if active[sy * sw + x / (2 * b)] {
                out.push(lum[y * w + x]);
            }
        }
    }
    out
}

/// np.min / np.max with NaN propagation.
fn np_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `np.percentile(lum, q)` for a Python float q: float32 interpolation.
pub fn percentile(lum: &[f32], q: f64) -> f32 {
    np_percentiles(lum, &[q]).map_or(f32::NAN, |p| p[0])
}

/// The anchor meter's measurement: the mean of the cells inside the `clip` percentile trim
/// (float32) averaged with the trim's midpoint.
pub fn anchor_measure(lum: &[f32], clip: f64) -> f64 {
    let p = np_percentiles_f64(lum, &[clip, 100.0 - clip]);
    let (lo, hi) = (p[0], p[1]);
    let inner: Vec<f32> = lum
        .iter()
        .copied()
        .filter(|&v| v as f64 >= lo && v as f64 <= hi)
        .collect();
    let mean = pairwise_sum_f32(&inner) / inner.len() as f32;
    0.5 * (mean as f64 + 0.5 * (lo + hi))
}

/// The textural range: the spread between the `clip` and `100 - clip` percentiles.
pub fn textural_range(lum: &[f32], clip: f64) -> f64 {
    let p = np_percentiles_f64(lum, &[clip, 100.0 - clip]);
    (p[1] - p[0]).abs()
}
