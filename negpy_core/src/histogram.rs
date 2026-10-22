//! The scope counts of negpy/features/exposure/analysis.py: integer counts, so only the bin each
//! sample lands in must match NumPy.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::analysis::luminance_chunk;

pub const DENSITY_HIST_BINS: usize = 120;
const DENSITY_HIST_RANGE: (f64, f64) = (-0.1, 1.1);
pub const WAVEFORM_BINS: usize = 128;
pub const WAVEFORM_COLS: usize = 256;
const SCAN_DENSITY_RANGE: (f64, f64) = (0.0, 3.0);

/// A stride-subsampled crop of an HxWxC float32 image: rows `y1..y2` and columns `x1..x2`, every
/// `step`-th from the first, as `img[y1:y2:step, x1:x2:step]` reads them.
#[derive(Clone, Copy)]
pub struct Samples<'a> {
    pub img: &'a [f32],
    pub w: usize,
    pub cn: usize,
    pub rows: (usize, usize),
    pub cols: (usize, usize),
    pub step: usize,
}

impl Samples<'_> {
    fn row_starts(&self) -> Vec<usize> {
        (self.rows.0..self.rows.1).step_by(self.step).collect()
    }

    /// The first three channels of row `y`'s samples, packed: a slice of the image when they
    /// already are, else a copy in `buf`.
    fn rgb_row<'b>(&'b self, y: usize, buf: &'b mut Vec<f32>) -> &'b [f32] {
        if self.step == 1 && self.cn == 3 {
            return &self.img[(y * self.w + self.cols.0) * 3..(y * self.w + self.cols.1) * 3];
        }
        buf.clear();
        for x in (self.cols.0..self.cols.1).step_by(self.step) {
            let p = (y * self.w + x) * self.cn;
            buf.extend_from_slice(&self.img[p..p + 3]);
        }
        buf
    }
}

/// `np.histogram(np.clip(values, lo, hi), bins, range=(lo, hi))` for float32 values with Python
/// float edges: NumPy bins in float32 (NEP 50), then corrects each index against the float32
/// `np.linspace` edges, `f32(i * step + lo)`. NaN is dropped.
#[derive(Clone, Copy)]
struct Binner {
    lo: f64,
    hi: f64,
    bins: usize,
}

const NO_BIN: u8 = u8::MAX;

impl Binner {
    /// Each value's bin, or NO_BIN for NaN; branch-free, so it vectorizes.
    fn bin_into(self, vals: &[f32], out: &mut [u8]) {
        bin_into(self, vals, out)
    }
}

#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn bin_into(b: Binner, vals: &[f32], out: &mut [u8]) {
    let (lo, hi, n) = (b.lo as f32, b.hi as f32, b.bins as i32);
    let (denom, step) = ((b.hi - b.lo) as f32, (b.hi - b.lo) / b.bins as f64);
    let edge = |i: i32| (i as f64 * step + b.lo) as f32;
    for (o, &v) in out.iter_mut().zip(vals) {
        let c = v.max(lo).min(hi);
        let i = ((((c - lo) / denom) * n as f32) as i32).min(n - 1);
        let down = c < edge(i);
        let up = !down & (i != n - 1) & (c >= edge(i + 1));
        let i = i - down as i32 + up as i32;
        *o = if v.is_nan() { NO_BIN } else { i as u8 };
    }
}

/// The parade bin of each value: np.nan_to_num and the negation in float32, the scan density
/// axis in float64, floored and clipped.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
fn parade_bins(vals: &[f32], out: &mut [u8]) {
    let (lo, hi) = SCAN_DENSITY_RANGE;
    for (o, &v) in out.iter_mut().zip(vals) {
        let v = if v.is_nan() {
            0.0
        } else {
            v.clamp(f32::MIN, f32::MAX)
        };
        let pos = (hi - (-v) as f64) / (hi - lo);
        *o = (pos * WAVEFORM_BINS as f64)
            .floor()
            .clamp(0.0, (WAVEFORM_BINS - 1) as f64) as u8;
    }
}

/// Neighbouring samples often share a bin, and one counter serializes on its own increments,
/// so each sample counts into lane `i % LANES` (u32: a row never overflows one).
const LANES: usize = 4;

fn fold_lanes(lanes: &[u32]) -> Vec<u64> {
    let n = lanes.len() / LANES;
    let mut out = vec![0u64; n];
    for lane in lanes.chunks_exact(n) {
        out.iter_mut().zip(lane).for_each(|(o, &c)| *o += c as u64);
    }
    out
}

/// `density_histogram`: (4, DENSITY_HIST_BINS) counts of R, G, B and luminance.
pub fn density_histogram(s: Samples) -> Vec<f64> {
    let binner = Binner {
        lo: DENSITY_HIST_RANGE.0,
        hi: DENSITY_HIST_RANGE.1,
        bins: DENSITY_HIST_BINS,
    };
    let counts = s
        .row_starts()
        .par_iter()
        .fold(
            || {
                let counts = vec![0u32; LANES * 4 * DENSITY_HIST_BINS];
                (counts, Vec::new(), Vec::new(), Vec::new(), Vec::new())
            },
            |(mut counts, mut buf, mut lum, mut bins, mut lbins), &y| {
                let rgb = s.rgb_row(y, &mut buf);
                let px = rgb.len() / 3;
                lum.resize(px, 0.0);
                luminance_chunk(rgb, &mut lum);
                bins.resize(3 * px, 0);
                lbins.resize(px, 0);
                binner.bin_into(rgb, &mut bins);
                binner.bin_into(&lum, &mut lbins);
                for (i, (p, &l)) in bins.as_chunks::<3>().0.iter().zip(&lbins).enumerate() {
                    let lane = (i % LANES) * 4 * DENSITY_HIST_BINS;
                    for (c, &b) in p.iter().chain([&l]).enumerate() {
                        if b != NO_BIN {
                            counts[lane + c * DENSITY_HIST_BINS + b as usize] += 1;
                        }
                    }
                }
                (counts, buf, lum, bins, lbins)
            },
        )
        .map(|(counts, ..)| fold_lanes(&counts))
        .reduce(
            || vec![0u64; 4 * DENSITY_HIST_BINS],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );
    counts.iter().map(|&c| c as f64).collect()
}

/// `density_parade`: (3, WAVEFORM_BINS, WAVEFORM_COLS) counts of each channel's scan density
/// (the negated log) per parade column, the column of the sample's x in the crop.
pub fn density_parade(s: Samples) -> Vec<f64> {
    let full_w = s.cols.1 - s.cols.0;
    let cols: Vec<usize> = (0..full_w)
        .step_by(s.step)
        .map(|x| (x * WAVEFORM_COLS / full_w).min(WAVEFORM_COLS - 1))
        .collect();
    let plane = WAVEFORM_BINS * WAVEFORM_COLS;
    let counts = s
        .row_starts()
        .par_iter()
        .fold(
            || (vec![0u32; LANES * 3 * plane], Vec::new(), Vec::new()),
            |(mut counts, mut buf, mut bins), &y| {
                let rgb = s.rgb_row(y, &mut buf);
                bins.resize(rgb.len(), 0);
                parade_bins(rgb, &mut bins);
                for (i, (p, &col)) in bins.as_chunks::<3>().0.iter().zip(&cols).enumerate() {
                    let lane = (i % LANES) * 3 * plane;
                    for (c, &b) in p.iter().enumerate() {
                        counts[lane + c * plane + b as usize * WAVEFORM_COLS + col] += 1;
                    }
                }
                (counts, buf, bins)
            },
        )
        .map(|(counts, ..)| fold_lanes(&counts))
        .reduce(
            || vec![0u64; 3 * plane],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );
    counts.iter().map(|&c| c as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_and_extremes_land_in_the_edge_bins() {
        let b = Binner {
            lo: DENSITY_HIST_RANGE.0,
            hi: DENSITY_HIST_RANGE.1,
            bins: DENSITY_HIST_BINS,
        };
        let step = (b.hi - b.lo) / b.bins as f64;
        let mut vals = vec![-5.0, 5.0, 1.1, f32::NAN];
        vals.extend((0..DENSITY_HIST_BINS).map(|i| (i as f64 * step + b.lo) as f32));
        let mut out = vec![0u8; vals.len()];
        b.bin_into(&vals, &mut out);
        let last = (DENSITY_HIST_BINS - 1) as u8;
        assert_eq!(out[..4], [0, last, last, NO_BIN]);
        for (i, &o) in out[4..].iter().enumerate() {
            assert_eq!(o as usize, i, "edge {i}");
        }
    }
}
