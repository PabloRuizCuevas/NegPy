//! The edge sampling of negpy/features/geometry/skew.py: `_fit_side`'s per-station profiles
//! and `_profile_peaks`, in NumPy's float32 and float64 order.

use rayon::prelude::*;

use crate::filter::{self, Border};

/// `_profile_peaks`: subpixel positions of the strong gradient steps along one profile, after a
/// sigma-1 Gaussian (reflect-101) and `np.gradient`.
pub fn profile_peaks(profile: &[f32]) -> Vec<f64> {
    let n = profile.len();
    if n < 5 {
        return Vec::new();
    }
    let mut blur = vec![0f32; n];
    if filter::gaussian_blur(
        profile,
        &mut blur,
        (n, 1, 1),
        (1, 0),
        (0.0, 1.0),
        Border::Reflect101,
    )
    .is_none()
    {
        return Vec::new();
    }
    let mut grad: Vec<f32> = (0..n)
        .map(|i| match i {
            0 => blur[1] - blur[0],
            _ if i == n - 1 => blur[n - 1] - blur[n - 2],
            _ => (blur[i + 1] - blur[i - 1]) / 2.0,
        })
        .map(f32::abs)
        .collect();
    let peak = grad.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sorted = grad.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    if peak <= 3.0 * median as f64 + 1e-6 {
        return Vec::new();
    }
    // A Python float meets the float32 gradient as float32.
    let (bar, tiny) = ((0.4 * peak) as f32, 1e-9f64 as f32);
    let g = std::mem::take(&mut grad);
    (1..n - 1)
        .filter(|&i| g[i] >= g[i - 1] && g[i] > g[i + 1] && g[i] >= bar)
        .map(|i| {
            let (a, b, c) = (g[i - 1], g[i], g[i + 1]);
            let denom = a - 2.0 * b + c;
            let frac = if denom.abs() > tiny {
                0.5 * (a - c) / denom
            } else {
                0.0
            };
            i as f64 + frac as f64
        })
        .collect()
}

/// Python's `round` of a float64: ties to even.
fn round_half_even(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `_fit_side`'s `sample(center, half, nearest)` on an HxW luminance plane read along rows
/// (`horizontal`) or columns: per station, the peaks of the 3-column mean profile within
/// `half` of its center, or only the one nearest the center. Returns (stations, positions).
pub fn sample(
    img: &[f32],
    (h, w): (usize, usize),
    horizontal: bool,
    (stations, centers): (&[f64], &[f64]),
    half: f64,
    nearest: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (vh, vw) = if horizontal { (h, w) } else { (w, h) };
    let at = |r: usize, c: usize| {
        if horizontal {
            img[r * w + c]
        } else {
            img[c * w + r]
        }
    };
    let found: Vec<(f64, Vec<f64>)> = stations
        .par_iter()
        .zip(centers.par_iter())
        .map(|(&x, &c)| {
            let x0 = round_half_even(x);
            let lo = ((c - half).floor() as i64).max(0);
            let hi = ((c + half).ceil() as i64 + 1).min(vh as i64);
            if hi - lo < 5 || x0 < 1 || x0 > vw as i64 - 2 {
                return (x, Vec::new());
            }
            let (lo, x0) = (lo as usize, x0 as usize);
            let profile: Vec<f32> = (lo..hi as usize)
                .map(|r| (at(r, x0 - 1) + at(r, x0) + at(r, x0 + 1)) / 3.0)
                .collect();
            let mut peaks = profile_peaks(&profile);
            if nearest && !peaks.is_empty() {
                let want = c - lo as f64;
                let best = peaks
                    .iter()
                    .enumerate()
                    .fold((0, f64::INFINITY), |(bi, bd), (i, &p)| {
                        let d = (p - want).abs();
                        if d < bd { (i, d) } else { (bi, bd) }
                    })
                    .0;
                peaks = vec![peaks[best]];
            }
            (x, peaks.into_iter().map(|p| lo as f64 + p).collect())
        })
        .collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, peaks) in found {
        xs.extend(std::iter::repeat_n(x, peaks.len()));
        ys.extend(peaks);
    }
    (xs, ys)
}
