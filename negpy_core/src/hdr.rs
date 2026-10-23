//! The bracket-merge kernels of negpy/features/hdr/logic.py.

use multiversion::multiversion;
use rayon::prelude::*;

/// `_count_clipped_rows`: per row, the pixels with any channel at or above `saturation`,
/// compared in float64 as Numba compares a float32 with a Python float.
pub fn count_clipped_rows(
    values: &[f32],
    (w, cn): (usize, usize),
    saturation: f64,
    out: &mut [i64],
) {
    // The smallest float32 at or above the threshold: the float32 compare is then the float64 one.
    let mut limit = saturation as f32;
    if (limit as f64) < saturation {
        limit = f32::from_bits(limit.to_bits() + 1);
    }
    out.par_iter_mut()
        .zip(values.par_chunks(w * cn))
        .for_each(|(n, row)| *n = count_row(row, cn, limit));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn count_row(row: &[f32], cn: usize, limit: f32) -> i64 {
    if cn == 3 {
        return row
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| ((p[0] >= limit) | (p[1] >= limit) | (p[2] >= limit)) as i64)
            .sum();
    }
    row.chunks(cn)
        .map(|p| p.iter().fold(false, |hit, &v| hit | (v >= limit)) as i64)
        .sum()
}

/// `_accumulate`: one frame added to the weighted sums in reference units. Weight is ratio^2,
/// rolled off per channel toward saturation; the order is Numba's vector loop.
pub fn accumulate(
    num: &mut [f32],
    den: &mut [f32],
    values: &[f32],
    ratio: f64,
    (saturation, rolloff_start): (f64, f64),
) {
    // The span is a Python float difference, rounded to float32 once.
    let inv_span = 1.0 / ((saturation - rolloff_start) as f32);
    let saturation = saturation as f32;
    let weight = (ratio as f32) * (ratio as f32);
    let inv_scale = 1.0 / (ratio.max(1e-9) as f32);
    num.par_chunks_mut(1 << 14)
        .zip(den.par_chunks_mut(1 << 14))
        .zip(values.par_chunks(1 << 14))
        .for_each(|((n, d), v)| {
            accumulate_block(n, d, v, (saturation, inv_span, weight, inv_scale))
        });
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn accumulate_block(
    num: &mut [f32],
    den: &mut [f32],
    values: &[f32],
    (sat, inv_span, weight, inv_scale): (f32, f32, f32, f32),
) {
    for ((n, d), &v) in num.iter_mut().zip(den.iter_mut()).zip(values) {
        let r = (sat - v) * inv_span;
        let rolloff = r.clamp(0.0, 1.0);
        let w = rolloff * weight;
        *n = (v * inv_scale).mul_add(w, *n);
        *d += w;
    }
}
