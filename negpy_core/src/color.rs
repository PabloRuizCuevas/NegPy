//! The working-space colour kernels of negpy/kernel/image/logic.py: the Adobe RGB TRC and
//! CIELAB. Numba's fastmath build computes them in float64 with FMA contraction; these copy its
//! operation order, and `mathf::pow_f64` stands in for glibc's pow.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::pow_f64;

const BLOCK: usize = 1 << 14;

/// `_oetf_encode_flat`: clamped to 0..1 (NaN stays), raised to `inv_gamma`.
pub fn oetf_encode(src: &[f32], dst: &mut [f32], inv_gamma: f32) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| encode_block(s, d, inv_gamma as f64));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn encode_block(src: &[f32], dst: &mut [f32], y: f64) {
    for (d, &v) in dst.iter_mut().zip(src) {
        // clamp keeps NaN, as the kernel did.
        let x = (v as f64).clamp(0.0, 1.0);
        *d = pow_f64(x, y) as f32;
    }
}

/// `_oetf_decode_flat`: negatives to 0 (NaN stays), raised to `gamma`.
pub fn oetf_decode(src: &[f32], dst: &mut [f32], gamma: f32) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| decode_block(s, d, gamma as f64));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn decode_block(src: &[f32], dst: &mut [f32], y: f64) {
    for (d, &v) in dst.iter_mut().zip(src) {
        let x = v as f64;
        *d = pow_f64(if x < 0.0 { 0.0 } else { x }, y) as f32;
    }
}

/// The constants of the Lab kernels, as float64.
#[derive(Clone, Copy)]
pub struct LabSpace {
    pub m: [[f64; 3]; 3],
    pub white: [f64; 3],
    pub eps: f64,
    pub kappa: f64,
}

impl LabSpace {
    pub fn new(m: [[f32; 3]; 3], white: [f32; 3], eps: f32, kappa: f32) -> Self {
        Self {
            m: m.map(|r| r.map(|v| v as f64)),
            white: white.map(|v| v as f64),
            eps: eps as f64,
            kappa: kappa as f64,
        }
    }
}

/// `np.float32(16/116)` as float64.
const C: f64 = (16.0f32 / 116.0f32) as f64;

/// One row of the 3x3 product as Numba contracts it.
#[inline(always)]
fn row(m: [f64; 3], x: f64, y: f64, z: f64) -> f64 {
    z.mul_add(m[2], y.mul_add(m[1], x * m[0]))
}

/// `_rgb_to_lab_kernel` over N x 3 linear RGB.
pub fn rgb_to_lab(src: &[f32], dst: &mut [f32], s: LabSpace) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| rgb_to_lab_block(p, d, s));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn rgb_to_lab_block(src: &[f32], dst: &mut [f32], s: LabSpace) {
    // Three passes over short runs: the cube root vectorizes only as a flat loop.
    const RUN: usize = 3 * 512;
    let mut t = [0f64; RUN];
    for (d, p) in dst.chunks_mut(RUN).zip(src.chunks(RUN)) {
        let t = &mut t[..p.len()];
        for (t, p) in t
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(p.as_chunks::<3>().0)
        {
            let [r, g, b] = p.map(|v| {
                let v = v as f64;
                if v < 0.0 { 0.0 } else { v }
            });
            *t = std::array::from_fn(|c| row(s.m[c], r, g, b) / s.white[c]);
        }
        for v in t.iter_mut() {
            let root = pow_f64(*v, 1.0 / 3.0);
            let lin = s.kappa.mul_add(*v, C);
            *v = if *v > s.eps { root } else { lin };
        }
        for (d, f) in d
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(t.as_chunks::<3>().0)
        {
            let [fx, fy, fz] = *f;
            d[0] = fy.mul_add(116.0, -16.0) as f32;
            d[1] = ((fx - fy) * 500.0) as f32;
            d[2] = ((fy - fz) * 200.0) as f32;
        }
    }
}

/// `_lab_to_rgb_kernel` over N x 3 Lab: Numba folds the divisions into reciprocal products
/// and `fy - c` into `L / 116 + (16/116 - c)`.
pub fn lab_to_rgb(src: &[f32], dst: &mut [f32], s: LabSpace) {
    dst.par_chunks_mut(BLOCK * 3)
        .zip(src.par_chunks(BLOCK * 3))
        .for_each(|(d, p)| lab_to_rgb_block(p, d, s));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn lab_to_rgb_block(src: &[f32], dst: &mut [f32], s: LabSpace) {
    let inv_kappa = 1.0 / s.kappa;
    let fy_gap = 16.0 / 116.0 - C;
    for (d, p) in dst
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
    {
        let [l, a, b] = p.map(|v| v as f64);
        let fy_l = l * (1.0 / 116.0);
        let fy = fy_l + 16.0 / 116.0;
        let fx = a.mul_add(0.002, fy);
        let fz = (-b).mul_add(0.005, fy);
        let (fx3, fy3, fz3) = (fx * fx * fx, fy * fy * fy, fz * fz * fz);
        let xr = if fx3 > s.eps {
            fx3 * s.white[0]
        } else {
            ((fx - C) * s.white[0]) * inv_kappa
        };
        let yr = if fy3 > s.eps {
            fy3
        } else {
            (fy_l + fy_gap) * inv_kappa
        } * s.white[1];
        let zr = if fz3 > s.eps {
            fz3
        } else {
            (fz - C) * inv_kappa
        } * s.white[2];
        *d = std::array::from_fn(|c| {
            let v = row(s.m[c], xr, yr, zr);
            if v > 0.0 { v as f32 } else { 0.0 }
        });
    }
}
