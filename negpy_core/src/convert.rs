//! The pixel-format kernels of negpy/kernel/image/logic.py: float to uint8/uint16 and back,
//! luminance to uint, and the RGB/RGBA copies of the texture paths.

use multiversion::multiversion;
use rayon::prelude::*;

const LUMA_R: f64 = 0.2126;
const LUMA_G: f64 = 0.7152;
const LUMA_B: f64 = 0.0722;
const BLOCK: usize = 1 << 14;

/// `v * scale + 0.5` clipped to 0..scale, NaN to 0, then truncated: the rounding of
/// `_to_uint8_jit` and `_to_uint16_jit`. A float32 times the scale is exact in float64.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn quantize_block<T: Level>(src: &[f32], dst: &mut [T]) {
    for (d, &v) in dst.iter_mut().zip(src) {
        // max and min drop NaN, so NaN becomes 0.
        *d = T::truncate((v as f64 * T::SCALE + 0.5).max(0.0).min(T::SCALE));
    }
}

pub fn to_uint<T: Level>(src: &[f32], dst: &mut [T]) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| quantize_block(s, d));
}

/// `uint8_to_float32`: the float32 value times 1/255 in float64, rounded to float32.
pub fn uint8_to_float32(src: &[u8], dst: &mut [f32]) {
    let inv = 1.0 / 255.0;
    crate::par_zip(dst, src, move |d, &v| *d = (v as f32 as f64 * inv) as f32);
}

/// `uint16_to_float32`: the float32 value times 1/65535 in float64, rounded to float32.
pub fn uint16_to_float32(src: &[u16], dst: &mut [f32]) {
    let inv = 1.0 / 65535.0;
    crate::par_zip(dst, src, move |d, &v| *d = (v as f32 as f64 * inv) as f32);
}

/// Numba's fastmath luminance in float64: two fused multiply-adds.
fn luma(p: &[f32]) -> f64 {
    LUMA_B.mul_add(
        p[2] as f64,
        LUMA_G.mul_add(p[1] as f64, LUMA_R * p[0] as f64),
    )
}

pub trait Level: Copy + Send {
    const SCALE: f64;
    fn truncate(v: f64) -> Self;
}

/// The integer part of `v` in 0..2^31: added to 2^52, it fills the low mantissa bits. The
/// saturating `as` cast keeps LLVM from vectorizing the loops; this rounds and adds instead.
fn trunc_bits(v: f64) -> u64 {
    (v.trunc() + 4503599627370496.0).to_bits()
}

impl Level for u8 {
    const SCALE: f64 = 255.0;
    fn truncate(v: f64) -> Self {
        trunc_bits(v) as u8
    }
}

impl Level for u16 {
    const SCALE: f64 = 65535.0;
    fn truncate(v: f64) -> Self {
        trunc_bits(v) as u16
    }
}

/// `_float_to_uint8_luma_jit` / `_float_to_uint16_luma_jit` over HxWx3 pixels. The
/// scale-and-round fuses too under fastmath.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn luma_block<T: Level>(src: &[f32], cn: usize, dst: &mut [T]) {
    let scale = T::SCALE;
    // NaN becomes 0, as Numba's cast of it does.
    let clip = |v: f64| v.max(0.0).min(scale);
    match cn {
        1 => {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = T::truncate(clip(v as f64 * scale + 0.5));
            }
        }
        3 => {
            for (d, p) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
                *d = T::truncate(clip(luma(p).mul_add(scale, 0.5)));
            }
        }
        _ => {
            for (d, p) in dst.iter_mut().zip(src.chunks_exact(cn)) {
                *d = T::truncate(clip(luma(p).mul_add(scale, 0.5)));
            }
        }
    }
}

pub fn luma_to_uint<T: Level>(src: &[f32], cn: usize, dst: &mut [T]) {
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK * cn))
        .for_each(|(d, s)| luma_block(s, cn, d));
}

/// The first three channels of each `cn`-channel pixel of `src` into the RGBA `dst`.
pub fn rgb_to_rgba(src: &[f32], cn: usize, dst: &mut [f32]) {
    dst.par_chunks_mut(4 * BLOCK)
        .zip(src.par_chunks(cn * BLOCK))
        .for_each(|(d, s)| {
            for (dp, sp) in d.as_chunks_mut::<4>().0.iter_mut().zip(s.chunks_exact(cn)) {
                dp[..3].copy_from_slice(&sp[..3]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_rounds_clips_and_zeroes_nan() {
        let src = [
            0.0,
            1.0,
            0.5 / 255.0,
            0.499 / 255.0,
            -1.0,
            2.0,
            f32::NAN,
            f32::INFINITY,
        ];
        let mut dst = [0u8; 8];
        to_uint(&src, &mut dst);
        assert_eq!(dst, [0, 255, 1, 0, 0, 255, 0, 255]);
        let mut wide = [0u16; 8];
        to_uint(&src, &mut wide);
        assert_eq!(wide[1], 65535);
        assert_eq!(wide[6], 0);
    }

    #[test]
    fn trunc_bits_is_the_integer_part() {
        for v in [0.0, 0.999, 1.0, 254.9, 65535.0, 2_147_483_647.0] {
            assert_eq!(trunc_bits(v) & 0xffff_ffff, v.trunc() as u64, "{v}");
        }
    }
}
