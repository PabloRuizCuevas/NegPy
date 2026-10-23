//! OpenCV's bilinear warps of a float32 image with replicated edges: `remap`, `warpAffine` and
//! `warpPerspective`. Each source coordinate is fixed point at 1/32 px, rounded as OpenCV rounds
//! it; the four weights are its table's float32 products, summed left to right.

use multiversion::multiversion;
use rayon::prelude::*;

/// An HxWxC float32 image.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub data: &'a [f32],
    pub h: usize,
    pub w: usize,
    pub cn: usize,
}

const BITS: i32 = 5;
const MASK: i32 = (1 << BITS) - 1;
const STEP: f32 = 1.0 / (1 << BITS) as f32;

/// The pixel at fixed-point source coordinates (x, y) into `out`, edges replicated. `CN` is
/// the channel count, or 0 for one known only at run time.
#[inline(always)]
fn sample<const CN: usize>(src: &Image, x: i32, y: i32, out: &mut [f32]) {
    let (fx, fy) = ((x & MASK) as f32 * STEP, (y & MASK) as f32 * STEP);
    let (ix, iy) = (x >> BITS, y >> BITS);
    let w0 = (1.0 - fy) * (1.0 - fx);
    let w1 = (1.0 - fy) * fx;
    let w2 = fy * (1.0 - fx);
    let w3 = fy * fx;
    let cn = if CN == 0 { src.cn } else { CN };
    let row = src.w * cn;
    let s = src.data;
    let (wl, hl) = (src.w as i32 - 1, src.h as i32 - 1);
    if (ix as u32) < wl as u32 && (iy as u32) < hl as u32 {
        let a = iy as usize * row + ix as usize * cn;
        let (p, q) = (&s[a..][..2 * cn], &s[a + row..][..2 * cn]);
        for (k, o) in out[..cn].iter_mut().enumerate() {
            *o = p[k] * w0 + p[cn + k] * w1 + q[k] * w2 + q[cn + k] * w3;
        }
        return;
    }
    let (x0, x1) = (ix.clamp(0, wl) as usize, (ix + 1).clamp(0, wl) as usize);
    let (y0, y1) = (iy.clamp(0, hl) as usize, (iy + 1).clamp(0, hl) as usize);
    let (a, b) = (y0 * row + x0 * cn, y0 * row + x1 * cn);
    let (c, d) = (y1 * row + x0 * cn, y1 * row + x1 * cn);
    for (k, o) in out[..cn].iter_mut().enumerate() {
        *o = s[a + k] * w0 + s[b + k] * w1 + s[c + k] * w2 + s[d + k] * w3;
    }
}

fn sample_row<const CN: usize>(src: &Image, xs: &[i32], ys: &[i32], row: &mut [f32]) {
    let cn = if CN == 0 { src.cn } else { CN };
    for ((o, &x), &y) in row.chunks_exact_mut(cn).zip(xs).zip(ys) {
        sample::<CN>(src, x, y, o);
    }
}

/// Fills a `dw`-wide output, row by row in parallel, from `coords(y, xs, ys)`, which writes the
/// fixed-point source coordinates of output row y.
fn warp(
    src: &Image,
    dw: usize,
    out: &mut [f32],
    coords: impl Fn(usize, &mut [i32], &mut [i32]) + Sync,
) {
    let cn = src.cn;
    out.par_chunks_mut(dw * cn).enumerate().for_each_init(
        || (vec![0i32; dw], vec![0i32; dw]),
        |(xs, ys), (y, row)| {
            coords(y, xs, ys);
            match cn {
                1 => sample_row::<1>(src, xs, ys, row),
                3 => sample_row::<3>(src, xs, ys, row),
                4 => sample_row::<4>(src, xs, ys, row),
                _ => sample_row::<0>(src, xs, ys, row),
            }
        },
    );
}

/// OpenCV's `cvRound` of a float64 into an int, ties to even, for |v| < 2^30: added to 1.5 * 2^52,
/// the integer fills the low mantissa bits. `round_ties_even` is a libm call on the x86-64
/// baseline, and the saturating `as` cast keeps LLVM from vectorizing.
#[inline(always)]
fn round_i32(v: f64) -> i32 {
    (v + 6755399441055744.0).to_bits() as u32 as i32
}

/// Coordinates far outside any image, held where `round_i32` stays exact.
const FAR: f64 = (1u32 << 30) as f64;

/// `cv2.remap(src, map_x, map_y, INTER_LINEAR, borderMode=BORDER_REPLICATE)`; the maps are
/// float32, one entry per output pixel, `dw` wide.
pub fn remap_linear(src: &Image, map_x: &[f32], map_y: &[f32], dw: usize, out: &mut [f32]) {
    warp(src, dw, out, move |y, xs, ys| {
        remap_coords(&map_x[y * dw..][..dw], &map_y[y * dw..][..dw], xs, ys)
    });
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn remap_coords(mx: &[f32], my: &[f32], xs: &mut [i32], ys: &mut [i32]) {
    let scale = (1 << BITS) as f32;
    for (((x, y), &a), &b) in xs.iter_mut().zip(ys.iter_mut()).zip(mx).zip(my) {
        *x = round_i32(((a * scale) as f64).clamp(-FAR, FAR));
        *y = round_i32(((b * scale) as f64).clamp(-FAR, FAR));
    }
}

/// The radial lens correction of a same-size image (`apply_radial_distortion`): the remap
/// source of output pixel p is c + (p - c) s f with f = 1 + k1 |(p - c) s|^2 inv_hd2, in float32
/// as NumPy builds the maps.
pub fn remap_radial(
    src: &Image,
    (cx, cy): (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
    out: &mut [f32],
) {
    let dw = src.w;
    warp(src, dw, out, move |y, xs, ys| {
        radial_coords(y as f32, (cx, cy), s, k1, inv_hd2, xs, ys)
    });
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn radial_coords(
    y: f32,
    (cx, cy): (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
    xs: &mut [i32],
    ys: &mut [i32],
) {
    let scale = (1 << BITS) as f32;
    let py = (y - cy) * s;
    for (i, (x, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
        let px = (i as f32 - cx) * s;
        let f = 1.0 + k1 * (px * px + py * py) * inv_hd2;
        *x = round_i32((((cx + px * f) * scale) as f64).clamp(-FAR, FAR));
        *yv = round_i32((((cy + py * f) * scale) as f64).clamp(-FAR, FAR));
    }
}

/// `cv2.warpAffine(src, m, (dw, dh), INTER_LINEAR, BORDER_REPLICATE)`; `m` maps source to
/// output unless `inverse_map` (OpenCV's WARP_INVERSE_MAP).
pub fn warp_affine(src: &Image, m: [f64; 6], inverse_map: bool, dw: usize, out: &mut [f32]) {
    let mut m = m;
    if !inverse_map {
        let d = m[0] * m[4] - m[1] * m[3];
        let d = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (a11, a22) = (m[4] * d, m[0] * d);
        m[0] = a11;
        m[1] *= -d;
        m[3] *= -d;
        m[4] = a22;
        let b1 = -m[0] * m[2] - m[1] * m[5];
        let b2 = -m[3] * m[2] - m[4] * m[5];
        m[2] = b1;
        m[5] = b2;
    }
    const AB_SCALE: f64 = 1024.0;
    const ROUND: i32 = 1 << (10 - BITS - 1);
    let adelta: Vec<i32> = (0..dw)
        .map(|x| round_i32((m[0] * x as f64 * AB_SCALE).clamp(-FAR, FAR)))
        .collect();
    let bdelta: Vec<i32> = (0..dw)
        .map(|x| round_i32((m[3] * x as f64 * AB_SCALE).clamp(-FAR, FAR)))
        .collect();
    warp(src, dw, out, move |y, xs, ys| {
        let x0 = round_i32(((m[1] * y as f64 + m[2]) * AB_SCALE).clamp(-FAR, FAR)) + ROUND;
        let y0 = round_i32(((m[4] * y as f64 + m[5]) * AB_SCALE).clamp(-FAR, FAR)) + ROUND;
        for (i, (x, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
            *x = x0.wrapping_add(adelta[i]) >> (10 - BITS);
            *yv = y0.wrapping_add(bdelta[i]) >> (10 - BITS);
        }
    });
}

/// OpenCV's closed-form 3x3 inverse (`cv::invert`, DECOMP_LU, n <= 3); None when singular.
fn invert3(s: [f64; 9]) -> Option<[f64; 9]> {
    let e = |i: usize, j: usize| s[i * 3 + j];
    let d = e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
        - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
        + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0));
    if d == 0.0 {
        return None;
    }
    let d = 1.0 / d;
    Some([
        (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1)) * d,
        (e(0, 2) * e(2, 1) - e(0, 1) * e(2, 2)) * d,
        (e(0, 1) * e(1, 2) - e(0, 2) * e(1, 1)) * d,
        (e(1, 2) * e(2, 0) - e(1, 0) * e(2, 2)) * d,
        (e(0, 0) * e(2, 2) - e(0, 2) * e(2, 0)) * d,
        (e(0, 2) * e(1, 0) - e(0, 0) * e(1, 2)) * d,
        (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0)) * d,
        (e(0, 1) * e(2, 0) - e(0, 0) * e(2, 1)) * d,
        (e(0, 0) * e(1, 1) - e(0, 1) * e(1, 0)) * d,
    ])
}

/// `cv2.warpPerspective(src, m, (dw, dh), INTER_LINEAR, BORDER_REPLICATE)`; `m` maps source to
/// output unless `inverse_map`. A singular `m` inverts to zeros, as `cv::invert` leaves it.
pub fn warp_perspective(
    src: &Image,
    m: [f64; 9],
    inverse_map: bool,
    (dw, dh): (usize, usize),
    out: &mut [f32],
) {
    let m = if inverse_map {
        m
    } else {
        invert3(m).unwrap_or([0.0; 9])
    };
    // OpenCV starts the projective sums again at each block of bw0 columns.
    let bh0 = dh.min(16);
    let bw0 = (1024 / bh0).min(dw);
    warp(src, dw, out, move |y, xs, ys| {
        let yf = y as f64;
        for (b, (xs, ys)) in xs.chunks_mut(bw0).zip(ys.chunks_mut(bw0)).enumerate() {
            let xb = (b * bw0) as f64;
            let x0 = m[0] * xb + m[1] * yf + m[2];
            let y0 = m[3] * xb + m[4] * yf + m[5];
            let w0 = m[6] * xb + m[7] * yf + m[8];
            persp_coords([x0, y0, w0], [m[0], m[3], m[6]], xs, ys);
        }
    });
}

/// One block's coordinates from the projective sums `start` at its first column and their
/// steps per column.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn persp_coords(start: [f64; 3], step: [f64; 3], xs: &mut [i32], ys: &mut [i32]) {
    for (i, (x, y)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
        let x1 = i as f64;
        let w = start[2] + step[2] * x1;
        let w = if w != 0.0 {
            (1 << BITS) as f64 / w
        } else {
            0.0
        };
        *x = round_i32(((start[0] + step[0] * x1) * w).clamp(-FAR, FAR));
        *y = round_i32(((start[1] + step[1] * x1) * w).clamp(-FAR, FAR));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(h: usize, w: usize, cn: usize) -> Vec<f32> {
        (0..h * w * cn)
            .map(|i| (i % 7) as f32 * 0.25 - 0.5)
            .collect()
    }

    #[test]
    fn identity_affine_copies() {
        let data = ramp(9, 13, 3);
        let src = Image {
            data: &data,
            h: 9,
            w: 13,
            cn: 3,
        };
        let mut out = vec![0.0; data.len()];
        warp_affine(&src, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], false, 13, &mut out);
        assert_eq!(out, data);
        warp_perspective(
            &src,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            false,
            (13, 9),
            &mut out,
        );
        assert_eq!(out, data);
    }

    #[test]
    fn remap_half_pixel_averages_and_replicates() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let src = Image {
            data: &data,
            h: 1,
            w: 4,
            cn: 1,
        };
        let mut out = vec![0.0; 3];
        remap_linear(&src, &[0.5, -3.0, 9.0], &[0.0; 3], 3, &mut out);
        assert_eq!(out, [0.5, 0.0, 3.0]);
    }

    #[test]
    fn invert3_matches_identity_product() {
        let m = [2.0, 0.5, 1.0, 0.25, 3.0, -1.0, 0.001, 0.002, 1.0];
        let inv = invert3(m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let p: f64 = (0..3).map(|k| m[i * 3 + k] * inv[k * 3 + j]).sum();
                assert!((p - (i == j) as i32 as f64).abs() < 1e-12);
            }
        }
        assert!(invert3([0.0; 9]).is_none());
    }
}
