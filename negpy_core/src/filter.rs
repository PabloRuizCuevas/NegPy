//! OpenCV's float32 separable filter engine (`sepFilter2D`, `GaussianBlur`), bit for bit with
//! opencv-python's AVX2 build: an 8-lane vector region with fused multiply-adds, then the scalar
//! loops in the arithmetic GCC compiled them to.

use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Border {
    Constant,
    Replicate,
    Reflect,
    Reflect101,
}

impl Border {
    /// From an OpenCV border code; `BORDER_ISOLATED` is ignored, since inputs are whole arrays.
    pub fn from_cv(code: i32) -> Option<Self> {
        match code & !16 {
            0 => Some(Border::Constant),
            1 => Some(Border::Replicate),
            2 => Some(Border::Reflect),
            4 => Some(Border::Reflect101),
            _ => None,
        }
    }
}

/// OpenCV's `borderInterpolate`; None for a constant (zero) border.
pub fn border_interpolate(p: isize, len: usize, border: Border) -> Option<usize> {
    let n = len as isize;
    if (0..n).contains(&p) {
        return Some(p as usize);
    }
    match border {
        Border::Constant => None,
        Border::Replicate => Some(if p < 0 { 0 } else { len - 1 }),
        Border::Reflect | Border::Reflect101 => {
            if len == 1 {
                return Some(0);
            }
            let delta = (border == Border::Reflect101) as isize;
            let mut p = p;
            while !(0..n).contains(&p) {
                p = if p < 0 {
                    -p - 1 + delta
                } else {
                    n - 1 - (p - n) - delta
                };
            }
            Some(p as usize)
        }
    }
}

include!("exptab.rs");

/// OpenCV's `exp(softdouble)` (softfloat.cpp `f64_exp`): a table and a polynomial, not libm.
fn soft_exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { x } else { 0.0 };
    }
    let a0p = f64::from_bits(0x3f83ce0f3e46f431);
    let a5 = 1.0 / a0p;
    let a4 = f64::from_bits(0x3fe62e42fefa39f1) / a0p;
    let a3 = f64::from_bits(0x3fcebfbdff82a45a) / a0p;
    let a2 = f64::from_bits(0x3fac6b08d81fec75) / a0p;
    let a1 = f64::from_bits(0x3f83b2a72b4f3cd3) / a0p;
    let a0 = f64::from_bits(0x3f55e7aa1566c2a4) / a0p;
    let prescale = f64::from_bits(0x3ff71547652b82fe) * 64.0;
    let postscale = 1.0 / 64.0;
    let max_val = 3000.0 * 64.0;

    let biased = ((x.to_bits() >> 52) & 0x7ff) as i64;
    let mut x0 = if biased > 1023 + 10 {
        if x < 0.0 { -max_val } else { max_val }
    } else {
        x * prescale
    };
    let val0 = x0.round_ties_even() as i32;
    let t = ((val0 >> 6) + 1023).clamp(0, 2047);
    let buf = f64::from_bits((t as u64) << 52);
    x0 = (x0 - x0.round_ties_even()) * postscale;
    buf * a0p
        * f64::from_bits(EXP_TAB[(val0 & 63) as usize])
        * (((((a0 * x0 + a1) * x0 + a2) * x0 + a3) * x0 + a4) * x0 + a5)
}

/// `cv2.getGaussianKernel(n, sigma, cv2.CV_32F)` (`getGaussianKernelBitExact`).
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        let fixed: Option<&[f64]> = match n {
            1 => Some(&[1.0]),
            3 => Some(&[0.25, 0.5, 0.25]),
            5 => Some(&[0.0625, 0.25, 0.375, 0.25, 0.0625]),
            7 => Some(&[
                0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
            ]),
            9 => Some(&[
                4.0 / 256.0,
                13.0 / 256.0,
                30.0 / 256.0,
                51.0 / 256.0,
                60.0 / 256.0,
                51.0 / 256.0,
                30.0 / 256.0,
                13.0 / 256.0,
                4.0 / 256.0,
            ]),
            _ => None,
        };
        if let Some(k) = fixed {
            return k.iter().map(|&v| v as f32).collect();
        }
    }
    let sigma_x = if sigma > 0.0 {
        sigma
    } else {
        (n as f64).mul_add(
            f64::from_bits(0x3fc3333333333333),
            f64::from_bits(0x3fd6666666666666),
        )
    };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let n2 = (n - 1) / 2;
    let mut values = Vec::with_capacity(n2);
    let mut sum = 0.0f64;
    let mut x = 1 - n as i64;
    for _ in 0..n2 {
        let t = soft_exp((x * x) as f64 * scale2x);
        values.push(t);
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut result = vec![0f64; n];
    for (i, v) in values.iter().enumerate() {
        let t = v * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
    }
    result[n2] = mul1;
    if n.is_multiple_of(2) {
        result[n2 + 1] = result[n2];
    }
    result.iter().map(|&v| v as f32).collect()
}

/// `createGaussianKernels`' size from sigma for a float image.
pub fn gaussian_ksize(sigma: f64) -> usize {
    ((sigma * 4.0 * 2.0 + 1.0).round_ties_even() as i64 | 1) as usize
}

#[derive(Clone, Copy, PartialEq)]
enum Symmetry {
    Symmetric,
    Asymmetric,
    None,
}

fn symmetry(k: &[f32]) -> Symmetry {
    let n = k.len();
    if (0..n).all(|i| k[i] == k[n - 1 - i]) {
        Symmetry::Symmetric
    } else if (0..n).all(|i| k[i] == -k[n - 1 - i]) {
        Symmetry::Asymmetric
    } else {
        Symmetry::None
    }
}

const LANES: usize = 8;

/// The 8-lane vector region of a row, `out.len()` a multiple of 8. Tap-outer slice loops keep
/// each output's order of operations and let the compiler vectorize.
#[inline(always)]
fn row_vector(p: &[f32], out: &mut [f32], cn: usize, k: &[f32], sym: Symmetry) {
    let n = out.len();
    let ks = k.len();
    let r = ks / 2;
    let sl = |o: isize| &p[((r * cn) as isize + o * cn as isize) as usize..][..n];
    if ks == 1 {
        out.iter_mut()
            .zip(&p[..n])
            .for_each(|(o, &x)| *o = k[0] * x);
    } else if sym == Symmetry::Symmetric && ks <= 5 {
        let (c, m1, p1) = (sl(0), sl(-1), sl(1));
        let (k0, k1) = (k[r], k[r + 1]);
        if ks == 3 {
            if k0.abs() == 2.0 && k1 == 1.0 {
                for i in 0..n {
                    out[i] = c[i].mul_add(k0, m1[i] + p1[i]);
                }
            } else {
                for i in 0..n {
                    out[i] = c[i].mul_add(k0, (m1[i] + p1[i]) * k1);
                }
            }
        } else {
            let (m2, p2, k2) = (sl(-2), sl(2), k[r + 2]);
            if k0 == -2.0 && k1 == 0.0 && k2 == 1.0 {
                for i in 0..n {
                    out[i] = c[i].mul_add(-2.0, m2[i] + p2[i]);
                }
            } else {
                for i in 0..n {
                    out[i] = (p2[i] + m2[i]).mul_add(k2, c[i].mul_add(k0, (m1[i] + p1[i]) * k1));
                }
            }
        }
    } else if sym == Symmetry::Asymmetric && ks == 3 {
        let (m1, p1, k1) = (sl(-1), sl(1), k[r + 1]);
        if k[r] == 0.0 && k1 == 1.0 {
            for i in 0..n {
                out[i] = p1[i] - m1[i];
            }
        } else {
            for i in 0..n {
                out[i] = (p1[i] - m1[i]) * k1;
            }
        }
    } else {
        let (blocks, rest) = out.split_at_mut(n / 16 * 16);
        for (b, chunk) in blocks.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            row_taps::<16>(p, chunk, b * 16, cn, k);
        }
        if !rest.is_empty() {
            row_taps::<8>(p, rest, n / 16 * 16, cn, k);
        }
    }
}

/// RowVec_32f's sum for `B` outputs from `base`, the accumulators held in registers.
#[inline(always)]
fn row_taps<const B: usize>(p: &[f32], out: &mut [f32], base: usize, cn: usize, k: &[f32]) {
    let mut acc = [0f32; B];
    let t0 = &p[base..base + B];
    for l in 0..B {
        acc[l] = t0[l].mul_add(k[0], 0.0);
    }
    for (j, &kj) in k.iter().enumerate().skip(1) {
        let t = &p[base + j * cn..base + j * cn + B];
        for l in 0..B {
            acc[l] = t[l].mul_add(kj, acc[l]);
        }
    }
    out.copy_from_slice(&acc);
}

/// One row: `p` is the source row padded by `r = k.len() / 2` pixels on each side.
#[inline(always)]
fn row_filter_impl(p: &[f32], out: &mut [f32], cn: usize, k: &[f32], sym: Symmetry) {
    let n = out.len();
    let ks = k.len();
    let r = ks / 2;
    let vec_end = n / LANES * LANES;
    let at = |i: usize, o: isize| p[(i as isize + (r * cn) as isize + o * cn as isize) as usize];
    row_vector(p, &mut out[..vec_end], cn, k, sym);
    let tail = out.iter_mut().enumerate().skip(vec_end);

    if ks == 1 {
        for (i, o) in tail {
            *o = k[0] * p[i];
        }
        return;
    }
    // After the vector region: a pair loop, then the odd last float alone, centre first
    // (fused for 3 taps, plain for 5). The pair loop matches the vector arithmetic.
    let single = |i: usize| i + 2 > n && (n - vec_end) % 2 == 1;
    let small = sym != Symmetry::None && ks <= 5;
    if small && sym == Symmetry::Symmetric {
        let (k0, k1) = (k[r], k[r + 1]);
        let k2 = if ks == 5 { k[r + 2] } else { 0.0 };
        for (i, o) in tail {
            let (s0, p1) = (at(i, 0), at(i, -1) + at(i, 1));
            *o = if ks == 3 {
                if single(i) {
                    k1.mul_add(at(i, 1) + at(i, -1), k0 * s0)
                } else if k0 == 2.0 && k1 == 1.0 {
                    (at(i, -1) + s0 * 2.0) + at(i, 1)
                } else if k0 == -2.0 && k1 == 1.0 {
                    (at(i, -1) - s0 * 2.0) + at(i, 1)
                } else {
                    s0.mul_add(k0, p1 * k1)
                }
            } else if single(i) {
                (k0 * s0 + k1 * (at(i, 1) + at(i, -1))) + k2 * (at(i, 2) + at(i, -2))
            } else if k0 == -2.0 && k1 == 0.0 && k2 == 1.0 {
                (-2.0 * s0 + at(i, -2)) + at(i, 2)
            } else {
                (at(i, 2) + at(i, -2)).mul_add(k2, s0.mul_add(k0, p1 * k1))
            };
        }
        return;
    }
    if small && ks == 3 {
        let (k0, k1) = (k[r], k[r + 1]);
        for (i, o) in tail {
            let d = at(i, 1) - at(i, -1);
            *o = if single(i) {
                k1.mul_add(d, k0 * at(i, 0))
            } else if k0 == 0.0 && k1 == 1.0 {
                d
            } else {
                d * k1
            };
        }
        return;
    }
    // RowFilter's scalar loops: four at a time (contracted), then one at a time, where GCC
    // unrolled the tap loop by four uncontracted and fused the remainder.
    let unrolled_end = vec_end + (n - vec_end) / 4 * 4;
    let plain_adds = (ks - 1) / 4 * 4;
    for (i, o) in tail {
        let tap = |j: usize| p[i + j * cn];
        *o = if i < unrolled_end {
            (1..ks).fold(k[0] * tap(0), |s, j| k[j].mul_add(tap(j), s))
        } else {
            (1..ks).fold(k[0] * tap(0), |s, j| {
                if j <= plain_adds {
                    s + k[j] * tap(j)
                } else {
                    k[j].mul_add(tap(j), s)
                }
            })
        };
    }
}

/// One output row from the row-filtered rows `rows[0..ks]` (top to bottom).
#[inline(always)]
fn column_filter_impl(rows: &[&[f32]], out: &mut [f32], k: &[f32], sym: Symmetry) {
    let n = out.len();
    let ks = k.len();
    let r = ks / 2;
    let vec_end = n / LANES * LANES;
    let (vec, tail) = out.split_at_mut(vec_end);
    match sym {
        Symmetry::Symmetric if ks == 3 => {
            let (f0, f1) = (k[1], k[2]);
            let (s0, s1, s2) = (rows[0], rows[1], rows[2]);
            if f0.abs() == 2.0 && f1 == 1.0 {
                for i in 0..vec_end {
                    vec[i] = s1[i].mul_add(f0, (s0[i] + s2[i]) + 0.0);
                }
            } else {
                for i in 0..vec_end {
                    vec[i] = (s0[i] + s2[i]).mul_add(f1, s1[i].mul_add(f0, 0.0));
                }
            }
            for (t, o) in tail.iter_mut().enumerate() {
                let i = vec_end + t;
                *o = if f0 == 2.0 && f1 == 1.0 {
                    ((s0[i] + s1[i] * 2.0) + s2[i]) + 0.0
                } else if f0 == -2.0 && f1 == 1.0 {
                    ((s0[i] - s1[i] * 2.0) + s2[i]) + 0.0
                } else {
                    (s0[i] + s2[i]).mul_add(f1, s1[i] * f0) + 0.0
                };
            }
        }
        Symmetry::Symmetric => {
            let c = rows[r];
            let (blocks, rest) = vec.split_at_mut(vec_end / 16 * 16);
            for (b, chunk) in blocks.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                column_taps::<16>(rows, chunk, b * 16, k);
            }
            if !rest.is_empty() {
                column_taps::<8>(rows, rest, vec_end / 16 * 16, k);
            }
            // GCC left the scalar column loops uncontracted: plain multiply, then add.
            for (t, o) in tail.iter_mut().enumerate() {
                let i = vec_end + t;
                *o = (1..=r).fold(k[r] * c[i] + 0.0, |s, j| {
                    s + k[r + j] * (rows[r + j][i] + rows[r - j][i])
                });
            }
        }
        // sep_filter2d admits only a 3-tap antisymmetric column kernel.
        Symmetry::Asymmetric | Symmetry::None => {
            let f1 = k[2];
            let (lo, hi) = if f1 < 0.0 {
                (rows[2], rows[0])
            } else {
                (rows[0], rows[2])
            };
            for i in 0..n {
                let o = if i < vec_end {
                    &mut vec[i]
                } else {
                    &mut tail[i - vec_end]
                };
                *o = if f1.abs() == 1.0 {
                    (hi[i] - lo[i]) + 0.0
                } else {
                    (rows[2][i] - rows[0][i]).mul_add(f1, 0.0)
                };
            }
        }
    }
}

/// SymmColumnVec_32f's sum for `B` outputs from `base`, the accumulators held in registers.
#[inline(always)]
fn column_taps<const B: usize>(rows: &[&[f32]], out: &mut [f32], base: usize, k: &[f32]) {
    let r = k.len() / 2;
    let mut acc = [0f32; B];
    let c = &rows[r][base..base + B];
    for l in 0..B {
        acc[l] = c[l].mul_add(k[r], 0.0);
    }
    for j in 1..=r {
        let (a, b, kj) = (
            &rows[r + j][base..base + B],
            &rows[r - j][base..base + B],
            k[r + j],
        );
        for l in 0..B {
            acc[l] = (a[l] + b[l]).mul_add(kj, acc[l]);
        }
    }
    out.copy_from_slice(&acc);
}

/// Copies of the row and column passes compiled for AVX2 + FMA, so `mul_add` is one
/// instruction and the slice loops vectorize; the plain copies serve other CPUs.
#[cfg(target_arch = "x86_64")]
mod fma {
    use super::Symmetry;

    #[target_feature(enable = "avx2,fma")]
    pub fn row_filter(p: &[f32], out: &mut [f32], cn: usize, k: &[f32], sym: Symmetry) {
        super::row_filter_impl(p, out, cn, k, sym)
    }

    #[target_feature(enable = "avx2,fma")]
    pub fn column_filter(rows: &[&[f32]], out: &mut [f32], k: &[f32], sym: Symmetry) {
        super::column_filter_impl(rows, out, k, sym)
    }
}

fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn row_filter(fast: bool, p: &[f32], out: &mut [f32], cn: usize, k: &[f32], sym: Symmetry) {
    #[cfg(target_arch = "x86_64")]
    if fast {
        // SAFETY: `fast` is true only when has_fma() found AVX2 and FMA on this CPU.
        return unsafe { fma::row_filter(p, out, cn, k, sym) };
    }
    let _ = fast;
    row_filter_impl(p, out, cn, k, sym)
}

fn column_filter(fast: bool, rows: &[&[f32]], out: &mut [f32], k: &[f32], sym: Symmetry) {
    #[cfg(target_arch = "x86_64")]
    if fast {
        // SAFETY: `fast` is true only when has_fma() found AVX2 and FMA on this CPU.
        return unsafe { fma::column_filter(rows, out, k, sym) };
    }
    let _ = fast;
    column_filter_impl(rows, out, k, sym)
}

/// `cv2.sepFilter2D(src, -1, kx, ky, borderType=border)` for float32, anchor at the center,
/// written into `dst`. None for a kernel OpenCV filters on a path this port does not cover.
pub fn sep_filter2d(
    src: &[f32],
    dst: &mut [f32],
    (h, w, cn): (usize, usize, usize),
    (kx, ky): (&[f32], &[f32]),
    border: Border,
) -> Option<()> {
    if h == 0
        || w == 0
        || cn == 0
        || src.len() != h * w * cn
        || dst.len() != src.len()
        || kx.len().is_multiple_of(2)
        || ky.len().is_multiple_of(2)
    {
        return None;
    }
    let (sx, sy) = (symmetry(kx), symmetry(ky));
    // Every caller passes a symmetric kernel, or Sobel's antisymmetric 3-tap one.
    let covered = |s: Symmetry, len: usize| {
        s == Symmetry::Symmetric || (s == Symmetry::Asymmetric && len == 3)
    };
    if !covered(sx, kx.len()) || !covered(sy, ky.len()) {
        return None;
    }
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    let n = w * cn;

    let fast = has_fma();
    let xmap: Vec<Option<usize>> = (0..w + 2 * rx)
        .map(|px| border_interpolate(px as isize - rx as isize, w, border))
        .collect();
    let row_pass = |padded: &mut [f32], sy: usize, out: &mut [f32]| {
        let row = &src[sy * n..(sy + 1) * n];
        padded[rx * cn..(rx + w) * cn].copy_from_slice(row);
        for px in (0..rx).chain(rx + w..w + 2 * rx) {
            let chunk = &mut padded[px * cn..(px + 1) * cn];
            match xmap[px] {
                Some(sxi) => chunk.copy_from_slice(&row[sxi * cn..(sxi + 1) * cn]),
                None => chunk.fill(0.0),
            }
        }
        row_filter(fast, padded, out, cn, kx, sx);
    };

    // Strips of output rows, each with the row-filtered rows it reads kept in cache.
    const STRIP: usize = 32;
    dst.par_chunks_mut(STRIP * n).enumerate().for_each_init(
        || {
            (
                vec![0f32; (w + 2 * rx) * cn],
                vec![0f32; (STRIP + 2 * ry) * n],
            )
        },
        |(padded, buf), (strip, out)| {
            let y0 = strip * STRIP;
            let count = out.len() / n;
            for t in 0..count + 2 * ry {
                let line = &mut buf[t * n..(t + 1) * n];
                match border_interpolate((y0 + t) as isize - ry as isize, h, border) {
                    Some(sy) => row_pass(padded, sy, line),
                    None => line.fill(0.0),
                }
            }
            for (yy, line) in out.chunks_exact_mut(n).enumerate() {
                let rows: Vec<&[f32]> = (0..ky.len())
                    .map(|j| &buf[(yy + j) * n..(yy + j + 1) * n])
                    .collect();
                column_filter(fast, &rows, line, ky, sy);
            }
        },
    );
    Some(())
}

/// `cv2.GaussianBlur(src, (kw, kh), sigma_x, sigmaY=sigma_y, borderType=border)` for float32.
pub fn gaussian_blur(
    src: &[f32],
    dst: &mut [f32],
    (h, w, cn): (usize, usize, usize),
    ksize: (usize, usize),
    (sigma_x, sigma_y): (f64, f64),
    border: Border,
) -> Option<()> {
    let (mut kw, mut kh) = ksize;
    let sigma_y = if sigma_y <= 0.0 { sigma_x } else { sigma_y };
    if kw == 0 && sigma_x > 0.0 {
        kw = gaussian_ksize(sigma_x);
    }
    if kh == 0 && sigma_y > 0.0 {
        kh = gaussian_ksize(sigma_y);
    }
    if kw.is_multiple_of(2) || kh.is_multiple_of(2) {
        return None;
    }
    if border != Border::Constant {
        if h == 1 {
            kh = 1;
        }
        if w == 1 {
            kw = 1;
        }
    }
    if kw == 1 && kh == 1 {
        dst.copy_from_slice(src);
        return Some(());
    }
    let kx = gaussian_kernel(kw, sigma_x.max(0.0));
    let ky = if kh == kw && (sigma_x - sigma_y).abs() < f64::EPSILON {
        kx.clone()
    } else {
        gaussian_kernel(kh, sigma_y.max(0.0))
    };
    sep_filter2d(src, dst, (h, w, cn), (&kx, &ky), border)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect101_and_reflect_borders() {
        let r101: Vec<_> = (-3..8)
            .map(|p| border_interpolate(p, 5, Border::Reflect101).unwrap())
            .collect();
        assert_eq!(r101, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
        let r: Vec<_> = (-2..7)
            .map(|p| border_interpolate(p, 5, Border::Reflect).unwrap())
            .collect();
        assert_eq!(r, vec![1, 0, 0, 1, 2, 3, 4, 4, 3]);
        assert_eq!(border_interpolate(-1, 5, Border::Constant), None);
        assert_eq!(border_interpolate(9, 5, Border::Replicate), Some(4));
    }

    #[test]
    fn soft_exp_is_close_to_exp() {
        for x in [-30.0, -7.5, -1.0, -0.125, 0.0, 0.5, 3.0] {
            let rel = (soft_exp(x) - f64::exp(x)).abs() / f64::exp(x);
            assert!(rel < 1e-14, "x={x} rel={rel}");
        }
    }

    #[test]
    fn kernels_sum_to_one() {
        for (n, sigma) in [(5, 0.0), (11, 0.0), (17, 2.0), (25, 3.0)] {
            let s: f64 = gaussian_kernel(n, sigma).iter().map(|&v| v as f64).sum();
            assert!((s - 1.0).abs() < 1e-6, "n={n}");
        }
    }

    #[test]
    fn flat_image_stays_flat() {
        let src = vec![0.5f32; 23 * 31 * 3];
        let mut out = vec![0f32; src.len()];
        gaussian_blur(
            &src,
            &mut out,
            (23, 31, 3),
            (5, 5),
            (0.0, 0.0),
            Border::Reflect101,
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 0.5));
    }
}
