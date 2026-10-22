//! OpenCV's erode, dilate and morphologyEx for rectangular kernels. A min or max has no
//! rounding, so a separable row-then-column pass gives OpenCV's bits.

use rayon::prelude::*;

use crate::filter::{Border, border_interpolate};

pub trait Pixel: Copy + PartialOrd + Send + Sync {
    /// Erode's identity, OpenCV's default constant border for it.
    const HIGH: Self;
    /// Dilate's identity.
    const LOW: Self;
    fn sub(self, other: Self) -> Self;
    fn lo(self, other: Self) -> Self;
    fn hi(self, other: Self) -> Self;
}

impl Pixel for u8 {
    const HIGH: Self = u8::MAX;
    const LOW: Self = u8::MIN;
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn lo(self, other: Self) -> Self {
        self.min(other)
    }
    fn hi(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Pixel for f32 {
    const HIGH: Self = f32::INFINITY;
    const LOW: Self = f32::NEG_INFINITY;
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn lo(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
    fn hi(self, other: Self) -> Self {
        if other > self { other } else { self }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    TopHat,
    BlackHat,
}

impl Op {
    /// From `cv2.MORPH_*`.
    pub fn from_cv(code: i32) -> Option<Self> {
        Some(match code {
            0 => Op::Erode,
            1 => Op::Dilate,
            2 => Op::Open,
            3 => Op::Close,
            4 => Op::Gradient,
            5 => Op::TopHat,
            6 => Op::BlackHat,
            _ => return None,
        })
    }
}

#[inline(always)]
fn pick<T: Pixel, const ERODE: bool>(a: T, b: T) -> T {
    if ERODE { a.lo(b) } else { a.hi(b) }
}

/// `out[i] = min (or max) of src[i + j * stride]` for `j < len`, from overlapping power-of-two
/// windows: a min or max is idempotent, so the overlap changes nothing and a window of
/// `len` costs log2(len) vector passes.
#[inline(always)]
fn window<T: Pixel, const ERODE: bool>(
    cur: &mut Vec<T>,
    scratch: &mut Vec<T>,
    out: &mut [T],
    len: usize,
    stride: usize,
) {
    let mut win = 1;
    while win * 2 <= len {
        let off = win * stride;
        let valid = cur.len() - off;
        scratch.resize(valid, cur[0]);
        for ((o, &x), &y) in scratch.iter_mut().zip(&cur[..valid]).zip(&cur[off..]) {
            *o = pick::<T, ERODE>(x, y);
        }
        std::mem::swap(cur, scratch);
        win *= 2;
    }
    let off = (len - win) * stride;
    for ((o, &x), &y) in out.iter_mut().zip(cur.iter()).zip(&cur[off..]) {
        *o = pick::<T, ERODE>(x, y);
    }
}

/// One erode (`ERODE`) or dilate pass with a `kw` x `kh` rectangle anchored at `(ax, ay)`.
fn rect_pass<T: Pixel, const ERODE: bool>(
    src: &[T],
    dst: &mut [T],
    (h, w, cn): (usize, usize, usize),
    (kw, kh): (usize, usize),
    (ax, ay): (usize, usize),
    border: Border,
) {
    let n = w * cn;
    let identity = if ERODE { T::HIGH } else { T::LOW };
    let xmap: Vec<Option<usize>> = (0..w + kw - 1)
        .map(|px| border_interpolate(px as isize - ax as isize, w, border))
        .collect();

    let mut rows = vec![identity; h * n];
    rows.par_chunks_mut(n).enumerate().for_each_init(
        || (Vec::new(), Vec::new()),
        |(padded, scratch): &mut (Vec<T>, Vec<T>), (y, out)| {
            padded.resize((w + kw - 1) * cn, identity);
            let row = &src[y * n..(y + 1) * n];
            padded[ax * cn..(ax + w) * cn].copy_from_slice(row);
            for px in (0..ax).chain(ax + w..w + kw - 1) {
                let chunk = &mut padded[px * cn..(px + 1) * cn];
                match xmap[px] {
                    Some(sx) => chunk.copy_from_slice(&row[sx * cn..(sx + 1) * cn]),
                    None => chunk.fill(identity),
                }
            }
            window::<T, ERODE>(padded, scratch, out, kw, cn);
        },
    );

    // The column pass doubles whole rows: row t of `a` holds the extreme of rows t..t+win.
    let total = h + kh - 1;
    let mut a = vec![identity; total * n];
    a.par_chunks_mut(n).enumerate().for_each(|(t, line)| {
        if let Some(sy) = border_interpolate(t as isize - ay as isize, h, border) {
            line.copy_from_slice(&rows[sy * n..(sy + 1) * n]);
        }
    });
    let mut b = vec![identity; total * n];
    let mut win = 1;
    while win * 2 <= kh {
        let valid = total - win * 2 + 1;
        b[..valid * n]
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(t, line)| {
                let (x, y) = (&a[t * n..(t + 1) * n], &a[(t + win) * n..(t + win + 1) * n]);
                for ((o, &p), &q) in line.iter_mut().zip(x).zip(y) {
                    *o = pick::<T, ERODE>(p, q);
                }
            });
        std::mem::swap(&mut a, &mut b);
        win *= 2;
    }
    let off = kh - win;
    dst.par_chunks_mut(n).enumerate().for_each(|(y, line)| {
        let (x, z) = (&a[y * n..(y + 1) * n], &a[(y + off) * n..(y + off + 1) * n]);
        for ((o, &p), &q) in line.iter_mut().zip(x).zip(z) {
            *o = pick::<T, ERODE>(p, q);
        }
    });
}

/// `cv2.erode`/`cv2.dilate` with a full `kw` x `kh` rectangle and `iterations`: OpenCV turns
/// iterations of a rectangle into one larger rectangle with the anchor scaled.
fn erode_dilate<T: Pixel>(
    src: &[T],
    dst: &mut [T],
    dims: (usize, usize, usize),
    ksize: (usize, usize),
    iterations: usize,
    erode: bool,
    border: Border,
) {
    let (kw, kh) = ksize;
    let (mut ax, mut ay) = (kw / 2, kh / 2);
    let (mut kw, mut kh) = (kw, kh);
    if iterations == 0 || (kw == 1 && kh == 1) {
        dst.copy_from_slice(src);
        return;
    }
    if iterations > 1 {
        ax *= iterations;
        ay *= iterations;
        kw += (iterations - 1) * (kw - 1);
        kh += (iterations - 1) * (kh - 1);
    }
    if erode {
        rect_pass::<T, true>(src, dst, dims, (kw, kh), (ax, ay), border);
    } else {
        rect_pass::<T, false>(src, dst, dims, (kw, kh), (ax, ay), border);
    }
}

/// `cv2.morphologyEx(src, op, ones((kh, kw)), iterations=..., borderType=...)`, the border value
/// OpenCV's default.
pub fn morphology<T: Pixel>(
    src: &[T],
    dst: &mut [T],
    dims: (usize, usize, usize),
    op: Op,
    ksize: (usize, usize),
    iterations: usize,
    border: Border,
) -> Option<()> {
    let (h, w, cn) = dims;
    if h == 0
        || w == 0
        || cn == 0
        || ksize.0 == 0
        || ksize.1 == 0
        || src.len() != h * w * cn
        || dst.len() != src.len()
    {
        return None;
    }
    let ed = |s: &[T], d: &mut [T], erode: bool| {
        erode_dilate(s, d, dims, ksize, iterations, erode, border)
    };
    let mut tmp = vec![src[0]; src.len()];
    match op {
        Op::Erode => ed(src, dst, true),
        Op::Dilate => ed(src, dst, false),
        Op::Open => {
            ed(src, &mut tmp, true);
            ed(&tmp, dst, false);
        }
        Op::Close => {
            ed(src, &mut tmp, false);
            ed(&tmp, dst, true);
        }
        Op::Gradient => {
            ed(src, &mut tmp, false);
            ed(src, dst, true);
            dst.par_iter_mut()
                .zip(tmp.par_iter())
                .for_each(|(d, &t)| *d = t.sub(*d));
        }
        Op::TopHat => {
            ed(src, &mut tmp, true);
            ed(&tmp, dst, false);
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, &s)| *d = s.sub(*d));
        }
        Op::BlackHat => {
            ed(src, &mut tmp, false);
            ed(&tmp, dst, true);
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, &s)| *d = d.sub(s));
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erode_takes_the_window_minimum() {
        let src: Vec<u8> = vec![5, 9, 1, 7, 3, 8];
        let mut dst = vec![0u8; 6];
        morphology(
            &src,
            &mut dst,
            (1, 6, 1),
            Op::Erode,
            (3, 1),
            1,
            Border::Constant,
        )
        .unwrap();
        assert_eq!(dst, vec![5, 1, 1, 1, 3, 3]);
        morphology(
            &src,
            &mut dst,
            (1, 6, 1),
            Op::Dilate,
            (3, 1),
            1,
            Border::Constant,
        )
        .unwrap();
        assert_eq!(dst, vec![9, 9, 9, 7, 8, 8]);
    }

    #[test]
    fn blackhat_of_a_flat_image_is_zero() {
        let src = vec![0.5f32; 7 * 9];
        let mut dst = vec![1f32; 7 * 9];
        morphology(
            &src,
            &mut dst,
            (7, 9, 1),
            Op::BlackHat,
            (3, 3),
            2,
            Border::Constant,
        )
        .unwrap();
        assert!(dst.iter().all(|&v| v == 0.0));
    }
}
