use multiversion::multiversion;
use rayon::prelude::*;

/// One source-to-destination contribution of OpenCV's area table (`DecimateAlpha`).
#[derive(Clone, Copy)]
struct Tap {
    di: usize,
    si: usize,
    alpha: f32,
}

/// OpenCV's `computeResizeAreaTab`, in its f64 arithmetic.
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<Tap> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(Tap {
                di: dx,
                si: (sx1 - 1) as usize,
                alpha: ((sx1 as f64 - fsx1) / cell) as f32,
            });
        }
        for sx in sx1..sx2 {
            tab.push(Tap {
                di: dx,
                si: sx as usize,
                alpha: (1.0 / cell) as f32,
            });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(Tap {
                di: dx,
                si: sx2 as usize,
                alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32,
            });
        }
    }
    tab
}

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA)` of a float32 image. `src` is
/// row-major with `cn` channels interleaved. An axis that grows makes it OpenCV's bilinear
/// variant. None for a 2x2 two-channel image.
pub fn resize_area(
    src: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    dh: usize,
    dw: usize,
) -> Option<Vec<f32>> {
    if h == 0 || w == 0 || dh == 0 || dw == 0 || cn == 0 || src.len() != h * w * cn {
        return None;
    }
    // OpenCV computes inv_scale = dsize / ssize, then scale = 1 / inv_scale.
    let scale_x = 1.0 / (dw as f64 / w as f64);
    let scale_y = 1.0 / (dh as f64 / h as f64);
    let int_x = (scale_x - scale_x.round()).abs() < f64::EPSILON;
    let int_y = (scale_y - scale_y.round()).abs() < f64::EPSILON;
    if scale_x < 1.0 || scale_y < 1.0 {
        let mut dst = vec![0f32; dh * dw * cn];
        resize_linear(src, (h, w, cn), (dh, dw), true, &mut dst)?;
        return Some(dst);
    }
    if dh == h && dw == w {
        return Some(src.to_vec());
    }
    if int_x && int_y {
        return resize_area_fast(
            src,
            w,
            cn,
            dh,
            dw,
            scale_x.round() as usize,
            scale_y.round() as usize,
        );
    }
    let xtab = area_tab(w, dw, scale_x);
    let ytab = area_tab(h, dh, scale_y);
    let dwidth = dw * cn;
    let mut dst = vec![0f32; dh * dwidth];

    // Each destination pixel's taps are consecutive in its table, so rows (and pixels) are
    // independent and one accumulator per pixel keeps OpenCV's order: ((0 + s0*a0) + s1*a1) ...
    let xspans = spans(&xtab, dw);
    let yspans = spans(&ytab, dh);

    dst.par_chunks_mut(dwidth)
        .enumerate()
        .for_each(|(dy, out)| {
            let mut first = true;
            for t in &ytab[yspans[dy]..yspans[dy + 1]] {
                let row = &src[t.si * w * cn..(t.si + 1) * w * cn];
                match cn {
                    1 => accumulate::<1>(row, &xtab, &xspans, t.alpha, first, out),
                    3 => accumulate::<3>(row, &xtab, &xspans, t.alpha, first, out),
                    _ => accumulate_any(row, cn, &xtab, &xspans, t.alpha, first, out),
                }
                first = false;
            }
        });
    Some(dst)
}

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST)`: each output pixel copies the
/// source pixel at floor(d * ssize / dsize), as OpenCV's `resizeNN` rounds it.
pub fn resize_nearest<T: Copy + Send + Sync>(
    src: &[T],
    (h, w, cn): (usize, usize, usize),
    (dh, dw): (usize, usize),
    dst: &mut [T],
) {
    let ofs = |s: usize, d: usize, i: usize| {
        ((i as f64 * (1.0 / (d as f64 / s as f64))).floor() as usize).min(s - 1)
    };
    // u32 offsets keep a wide row's table in L1.
    let xs: Vec<u32> = (0..dw).map(|x| (ofs(w, dw, x) * cn) as u32).collect();
    let dwidth = dw * cn;
    const ROWS: usize = 16;
    dst.par_chunks_mut(ROWS * dwidth)
        .enumerate()
        .for_each(|(block, out)| {
            let mut prev = usize::MAX;
            for i in 0..out.len() / dwidth {
                let sy = ofs(h, dh, block * ROWS + i);
                let (done, rest) = out.split_at_mut(i * dwidth);
                let o = &mut rest[..dwidth];
                // An upscale repeats a source row: copy the output row it gave.
                if sy == prev {
                    o.copy_from_slice(&done[(i - 1) * dwidth..]);
                    continue;
                }
                prev = sy;
                let row = &src[sy * w * cn..][..w * cn];
                match cn {
                    1 => nearest_row::<T, 1>(row, &xs, o),
                    3 => nearest_row::<T, 3>(row, &xs, o),
                    4 => nearest_row::<T, 4>(row, &xs, o),
                    _ => {
                        for (o, &x) in o.chunks_exact_mut(cn).zip(&xs) {
                            o.copy_from_slice(&row[x as usize..][..cn]);
                        }
                    }
                }
            }
        });
}

fn nearest_row<T: Copy, const CN: usize>(row: &[T], xs: &[u32], out: &mut [T]) {
    let (pixels, _) = row.as_chunks::<CN>();
    let last = pixels.len() - 1;
    for (o, &x) in out.as_chunks_mut::<CN>().0.iter_mut().zip(xs) {
        *o = pixels[(x as usize / CN).min(last)];
    }
}

/// One axis of OpenCV's bilinear resize (`hal::resize`): per destination index, the first
/// source index and the float32 weight of the next one. `area` takes INTER_AREA's upscale
/// fractions. Without `clamp` (the y axis) a fraction outside the source stays and both rows
/// clip to the edge.
fn linear_axis(ssize: usize, dsize: usize, area: bool, clamp: bool) -> Vec<(i64, f32)> {
    let inv = dsize as f64 / ssize as f64;
    let scale = 1.0 / inv;
    (0..dsize)
        .map(|d| {
            let (mut s, mut f) = if area {
                let s = (d as f64 * scale).floor() as i64;
                let f = ((d + 1) as f64 - (s + 1) as f64 * inv) as f32;
                (s, if f <= 0.0 { 0.0 } else { f - f.floor() })
            } else {
                let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
                let s = f.floor();
                (s as i64, f - s)
            };
            if clamp && s < 0 {
                (s, f) = (0, 0.0);
            }
            if clamp && s >= ssize as i64 - 1 {
                (s, f) = (ssize as i64 - 1, 0.0);
            }
            (s, f)
        })
        .collect()
}

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)` of a float32 image, or with
/// `area` INTER_AREA where either axis grows, as OpenCV runs it without IPP: a row pass then a
/// column pass, each `a * (1 - f) + b * f` in float32. An exact halving is INTER_AREA's mean.
pub fn resize_linear(
    src: &[f32],
    (h, w, cn): (usize, usize, usize),
    (dh, dw): (usize, usize),
    area: bool,
    dst: &mut [f32],
) -> Option<()> {
    if h == 0 || w == 0 || dh == 0 || dw == 0 || cn == 0 || src.len() != h * w * cn {
        return None;
    }
    if dst.len() != dh * dw * cn {
        return None;
    }
    if !area && dw * 2 == w && dh * 2 == h {
        dst.copy_from_slice(&resize_area(src, h, w, cn, dh, dw)?);
        return Some(());
    }
    let xs = linear_axis(w, dw, area, true);
    let ys = linear_axis(h, dh, area, false);
    // From xmax on, OpenCV reads one source pixel (weight 1) instead of two.
    let xmax = xs
        .iter()
        .position(|&(s, _)| s + 1 >= w as i64)
        .unwrap_or(dw);
    let xs: Vec<(usize, f32, f32)> = xs.iter().map(|&(s, f)| (s as usize, 1.0 - f, f)).collect();
    let row_pass = |sy: usize, out: &mut [f32]| {
        let row = &src[sy * w * cn..][..w * cn];
        match cn {
            1 => linear_row::<1>(row, &xs, xmax, out),
            3 => linear_row::<3>(row, &xs, xmax, out),
            4 => linear_row::<4>(row, &xs, xmax, out),
            _ => linear_row_any(row, cn, &xs, xmax, out),
        }
    };
    let dwidth = dw * cn;
    let clip = |v: i64| v.clamp(0, h as i64 - 1) as usize;
    // A block of output rows reuses each row pass; a few blocks per thread keep the
    // repeated passes at block starts rare.
    let block = dh.div_ceil(4 * rayon::current_num_threads()).max(16);
    dst.par_chunks_mut(block * dwidth)
        .zip(ys.par_chunks(block))
        .for_each(|(out, ys)| {
            let mut cache = [
                (usize::MAX, vec![0f32; dwidth]),
                (usize::MAX, vec![0f32; dwidth]),
            ];
            for (o, &(sy, fy)) in out.chunks_exact_mut(dwidth).zip(ys) {
                let (r0, r1) = (clip(sy), clip(sy + 1));
                if cache[0].0 != r0 && cache[1].0 == r0 {
                    cache.swap(0, 1);
                }
                let (c0, c1) = cache.split_at_mut(1);
                let (c0, c1) = (&mut c0[0], &mut c1[0]);
                if c0.0 != r0 {
                    c0.0 = r0;
                    row_pass(r0, &mut c0.1);
                }
                if c1.0 != r1 {
                    c1.0 = r1;
                    if r1 == r0 {
                        c1.1.copy_from_slice(&c0.1);
                    } else {
                        row_pass(r1, &mut c1.1);
                    }
                }
                blend_rows(&c0.1, &c1.1, (1.0 - fy, fy), o);
            }
        });
    Some(())
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn blend_rows(r0: &[f32], r1: &[f32], (b0, b1): (f32, f32), out: &mut [f32]) {
    for ((o, &a), &b) in out.iter_mut().zip(r0).zip(r1) {
        *o = a * b0 + b * b1;
    }
}

/// One row of the bilinear row pass; from `xmax` on, OpenCV weights one source pixel.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn linear_row<const CN: usize>(
    row: &[f32],
    xs: &[(usize, f32, f32)],
    xmax: usize,
    out: &mut [f32],
) {
    let last = row.len() - CN;
    let (two, one) = out.split_at_mut(xmax * CN);
    for (o, &(sx, a0, a1)) in two.as_chunks_mut::<CN>().0.iter_mut().zip(&xs[..xmax]) {
        let i = (sx * CN).min(last - CN);
        let (p, q) = (&row[i..][..CN], &row[i + CN..][..CN]);
        for k in 0..CN {
            o[k] = p[k] * a0 + q[k] * a1;
        }
    }
    for (o, &(sx, a0, _)) in one.as_chunks_mut::<CN>().0.iter_mut().zip(&xs[xmax..]) {
        let p = &row[(sx * CN).min(last)..][..CN];
        for k in 0..CN {
            o[k] = p[k] * a0;
        }
    }
}

fn linear_row_any(row: &[f32], cn: usize, xs: &[(usize, f32, f32)], xmax: usize, out: &mut [f32]) {
    for (dx, (o, &(sx, a0, a1))) in out.chunks_exact_mut(cn).zip(xs).enumerate() {
        let p = &row[sx * cn..];
        for (k, o) in o.iter_mut().enumerate() {
            *o = if dx < xmax {
                p[k] * a0 + p[cn + k] * a1
            } else {
                p[k] * a0
            };
        }
    }
}

/// OpenCV's `resizeAreaFast_` for an integer scale. A 2x2 one- or four-channel image takes its
/// SIMD order, ((a + b) + (c + d)) * 0.25, over the columns the 4-lane vector loop reaches; the
/// rest sums each cell row by row, four at a time: sum + (((a + b) + c) + d).
fn resize_area_fast(
    src: &[f32],
    w: usize,
    cn: usize,
    dh: usize,
    dw: usize,
    sx: usize,
    sy: usize,
) -> Option<Vec<f32>> {
    let simd_cols = match (sx, sy, cn) {
        (2, 2, 1) => dw / 4 * 4,
        (2, 2, 4) => dw,
        (2, 2, 2) => return None,
        _ => 0,
    };
    let area = sx * sy;
    let scale = 1.0f32 / area as f32;
    let row_len = w * cn;
    let ofs: Vec<usize> = (0..sy)
        .flat_map(|y| (0..sx).map(move |x| y * row_len + x * cn))
        .collect();
    let mut dst = vec![0f32; dh * dw * cn];
    dst.par_chunks_mut(dw * cn)
        .enumerate()
        .for_each(|(dy, out)| {
            let base = dy * sy * row_len;
            for dx in 0..dw {
                for c in 0..cn {
                    let s0 = base + dx * sx * cn + c;
                    out[dx * cn + c] = if dx < simd_cols {
                        let (a, b) = (src[s0], src[s0 + cn]);
                        let (p, q) = (src[s0 + row_len], src[s0 + row_len + cn]);
                        ((a + b) + (p + q)) * 0.25
                    } else {
                        let mut sum = 0f32;
                        let mut k = 0;
                        while k + 4 <= area {
                            sum += ((src[s0 + ofs[k]] + src[s0 + ofs[k + 1]])
                                + src[s0 + ofs[k + 2]])
                                + src[s0 + ofs[k + 3]];
                            k += 4;
                        }
                        for o in &ofs[k..] {
                            sum += src[s0 + o];
                        }
                        sum * scale
                    };
                }
            }
        });
    Some(dst)
}

/// Start index in `tab` of each destination index's taps, plus the end.
fn spans(tab: &[Tap], dsize: usize) -> Vec<usize> {
    let mut starts = vec![tab.len(); dsize + 1];
    for (j, t) in tab.iter().enumerate().rev() {
        starts[t.di] = j;
    }
    starts
}

/// One source row's horizontal pass, weighted by `beta` into `out` (OpenCV's buf and sum).
fn accumulate<const CN: usize>(
    row: &[f32],
    xtab: &[Tap],
    xspans: &[usize],
    beta: f32,
    first: bool,
    out: &mut [f32],
) {
    for (dx, px) in out.as_chunks_mut::<CN>().0.iter_mut().enumerate() {
        let mut acc = [0f32; CN];
        for tap in &xtab[xspans[dx]..xspans[dx + 1]] {
            let s = &row[tap.si * CN..tap.si * CN + CN];
            for c in 0..CN {
                acc[c] += s[c] * tap.alpha;
            }
        }
        for c in 0..CN {
            px[c] = if first {
                beta * acc[c]
            } else {
                px[c] + beta * acc[c]
            };
        }
    }
}

fn accumulate_any(
    row: &[f32],
    cn: usize,
    xtab: &[Tap],
    xspans: &[usize],
    beta: f32,
    first: bool,
    out: &mut [f32],
) {
    for (dx, px) in out.chunks_exact_mut(cn).enumerate() {
        for (c, p) in px.iter_mut().enumerate() {
            let mut acc = 0f32;
            for tap in &xtab[xspans[dx]..xspans[dx + 1]] {
                acc += row[tap.si * cn + c] * tap.alpha;
            }
            *p = if first { beta * acc } else { *p + beta * acc };
        }
    }
}

/// The kernels of `cv2.INTER_CUBIC` and `cv2.INTER_LANCZOS4`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kernel {
    Cubic,
    Lanczos4,
}

impl Kernel {
    fn taps(self) -> usize {
        match self {
            Kernel::Cubic => 4,
            Kernel::Lanczos4 => 8,
        }
    }

    /// OpenCV's `interpolateCubic` and `interpolateLanczos4` in their float arithmetic.
    fn weights(self, x: f32) -> [f32; 8] {
        let mut c = [0f32; 8];
        match self {
            Kernel::Cubic => {
                let a = -0.75f32;
                c[0] = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
                c[1] = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
                c[2] = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
                c[3] = 1.0 - c[0] - c[1] - c[2];
            }
            Kernel::Lanczos4 => {
                let s45 = std::f64::consts::FRAC_1_SQRT_2;
                let cs = [
                    (1.0, 0.0),
                    (-s45, -s45),
                    (0.0, 1.0),
                    (s45, -s45),
                    (-1.0, 0.0),
                    (s45, s45),
                    (0.0, -1.0),
                    (-s45, s45),
                ];
                let y0 = -(x as f64 + 3.0) * std::f64::consts::PI * 0.25;
                let (s0, c0) = (libm::sin(y0), libm::cos(y0));
                let mut sum = 0f32;
                for (i, (k, &(a, b))) in c.iter_mut().zip(&cs).enumerate() {
                    let t = x + 3.0 - i as f32;
                    *k = if t.abs() >= 1e-6 {
                        let y = -t as f64 * std::f64::consts::PI * 0.25;
                        ((a * s0 + b * c0) / (y * y)) as f32
                    } else {
                        1e30
                    };
                    sum += *k;
                }
                let inv = 1.0 / sum;
                for k in &mut c {
                    *k *= inv;
                }
            }
        }
        c
    }
}

/// Per output position along one axis: the first source tap (before clamping) and its weights.
fn kernel_axis(src: usize, dst: usize, kernel: Kernel) -> Vec<(i64, [f32; 8])> {
    let scale = 1.0 / (dst as f64 / src as f64);
    let back = (kernel.taps() / 2 - 1) as i64;
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            (s as i64 - back, kernel.weights(f - s))
        })
        .collect()
}

/// One output row of the vertical pass: per element the float32 sum of its taps in order.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2",
    "aarch64+neon"
))]
fn column_pass<const N: usize>(out: &mut [f32], rows: [&[f32]; N], wt: [f32; N]) {
    for (i, d) in out.iter_mut().enumerate() {
        let mut v = rows[0][i] * wt[0];
        for k in 1..N {
            v += rows[k][i] * wt[k];
        }
        *d = v;
    }
}

/// One row of the horizontal pass for `C` channels and `N` taps: each output the float32 sum of
/// its taps in order.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2",
    "aarch64+neon"
))]
fn row_pass<const C: usize, const N: usize>(
    s: &[f32],
    taps: &[usize],
    wts: &[f32],
    out: &mut [f32],
) {
    let px = s.as_chunks::<C>().0;
    let (taps, wts) = (taps.as_chunks::<N>().0, wts.as_chunks::<N>().0);
    for ((o, t), wt) in out.as_chunks_mut::<C>().0.iter_mut().zip(taps).zip(wts) {
        let p = px[t[0]];
        let mut acc: [f32; C] = std::array::from_fn(|c| p[c] * wt[0]);
        for k in 1..N {
            let p = px[t[k]];
            for c in 0..C {
                acc[c] += p[c] * wt[k];
            }
        }
        *o = acc;
    }
}

fn row_pass_any(s: &[f32], taps: &[usize], wts: &[f32], n: usize, cn: usize, out: &mut [f32]) {
    for ((o, t), wt) in out
        .chunks_exact_mut(cn)
        .zip(taps.chunks_exact(n))
        .zip(wts.chunks_exact(n))
    {
        o.fill(0.0);
        for (&sx, &wk) in t.iter().zip(wt) {
            for (d, &v) in o.iter_mut().zip(&s[sx * cn..][..cn]) {
                *d += v * wk;
            }
        }
    }
}

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_CUBIC or cv2.INTER_LANCZOS4)` of a float32
/// image: OpenCV's taps and weights, edges replicated, each pass summed in float32 tap by tap.
pub fn resize_kernel(
    src: &[f32],
    dims: (usize, usize, usize),
    dsize: (usize, usize),
    kernel: Kernel,
    dst: &mut [f32],
) -> Option<()> {
    resize_kernel_as(src, dims, dsize, kernel, dst, |v| v, |v| v)
}

/// The kernel resize, reading `S` and writing `D` through float32 inside one parallel pass.
fn resize_kernel_as<S: Copy + Sync, D: Copy + Send>(
    src: &[S],
    (h, w, cn): (usize, usize, usize),
    (dh, dw): (usize, usize),
    kernel: Kernel,
    dst: &mut [D],
    load: impl Fn(S) -> f32 + Sync,
    store: impl Fn(f32) -> D + Sync,
) -> Option<()> {
    if h == 0
        || w == 0
        || dh == 0
        || dw == 0
        || cn == 0
        || src.len() != h * w * cn
        || dst.len() != dh * dw * cn
    {
        return None;
    }
    let n = kernel.taps();
    let (xs, ys) = (kernel_axis(w, dw, kernel), kernel_axis(h, dh, kernel));
    let clamp = |i: i64, len: usize| i.clamp(0, len as i64 - 1) as usize;
    // Source column of every tap, clamped once: replicated edges.
    let taps: Vec<usize> = xs
        .iter()
        .flat_map(|&(x0, _)| (0..n).map(move |k| clamp(x0 + k as i64, w)))
        .collect();
    let wts: Vec<f32> = xs
        .iter()
        .flat_map(|(_, wt)| wt[..n].iter().copied())
        .collect();
    let hrow = |y: usize, line: &mut Vec<f32>, out: &mut [f32]| {
        line.clear();
        line.extend(src[y * w * cn..][..w * cn].iter().map(|&v| load(v)));
        let s = line.as_slice();
        match (cn, n) {
            (1, 4) => row_pass::<1, 4>(s, &taps, &wts, out),
            (3, 4) => row_pass::<3, 4>(s, &taps, &wts, out),
            (4, 4) => row_pass::<4, 4>(s, &taps, &wts, out),
            (1, 8) => row_pass::<1, 8>(s, &taps, &wts, out),
            (3, 8) => row_pass::<3, 8>(s, &taps, &wts, out),
            (4, 8) => row_pass::<4, 8>(s, &taps, &wts, out),
            _ => row_pass_any(s, &taps, &wts, n, cn, out),
        }
    };
    // Bands of output rows; each filters the source rows it reads into its own buffer, so a
    // band's halo is filtered twice: bands as wide as the thread count allows.
    let band = (dh / (4 * rayon::current_num_threads())).clamp(16, 128);
    let row_len = dw * cn;
    dst.par_chunks_mut(band * row_len)
        .zip(ys.par_chunks(band))
        .for_each(|(band, ys)| {
            let lo = clamp(ys[0].0, h);
            let hi = clamp(ys[ys.len() - 1].0 + n as i64 - 1, h);
            let mut rows = vec![0f32; (hi - lo + 1) * row_len];
            let mut line = Vec::with_capacity(w * cn);
            for (i, r) in rows.chunks_mut(row_len).enumerate() {
                hrow(lo + i, &mut line, r);
            }
            let mut out = vec![0f32; row_len];
            for (o, (y0, wt)) in band.chunks_mut(row_len).zip(ys) {
                let r = |k: usize| &rows[(clamp(y0 + k as i64, h) - lo) * row_len..][..row_len];
                if n == 4 {
                    column_pass(
                        &mut out,
                        [r(0), r(1), r(2), r(3)],
                        [wt[0], wt[1], wt[2], wt[3]],
                    );
                } else {
                    column_pass(&mut out, std::array::from_fn::<_, 8, _>(r), *wt);
                }
                for (d, &v) in o.iter_mut().zip(&out) {
                    *d = store(v);
                }
            }
        });
    Some(())
}

/// `expand_mask_plane` times `scale`: a ph x pw plane resized bilinearly onto rows y1..y2 and
/// columns x1..x2 of an oh x ow plane, the edges replicated outside, each value times `scale`
/// in float32.
pub fn expand_plane(
    plane: &[f32],
    (ph, pw): (usize, usize),
    (oh, ow): (usize, usize),
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    scale: f32,
    dst: &mut [f32],
) -> Option<()> {
    let (ih, iw) = (y2.checked_sub(y1)?, x2.checked_sub(x1)?);
    if ih == 0 || iw == 0 || y2 > oh || x2 > ow || dst.len() != oh * ow {
        return None;
    }
    let mut inner = vec![0f32; ih * iw];
    resize_linear(plane, (ph, pw, 1), (ih, iw), false, &mut inner)?;
    dst.par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
        let src = &inner[(y.clamp(y1, y2 - 1) - y1) * iw..][..iw];
        for (x, d) in row.iter_mut().enumerate() {
            *d = src[x.clamp(x1, x2 - 1) - x1] * scale;
        }
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_stays_flat() {
        let src = vec![0.25f32; 37 * 53 * 3];
        let out = resize_area(&src, 37, 53, 3, 11, 17).unwrap();
        assert!(out.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn upscale_of_a_flat_image_stays_flat() {
        let src = vec![0.5f32; 40 * 40];
        let out = resize_area(&src, 40, 40, 1, 80, 30).unwrap();
        assert!(out.iter().all(|&v| v == 0.5));
        let mut lin = vec![0f32; 90 * 70 * 3];
        resize_linear(
            &vec![0.5f32; 40 * 40 * 3],
            (40, 40, 3),
            (90, 70),
            false,
            &mut lin,
        )
        .unwrap();
        assert!(lin.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn taps_cover_each_destination_with_unit_weight() {
        let tab = area_tab(1600, 1059, 1600.0 / 1059.0);
        let mut total = vec![0f64; 1059];
        for t in &tab {
            total[t.di] += t.alpha as f64;
        }
        // OpenCV drops a fragment under 1e-3 of a source pixel, so a cell can fall short by that much.
        assert!(total.iter().all(|&s| (s - 1.0).abs() < 2e-3));
    }
}
