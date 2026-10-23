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
        resize_linear(&vec![0.5f32; 40 * 40 * 3], (40, 40, 3), (90, 70), false, &mut lin).unwrap();
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
