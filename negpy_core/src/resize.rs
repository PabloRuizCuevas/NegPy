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

/// `cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA)` for a float32 downscale. `src` is
/// row-major with `cn` channels interleaved. None for an upscale or a 2x2 two-channel image.
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
        return None;
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
    fn upscale_is_not_covered() {
        let src = vec![0f32; 40 * 40];
        assert!(resize_area(&src, 40, 40, 1, 20, 20).is_some());
        assert!(resize_area(&src, 40, 40, 1, 80, 30).is_none());
        assert!(resize_area(&src, 40, 40, 1, 20, 30).is_some());
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
