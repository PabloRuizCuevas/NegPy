//! OpenCV's uint8 grayscale operations, all integer arithmetic: BGR to gray, the fixed-point
//! GaussianBlur, Otsu's threshold and Canny.

use rayon::prelude::*;

use crate::filter::{Border, border_interpolate, gaussian_kernel_f64};

/// `cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)`: 15-bit fixed-point weights, rounded.
pub fn bgr_to_gray(src: &[u8], dst: &mut [u8]) {
    dst.par_iter_mut()
        .zip(src.par_chunks_exact(3))
        .for_each(|(d, p)| {
            *d = ((p[0] as u32 * 3735 + p[1] as u32 * 19235 + p[2] as u32 * 9798 + (1 << 14)) >> 15)
                as u8;
        });
}

/// `getGaussianKernelFixedPoint_ED`: the bit-exact kernel in 8 fractional bits, rounding errors
/// carried along the half kernel and the centre taking what makes the sum exactly 256.
pub fn gaussian_kernel_u8(n: usize, sigma: f64) -> Vec<u32> {
    let k = gaussian_kernel_f64(n, sigma);
    let mut out = vec![0u32; n];
    let mut err = 0.0f64;
    let mut sum = 0i64;
    for i in 0..n / 2 {
        let adj = k[i] * 256.0 + err;
        let v = adj.round_ties_even() as i64;
        err = adj - v as f64;
        out[i] = v as u32;
        out[n - 1 - i] = v as u32;
        sum += v;
    }
    out[n / 2] = (256 - 2 * sum) as u32;
    out
}

/// `cv2.GaussianBlur` for uint8 (`GaussianBlurFixedPoint`): integer rows, then integer columns,
/// rounded once from 16 fractional bits.
pub fn gaussian_blur_u8(
    src: &[u8],
    dst: &mut [u8],
    (h, w, cn): (usize, usize, usize),
    ksize: (usize, usize),
    (sigma_x, sigma_y): (f64, f64),
    border: Border,
) -> Option<()> {
    if h == 0 || w == 0 || cn == 0 || src.len() != h * w * cn || dst.len() != src.len() {
        return None;
    }
    let (mut kw, mut kh) = ksize;
    let sigma_y = if sigma_y <= 0.0 { sigma_x } else { sigma_y };
    let size_of = |s: f64| ((s * 3.0 * 2.0 + 1.0).round_ties_even() as i64 | 1) as usize;
    if kw == 0 && sigma_x > 0.0 {
        kw = size_of(sigma_x);
    }
    if kh == 0 && sigma_y > 0.0 {
        kh = size_of(sigma_y);
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
    let kx = gaussian_kernel_u8(kw, sigma_x.max(0.0));
    let ky = if kh == kw && (sigma_x - sigma_y).abs() < f64::EPSILON {
        kx.clone()
    } else {
        gaussian_kernel_u8(kh, sigma_y.max(0.0))
    };
    let (rx, ry) = (kw / 2, kh / 2);
    let n = w * cn;
    let xmap: Vec<Option<usize>> = (0..w + 2 * rx)
        .map(|px| border_interpolate(px as isize - rx as isize, w, border))
        .collect();

    let mut rows = vec![0u32; h * n];
    rows.par_chunks_mut(n).enumerate().for_each(|(y, out)| {
        let row = &src[y * n..(y + 1) * n];
        let px = |p: usize, c: usize| xmap[p].map_or(0, |sx| row[sx * cn + c] as u32);
        for x in 0..w {
            for c in 0..cn {
                out[x * cn + c] = (0..kw).map(|j| kx[j] * px(x + j, c)).sum();
            }
        }
    });
    dst.par_chunks_mut(n).enumerate().for_each(|(y, out)| {
        let lines: Vec<Option<&[u32]>> = (0..kh)
            .map(|j| {
                border_interpolate(y as isize + j as isize - ry as isize, h, border)
                    .map(|sy| &rows[sy * n..(sy + 1) * n])
            })
            .collect();
        for (i, o) in out.iter_mut().enumerate() {
            let v: u32 = lines
                .iter()
                .zip(&ky)
                .map(|(l, &k)| l.map_or(0, |l| k * l[i]))
                .sum();
            *o = ((v + (1 << 15)) >> 16).min(255) as u8;
        }
    });
    Some(())
}

/// Otsu's threshold for uint8 (`getThreshVal_Otsu`), in its double arithmetic and first-max rule.
pub fn otsu_threshold(src: &[u8]) -> f64 {
    let mut hist = [0u64; 256];
    for &v in src {
        hist[v as usize] += 1;
    }
    let scale = 1.0 / src.len() as f64;
    let mut mu = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        mu += i as f64 * c as f64;
    }
    mu *= scale;
    let (mut mu1, mut q1, mut max_sigma, mut max_val) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let eps = f32::EPSILON as f64;
    for (i, &c) in hist.iter().enumerate() {
        let p = c as f64 * scale;
        mu1 *= q1;
        q1 += p;
        let q2 = 1.0 - q1;
        if q1.min(q2) < eps || q1.max(q2) > 1.0 - eps {
            continue;
        }
        mu1 = (mu1 + i as f64 * p) / q1;
        let mu2 = (mu - q1 * mu1) / q2;
        let sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if sigma > max_sigma {
            max_sigma = sigma;
            max_val = i as f64;
        }
    }
    max_val
}

/// `cv2.threshold` with THRESH_BINARY (`invert` false) or THRESH_BINARY_INV on uint8.
pub fn threshold_binary(src: &[u8], dst: &mut [u8], thresh: f64, maxval: u8, invert: bool) {
    // OpenCV compares against cvFloor(thresh) for uint8.
    let t = thresh.floor().clamp(-1.0, 255.0) as i32;
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, &s)| {
        *d = if (s as i32 > t) != invert { maxval } else { 0 };
    });
}

/// `cv2.Canny(src, low, high)` for a single-channel uint8 image, aperture 3, L1 gradient.
pub fn canny(src: &[u8], dst: &mut [u8], h: usize, w: usize, low: f64, high: f64) -> Option<()> {
    if h == 0 || w == 0 || src.len() != h * w || dst.len() != src.len() {
        return None;
    }
    let (low, high) = if low > high { (high, low) } else { (low, high) };
    let (low, high) = (low.floor() as i32, high.floor() as i32);
    let at = |y: isize, x: isize| {
        src[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize] as i32
    };

    // Sobel 3x3 in 16-bit integers with a replicated border, then |dx| + |dy|.
    let mut dx = vec![0i32; h * w];
    let mut dy = vec![0i32; h * w];
    dx.par_chunks_mut(w)
        .zip(dy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (rx, ry))| {
            let y = y as isize;
            for x in 0..w {
                let xi = x as isize;
                let col = |c: isize| at(y - 1, c) + 2 * at(y, c) + at(y + 1, c);
                let row = |r: isize| at(r, xi - 1) + 2 * at(r, xi) + at(r, xi + 1);
                rx[x] = col(xi + 1) - col(xi - 1);
                ry[x] = row(y + 1) - row(y - 1);
            }
        });
    let mag = |y: isize, x: isize| -> i32 {
        if y < 0 || y >= h as isize || x < 0 || x >= w as isize {
            0
        } else {
            let i = y as usize * w + x as usize;
            dx[i].abs() + dy[i].abs()
        }
    };

    // Map: 0 may be an edge, 1 cannot, 2 is one. A one-pixel frame of 1 surrounds the image.
    const TG22: i64 = 13573;
    let mw = w + 2;
    let mut map = vec![1u8; (h + 2) * mw];
    map.par_chunks_mut(mw)
        .enumerate()
        .skip(1)
        .take(h)
        .for_each(|(my, line)| {
            let y = my as isize - 1;
            for x in 0..w {
                let xi = x as isize;
                let m = mag(y, xi);
                let mut v = 1u8;
                if m > low {
                    let i = y as usize * w + x;
                    let (xs, ys) = (dx[i], dy[i]);
                    let ax = xs.abs() as i64;
                    let ay = (ys.abs() as i64) << 15;
                    let tg22x = ax * TG22;
                    let peak = if ay < tg22x {
                        m > mag(y, xi - 1) && m >= mag(y, xi + 1)
                    } else if ay > tg22x + (ax << 16) {
                        m > mag(y - 1, xi) && m >= mag(y + 1, xi)
                    } else {
                        let s: isize = if (xs ^ ys) < 0 { -1 } else { 1 };
                        m > mag(y - 1, xi - s) && m > mag(y + 1, xi + s)
                    };
                    if peak {
                        v = if m > high { 2 } else { 0 };
                    }
                }
                line[x + 1] = v;
            }
        });

    // Hysteresis: grow every edge pixel into 8-connected candidates.
    let mut stack: Vec<usize> = map
        .iter()
        .enumerate()
        .filter(|(_, v)| **v == 2)
        .map(|(i, _)| i)
        .collect();
    while let Some(i) = stack.pop() {
        for o in [
            i - mw - 1,
            i - mw,
            i - mw + 1,
            i - 1,
            i + 1,
            i + mw - 1,
            i + mw,
            i + mw + 1,
        ] {
            if map[o] == 0 {
                map[o] = 2;
                stack.push(o);
            }
        }
    }
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let line = &map[(y + 1) * mw + 1..(y + 1) * mw + 1 + w];
        for (o, &v) in out.iter_mut().zip(line) {
            *o = if v == 2 { 255 } else { 0 };
        }
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_kernels_sum_to_256() {
        for (n, s) in [(5, 0.0), (7, 0.0), (11, 0.0), (13, 2.0), (25, 4.0)] {
            assert_eq!(gaussian_kernel_u8(n, s).iter().sum::<u32>(), 256, "n={n}");
        }
        assert_eq!(gaussian_kernel_u8(5, 0.0), vec![16, 64, 96, 64, 16]);
    }

    #[test]
    fn otsu_splits_two_levels() {
        let mut img = vec![20u8; 500];
        img.extend(vec![200u8; 500]);
        let t = otsu_threshold(&img);
        assert!((20.0..200.0).contains(&t));
    }

    #[test]
    fn canny_finds_a_step() {
        let (h, w) = (20, 20);
        let src: Vec<u8> = (0..h * w)
            .map(|i| if i % w < 10 { 0 } else { 255 })
            .collect();
        let mut dst = vec![0u8; h * w];
        canny(&src, &mut dst, h, w, 40.0, 160.0).unwrap();
        assert!(dst.iter().any(|&v| v == 255));
        assert!((0..h).all(|y| dst[y * w] == 0));
    }
}
