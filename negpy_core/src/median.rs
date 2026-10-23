//! `cv2.medianBlur` for single-channel uint8: the median of a `k` x `k` window, edges
//! replicated as OpenCV does. A median of integers is exact, so only the border rule matters.

use rayon::prelude::*;

fn bump<const N: usize>(hist: &mut [u16; N], bin: usize, up: bool) {
    hist[bin] = if up { hist[bin] + 1 } else { hist[bin] - 1 };
}

/// Perreault and Hebert's constant-time median: column histograms slid down the rows, a 16-bin
/// coarse window histogram slid along each row, and each 16-bin fine segment brought up to date
/// only when the search enters it.
pub fn median_blur_u8(src: &[u8], dst: &mut [u8], h: usize, w: usize, k: usize) -> Option<()> {
    if h == 0
        || w == 0
        || k == 0
        || k.is_multiple_of(2)
        || src.len() != h * w
        || dst.len() != src.len()
    {
        return None;
    }
    if k == 1 {
        dst.copy_from_slice(src);
        return Some(());
    }
    let r = (k / 2) as isize;
    let rank = (k * k / 2) as u32;
    let cy = |v: isize| v.clamp(0, h as isize - 1) as usize;
    let cx = |v: isize| v.clamp(0, w as isize - 1) as usize;
    const BAND: usize = 256;
    dst.par_chunks_mut(BAND * w)
        .enumerate()
        .for_each(|(bi, band)| {
            let y0 = bi * BAND;
            // Column histograms over rows y0 - r ..= y0 + r: fine (256 bins) and coarse (16).
            let mut fine_col = vec![[0u16; 256]; w];
            let mut coarse_col = vec![[0u16; 16]; w];
            for dy in -r..=r {
                for (x, &v) in src[cy(y0 as isize + dy) * w..][..w].iter().enumerate() {
                    bump(&mut fine_col[x], v as usize, true);
                    bump(&mut coarse_col[x], (v >> 4) as usize, true);
                }
            }
            for (dy, out) in band.chunks_mut(w).enumerate() {
                let y = (y0 + dy) as isize;
                if dy > 0 {
                    let gone = &src[cy(y - r - 1) * w..][..w];
                    let new = &src[cy(y + r) * w..][..w];
                    for x in 0..w {
                        bump(&mut fine_col[x], gone[x] as usize, false);
                        bump(&mut coarse_col[x], (gone[x] >> 4) as usize, false);
                        bump(&mut fine_col[x], new[x] as usize, true);
                        bump(&mut coarse_col[x], (new[x] >> 4) as usize, true);
                    }
                }
                let mut coarse = [0u32; 16];
                for dx in -r..=r {
                    for (c, &v) in coarse.iter_mut().zip(&coarse_col[cx(dx)]) {
                        *c += v as u32;
                    }
                }
                let mut fine = [[0u32; 16]; 16];
                // The output column each fine segment was last summed for.
                let mut synced = [isize::MIN; 16];
                for (x, o) in out.iter_mut().enumerate() {
                    let xi = x as isize;
                    let mut seen = 0u32;
                    let mut b = 0;
                    while seen + coarse[b] <= rank {
                        seen += coarse[b];
                        b += 1;
                    }
                    let seg = &mut fine[b];
                    let lo = b << 4;
                    if synced[b] < xi - 2 * r {
                        *seg = [0; 16];
                        for dx in -r..=r {
                            for (f, &v) in seg.iter_mut().zip(&fine_col[cx(xi + dx)][lo..lo + 16]) {
                                *f += v as u32;
                            }
                        }
                    } else {
                        for j in synced[b] + 1..=xi {
                            let (gone, new) = (&fine_col[cx(j - r - 1)], &fine_col[cx(j + r)]);
                            for i in 0..16 {
                                seg[i] = (seg[i] + new[lo + i] as u32) - gone[lo + i] as u32;
                            }
                        }
                    }
                    synced[b] = xi;
                    let mut v = 0;
                    while seen + seg[v] <= rank {
                        seen += seg[v];
                        v += 1;
                    }
                    *o = (lo + v) as u8;
                    if x + 1 < w {
                        let (gone, new) = (&coarse_col[cx(xi - r)], &coarse_col[cx(xi + r + 1)]);
                        for i in 0..16 {
                            coarse[i] = (coarse[i] + new[i] as u32) - gone[i] as u32;
                        }
                    }
                }
            }
        });
    Some(())
}
