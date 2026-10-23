//! Ports of the preview and capture metering in negpy/services/capture, negpy/services/scanning
//! and the scan preview widgets.

use rayon::prelude::*;

use crate::stats::{np_percentiles, pairwise_sum_f32};

/// How `stretch_preview` maps a channel between its 1st and 99th percentiles.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stretch {
    /// `preview_positive` of a positive: `clip((v - lo) / (hi - lo), 0, 1) * 255`.
    Positive,
    /// `preview_positive` of a negative: `clip((hi - v) / (hi - lo), 0, 1) * 255`.
    Negative,
    /// `_preview_u8`: `clip((v - lo) / (hi - lo) * 255, 0, 255)`.
    Prescan,
}

/// The scan preview stretch of an HxWxC float32 image into `out_cn` uint8 channels, output channel
/// c reading input channel min(c, C - 1). A channel whose 99th percentile is not above its 1st is 0.
pub fn stretch_preview(img: &[f32], cn: usize, out_cn: usize, mode: Stretch, dst: &mut [u8]) {
    let bounds: Vec<Option<(f32, f32)>> = (0..cn)
        .into_par_iter()
        .map(|c| {
            let ch: Vec<f32> = img.iter().skip(c).step_by(cn).copied().collect();
            let p = np_percentiles(&ch, &[1.0, 99.0])?;
            (p[1] > p[0]).then_some((p[0], p[1]))
        })
        .collect();
    dst.par_chunks_mut(out_cn * 4096)
        .zip(img.par_chunks(cn * 4096))
        .for_each(|(d, s)| {
            for (o, p) in d.chunks_exact_mut(out_cn).zip(s.chunks_exact(cn)) {
                for (c, v) in o.iter_mut().enumerate() {
                    let src = c.min(cn - 1);
                    let Some((lo, hi)) = bounds[src] else {
                        *v = 0;
                        continue;
                    };
                    let x = p[src];
                    // NumPy's float32 to uint8 cast truncates.
                    *v = match mode {
                        Stretch::Positive => ((x - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0,
                        Stretch::Negative => ((hi - x) / (hi - lo)).clamp(0.0, 1.0) * 255.0,
                        Stretch::Prescan => ((x - lo) / (hi - lo) * 255.0).clamp(0.0, 255.0),
                    } as u8;
                }
            }
        });
}

/// `sharpness`: the float32 mean square of the 4-neighbour Laplacian over the squared float32
/// mean level, or 0 below `min_mean`.
pub fn sharpness(img: &[f32], h: usize, w: usize, min_mean: f64) -> f64 {
    let mean = (pairwise_sum_f32(img) / img.len() as f32) as f64;
    if mean < min_mean {
        return 0.0;
    }
    let mut sq = vec![0f32; (h - 2) * (w - 2)];
    sq.par_chunks_mut(w - 2).enumerate().for_each(|(i, row)| {
        let y = i + 1;
        let at = |yy: usize, x: usize| img[yy * w + x];
        for (j, d) in row.iter_mut().enumerate() {
            let x = j + 1;
            let lap = 4.0 * at(y, x) - at(y - 1, x) - at(y + 1, x) - at(y, x - 1) - at(y, x + 1);
            *d = lap * lap;
        }
    });
    (pairwise_sum_f32(&sq) / sq.len() as f32) as f64 / (mean * mean)
}

/// The share of pixels of an image with `cn` channels where any of the first three is at least `level`.
pub fn any_channel_at_least(img: &[f32], cn: usize, level: f32) -> f64 {
    let n = img.len() / cn;
    let hits: usize = img
        .par_chunks(cn * 4096)
        .map(|s| {
            s.chunks_exact(cn)
                .filter(|p| p[..cn.min(3)].iter().any(|&v| v >= level))
                .count()
        })
        .sum();
    hits as f64 / n as f64
}

/// `_to_grey`: `(r + g + b + 1) // 3` per pixel, summed in uint32.
pub fn to_grey<T: Copy + Send + Sync + Into<u32> + TryFrom<u32>>(
    rgb: &[T],
    cn: usize,
    dst: &mut [T],
) {
    dst.par_chunks_mut(4096)
        .zip(rgb.par_chunks(cn * 4096))
        .for_each(|(d, s)| {
            for (o, p) in d.iter_mut().zip(s.chunks_exact(cn)) {
                let sum: u32 = p.iter().map(|&v| v.into()).sum();
                // The average of the channels fits the type.
                *o = T::try_from((sum + 1) / 3).unwrap_or(p[0]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_frame_has_no_sharpness() {
        let img = vec![10f32; 25];
        assert_eq!(sharpness(&img, 5, 5, 4.0), 0.0);
        assert_eq!(sharpness(&[1f32; 25], 5, 5, 4.0), 0.0);
    }

    #[test]
    fn grey_rounds_up_from_two_thirds() {
        let mut out = [0u16; 2];
        to_grey(&[1u16, 1, 0, 2, 2, 1], 3, &mut out);
        assert_eq!(out, [1, 2]);
    }
}
