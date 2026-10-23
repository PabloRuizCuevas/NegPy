//! The panorama gain compensation of negpy/features/stitch/logic.py.

/// `gain_compensate`'s per-channel gains: the float64 mean of `reference` over the overlap
/// over that of `moving`, rows added in order. `None` under `min_overlap` pixels or for a
/// mean under 1e-6, where the part stays as it is.
pub fn compensation_gains(
    reference: &[f32],
    ref_mask: &[bool],
    moving: &[f32],
    mov_mask: &[bool],
    min_overlap: usize,
) -> Option<[f32; 3]> {
    let (mut a, mut b, mut n) = ([0f64; 3], [0f64; 3], 0usize);
    for (p, (&r, &m)) in ref_mask.iter().zip(mov_mask).enumerate() {
        if r && m {
            for c in 0..3 {
                a[c] += reference[3 * p + c] as f64;
                b[c] += moving[3 * p + c] as f64;
            }
            n += 1;
        }
    }
    if n < min_overlap {
        return None;
    }
    let (a, b) = (a.map(|v| v / n as f64), b.map(|v| v / n as f64));
    if b.iter().any(|&v| v < 1e-6) {
        return None;
    }
    Some([0, 1, 2].map(|c| (a[c] / b[c]) as f32))
}
