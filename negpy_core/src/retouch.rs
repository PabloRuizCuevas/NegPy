//! The per-pixel passes of the retouch score fill (negpy/features/retouch/logic.py), in its
//! float32 operation order.

use multiversion::multiversion;
use rayon::prelude::*;

const BLOCK: usize = 1 << 13;

/// Python's `max(a, b)`, which Numba keeps: `a` unless `b` is greater.
fn py_max(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: `a` unless `b` is smaller.
fn py_min(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn fill_rung_block(
    num: &[f32],
    den: &[f32],
    fill: &mut [f32],
    first: bool,
    reject: bool,
    (floor, tau): (f32, f32),
) {
    let inv_span = 1.0 - floor;
    for ((f, n), &dv) in fill
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(num.as_chunks::<3>().0)
        .zip(den)
    {
        let d = py_max(dv, 1e-6);
        if first {
            for c in 0..3 {
                f[c] = n[c] / d;
            }
        } else {
            let mass = if reject { (dv - floor) / inv_span } else { dv };
            let conf = py_min(py_max(mass / tau, 0.0), 1.0);
            let keep = 1.0 - conf;
            for c in 0..3 {
                f[c] = f[c] * keep + (n[c] / d) * conf;
            }
        }
    }
}

/// `_fill_rung`: one rung of the fill ladder in place over HxW pixels of three channels.
pub fn fill_rung(
    num: &[f32],
    den: &[f32],
    fill: &mut [f32],
    first: bool,
    reject_floor_mass: bool,
    constants: (f32, f32),
) {
    fill.par_chunks_mut(3 * BLOCK)
        .zip(num.par_chunks(3 * BLOCK).zip(den.par_chunks(BLOCK)))
        .for_each(|(f, (n, d))| fill_rung_block(n, d, f, first, reject_floor_mass, constants));
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn blend_block(
    src: &[f32],
    score: &[f32],
    fill: &mut [f32],
    a_out: &mut [f32],
    (hi, span): (f32, f32),
) {
    for (((f, s), &sc), ao) in fill
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<3>().0)
        .zip(score)
        .zip(a_out.iter_mut())
    {
        let a = py_min(py_max((hi - sc) / span, 0.0), 1.0);
        let a = a * a * (3.0 - 2.0 * a);
        *ao = a;
        let keep = 1.0 - a;
        for c in 0..3 {
            f[c] = s[c] * keep + f[c] * a;
        }
    }
}

/// `_blend_fill`: `fill` becomes `src * (1 - a) + fill * a` under the smoothstep write ramp of
/// the score, the ramp kept in `a_out`.
pub fn blend_fill(
    src: &[f32],
    score: &[f32],
    fill: &mut [f32],
    a_out: &mut [f32],
    ramp: (f32, f32),
) {
    fill.par_chunks_mut(3 * BLOCK)
        .zip(a_out.par_chunks_mut(BLOCK))
        .zip(src.par_chunks(3 * BLOCK).zip(score.par_chunks(BLOCK)))
        .for_each(|((f, ao), (s, sc))| blend_block(s, sc, f, ao, ramp));
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn deficit_block(out: &mut [f32], a: &[f32], blur_src: &[f32], blur_out: &[f32]) {
    for (((o, &av), bs), bo) in out
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(a)
        .zip(blur_src.as_chunks::<3>().0)
        .zip(blur_out.as_chunks::<3>().0)
    {
        for c in 0..3 {
            let v = o[c] + av * py_max(bs[c] - bo[c], 0.0);
            o[c] = py_max(v, 0.0);
        }
    }
}

/// `_add_floor_deficit`: `out += a * max(blur_src - blur_out, 0)`, then clamped at zero.
pub fn add_floor_deficit(out: &mut [f32], a: &[f32], blur_src: &[f32], blur_out: &[f32]) {
    out.par_chunks_mut(3 * BLOCK)
        .zip(a.par_chunks(BLOCK))
        .zip(
            blur_src
                .par_chunks(3 * BLOCK)
                .zip(blur_out.par_chunks(3 * BLOCK)),
        )
        .for_each(|((o, av), (bs, bo))| deficit_block(o, av, bs, bo));
}
