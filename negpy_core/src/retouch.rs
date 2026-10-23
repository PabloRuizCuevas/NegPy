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

/// `_hair_encode`: `clip((crop - lo) * inv_span, 0, 1) ** inv_gamma` to 8-bit for the first
/// three channels, in float32 (no fastmath), truncated as `np.uint8` truncates.
pub fn hair_encode(
    crop: &[f32],
    cn: usize,
    lo: f32,
    inv_span: f32,
    inv_gamma: f32,
    out: &mut [u8],
) {
    out.par_chunks_mut(cn * 1024)
        .zip(crop.par_chunks(cn * 1024))
        .for_each(|(o, c)| hair_encode_block(c, o, cn, (lo, inv_span, inv_gamma)));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn hair_encode_block(
    crop: &[f32],
    out: &mut [u8],
    cn: usize,
    (lo, inv_span, inv_gamma): (f32, f32, f32),
) {
    let encode = |v: f32| {
        let v = ((v - lo) * inv_span).clamp(0.0, 1.0);
        let p = crate::mathf::pow_f64(v as f64, inv_gamma as f64) as f32;
        (p * 255.0 + 0.5) as u8
    };
    if cn == 3 {
        for (o, &v) in out.iter_mut().zip(crop) {
            *o = encode(v);
        }
        return;
    }
    for (o, c) in out.chunks_mut(cn).zip(crop.chunks(cn)) {
        for k in 0..3 {
            o[k] = encode(c[k]);
        }
    }
}

/// The scalars of `_ir_clean_base`, float32 as NumPy rounds them.
pub struct CleanBase {
    pub win: usize,
    pub identity: f32,
    pub sigma: f32,
    pub min_support: f32,
}

/// `_ir_clean_base(img_det, ratio)` of an HxWx3 image: the clean-film mean minus `sigma` of its
/// spread where enough of the window is clean, else the box mean of the window maximum.
pub fn ir_clean_base(
    img: &[f32],
    ratio: &[f32],
    (h, w): (usize, usize),
    c: &CleanBase,
    dst: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter};
    use crate::morph::{self, Op};
    let n = h * w;
    if ratio.len() != n || img.len() != 3 * n || dst.len() != img.len() {
        return None;
    }
    let win = (c.win, c.win);
    let identity = c.identity;
    let clean: Vec<f32> = ratio
        .par_iter()
        .map(move |&r| if r >= identity { 1.0 } else { 0.0 })
        .collect();
    let (mut p1, mut p2) = (vec![0f32; 3 * n], vec![0f32; 3 * n]);
    p1.par_chunks_mut(3 * BLOCK)
        .zip(p2.par_chunks_mut(3 * BLOCK))
        .zip(img.par_chunks(3 * BLOCK).zip(clean.par_chunks(BLOCK)))
        .for_each(|((p1, p2), (px, wc))| {
            for (i, &k) in wc.iter().enumerate() {
                for j in 3 * i..3 * i + 3 {
                    let v = px[j];
                    p1[j] = v * k;
                    p2[j] = v * v * k;
                }
            }
        });
    let border = Border::Reflect101;
    let mut den = vec![0f32; n];
    box_filter(&clean, &mut den, (h, w, 1), win, true, border)?;
    let (mut b1, mut b2) = (vec![0f32; 3 * n], vec![0f32; 3 * n]);
    box_filter(&p1, &mut b1, (h, w, 3), win, true, border)?;
    box_filter(&p2, &mut b2, (h, w, 3), win, true, border)?;
    // p1 holds the dilated image, then its box mean.
    morph::morphology(
        img,
        &mut p1,
        (h, w, 3),
        Op::Dilate,
        win,
        1,
        Border::Constant,
    )?;
    box_filter(&p1, &mut p2, (h, w, 3), win, true, border)?;
    let (sigma, support) = (c.sigma, c.min_support);
    dst.par_chunks_mut(3 * BLOCK)
        .zip(den.par_chunks(BLOCK))
        .zip(b1.par_chunks(3 * BLOCK).zip(b2.par_chunks(3 * BLOCK)))
        .zip(p2.par_chunks(3 * BLOCK))
        .for_each(|(((o, d), (b1, b2)), dil)| clean_base_block(o, d, b1, b2, dil, sigma, support));
    Some(())
}

#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn clean_base_block(
    out: &mut [f32],
    den: &[f32],
    b1: &[f32],
    b2: &[f32],
    dil: &[f32],
    sigma: f32,
    support: f32,
) {
    for (i, &d) in den.iter().enumerate() {
        let d = py_max(d, 1e-6);
        for j in 3 * i..3 * i + 3 {
            let mean = b1[j] / d;
            let var = b2[j] / d - mean * mean;
            let var = if var.is_nan() || var > 0.0 { var } else { 0.0 };
            let base = mean - sigma * var.sqrt();
            out[j] = if d > support { base } else { dil[j] };
        }
    }
}

/// `_ir_live(plane)`: 1 where film lies under the head, 0 in the below-`floor` regions of at
/// least `min_area` of the frame.
pub fn ir_live(plane: &[f32], (h, w): (usize, usize), floor: f32, min_area: f64) -> Vec<bool> {
    let dead: Vec<u8> = plane.par_iter().map(move |&v| (v < floor) as u8).collect();
    if !dead.par_iter().any(|&d| d != 0) {
        return vec![true; h * w];
    }
    let cc = crate::components::connected_components(&dead, h, w, 8);
    let bar = min_area * (h * w) as f64;
    let holder: Vec<bool> = (0..cc.count)
        .map(|l| l > 0 && cc.stats[l][4] as f64 >= bar)
        .collect();
    cc.labels.par_iter().map(|&l| !holder[l as usize]).collect()
}

/// `_fit_sample(mask)`: the values under `mask`, every `step`-th one so at most `cap` remain.
fn fit_sample(values: &[f32], mask: &[bool], cap: usize) -> Vec<f32> {
    let n = mask.par_iter().filter(|&&m| m).count();
    let step = n.div_ceil(cap).max(1);
    values
        .iter()
        .zip(mask)
        .filter(|&(_, &m)| m)
        .step_by(step)
        .map(|(&v, _)| v)
        .collect()
}

/// The median of `sample` and 1.4826 times its median absolute deviation, as Python floats.
fn median_and_sigma(sample: &[f32]) -> (f64, f64) {
    let med = crate::stats::median(sample);
    let dev: Vec<f32> = sample.par_iter().map(move |&v| (v - med).abs()).collect();
    (med as f64, 1.4826 * crate::stats::median(&dev) as f64)
}

/// `_ir_glare(plane, live)`: 1 more than `sigmas` deviations over the live median, `None` when
/// fewer than 500 samples or no pixel qualifies.
pub fn ir_glare(plane: &[f32], live: &[bool], cap: usize, sigmas: f64) -> Option<Vec<bool>> {
    let sample = fit_sample(plane, live, cap);
    if sample.len() < 500 {
        return None;
    }
    let (med, sigma) = median_and_sigma(&sample);
    let bar = (med + sigmas * sigma) as f32;
    let glare: Vec<bool> = plane.par_iter().map(move |&v| v > bar).collect();
    glare.par_iter().any(|&g| g).then_some(glare)
}

/// `normalize_ir(plane, glare)`: the plane over the box mean of its `win` x `win` maximum,
/// glare pixels zeroed for the base.
pub fn normalize_ir(
    plane: &[f32],
    glare: Option<&[bool]>,
    (h, w): (usize, usize),
    win: usize,
    dst: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter};
    use crate::morph::{self, Op};
    let n = h * w;
    if plane.len() != n || dst.len() != n || glare.is_some_and(|g| g.len() != n) {
        return None;
    }
    let src: std::borrow::Cow<[f32]> = match glare {
        Some(g) => plane
            .par_iter()
            .zip(g)
            .map(|(&v, &g)| if g { 0.0 } else { v })
            .collect::<Vec<f32>>()
            .into(),
        None => plane.into(),
    };
    let mut dil = vec![0f32; n];
    morph::morphology(
        &src,
        &mut dil,
        (h, w, 1),
        Op::Dilate,
        (win, win),
        1,
        Border::Constant,
    )?;
    box_filter(&dil, dst, (h, w, 1), (win, win), true, Border::Reflect101)?;
    crate::par_zip(dst, plane, move |d, &p| *d = p / py_max(*d, 1e-4));
    Some(())
}

/// The scalars of `_ir_normalize_ratio`, as Python floats.
pub struct RatioNorm {
    pub identity: f64,
    pub noise_sigma: f64,
    pub pivot_lo: f64,
    pub ref_sigma: f64,
    pub scale_max: f64,
}

/// `np.clip(v, lo, hi)` of Python floats: a NaN comes through.
fn clip_f64(v: f64, lo: f64, hi: f64) -> f64 {
    let m = if v.is_nan() || v > lo { v } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

/// `_ir_normalize_ratio(ratio, live)` in place; returns the (pivot, sigma) it measured, or
/// `None` when fewer than 500 live pixels leave the ratio alone.
pub fn ir_normalize_ratio(ratio: &mut [f32], live: &[bool], c: &RatioNorm) -> Option<(f64, f64)> {
    let sample: Vec<f32> = ratio
        .par_iter()
        .zip(live)
        .filter(|&(_, &l)| l)
        .map(|(&v, _)| v)
        .collect();
    if sample.len() < 500 {
        return None;
    }
    let (med, sigma) = median_and_sigma(&sample);
    let pivot = clip_f64(med - c.noise_sigma * sigma, c.pivot_lo, c.identity);
    let scale = c.identity / pivot;
    let spread = sigma * scale;
    let k = clip_f64(
        c.ref_sigma / if 1e-6 > spread { 1e-6 } else { spread },
        1.0,
        c.scale_max,
    );
    let (s32, k32, id32) = (scale as f32, k as f32, c.identity as f32);
    let stretch = k > 1.0;
    ratio.par_chunks_mut(BLOCK).for_each(move |run| {
        for v in run {
            let r = *v * s32;
            *v = if stretch && r < id32 {
                id32 - (id32 - r) * k32
            } else {
                r
            };
        }
    });
    Some((pivot, sigma))
}

/// `np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5)`: `_ir_decontaminate`'s division,
/// in float32 (lstsq keeps float32).
pub fn ir_ghost_divide(ratio: &[f32], vis_log: &[f32], b: [f32; 3], dst: &mut [f32]) {
    dst.par_chunks_mut(BLOCK)
        .zip(ratio.par_chunks(BLOCK).zip(vis_log.par_chunks(3 * BLOCK)))
        .for_each(move |(o, (r, v))| ghost_divide_block(o, r, v, b));
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn ghost_divide_block(out: &mut [f32], ratio: &[f32], vis_log: &[f32], b: [f32; 3]) {
    for ((o, &r), v) in out.iter_mut().zip(ratio).zip(vis_log.as_chunks::<3>().0) {
        let s = ((-0.0 + v[0] * b[0]) + v[1] * b[1]) + v[2] * b[2];
        let q = r / crate::mathf::np_expf(s);
        let m = if q.is_nan() || q > 0.0 { q } else { 0.0 };
        *o = if m.is_nan() || m < 1.5 { m } else { 1.5 };
    }
}

/// `score_weighted_fill(img, score, scales, reject_floor_mass)` of an HxWx3 image: box means
/// of the score-weighted image over the score at every support but the last, which is a
/// Gaussian, each rung blended over the coarser ones by `fill_rung`. `reject_floor_mass` counts
/// a rung's confidence above the score floor: without the original-floor rule, a support that
/// sees only defect would otherwise win with the defect as its candidate.
pub fn score_weighted_fill(
    img: &[f32],
    score: &[f32],
    (h, w): (usize, usize),
    scales: &[usize],
    reject_floor_mass: bool,
    constants: (f32, f32),
    fill: &mut [f32],
) -> Option<()> {
    use crate::filter::{Border, box_filter, gaussian_blur};
    let n = h * w;
    if score.len() != n || img.len() != 3 * n || fill.len() != img.len() {
        return None;
    }
    let mut weighted = vec![0f32; 3 * n];
    weighted
        .par_chunks_mut(3 * BLOCK)
        .zip(img.par_chunks(3 * BLOCK).zip(score.par_chunks(BLOCK)))
        .for_each(|(o, (px, s))| {
            for (i, &k) in s.iter().enumerate() {
                for j in 3 * i..3 * i + 3 {
                    o[j] = px[j] * k;
                }
            }
        });
    let (mut num, mut den) = (vec![0f32; 3 * n], vec![0f32; n]);
    let border = Border::Reflect101;
    for (i, &k) in scales.iter().enumerate() {
        if i == scales.len() - 1 {
            gaussian_blur(&weighted, &mut num, (h, w, 3), (k, k), (0.0, 0.0), border)?;
            gaussian_blur(score, &mut den, (h, w, 1), (k, k), (0.0, 0.0), border)?;
        } else {
            box_filter(&weighted, &mut num, (h, w, 3), (k, k), true, border)?;
            box_filter(score, &mut den, (h, w, 1), (k, k), true, border)?;
        }
        fill_rung(&num, &den, fill, i == 0, reject_floor_mass, constants);
    }
    Some(())
}

/// `_borrow_clean_grain` added into `out` at every pixel `clean` leaves out: the detail of the
/// pixel mirrored through the nearest clean one (that clean pixel itself when the mirror is
/// not clean), high-passed at `sigma`, times `a` when given. Returns the blur of `src`, or
/// `None` without touching `out` when no pixel or every pixel is clean.
pub fn add_clean_grain(
    src: &[f32],
    clean: &[bool],
    (h, w): (usize, usize),
    sigma: f64,
    a: Option<&[f32]>,
    out: &mut [f32],
) -> Option<Vec<f32>> {
    if !clean.par_iter().any(|&c| c) || clean.par_iter().all(|&c| c) {
        return None;
    }
    let holes: Vec<u8> = clean.par_iter().map(|&c| !c as u8).collect();
    let (_, labels) = crate::distance::distance_transform_labels(&holes, h, w);
    // The labels number the clean pixels from 1 in raster order.
    let donors: Vec<usize> = (0..h * w).filter(|&p| clean[p]).collect();
    let mut blur = vec![0f32; 3 * h * w];
    crate::filter::gaussian_blur(
        src,
        &mut blur,
        (h, w, 3),
        (0, 0),
        (sigma, sigma),
        crate::filter::Border::Reflect101,
    )?;
    let (hi, wi) = (h as isize - 1, w as isize - 1);
    out.par_chunks_mut(3 * w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let p = y * w + x;
            if clean[p] {
                continue;
            }
            let nearest = donors[labels[p] as usize - 1];
            let (ny, nx) = ((nearest / w) as isize, (nearest % w) as isize);
            let my = (2 * ny - y as isize).clamp(0, hi) as usize;
            let mx = (2 * nx - x as isize).clamp(0, wi) as usize;
            let mirrored = my * w + mx;
            let take = if clean[mirrored] { mirrored } else { nearest };
            for c in 0..3 {
                let g = src[3 * take + c] - blur[3 * take + c];
                row[3 * x + c] += match a {
                    Some(a) => a[p] * g,
                    None => g,
                };
            }
        }
    });
    Some(blur)
}

/// The scalars of `apply_score_repair`, float32 as NumPy rounds them.
pub struct ScoreRepair {
    pub score_floor: f32,
    pub tau: f32,
    pub write_hi: f32,
    /// `np.float32(write_hi - write_lo)`, subtracted as Python floats.
    pub write_span: f32,
}

/// `apply_score_repair` of an HxWx3 image and a score at its resolution, with the fill
/// supports and grain `sigma` already chosen.
#[allow(clippy::too_many_arguments)]
pub fn apply_score_repair(
    src: &[f32],
    score: &[f32],
    (h, w): (usize, usize),
    supports: &[usize],
    floor: bool,
    sigma: f64,
    c: &ScoreRepair,
    out: &mut [f32],
) -> Option<()> {
    let n = h * w;
    score_weighted_fill(
        src,
        score,
        (h, w),
        supports,
        !floor,
        (c.score_floor, c.tau),
        out,
    )?;
    let mut a = vec![0f32; n];
    blend_fill(src, score, out, &mut a, (c.write_hi, c.write_span));
    let hi = c.write_hi;
    let clean: Vec<bool> = score.par_iter().map(move |&s| s >= hi).collect();
    let blur_src = add_clean_grain(src, &clean, (h, w), sigma, Some(&a), out);
    if floor {
        let gauss = |img: &[f32]| {
            let mut b = vec![0f32; 3 * n];
            crate::filter::gaussian_blur(
                img,
                &mut b,
                (h, w, 3),
                (0, 0),
                (sigma, sigma),
                crate::filter::Border::Reflect101,
            )
            .map(|_| b)
        };
        let blur_src = match blur_src {
            Some(b) => b,
            None => gauss(src)?,
        };
        let blur_out = gauss(out)?;
        add_floor_deficit(out, &a, &blur_src, &blur_out);
    } else {
        out.par_chunks_mut(BLOCK).for_each(|run| {
            for v in run {
                *v = if v.is_nan() || *v > 0.0 { *v } else { 0.0 };
            }
        });
    }
    Some(())
}
