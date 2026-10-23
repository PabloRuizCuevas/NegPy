use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::retouch;

fn plane<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>, n: usize) -> PyResult<&'a [f32]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    if s.len() != n {
        return Err(PyValueError::new_err("array sizes disagree"));
    }
    Ok(s)
}

/// `_hair_encode(crop, lo, inv_span, out_u8)` of an HxWxC float32 tile.
#[pyfunction]
fn hair_encode(
    py: Python<'_>,
    crop: numpy::PyReadonlyArrayDyn<'_, f32>,
    lo: f32,
    inv_span: f32,
    inv_gamma: f32,
    mut out: numpy::PyReadwriteArrayDyn<'_, u8>,
) -> PyResult<()> {
    use numpy::PyUntypedArrayMethods;
    let [_, _, cn] = crop.shape()[..] else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "needs an HxWxC tile",
        ));
    };
    if cn < 3 || out.shape() != crop.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "needs 3 channels and a matching out",
        ));
    }
    let src = crop
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("needs a contiguous tile"))?;
    let dst = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("needs a contiguous out"))?;
    super::detach(py, dst.len(), || {
        crate::retouch::hair_encode(src, cn, lo, inv_span, inv_gamma, dst)
    });
    Ok(())
}

/// `_ir_clean_base(img_det, ratio)` with (window, identity, sigma, min_support).
#[pyfunction]
fn ir_clean_base<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    constants: (usize, f32, f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    if ratio.shape() != [h, w] {
        return Err(PyValueError::new_err("needs an HxW ratio"));
    }
    let (src, ratio) = (plane(&img, 3 * h * w)?, plane(&ratio, h * w)?);
    let (win, identity, sigma, min_support) = constants;
    let c = retouch::CleanBase {
        win,
        identity,
        sigma,
        min_support,
    };
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            retouch::ir_clean_base(src, ratio, (h, w), &c, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid clean-base input"))?;
    }
    Ok(out)
}

fn plane2<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let [h, w] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW plane"));
    };
    Ok((plane(a, h * w)?, (h, w)))
}

fn mask<'a>(a: &'a PyReadonlyArrayDyn<'_, bool>, n: usize) -> PyResult<&'a [bool]> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    if s.len() != n {
        return Err(PyValueError::new_err("array sizes disagree"));
    }
    Ok(s)
}

fn bool_array<'py>(
    py: Python<'py>,
    data: Vec<bool>,
    (h, w): (usize, usize),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<bool>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    data.into_pyarray(py).reshape(vec![h, w])
}

/// `_ir_live(plane)` with (dead floor, minimum holder area as a fraction of the frame).
#[pyfunction]
fn ir_live<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    floor: f32,
    min_area: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<bool>>> {
    let (src, dims) = plane2(&plane)?;
    let live = super::detach(py, src.len(), || {
        retouch::ir_live(src, dims, floor, min_area)
    });
    bool_array(py, live, dims)
}

/// `_ir_glare(plane, live)` with (sample cap, sigmas).
#[pyfunction]
fn ir_glare<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    live: PyReadonlyArrayDyn<'py, bool>,
    cap: usize,
    sigmas: f64,
) -> PyResult<Option<Bound<'py, numpy::PyArrayDyn<bool>>>> {
    let (src, dims) = plane2(&plane)?;
    let live = mask(&live, src.len())?;
    super::detach(py, src.len(), || retouch::ir_glare(src, live, cap, sigmas))
        .map(|g| bool_array(py, g, dims))
        .transpose()
}

/// `normalize_ir(plane, glare)` for a float32 plane, with the base window in pixels.
#[pyfunction]
#[pyo3(signature = (plane, glare, win))]
fn normalize_ir<'py>(
    py: Python<'py>,
    plane: PyReadonlyArrayDyn<'py, f32>,
    glare: Option<PyReadonlyArrayDyn<'py, bool>>,
    win: usize,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let (src, dims) = plane2(&plane)?;
    let glare = glare.as_ref().map(|g| mask(g, src.len())).transpose()?;
    let out = PyArrayDyn::<f32>::zeros(py, plane.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || {
            retouch::normalize_ir(src, glare, dims, win, dst)
        })
        .ok_or_else(|| PyValueError::new_err("invalid normalize_ir input"))?;
    }
    Ok(out)
}

/// `_ir_normalize_ratio(ratio, live)` in place, with (identity, noise sigma, pivot floor,
/// reference sigma, scale cap); returns (pivot, sigma), or None when it left the ratio alone.
#[pyfunction]
fn ir_normalize_ratio(
    py: Python<'_>,
    mut ratio: PyReadwriteArrayDyn<'_, f32>,
    live: PyReadonlyArrayDyn<'_, bool>,
    constants: (f64, f64, f64, f64, f64),
) -> PyResult<Option<(f64, f64)>> {
    let r = ratio
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    let live = mask(&live, r.len())?;
    let (identity, noise_sigma, pivot_lo, ref_sigma, scale_max) = constants;
    let c = retouch::RatioNorm {
        identity,
        noise_sigma,
        pivot_lo,
        ref_sigma,
        scale_max,
    };
    Ok(super::detach(py, r.len(), || {
        retouch::ir_normalize_ratio(r, live, &c)
    }))
}

/// `_ir_decontaminate`'s division of an HxW ratio by the ghost of HxWx3 `vis_log`.
#[pyfunction]
fn ir_ghost_divide<'py>(
    py: Python<'py>,
    ratio: PyReadonlyArrayDyn<'py, f32>,
    vis_log: PyReadonlyArrayDyn<'py, f32>,
    b: [f32; 3],
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let (r, (h, w)) = plane2(&ratio)?;
    if vis_log.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("needs an HxWx3 vis_log"));
    }
    let v = plane(&vis_log, 3 * h * w)?;
    let out = PyArrayDyn::<f32>::zeros(py, ratio.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || retouch::ir_ghost_divide(r, v, b, dst));
    }
    Ok(out)
}

fn image3<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], (usize, usize))> {
    let [h, w, 3] = a.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    Ok((plane(a, 3 * h * w)?, (h, w)))
}

fn new_image<'py>(
    py: Python<'py>,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) -> Option<()> + Send,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{PyArrayDyn, PyArrayMethods};
    let out = PyArrayDyn::<f32>::zeros(py, shape, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst))
            .ok_or_else(|| PyValueError::new_err("invalid input"))?;
    }
    Ok(out)
}

/// `out[idx] += _borrow_clean_grain(src, clean, sigma, idx)[0]` for every pixel `clean` leaves
/// out, in place.
#[pyfunction]
fn add_clean_grain(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    src: PyReadonlyArrayDyn<'_, f32>,
    clean: PyReadonlyArrayDyn<'_, bool>,
    sigma: f64,
) -> PyResult<()> {
    let (s, dims) = image3(&src)?;
    let clean = mask(&clean, dims.0 * dims.1)?;
    if out.shape() != src.shape() {
        return Err(PyValueError::new_err("needs a matching out"));
    }
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    super::detach(py, s.len(), || {
        retouch::add_clean_grain(s, clean, dims, sigma, None, o)
    });
    Ok(())
}

/// `apply_score_repair` of an HxWx3 image and a score at its resolution, with the supports, the
/// floor rule, the grain sigma and (score floor, tau, write hi, write hi - write lo).
#[pyfunction]
fn apply_score_repair<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    score: PyReadonlyArrayDyn<'py, f32>,
    supports: Vec<usize>,
    floor: bool,
    sigma: f64,
    constants: (f32, f32, f32, f32),
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    let (src, dims) = image3(&img)?;
    let sc = plane(&score, dims.0 * dims.1)?;
    let (score_floor, tau, write_hi, write_span) = constants;
    let c = retouch::ScoreRepair {
        score_floor,
        tau,
        write_hi,
        write_span,
    };
    new_image(py, img.shape(), |dst| {
        retouch::apply_score_repair(src, sc, dims, &supports, floor, sigma, &c, dst)
    })
}

/// `repair_components` into `out`, which holds the base, with the supports, the floor rule,
/// the grain sigma, (score floor, tau, write hi, write hi - write lo), the crop pad and the
/// component cap; returns False, leaving `out`, when the score holds no defect.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (img, score, out, supports, floor, sigma, constants, pad, max_components, dirty=None))]
fn repair_components(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    score: PyReadonlyArrayDyn<'_, f32>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    supports: Vec<usize>,
    floor: bool,
    sigma: f64,
    constants: (f32, f32, f32, f32),
    pad: usize,
    max_components: usize,
    dirty: Option<PyReadonlyArrayDyn<'_, bool>>,
) -> PyResult<bool> {
    let (src, dims) = image3(&img)?;
    let n = dims.0 * dims.1;
    let sc = plane(&score, n)?;
    let dirty = dirty.as_ref().map(|d| mask(d, n)).transpose()?;
    if out.shape() != img.shape() {
        return Err(PyValueError::new_err("needs a matching out"));
    }
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    let (score_floor, tau, write_hi, write_span) = constants;
    let c = retouch::ScoreRepair {
        score_floor,
        tau,
        write_hi,
        write_span,
    };
    super::detach(py, src.len(), || {
        retouch::repair_components(
            src,
            sc,
            dims,
            &supports,
            floor,
            sigma,
            &c,
            pad,
            max_components,
            dirty,
            o,
        )
    })
    .ok_or_else(|| PyValueError::new_err("invalid repair input"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(repair_components, m)?)?;
    m.add_function(wrap_pyfunction!(add_clean_grain, m)?)?;
    m.add_function(wrap_pyfunction!(apply_score_repair, m)?)?;
    m.add_function(wrap_pyfunction!(ir_ghost_divide, m)?)?;
    m.add_function(wrap_pyfunction!(ir_live, m)?)?;
    m.add_function(wrap_pyfunction!(ir_glare, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ir, m)?)?;
    m.add_function(wrap_pyfunction!(ir_normalize_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(ir_clean_base, m)?)?;
    m.add_function(wrap_pyfunction!(hair_encode, m)?)
}
