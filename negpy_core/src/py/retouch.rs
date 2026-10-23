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

fn pixels(a: &PyReadwriteArrayDyn<'_, f32>) -> PyResult<usize> {
    match a.shape()[..] {
        [h, w, 3] => Ok(h * w),
        _ => Err(PyValueError::new_err("needs an HxWx3 array")),
    }
}

/// `_fill_rung(num, den, fill, first, reject_floor_mass)` in place, with (score floor, tau).
#[pyfunction]
fn fill_rung(
    py: Python<'_>,
    num: PyReadonlyArrayDyn<'_, f32>,
    den: PyReadonlyArrayDyn<'_, f32>,
    mut fill: PyReadwriteArrayDyn<'_, f32>,
    first: bool,
    reject_floor_mass: bool,
    constants: (f32, f32),
) -> PyResult<()> {
    let n = pixels(&fill)?;
    let (num, den) = (plane(&num, 3 * n)?, plane(&den, n)?);
    let fill = fill
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    py.detach(|| retouch::fill_rung(num, den, fill, first, reject_floor_mass, constants));
    Ok(())
}

/// `_blend_fill(src, score, fill, a_out)` in place, with (write ramp top, ramp span).
#[pyfunction]
fn blend_fill(
    py: Python<'_>,
    src: PyReadonlyArrayDyn<'_, f32>,
    score: PyReadonlyArrayDyn<'_, f32>,
    mut fill: PyReadwriteArrayDyn<'_, f32>,
    mut a_out: PyReadwriteArrayDyn<'_, f32>,
    ramp: (f32, f32),
) -> PyResult<()> {
    let n = pixels(&fill)?;
    let (src, score) = (plane(&src, 3 * n)?, plane(&score, n)?);
    let fill = fill
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    let a_out = a_out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    if a_out.len() != n {
        return Err(PyValueError::new_err("array sizes disagree"));
    }
    py.detach(|| retouch::blend_fill(src, score, fill, a_out, ramp));
    Ok(())
}

/// `_add_floor_deficit(out, a, blur_src, blur_out)` in place.
#[pyfunction]
fn add_floor_deficit(
    py: Python<'_>,
    mut out: PyReadwriteArrayDyn<'_, f32>,
    a: PyReadonlyArrayDyn<'_, f32>,
    blur_src: PyReadonlyArrayDyn<'_, f32>,
    blur_out: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<()> {
    let n = pixels(&out)?;
    let (a, bs, bo) = (
        plane(&a, n)?,
        plane(&blur_src, 3 * n)?,
        plane(&blur_out, 3 * n)?,
    );
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
    py.detach(|| retouch::add_floor_deficit(out, a, bs, bo));
    Ok(())
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
    py.detach(|| crate::retouch::hair_encode(src, cn, lo, inv_span, inv_gamma, dst));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hair_encode, m)?)?;
    m.add_function(wrap_pyfunction!(fill_rung, m)?)?;
    m.add_function(wrap_pyfunction!(blend_fill, m)?)?;
    m.add_function(wrap_pyfunction!(add_floor_deficit, m)?)
}
