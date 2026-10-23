use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::capture::{self, Stretch};
use crate::stats::{np_percentiles, np_percentiles_f64};

fn contiguous<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

fn dims(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match shape {
        [h, w] => Ok((*h, *w, 1)),
        [h, w, c] if *c > 0 => Ok((*h, *w, *c)),
        _ => Err(PyValueError::new_err("needs an HxW or HxWxC array")),
    }
}

/// The scan preview stretch of a float32 image: `mode` "positive", "negative" (`preview_positive`)
/// or "prescan" (`_preview_u8`, always three channels out).
#[pyfunction]
fn stretch_preview<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    mode: &str,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let (h, w, cn) = dims(img.shape())?;
    let mode = match mode {
        "positive" => Stretch::Positive,
        "negative" => Stretch::Negative,
        "prescan" => Stretch::Prescan,
        _ => {
            return Err(PyValueError::new_err(
                "mode must be positive, negative or prescan",
            ));
        }
    };
    let out_cn = if mode == Stretch::Prescan { 3 } else { cn };
    let src = contiguous(&img)?;
    let out = PyArrayDyn::<u8>::zeros(py, vec![h, w, out_cn], false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, src.len(), || {
            capture::stretch_preview(src, cn, out_cn, mode, dst)
        });
    }
    Ok(out)
}

/// `sharpness` of an HxW float32 frame.
#[pyfunction]
fn sharpness(py: Python<'_>, gray: PyReadonlyArrayDyn<'_, f32>, min_mean: f64) -> PyResult<f64> {
    let [h, w] = gray.shape()[..] else {
        return Err(PyValueError::new_err("sharpness needs a 2-D array"));
    };
    if h < 3 || w < 3 {
        return Ok(0.0);
    }
    let src = contiguous(&gray)?;
    Ok(super::detach(py, src.len(), || {
        capture::sharpness(src, h, w, min_mean)
    }))
}

/// The share of pixels of an HxWxC float32 image where any of the first three channels is at least `level`.
#[pyfunction]
fn any_channel_at_least(
    py: Python<'_>,
    img: PyReadonlyArrayDyn<'_, f32>,
    level: f32,
) -> PyResult<f64> {
    let (_, _, cn) = dims(img.shape())?;
    let src = contiguous(&img)?;
    if src.is_empty() {
        return Ok(f64::NAN);
    }
    Ok(super::detach(py, src.len(), || {
        capture::any_channel_at_least(src, cn, level)
    }))
}

/// `(np.mean(patch >= level), np.percentile(patch, q))` of a uint16, float32 or float64 patch, the
/// percentile in float32 for float32 and in float64 otherwise, as NumPy computes them.
#[pyfunction]
fn patch_level_stats(
    py: Python<'_>,
    patch: &Bound<'_, PyAny>,
    level: f64,
    q: f64,
) -> PyResult<(f64, f64)> {
    fn stats(v: &[f32], level: f32, q: f64, wide: bool) -> (f64, f64) {
        let hits = v.iter().filter(|&&x| x >= level).count();
        let p = if wide {
            np_percentiles_f64(v, &[q])[0]
        } else {
            np_percentiles(v, &[q]).map_or(f64::NAN, |p| p[0] as f64)
        };
        (hits as f64 / v.len() as f64, p)
    }
    if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<u16>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            let v: Vec<f32> = src.iter().map(|&x| x as f32).collect();
            stats(&v, level as f32, q, true)
        }))
    } else if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<f32>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            stats(src, level as f32, q, false)
        }))
    } else if let Ok(a) = patch.extract::<PyReadonlyArrayDyn<f64>>() {
        let src = contiguous(&a)?;
        if src.is_empty() {
            return Ok((0.0, 0.0));
        }
        Ok(super::detach(py, src.len(), || {
            let hits = src.iter().filter(|&&x| x >= level).count();
            let p = if src.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                crate::stats::quantile_f64(&mut src.to_vec(), q / 100.0).unwrap_or(f64::NAN)
            };
            (hits as f64 / src.len() as f64, p)
        }))
    } else {
        Err(PyTypeError::new_err(
            "patch_level_stats needs uint16, float32 or float64",
        ))
    }
}

/// `_to_grey` of a uint8 or uint16 HxWxC image.
#[pyfunction]
fn to_grey<'py>(py: Python<'py>, rgb: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    fn run<'py, T: numpy::Element + Copy + Send + Sync + Into<u32> + TryFrom<u32>>(
        py: Python<'py>,
        rgb: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a: PyReadonlyArrayDyn<'py, T> = rgb.extract()?;
        let (h, w, cn) = dims(a.shape())?;
        let src = contiguous(&a)?;
        let out = PyArrayDyn::<T>::zeros(py, vec![h, w], false);
        {
            let mut rw = out.readwrite();
            let dst = rw
                .as_slice_mut()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            super::detach(py, src.len(), || capture::to_grey(src, cn, dst));
        }
        Ok(out.into_any())
    }
    if rgb.extract::<PyReadonlyArrayDyn<u8>>().is_ok() {
        run::<u8>(py, rgb)
    } else if rgb.extract::<PyReadonlyArrayDyn<u16>>().is_ok() {
        run::<u16>(py, rgb)
    } else {
        Err(PyTypeError::new_err("to_grey needs uint8 or uint16"))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stretch_preview, m)?)?;
    m.add_function(wrap_pyfunction!(sharpness, m)?)?;
    m.add_function(wrap_pyfunction!(any_channel_at_least, m)?)?;
    m.add_function(wrap_pyfunction!(patch_level_stats, m)?)?;
    m.add_function(wrap_pyfunction!(to_grey, m)?)
}
