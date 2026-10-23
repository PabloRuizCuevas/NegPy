use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{self, Level};

fn contiguous<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

fn empty<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), dtype))?
        .cast_into::<PyArrayDyn<T>>()?)
}

/// Fills a new array of `shape` and `dtype` from `fill(dst)` with the GIL released.
fn filled<'py, T: numpy::Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
    fill: impl FnOnce(&mut [T]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let out = empty::<T>(py, shape, dtype)?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, dst.len(), || fill(dst));
    }
    Ok(out)
}

/// `_to_uint8_jit` of a contiguous float32 array.
#[pyfunction]
fn to_uint8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "uint8", |d| convert::to_uint(src, d))
}

/// `_to_uint16_jit` of a contiguous float32 array.
#[pyfunction]
fn to_uint16<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "uint16", |d| convert::to_uint(src, d))
}

/// `uint8_to_float32` of a contiguous uint8 array.
#[pyfunction]
fn uint8_to_float32<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| {
        convert::uint8_to_float32(src, d)
    })
}

/// `uint16_to_float32` of a contiguous uint16 array.
#[pyfunction]
fn uint16_to_float32<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u16>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let src = contiguous(&img)?;
    filled(py, img.shape(), "float32", |d| {
        convert::uint16_to_float32(src, d)
    })
}

fn luma_to<'py, T: Level + numpy::Element>(
    py: Python<'py>,
    img: &PyReadonlyArrayDyn<'py, f32>,
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, c] if c >= 3 => (h, w, c),
        _ => {
            return Err(PyValueError::new_err("needs an HxW or HxWxC array, C >= 3"));
        }
    };
    let src = contiguous(img)?;
    filled(py, &[h, w], dtype, |d| convert::luma_to_uint(src, cn, d))
}

/// `float_to_uint_luma(img, bit_depth)` of a contiguous float32 HxW or HxWxC array.
#[pyfunction]
fn luma_to_uint<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    bit_depth: u8,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(if bit_depth == 16 {
        luma_to::<u16>(py, &img, "uint16")?.into_any()
    } else {
        luma_to::<u8>(py, &img, "uint8")?.into_any()
    })
}

/// `rgb_to_rgba_into(src, dst)`: the first three channels of `src` into the HxWx4 `dst`.
#[pyfunction]
fn rgb_to_rgba_into(
    py: Python<'_>,
    src: PyReadonlyArray3<'_, f32>,
    mut dst: PyReadwriteArray3<'_, f32>,
) -> PyResult<()> {
    let (s, mut d) = (src.as_array(), dst.as_array_mut());
    if s.dim().0 != d.dim().0 || s.dim().1 != d.dim().1 || s.dim().2 < 3 || d.dim().2 != 4 {
        return Err(PyValueError::new_err("needs HxWx3 into HxWx4"));
    }
    let cn = s.dim().2;
    if let (Some(src), Some(dst)) = (s.as_slice(), d.as_slice_mut()) {
        super::detach(py, dst.len(), || convert::rgb_to_rgba(src, cn, dst));
        return Ok(());
    }
    super::detach(py, 0, || {
        for (srow, mut drow) in s.outer_iter().zip(d.outer_iter_mut()) {
            for (sp, mut dp) in srow.outer_iter().zip(drow.outer_iter_mut()) {
                for c in 0..3 {
                    dp[c] = sp[c];
                }
            }
        }
    });
    Ok(())
}

/// `rgba_to_rgb_into(src, dst, oy, ox)`: the RGB of `src[oy:oy+h, ox:ox+w]` into `dst`.
#[pyfunction]
fn rgba_to_rgb_into(
    py: Python<'_>,
    src: PyReadonlyArray3<'_, f32>,
    mut dst: PyReadwriteArray3<'_, f32>,
    oy: usize,
    ox: usize,
) -> PyResult<()> {
    let (s, mut d) = (src.as_array(), dst.as_array_mut());
    let (h, w, dc) = d.dim();
    if oy + h > s.dim().0 || ox + w > s.dim().1 || s.dim().2 < 3 || dc < 3 {
        return Err(PyValueError::new_err("region outside the source"));
    }
    super::detach(py, 0, || {
        for (y, mut drow) in d.outer_iter_mut().enumerate() {
            let srow = s.index_axis(numpy::ndarray::Axis(0), oy + y);
            // A padded readback has whole RGBA rows: copy them as slices when it does.
            if let (Some(sr), Some(dr)) = (srow.as_slice(), drow.as_slice_mut()) {
                let sc = srow.dim().1;
                for (dp, sp) in dr.chunks_exact_mut(dc).zip(sr[ox * sc..].chunks_exact(sc)) {
                    dp[..3].copy_from_slice(&sp[..3]);
                }
                continue;
            }
            for (x, mut dp) in drow.outer_iter_mut().enumerate() {
                for c in 0..3 {
                    dp[c] = srow[[ox + x, c]];
                }
            }
        }
    });
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(to_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(uint8_to_float32, m)?)?;
    m.add_function(wrap_pyfunction!(uint16_to_float32, m)?)?;
    m.add_function(wrap_pyfunction!(luma_to_uint, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_rgba_into, m)?)?;
    m.add_function(wrap_pyfunction!(rgba_to_rgb_into, m)?)
}
