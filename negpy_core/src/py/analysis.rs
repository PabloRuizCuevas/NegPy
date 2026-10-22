use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::analysis;

fn empty<'py, T: Element>(
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

fn slice<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `get_luminance` of an HxWx3 float32 image.
#[pyfunction]
fn luminance<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = rgb.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("luminance needs an HxWx3 array"));
    }
    let data = slice(&rgb)?;
    let out = empty::<f32>(py, &shape[..2], "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::luminance(data, dst));
    }
    Ok(out)
}

/// `_normalize_to_uint8` of a float32 image.
#[pyfunction]
fn normalize_to_uint8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = img.shape().to_vec();
    let data = slice(&img)?;
    let out = empty::<u8>(py, &shape, "uint8")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::normalize_to_uint8(data, dst));
    }
    Ok(out)
}

/// `_detection_luma` of a float32 luminance plane.
#[pyfunction]
fn detection_luma<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = lum.shape().to_vec();
    let data = slice(&lum)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::detection_luma(data, dst))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("percentiles of an empty array")
            })?;
    }
    Ok(out)
}

/// `_base_blocks(det, band)`: an Nx3 float32 array of the base blocks' means, or None.
#[pyfunction]
fn base_blocks<'py>(
    py: Python<'py>,
    det: PyReadonlyArrayDyn<'py, f32>,
    band: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
    let shape = det.shape().to_vec();
    let [h, w, 3] = shape[..] else {
        return Err(PyValueError::new_err(
            "flat_block_means needs an HxWx3 array",
        ));
    };
    if band.shape() != [h, w] {
        return Err(PyValueError::new_err("band must be HxW"));
    }
    let data = slice(&det)?;
    let mask: Vec<u8> = band
        .as_slice()
        .map_err(|_| PyValueError::new_err("band must be contiguous"))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    let found = py.detach(|| analysis::base_blocks(data, &mask, h, w));
    found
        .map(|rows| {
            let flat: Vec<Vec<f32>> = rows.iter().map(|r| r.to_vec()).collect();
            numpy::PyArray2::from_vec2(py, &flat).map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .transpose()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(luminance, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_to_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(detection_luma, m)?)?;
    m.add_function(wrap_pyfunction!(base_blocks, m)?)
}
