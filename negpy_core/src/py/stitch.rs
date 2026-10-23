use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::stitch;

fn slices<'a, T: numpy::Element>(
    arrays: &'a [PyReadonlyArrayDyn<'_, T>],
    n: usize,
) -> PyResult<Vec<&'a [T]>> {
    arrays
        .iter()
        .map(|a| {
            let s = a
                .as_slice()
                .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
            if s.len() != n {
                return Err(PyValueError::new_err("array sizes disagree"));
            }
            Ok(s)
        })
        .collect()
}

/// `gain_compensate`'s gains for `moving` against `reference`, or None to leave it.
#[pyfunction]
fn compensation_gains(
    py: Python<'_>,
    reference: PyReadonlyArrayDyn<'_, f32>,
    ref_mask: PyReadonlyArrayDyn<'_, bool>,
    moving: PyReadonlyArrayDyn<'_, f32>,
    mov_mask: PyReadonlyArrayDyn<'_, bool>,
    min_overlap: usize,
) -> PyResult<Option<[f32; 3]>> {
    let [h, w, 3] = reference.shape()[..] else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    let n = h * w;
    let images = [reference, moving];
    let coverage = [ref_mask, mov_mask];
    let imgs = slices(&images, 3 * n)?;
    let masks = slices(&coverage, n)?;
    Ok(super::detach(py, n, || {
        stitch::compensation_gains(imgs[0], masks[0], imgs[1], masks[1], min_overlap)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compensation_gains, m)?)
}
