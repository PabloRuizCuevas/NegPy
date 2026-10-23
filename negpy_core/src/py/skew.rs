use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::skew;

type Samples<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// `_fit_side`'s `sample(center, half, nearest)` on an HxW float32 luminance plane, read along
/// rows (`horizontal`) or columns: (stations, positions) of the profile peaks.
#[pyfunction]
fn skew_sample<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    horizontal: bool,
    stations: PyReadonlyArray1<'py, f64>,
    centers: PyReadonlyArray1<'py, f64>,
    half: f64,
    nearest: bool,
) -> PyResult<Samples<'py>> {
    let (h, w) = (lum.shape()[0], lum.shape()[1]);
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let img = lum.as_slice().map_err(not_contiguous)?;
    let (st, ce) = (
        stations.as_slice().map_err(not_contiguous)?,
        centers.as_slice().map_err(not_contiguous)?,
    );
    if st.len() != ce.len() {
        return Err(PyValueError::new_err("needs one center per station"));
    }
    let (xs, ys) = super::detach(py, || {
        skew::sample(img, (h, w), horizontal, (st, ce), half, nearest)
    });
    Ok((xs.into_pyarray(py), ys.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skew_sample, m)?)
}
