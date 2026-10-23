use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{lsd, skew};

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
    let (xs, ys) = super::detach(py, img.len(), || {
        skew::sample(img, (h, w), horizontal, (st, ce), half, nearest)
    });
    Ok((xs.into_pyarray(py), ys.into_pyarray(py)))
}

/// `_coarse_angles` on an HxW float32 luminance plane: the dominant folded segment angles in
/// degrees, strongest first.
#[pyfunction]
fn coarse_angles(
    py: Python<'_>,
    lum: PyReadonlyArray2<'_, f32>,
    min_segment: f64,
    max_skew: f64,
    angle_bin: f64,
) -> PyResult<Vec<f64>> {
    let (h, w) = (lum.shape()[0], lum.shape()[1]);
    let img = lum
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    if img.is_empty() || angle_bin <= 0.0 || max_skew <= 0.0 {
        return Err(PyValueError::new_err(
            "needs a non-empty plane and positive bins",
        ));
    }
    Ok(super::detach(py, img.len(), || {
        skew::coarse_angles(img, (h, w), min_segment, max_skew, angle_bin)
    }))
}

/// `cv2.createLineSegmentDetector().detect(img)[0]` for an HxW uint8 image, as an (N, 4)
/// float32 array of (x1, y1, x2, y2).
#[pyfunction]
fn line_segments<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let lines = super::detach(py, src.len(), || lsd::line_segments(src, (h, w)));
    let n = lines.len();
    let flat: Vec<f32> = lines.into_iter().flatten().collect();
    flat.into_pyarray(py).reshape([n, 4])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skew_sample, m)?)?;
    m.add_function(wrap_pyfunction!(coarse_angles, m)?)?;
    m.add_function(wrap_pyfunction!(line_segments, m)?)
}
