use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::histogram::{self, DENSITY_HIST_BINS, Samples, WAVEFORM_BINS, WAVEFORM_COLS};

type Counts<'py> = Bound<'py, numpy::PyArrayDyn<f64>>;

/// The samples `img[y1:y2:step, x1:x2:step, :3]` of an HxWxC float32 image, C >= 3.
fn samples<'a>(
    img: &'a PyReadonlyArrayDyn<'_, f32>,
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    step: usize,
) -> PyResult<Samples<'a>> {
    let [h, w, cn] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC array"));
    };
    if cn < 3 || step == 0 || y2 > h || x2 > w || y1 >= y2 || x1 >= x2 {
        return Err(PyValueError::new_err("invalid channels, crop or step"));
    }
    let img = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok(Samples {
        img,
        w,
        cn,
        rows: (y1, y2),
        cols: (x1, x2),
        step,
    })
}

/// `density_histogram` of a crop (y1, y2, x1, x2) sampled every `step`: (4, bins) counts.
#[pyfunction]
fn density_histogram<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    roi: (usize, usize, usize, usize),
    step: usize,
) -> PyResult<Counts<'py>> {
    let s = samples(&img, roi, step)?;
    let counts = py.detach(|| histogram::density_histogram(s));
    numpy::PyArray1::from_vec(py, counts).reshape(vec![4, DENSITY_HIST_BINS])
}

/// `density_parade` of a crop (y1, y2, x1, x2) sampled every `step`: (3, bins, cols) counts.
#[pyfunction]
fn density_parade<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    roi: (usize, usize, usize, usize),
    step: usize,
) -> PyResult<Counts<'py>> {
    let s = samples(&img, roi, step)?;
    let counts = py.detach(|| histogram::density_parade(s));
    numpy::PyArray1::from_vec(py, counts).reshape(vec![3, WAVEFORM_BINS, WAVEFORM_COLS])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(density_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(density_parade, m)?)
}
