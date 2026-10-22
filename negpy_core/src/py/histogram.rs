use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::histogram::{
    self, COLOR_HIST_BINS, DENSITY_HIST_BINS, OUTPUT_HIST_BINS, Samples, WAVEFORM_BINS,
    WAVEFORM_COLS, Waveform,
};

type Counts<'py> = Bound<'py, numpy::PyArrayDyn<f64>>;

/// The samples `img[y1:y2:step, x1:x2:step, :3]` of an HxWxC float32 image, C >= 3.
fn samples<'a, T: numpy::Element>(
    img: &'a PyReadonlyArrayDyn<'_, T>,
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    step: usize,
) -> PyResult<Samples<'a, T>> {
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

/// The whole frame of an HxWxC image, sampled every `step`.
fn frame<'a, T: numpy::Element>(
    img: &'a PyReadonlyArrayDyn<'_, T>,
    step: usize,
) -> PyResult<Samples<'a, T>> {
    let (h, w) = (img.shape()[0], img.shape().get(1).copied().unwrap_or(0));
    samples(img, (0, h, 0, w), step)
}

/// `output_histogram` of an HxWx3 float32 or uint8 frame sampled every `step`: (4, 256) counts.
#[pyfunction]
fn output_histogram<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    step: usize,
) -> PyResult<Counts<'py>> {
    let counts = if let Ok(a) = img.extract::<PyReadonlyArrayDyn<'py, u8>>() {
        let s = frame(&a, step)?;
        py.detach(|| histogram::output_histogram_u8(s))
    } else {
        let a = img.extract::<PyReadonlyArrayDyn<'py, f32>>()?;
        let s = frame(&a, step)?;
        py.detach(|| histogram::output_histogram(s))
    };
    numpy::PyArray1::from_vec(py, counts).reshape(vec![4, OUTPUT_HIST_BINS])
}

/// `color_histogram` of an HxWx3 float32 frame sampled every `step`: (32, 32, 32) counts.
#[pyfunction]
fn color_histogram<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    step: usize,
) -> PyResult<Counts<'py>> {
    let s = frame(&img, step)?;
    let counts = py.detach(|| histogram::color_histogram(s));
    numpy::PyArray1::from_vec(py, counts).reshape(vec![COLOR_HIST_BINS; 3])
}

/// `waveform` ("luma"), `waveform_color` ("color") or `waveform_parade` ("parade") of an
/// HxWxC float32 frame sampled every `step`.
#[pyfunction]
fn waveform<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    step: usize,
    kind: &str,
) -> PyResult<Counts<'py>> {
    let kind = match kind {
        "luma" => Waveform::Luma,
        "color" => Waveform::Color,
        "parade" => Waveform::Parade,
        _ => return Err(PyValueError::new_err(format!("unknown waveform {kind:?}"))),
    };
    let s = frame(&img, step)?;
    let counts = py.detach(|| histogram::waveform(s, kind));
    let shape = if kind == Waveform::Luma {
        vec![WAVEFORM_BINS, WAVEFORM_COLS]
    } else {
        vec![3, WAVEFORM_BINS, WAVEFORM_COLS]
    };
    numpy::PyArray1::from_vec(py, counts).reshape(shape)
}

/// `scan_full_scale_columns` of a crop (y1, y2, x1, x2) sampled every `step`: (3, cols) counts.
#[pyfunction]
fn scan_full_scale_columns<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    roi: (usize, usize, usize, usize),
    step: usize,
    level: f32,
) -> PyResult<Counts<'py>> {
    let s = samples(&img, roi, step)?;
    let counts = py.detach(|| histogram::scan_full_scale_columns(s, level));
    numpy::PyArray1::from_vec(py, counts).reshape(vec![3, WAVEFORM_COLS])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(density_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(density_parade, m)?)?;
    m.add_function(wrap_pyfunction!(output_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(color_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(waveform, m)?)?;
    m.add_function(wrap_pyfunction!(scan_full_scale_columns, m)?)
}
