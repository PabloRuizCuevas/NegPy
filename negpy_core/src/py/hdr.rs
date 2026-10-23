use numpy::{PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::hdr;

fn contiguous<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs contiguous arrays"))
}

/// `_count_clipped_rows(values, out)` of an HxWxC float32 frame.
#[pyfunction]
fn count_clipped_rows(
    py: Python<'_>,
    values: PyReadonlyArrayDyn<'_, f32>,
    saturation: f64,
    mut out: PyReadwrite1<'_>,
) -> PyResult<()> {
    let [h, w, cn] = values.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWxC frame"));
    };
    if out.len() != h {
        return Err(PyValueError::new_err("out needs one entry per row"));
    }
    let v = contiguous(&values)?;
    let dst = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs a contiguous out"))?;
    py.detach(|| hdr::count_clipped_rows(v, (w, cn), saturation, dst));
    Ok(())
}

type PyReadwrite1<'py> = PyReadwriteArray1<'py, i64>;

/// `_accumulate(num, den, values, ratio)`, with the module's saturation constants.
#[pyfunction]
fn hdr_accumulate(
    py: Python<'_>,
    mut num: PyReadwriteArrayDyn<'_, f32>,
    mut den: PyReadwriteArrayDyn<'_, f32>,
    values: PyReadonlyArrayDyn<'_, f32>,
    ratio: f64,
    saturation: f64,
    rolloff_start: f64,
) -> PyResult<()> {
    if num.shape() != values.shape() || den.shape() != values.shape() {
        return Err(PyValueError::new_err("the sums do not match the frame"));
    }
    let v = contiguous(&values)?;
    let n = num
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous sums"))?;
    let d = den
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("needs contiguous sums"))?;
    py.detach(|| hdr::accumulate(n, d, v, ratio, (saturation, rolloff_start)));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_clipped_rows, m)?)?;
    m.add_function(wrap_pyfunction!(hdr_accumulate, m)?)
}
