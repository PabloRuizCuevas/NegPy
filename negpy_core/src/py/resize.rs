use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::resize;

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_AREA)` for a contiguous float32 downscale.
#[pyfunction]
fn resize_area<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dsize: (usize, usize),
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = src.shape().to_vec();
    let (h, w, cn) = match shape.as_slice() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => {
            return Err(PyValueError::new_err(
                "resize_area needs a 2-D or 3-D array",
            ));
        }
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_area needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let out = py
        .detach(|| resize::resize_area(data, h, w, cn, dh, dw))
        .ok_or_else(|| {
            PyValueError::new_err(
                "resize_area covers a float32 downscale, not a 2x2 two-channel one",
            )
        })?;
    let dims: Vec<usize> = if shape.len() == 2 {
        vec![dh, dw]
    } else {
        vec![dh, dw, cn]
    };
    let arr = numpy::ndarray::ArrayD::from_shape_vec(dims, out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_area, m)?)
}
