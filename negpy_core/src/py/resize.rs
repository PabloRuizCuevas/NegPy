use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::resize;

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_AREA)` of a contiguous float32 image.
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
    let out =
        super::detach(py, || resize::resize_area(data, h, w, cn, dh, dw)).ok_or_else(|| {
            PyValueError::new_err("resize_area does not cover a 2x2 two-channel downscale")
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

/// `cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR)` of a contiguous float32 image, as
/// OpenCV computes it without IPP.
#[pyfunction]
fn resize_linear<'py>(
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
                "resize_linear needs a 2-D or 3-D array",
            ));
        }
    };
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("resize_linear needs a contiguous array"))?;
    let (dw, dh) = dsize;
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&shape[2..]);
    // NumPy's allocator asks for huge pages, so the output faults in far fewer pages.
    let out = PyArrayDyn::<f32>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, || {
            resize::resize_linear(data, (h, w, cn), (dh, dw), false, dst)
        })
        .ok_or_else(|| {
            PyValueError::new_err(
                "resize_linear needs a non-empty size and covers no 2x2 two-channel halving",
            )
        })?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_area, m)?)?;
    m.add_function(wrap_pyfunction!(resize_linear, m)?)
}
