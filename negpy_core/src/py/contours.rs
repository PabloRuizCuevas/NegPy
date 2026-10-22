use numpy::{PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::contours;

/// `cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` for a uint8 image:
/// (contours as Nx1x2 int32 arrays, hierarchy as 1xKx4 int32), in OpenCV's order.
#[pyfunction]
fn find_external_contours<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyAny>)> {
    let arr = src.as_array();
    let (h, w) = arr.dim();
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let found = py.detach(|| contours::find_external_contours(data, h, w));
    let k = found.len();
    let arrays: Vec<Bound<'py, PyArray3<i32>>> = found
        .iter()
        .map(|c| {
            let flat: Vec<Vec<Vec<i32>>> = c.iter().map(|&(x, y)| vec![vec![x, y]]).collect();
            PyArray3::from_vec3(py, &flat)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let hierarchy: Vec<Vec<i32>> = (0..k)
        .map(|i| {
            vec![
                if i + 1 < k { i as i32 + 1 } else { -1 },
                i as i32 - 1,
                -1,
                -1,
            ]
        })
        .collect();
    let np = py.import("numpy")?;
    let hier = if k == 0 {
        py.None().into_bound(py)
    } else {
        np.getattr("array")?.call1((vec![hierarchy], "int32"))?
    };
    Ok((PyTuple::new(py, arrays)?, hier))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_external_contours, m)?)
}
