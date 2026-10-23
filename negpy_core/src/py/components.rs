use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::components;

type Labelled<'py> = (
    usize,
    Bound<'py, numpy::PyArrayDyn<i32>>,
    Bound<'py, numpy::PyArray2<i32>>,
    Bound<'py, numpy::PyArray2<f64>>,
);

/// `cv2.connectedComponentsWithStats(mask, connectivity=...)` of an HxW uint8 mask:
/// (count, labels, stats, centroids).
#[pyfunction]
#[pyo3(signature = (mask, connectivity=8))]
fn connected_components_with_stats<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    connectivity: u8,
) -> PyResult<Labelled<'py>> {
    let [h, w] = mask.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxW array"));
    };
    if connectivity != 4 && connectivity != 8 {
        return Err(PyValueError::new_err("connectivity must be 4 or 8"));
    }
    let data = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    let c = py.detach(|| components::connected_components(data, h, w, connectivity));
    let labels = numpy::PyArray1::from_vec(py, c.labels).reshape(vec![h, w])?;
    let stats =
        numpy::PyArray2::from_vec2(py, &c.stats.iter().map(|s| s.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let centroids = numpy::PyArray2::from_vec2(
        py,
        &c.centroids.iter().map(|s| s.to_vec()).collect::<Vec<_>>(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((c.count, labels, stats, centroids))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connected_components_with_stats, m)?)
}
