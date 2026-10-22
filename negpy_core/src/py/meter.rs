use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::meter::{self, Activity, ChannelNorm};

/// Per channel (floor, floor is float64, span, span is float64), as the caller's types decide.
type Norm = [(f64, bool, f64, bool); 3];

/// The data and size of an HxWx3 float32 grid.
fn grid_of<'a>(grid: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<(&'a [f32], usize, usize)> {
    let [h, w, 3] = grid.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 array"));
    };
    let data = grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))?;
    Ok((data, h, w))
}

fn activity((block, gate, min_fraction): (usize, f32, f64)) -> PyResult<Activity> {
    if block == 0 {
        return Err(PyValueError::new_err("block must be positive"));
    }
    Ok(Activity {
        block,
        gate,
        min_fraction,
    })
}

/// The textured luma of a grid, normalized when `norm` is given.
fn textured((data, h, w): (&[f32], usize, usize), norm: Option<Norm>, a: Activity) -> Vec<f32> {
    let norm = norm.map(|n| {
        n.map(|(floor, floor_wide, span, span_wide)| ChannelNorm {
            floor,
            floor_wide,
            span,
            span_wide,
        })
    });
    meter::textured_cells(meter::luma(data, norm.as_ref()), h, w, a)
}

/// `np.percentile` (float32 q) of the grid's textured normalized luma: shadow reach, highlight hold.
#[pyfunction]
fn textured_percentile(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    norm: Norm,
    gate: (usize, f32, f64),
    q: f64,
) -> PyResult<f32> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(py.detach(|| meter::percentile(&textured(g, Some(norm), a), q)))
}

/// The anchor meter's measurement over the grid's textured normalized luma.
#[pyfunction]
fn textured_anchor(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    norm: Norm,
    gate: (usize, f32, f64),
    clip: f64,
) -> PyResult<f64> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(py.detach(|| meter::anchor_measure(&textured(g, Some(norm), a), clip)))
}

/// The textural range of the grid's textured raw-log luma.
#[pyfunction]
fn textured_range(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    gate: (usize, f32, f64),
    clip: f64,
) -> PyResult<f64> {
    let (g, a) = (grid_of(&grid)?, activity(gate)?);
    Ok(py.detach(|| meter::textural_range(&textured(g, None, a), clip)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(textured_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(textured_anchor, m)?)?;
    m.add_function(wrap_pyfunction!(textured_range, m)?)
}
