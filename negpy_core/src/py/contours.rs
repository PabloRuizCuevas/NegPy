use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::contours;

type PyRotatedRect = ((f32, f32), (f32, f32), f32);

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

fn int_points(pts: &numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<(i32, i32)>> {
    let flat = pts
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be contiguous"))?;
    if flat.len() % 2 != 0 {
        return Err(PyValueError::new_err("points must be pairs"));
    }
    Ok(flat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0], p[1]))
        .collect())
}

/// `cv2.minAreaRect(points)` of int32 points: ((cx, cy), (w, h), angle).
#[pyfunction]
fn min_area_rect(pts: numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<PyRotatedRect> {
    let r = crate::shapes::min_area_rect(&int_points(&pts)?);
    Ok((r.center, r.size, r.angle))
}

/// `cv2.boxPoints(rect)`: four (x, y) float32 corners.
#[pyfunction]
fn box_points(rect: PyRotatedRect) -> Vec<(f32, f32)> {
    let r = crate::shapes::RotatedRect {
        center: rect.0,
        size: rect.1,
        angle: rect.2,
    };
    crate::shapes::box_points(&r).to_vec()
}

/// `cv2.convexHull(points)` of int32 points, as a list of (x, y).
#[pyfunction]
fn convex_hull(pts: numpy::PyReadonlyArrayDyn<'_, i32>) -> PyResult<Vec<(i32, i32)>> {
    Ok(crate::shapes::convex_hull(&int_points(&pts)?))
}

/// `cv2.fillConvexPoly(zeros((h, w), uint8), points, 1)` as a new mask.
#[pyfunction]
fn fill_convex_poly<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    pts: numpy::PyReadonlyArrayDyn<'_, i32>,
) -> PyResult<Bound<'py, numpy::PyArray2<u8>>> {
    let (h, w) = shape;
    let v: Vec<(i64, i64)> = int_points(&pts)?
        .iter()
        .map(|&(x, y)| (x as i64, y as i64))
        .collect();
    let mut mask = vec![0u8; h * w];
    crate::shapes::fill_convex_poly(&mut mask, w, h, &v, 1);
    numpy::PyArray1::from_vec(py, mask).reshape([h, w])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_external_contours, m)?)?;
    m.add_function(wrap_pyfunction!(min_area_rect, m)?)?;
    m.add_function(wrap_pyfunction!(box_points, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(fill_convex_poly, m)?)
}
