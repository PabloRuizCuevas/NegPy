use numpy::{Element, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::analysis;

fn empty<'py, T: Element>(
    py: Python<'py>,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(py
        .import("numpy")?
        .getattr("empty")?
        .call1((shape.to_vec(), dtype))?
        .cast_into::<PyArrayDyn<T>>()?)
}

fn slice<'a>(a: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `get_luminance` of an HxWx3 float32 image.
#[pyfunction]
fn luminance<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = rgb.shape().to_vec();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("luminance needs an HxWx3 array"));
    }
    let data = slice(&rgb)?;
    let out = empty::<f32>(py, &shape[..2], "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::luminance(data, dst));
    }
    Ok(out)
}

/// `_normalize_to_uint8` of a float32 image.
#[pyfunction]
fn normalize_to_uint8<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = img.shape().to_vec();
    let data = slice(&img)?;
    let out = empty::<u8>(py, &shape, "uint8")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::normalize_to_uint8(data, dst));
    }
    Ok(out)
}

/// `_detection_luma` of a float32 luminance plane.
#[pyfunction]
fn detection_luma<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = lum.shape().to_vec();
    let data = slice(&lum)?;
    let out = empty::<f32>(py, &shape, "float32")?;
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| analysis::detection_luma(data, dst))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("percentiles of an empty array")
            })?;
    }
    Ok(out)
}

/// `_base_blocks(det, band)`: an Nx3 float32 array of the base blocks' means, or None.
#[pyfunction]
fn base_blocks<'py>(
    py: Python<'py>,
    det: PyReadonlyArrayDyn<'py, f32>,
    band: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<Option<Bound<'py, numpy::PyArray2<f32>>>> {
    let shape = det.shape().to_vec();
    let [h, w, 3] = shape[..] else {
        return Err(PyValueError::new_err(
            "flat_block_means needs an HxWx3 array",
        ));
    };
    if band.shape() != [h, w] {
        return Err(PyValueError::new_err("band must be HxW"));
    }
    let data = slice(&det)?;
    let mask: Vec<u8> = band
        .as_slice()
        .map_err(|_| PyValueError::new_err("band must be contiguous"))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    let found = py.detach(|| analysis::base_blocks(data, &mask, h, w));
    found
        .map(|rows| {
            let flat: Vec<Vec<f32>> = rows.iter().map(|r| r.to_vec()).collect();
            numpy::PyArray2::from_vec2(py, &flat).map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .transpose()
}

/// `_find_rebate_level(lum, film_roi, bed)` for a 2-D float32 plane: (level, spread) or None.
#[pyfunction]
#[pyo3(signature = (lum, film_roi, bed=None))]
fn find_rebate_level(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
    bed: Option<f64>,
) -> PyResult<Option<(f64, f64)>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err("find_rebate_level needs a 2-D array"));
    };
    let data = slice(&lum)?;
    Ok(py.detach(|| analysis::find_rebate_level(data, h, w, film_roi, bed)))
}

/// `_walk_border_depth(lines, limit, bridge, relative_step)` for an L x N float32 array.
#[pyfunction]
fn walk_border_depth<'py>(
    py: Python<'py>,
    lines: PyReadonlyArrayDyn<'py, f32>,
    limit: usize,
    bridge: i64,
    relative_step: bool,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    let [rows, n] = lines.shape()[..] else {
        return Err(PyValueError::new_err("walk_border_depth needs a 2-D array"));
    };
    let data = slice(&lines)?;
    let out = py
        .detach(|| analysis::walk_border_depth(data, rows, n, limit, bridge, relative_step))
        .ok_or_else(|| PyValueError::new_err("walk_border_depth needs at least limit + 3 rows"))?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// `measure_film_edges(lum, film_roi)`: (top, bottom, left, right) border fractions.
#[pyfunction]
fn measure_film_edges(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<(f64, f64, f64, f64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "measure_film_edges needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let [t, b, l, r] = py.detach(|| analysis::measure_film_edges(data, h, w, film_roi));
    Ok((t, b, l, r))
}

/// `_trim_film_edges(lum, film_roi)`.
#[pyfunction]
fn trim_film_edges(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (i64, i64, i64, i64),
) -> PyResult<(i64, i64, i64, i64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err("trim_film_edges needs a 2-D array"));
    };
    let data = slice(&lum)?;
    Ok(py.detach(|| analysis::trim_film_edges(data, h, w, film_roi)))
}

/// `measure_film_border(lum, film_roi)`: (top, bottom, left, right) border fractions.
#[pyfunction]
fn measure_film_border(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<(f64, f64, f64, f64)> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "measure_film_border needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let [t, b, l, r] = py.detach(|| analysis::measure_film_border(data, h, w, film_roi));
    Ok((t, b, l, r))
}

/// `_roi_from_measured_border(lum, film_roi)`.
#[pyfunction]
fn roi_from_measured_border(
    py: Python<'_>,
    lum: PyReadonlyArrayDyn<'_, f32>,
    film_roi: (i64, i64, i64, i64),
) -> PyResult<Option<(i64, i64, i64, i64)>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "roi_from_measured_border needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    Ok(py.detach(|| analysis::roi_from_measured_border(data, h, w, film_roi)))
}

/// `_smooth_signal(signal, window)` for a 1-D float32 signal.
#[pyfunction]
fn smooth_signal<'py>(
    py: Python<'py>,
    signal: PyReadonlyArrayDyn<'py, f32>,
    window: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
    let data = slice(&signal)?.to_vec();
    let out = py
        .detach(|| analysis::smooth_signal(&data, window))
        .ok_or_else(|| {
            PyValueError::new_err(
                "smooth_signal covers a window of at most 31, no longer than the signal",
            )
        })?;
    Ok(numpy::PyArray1::from_vec(py, out))
}

/// `_snap_edge_to_gradient(profile, idx, is_start, window_out, window_in, min_dominance, min_window_px)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn snap_edge_to_gradient(
    py: Python<'_>,
    profile: PyReadonlyArrayDyn<'_, f32>,
    idx: i64,
    is_start: bool,
    window_out: f64,
    window_in: f64,
    min_dominance: f64,
    min_window_px: i64,
) -> PyResult<i64> {
    let data = slice(&profile)?;
    let snap = analysis::Snap {
        window_out,
        window_in,
        min_dominance,
        min_window_px,
    };
    Ok(py.detach(|| analysis::snap_edge_to_gradient(data, idx, is_start, snap)))
}

type Refined<'py> = (
    (i64, i64, i64, i64),
    Bound<'py, numpy::PyArray1<f32>>,
    Bound<'py, numpy::PyArray1<f32>>,
);

/// `_refine_film_roi_by_tiers(lum, film_roi)`: (roi, row occupancy, column occupancy) or None.
#[pyfunction]
fn refine_film_roi_by_tiers<'py>(
    py: Python<'py>,
    lum: PyReadonlyArrayDyn<'py, f32>,
    film_roi: (usize, usize, usize, usize),
) -> PyResult<Option<Refined<'py>>> {
    let [h, w] = lum.shape()[..] else {
        return Err(PyValueError::new_err(
            "refine_film_roi_by_tiers needs a 2-D array",
        ));
    };
    let data = slice(&lum)?;
    let found = py.detach(|| analysis::refine_film_roi_by_tiers(data, h, w, film_roi));
    Ok(found.map(|r| {
        (
            r.roi,
            numpy::PyArray1::from_vec(py, r.rows),
            numpy::PyArray1::from_vec(py, r.cols),
        )
    }))
}

/// `_refine_frame_bounds(image)`'s (left, top, right, bottom) for an HxW or HxWx3 float32 image.
#[pyfunction]
fn refine_frame_bounds(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, f32>,
) -> PyResult<(i64, i64, i64, i64)> {
    let (h, w, cn) = match image.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "refine_frame_bounds needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = slice(&image)?;
    py.detach(|| analysis::refine_frame_bounds(data, h, w, cn))
        .ok_or_else(|| {
            PyValueError::new_err("refine_frame_bounds needs an image at least 31 px on each side")
        })
}

type Gated<'py> = (
    analysis::Roi,
    Option<Bound<'py, numpy::PyArray1<f32>>>,
    Option<Bound<'py, numpy::PyArray1<f32>>>,
);

/// `_refine_roi_to_image(img, film_roi)` for an HxW or HxWx3 float32 image: (roi, row
/// occupancy or None, column occupancy or None).
#[pyfunction]
fn refine_roi_to_image<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    film_roi: analysis::Roi,
) -> PyResult<Gated<'py>> {
    let (h, w, cn) = match img.shape()[..] {
        [h, w] => (h, w, 1),
        [h, w, 3] => (h, w, 3),
        _ => {
            return Err(PyValueError::new_err(
                "refine_roi_to_image needs an HxW or HxWx3 array",
            ));
        }
    };
    let data = slice(&img)?;
    let (roi, tiers) = py
        .detach(|| analysis::refine_roi_to_image(data, h, w, cn, film_roi))
        .ok_or_else(|| {
            PyValueError::new_err("refine_roi_to_image: image too small for the Sobel path")
        })?;
    Ok(match tiers {
        Some(t) => (
            roi,
            Some(numpy::PyArray1::from_vec(py, t.rows)),
            Some(numpy::PyArray1::from_vec(py, t.cols)),
        ),
        None => (roi, None, None),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(luminance, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_to_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(detection_luma, m)?)?;
    m.add_function(wrap_pyfunction!(base_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(find_rebate_level, m)?)?;
    m.add_function(wrap_pyfunction!(walk_border_depth, m)?)?;
    m.add_function(wrap_pyfunction!(measure_film_edges, m)?)?;
    m.add_function(wrap_pyfunction!(trim_film_edges, m)?)?;
    m.add_function(wrap_pyfunction!(measure_film_border, m)?)?;
    m.add_function(wrap_pyfunction!(roi_from_measured_border, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_signal, m)?)?;
    m.add_function(wrap_pyfunction!(snap_edge_to_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(refine_film_roi_by_tiers, m)?)?;
    m.add_function(wrap_pyfunction!(refine_frame_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(refine_roi_to_image, m)?)
}
