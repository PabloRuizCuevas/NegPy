use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::stitch;

fn slices<'a, T: numpy::Element>(
    arrays: &'a [PyReadonlyArrayDyn<'_, T>],
    n: usize,
) -> PyResult<Vec<&'a [T]>> {
    arrays
        .iter()
        .map(|a| {
            let s = a
                .as_slice()
                .map_err(|_| PyValueError::new_err("needs contiguous arrays"))?;
            if s.len() != n {
                return Err(PyValueError::new_err("array sizes disagree"));
            }
            Ok(s)
        })
        .collect()
}

/// `gain_compensate`'s gains for `moving` against `reference`, or None to leave it.
#[pyfunction]
fn compensation_gains(
    py: Python<'_>,
    reference: PyReadonlyArrayDyn<'_, f32>,
    ref_mask: PyReadonlyArrayDyn<'_, bool>,
    moving: PyReadonlyArrayDyn<'_, f32>,
    mov_mask: PyReadonlyArrayDyn<'_, bool>,
    min_overlap: usize,
) -> PyResult<Option<[f32; 3]>> {
    let [h, w, 3] = reference.shape()[..] else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    let n = h * w;
    let images = [reference, moving];
    let coverage = [ref_mask, mov_mask];
    let imgs = slices(&images, 3 * n)?;
    let masks = slices(&coverage, n)?;
    Ok(super::detach(py, n, || {
        stitch::compensation_gains(imgs[0], masks[0], imgs[1], masks[1], min_overlap)
    }))
}

/// `cv2.detail_DpSeamFinder("COLOR").find(images, [(0, 0)] * n, masks)` of HxWx3 float32 parts
/// on one canvas: the cut masks, as new uint8 arrays.
#[pyfunction]
fn dp_seams<'py>(
    py: Python<'py>,
    images: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, u8>>,
) -> PyResult<Vec<Bound<'py, numpy::PyArrayDyn<u8>>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let Some(&[h, w, 3]) = images.first().map(|a| a.shape()) else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    if images.len() != masks.len() {
        return Err(PyValueError::new_err("needs one mask per part"));
    }
    let imgs = slices(&images, 3 * h * w)?;
    let mut cut: Vec<Vec<u8>> = slices(&masks, h * w)?
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    super::detach(py, h * w, || crate::seam::dp_seams(&imgs, &mut cut, (h, w)));
    cut.into_iter()
        .map(|m| m.into_pyarray(py).reshape(vec![h, w]))
        .collect()
}

type BlendInputs<'a> = (Vec<&'a [f32]>, Vec<&'a [bool]>, (usize, usize));

fn blend_inputs<'a>(
    warped: &'a [PyReadonlyArrayDyn<'_, f32>],
    masks: &'a [PyReadonlyArrayDyn<'_, bool>],
) -> PyResult<BlendInputs<'a>> {
    let Some(&[h, w, 3]) = warped.first().map(|a| a.shape()) else {
        return Err(PyValueError::new_err("needs HxWx3 parts"));
    };
    if warped.len() != masks.len() {
        return Err(PyValueError::new_err("needs one mask per part"));
    }
    Ok((slices(warped, 3 * h * w)?, slices(masks, h * w)?, (h, w)))
}

/// `feather_blend(warped, masks)` of HxWx3 float32 parts and HxW bool masks.
#[pyfunction]
fn feather_blend<'py>(
    py: Python<'py>,
    warped: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, bool>>,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let (imgs, m, (h, w)) = blend_inputs(&warped, &masks)?;
    let out = super::detach(py, h * w, || stitch::feather_blend(&imgs, &m, (h, w)));
    out.into_pyarray(py).reshape(vec![h, w, 3])
}

/// `seam_blend(warped, masks, seams, feather_px)` with the band already floored at 1 px.
#[pyfunction]
fn seam_blend<'py>(
    py: Python<'py>,
    warped: Vec<PyReadonlyArrayDyn<'py, f32>>,
    masks: Vec<PyReadonlyArrayDyn<'py, bool>>,
    seams: Vec<PyReadonlyArrayDyn<'py, bool>>,
    feather: f64,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<f32>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    let (imgs, m, (h, w)) = blend_inputs(&warped, &masks)?;
    if seams.len() != masks.len() {
        return Err(PyValueError::new_err("needs one seam per part"));
    }
    let s = slices(&seams, h * w)?;
    let out = super::detach(py, h * w, || {
        stitch::seam_blend(&imgs, &m, &s, (h, w), feather)
    });
    out.into_pyarray(py).reshape(vec![h, w, 3])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(feather_blend, m)?)?;
    m.add_function(wrap_pyfunction!(seam_blend, m)?)?;
    m.add_function(wrap_pyfunction!(dp_seams, m)?)?;
    m.add_function(wrap_pyfunction!(compensation_gains, m)?)
}
