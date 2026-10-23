use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::color::space;
use crate::lab;

/// `_usm_sharpening_block(img, amount, radius, masking)` of an HxWx3 float32 image: `gain` is
/// `np.float32(amount * 2.5)`, `k` the blur taps, then the working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn usm_sharpen<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    gain: f32,
    masking: f64,
    k: PyReadonlyArray1<'py, f32>,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let k = k.as_slice().map_err(not_contiguous)?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, || {
            lab::usm_sharpen(src, (h, w), (gain, masking), k, spaces, dst)
        })
        .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

/// `apply_chroma_denoise` of an HxWx3 float32 image at a known low-pass `sigma` with blur taps
/// `k`, then the working Lab constants.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn chroma_denoise<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    sigma: f64,
    k: PyReadonlyArray1<'py, f32>,
    to_xyz: PyReadonlyArray2<'py, f32>,
    to_rgb: PyReadonlyArray2<'py, f32>,
    white: PyReadonlyArray1<'py, f32>,
    eps: f32,
    kappa: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [h, w, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("needs an HxWx3 image"));
    };
    let spaces = (
        space(&to_xyz, &white, eps, kappa)?,
        space(&to_rgb, &white, eps, kappa)?,
    );
    let not_contiguous = |_| PyValueError::new_err("needs contiguous arrays");
    let src = img.as_slice().map_err(not_contiguous)?;
    let k = k.as_slice().map_err(not_contiguous)?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, || {
            lab::chroma_denoise(src, (h, w), sigma, k, spaces, dst)
        })
        .ok_or_else(|| PyValueError::new_err("the image is too small for the blur"))?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(usm_sharpen, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_denoise, m)?)
}
