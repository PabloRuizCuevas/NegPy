use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::warp::{self, Image};

fn image<'a>(src: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<Image<'a>> {
    let (h, w, cn) = match src.shape() {
        [h, w] => (*h, *w, 1),
        [h, w, c] => (*h, *w, *c),
        _ => return Err(PyValueError::new_err("needs an HxW or HxWxC image")),
    };
    if h == 0 || w == 0 || cn == 0 {
        return Err(PyValueError::new_err("needs a non-empty image"));
    }
    let data = src
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    Ok(Image { data, h, w, cn })
}

/// A new float32 output of `dh` x `dw` with the source's channel layout, filled by `f`.
fn output<'py>(
    py: Python<'py>,
    src: &PyReadonlyArrayDyn<'py, f32>,
    (dw, dh): (usize, usize),
    f: impl FnOnce(&mut [f32]) + Send,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let mut dims = vec![dh, dw];
    dims.extend_from_slice(&src.shape()[2..]);
    let out = PyArrayDyn::<f32>::zeros(py, dims, false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        super::detach(py, || f(dst));
    }
    Ok(out)
}

fn matrix<const N: usize>(m: &PyReadonlyArray2<'_, f64>, rows: usize) -> PyResult<[f64; N]> {
    let a = m.as_array();
    if a.shape() != [rows, 3] {
        return Err(PyValueError::new_err(format!("needs a {rows}x3 matrix")));
    }
    Ok(std::array::from_fn(|i| a[[i / 3, i % 3]]))
}

/// `cv2.remap(src, map_x, map_y, INTER_LINEAR, borderMode=BORDER_REPLICATE)` of a float32 image.
#[pyfunction]
fn remap_linear<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    if map_x.shape() != map_y.shape() {
        return Err(PyValueError::new_err("the maps differ in size"));
    }
    let (dh, dw) = (map_x.shape()[0], map_x.shape()[1]);
    let not_contiguous = |_| PyValueError::new_err("needs contiguous maps");
    let mx = map_x.as_slice().map_err(not_contiguous)?;
    let my = map_y.as_slice().map_err(not_contiguous)?;
    output(py, &src, (dw, dh), |dst| {
        warp::remap_linear(&img, mx, my, dw, dst)
    })
}

/// `apply_radial_distortion` of a float32 image: the remap with the scale-to-fill `s`, the
/// center and 1 / halfdiag^2, each rounded to float32.
#[pyfunction]
fn remap_radial<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    center: (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    output(py, &src, (img.w, img.h), |dst| {
        warp::remap_radial(&img, center, s, k1, inv_hd2, dst)
    })
}

/// `cv2.warpAffine(src, m, dsize, INTER_LINEAR, borderMode=BORDER_REPLICATE)` of a float32 image.
#[pyfunction]
#[pyo3(signature = (src, m, dsize, inverse_map=false))]
fn warp_affine<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f64>,
    dsize: (usize, usize),
    inverse_map: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let m = matrix::<6>(&m, 2)?;
    output(py, &src, dsize, |dst| {
        warp::warp_affine(&img, m, inverse_map, dsize.0, dst)
    })
}

/// `cv2.warpPerspective(src, m, dsize, INTER_LINEAR, borderMode=BORDER_REPLICATE)` of a float32
/// image.
#[pyfunction]
#[pyo3(signature = (src, m, dsize, inverse_map=false))]
fn warp_perspective<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    m: PyReadonlyArray2<'py, f64>,
    dsize: (usize, usize),
    inverse_map: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let img = image(&src)?;
    let m = matrix::<9>(&m, 3)?;
    output(py, &src, dsize, |dst| {
        warp::warp_perspective(&img, m, inverse_map, dsize, dst)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(remap_linear, m)?)?;
    m.add_function(wrap_pyfunction!(remap_radial, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(warp_perspective, m)?)
}
