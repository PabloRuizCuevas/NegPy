pub mod analysis;
pub mod bounds;
pub mod contours;
pub mod detect;
pub mod filter;
pub mod gray;
pub mod histogram;
pub mod meter;
pub mod morph;
pub mod resize;
pub mod shapes;
pub mod stats;

/// `f(dst[i], src[i])` for every i, in parallel blocks: a rayon task per element costs more than
/// the work. Pass a `move` closure: a capture read through a reference may alias `dst`, so the
/// loop reloads it per element and does not vectorize.
pub(crate) fn par_zip<T: Send, U: Sync, F: Fn(&mut T, &U) + Sync>(dst: &mut [T], src: &[U], f: F) {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 14;
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| d.iter_mut().zip(s).for_each(|(d, s)| f(d, s)));
}

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn negpy_core(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    py::stats::register(m)?;
    py::resize::register(m)?;
    py::filter::register(m)?;
    py::morph::register(m)?;
    py::gray::register(m)?;
    py::contours::register(m)?;
    py::histogram::register(m)?;
    py::meter::register(m)?;
    py::bounds::register(m)?;
    py::analysis::register(m)
}
