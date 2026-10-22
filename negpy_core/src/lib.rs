pub mod analysis;
pub mod contours;
pub mod filter;
pub mod gray;
pub mod morph;
pub mod resize;
pub mod stats;

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
    py::analysis::register(m)
}
