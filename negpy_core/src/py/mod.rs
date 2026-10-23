pub mod analysis;
pub mod bounds;
pub mod color;
pub mod components;
pub mod contours;
pub mod convert;
pub mod display;
pub mod filter;
pub mod gray;
pub mod hdr;
pub mod histogram;
pub mod meter;
pub mod morph;
pub mod print_curve;
pub mod resize;
pub mod retouch;
pub mod sensor;
pub mod stats;
pub mod toning;

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;

/// The "Multi-core CPU Rendering" setting: off runs every call on one worker thread.
static PARALLEL: AtomicBool = AtomicBool::new(true);

/// Turns multi-core work on or off for the calls that start after it.
#[pyfunction]
fn set_parallel(enabled: bool) {
    PARALLEL.store(enabled, Ordering::Relaxed);
}

/// `py.detach(f)`, on the global rayon pool or, with multi-core off, on a pool of one thread
/// (the global pool if that one cannot start).
pub(crate) fn detach<T: Send>(py: Python<'_>, f: impl FnOnce() -> T + Send) -> T {
    static SERIAL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    py.detach(|| {
        let serial =
            SERIAL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).build().ok());
        match serial {
            Some(pool) if !PARALLEL.load(Ordering::Relaxed) => pool.install(f),
            _ => f(),
        }
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_parallel, m)?)
}
