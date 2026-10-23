pub mod altprocess;
pub mod analysis;
pub mod autocrop;
pub mod bounds;
pub mod color;
pub mod components;
pub mod contours;
pub mod convert;
pub mod display;
pub mod filter;
pub mod finish;
pub mod flatfield;
pub mod gray;
pub mod hdr;
pub mod histogram;
pub mod lab;
pub mod lens;
pub mod meter;
pub mod morph;
pub mod print_curve;
pub mod register;
pub mod resize;
pub mod retouch;
pub mod sensor;
pub mod skew;
pub mod stats;
pub mod stitch;
pub mod toning;
pub mod warp;

use pyo3::Python;
use pyo3::marker::Ungil;

/// Work under this many elements stays on the calling thread: waking other workers for each
/// parallel pass of a tiny image costs more than the passes.
const SMALL: usize = 1 << 16;

thread_local! {
    /// A one-thread pool made of the calling thread itself, so its parallel passes run inline.
    static LOCAL: Option<rayon::ThreadPool> = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .ok();
}

/// Every core. A thread that joined `LOCAL` stays in it, so large work enters this pool
/// explicitly rather than through rayon's global one.
fn all_cores() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .build()
            .expect("cannot start the negpy_core thread pool")
    })
}

/// `py.detach(f)` for work of `n` elements: the GIL released, and every binding's work run
/// through here so that each call enters a pool once.
pub fn detach<T: Ungil + Send>(
    py: Python<'_>,
    n: usize,
    f: impl FnOnce() -> T + Ungil + Send,
) -> T {
    py.detach(|| {
        LOCAL.with(|local| match local {
            Some(p) if n < SMALL => p.install(f),
            _ => all_cores().install(f),
        })
    })
}
