# Version-bump ledger

A port whose output cannot match the Python version bit for bit is listed here. The release
that ships negpy_core bumps `BOUNDS_VERSION` and `THUMBNAIL_RENDER_VERSION` one time for all
entries, so users render their thumbnails again only once.

Transcendental functions in array loops come from `src/mathf.rs`, scalar ones from the pure-Rust
`libm` crate: the same bits on every platform. NumPy computes them through SVML on an AVX-512 CPU
and through libm elsewhere, so the Python results already differ by CPU. Replay accepts the
tolerance listed for each entry.

| Function | Differs because | Tolerance |
|---|---|---|
| `features/exposure/normalization.py::_base_blocks` | Blocks of equal log brightness come out in block order, not in NumPy's SIMD argsort order. log10 is libm's, not SVML's, so a block on a group gap can change group in rare cases. | rows in any order |
| `negpy_core.distance_transform` (DIST_L2, mask 3), once a port calls it | opencv-python runs IPP's float32 chamfer, whose 3x3 pass picks a path one float32 unit longer than the two-pass order at rare pixels; OpenCV without IPP (Apple Silicon wheels) returns its own fixed-point values. | ulp=1 |
| `features/exposure/normalization.py::to_log_density` (through `prefilter_log_grid`) | log10 is correctly rounded (`mathf::log10f`); NumPy's float32 log10 (SVML) is up to 3 ULP off on about half the inputs. | ulp=3 |
