# Version-bump ledger

A port whose output cannot match the Python version bit for bit is listed here. The release
that ships negpy_core bumps `BOUNDS_VERSION` and `THUMBNAIL_RENDER_VERSION` one time for all
entries, so users render their thumbnails again only once.

Transcendental functions use the pure-Rust `libm` crate, the same bits on every platform.
NumPy computes them through SVML on an AVX-512 CPU and through libm elsewhere, so the Python
results already differ by CPU. Replay accepts the tolerance listed for each entry.

| Function | Differs because | Tolerance |
|---|---|---|
| `features/exposure/normalization.py::_base_blocks` | Blocks of equal log brightness come out in block order, not in NumPy's SIMD argsort order. log10 is libm's, not SVML's, so a block on a group gap can change group in rare cases. | rows in any order |
