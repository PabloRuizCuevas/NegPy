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
| `kernel/image/logic.py::_rgb_to_lab_kernel` (and `rgb_to_lab_working`) | The cube root is `mathf::pow_f64`, not glibc's correctly rounded pow; a* and b* near zero take the float64 difference in a few ULP. | abs=1e-6 |
| `kernel/image/logic.py::_lab_to_rgb_kernel` (and `lab_to_rgb_working`) | One value in the corpus rounds the other way; the float64 operation order copies Numba's fastmath build. | ulp=1 |
| `kernel/image/logic.py::_skin_chroma_rein_kernel` (and `skin_chroma_rein`) | `atan2f` and `exp` are `mathf`'s, not glibc's; Numba's own serial and parallel builds of the kernel already differ in the last bit. | ulp=1 |
| `features/exposure/analysis.py::vectorscope_histogram` | atan2 and log1p are `mathf`'s, not SVML's; a sample on a bin edge can change bin (none did in 39 recorded frames). | abs=2 |
| `features/exposure/logic.py::apply_characteristic_curve` (the print curve) | `exp`, `log1p`, `tanh` and 10^x are `mathf`'s, not glibc's; 2 of 205 M recorded values round the other way. Every feature branch (masks, flash, diffusion, dye mix, separation damping, the print LUTs) matched the Numba kernel bit for bit on test frames. | ulp=1 |
| `features/process/sensor.py::_log_unmix_kernel` (density unmix) | `logf` and `expf` are `mathf`'s (correctly rounded) instead of glibc's; the log-domain matrix and the exp amplify a one-ULP log to up to 16 ULP on 0.2 % of values. The other three unmix kernels match Numba bit for bit. | ulp=16 |
| `features/geometry/logic.py::apply_fine_rotation`, `apply_keystone`, `apply_radial_distortion`, frames of 32767 px or more on a side | OpenCV cannot warp such a frame, so the Python version warped it in tiles from float64 source maps. negpy_core warps the whole frame with OpenCV's fixed-point coordinates, which can move a sample by 1/32 px. Smaller frames match `cv2` bit for bit. | none (the corpus has no such frame) |
| `negpy_core.resize_linear` (bilinear `cv2.resize` of float32: the contrast mask plane, the two-scale unmix layers, dust score and IR gain maps, flat-field gains) | opencv-python on x86-64 runs IPP's float32 bilinear resize, whose bits depend on the CPU it dispatches for; Apple Silicon wheels run OpenCV's own code. negpy_core copies OpenCV's own code. Upscales differ by up to 3 ULP, downscales more. No recorded call reaches these resizes. | ulp=3 (upscale) |
| `features/lab/logic.py::apply_rl_sharpening` | L* takes a correctly rounded cube root (`mathf::cbrtf`); NumPy's float32 `np.cbrt` runs SVML here, up to 2 ULP off on a third of inputs. With a correctly rounded cube root the port matches bit for bit. | abs=1e-6 |
| `features/sabattier/logic.py::apply_sabattier`, `features/cyanotype/logic.py::apply_cyanotype`, `features/lith/logic.py::apply_lith` | log10, exp, log1p, 10^x and the cube root are `mathf`'s, correctly rounded (or within an ULP of it); NumPy's float32 versions run SVML or its own SIMD code. With correctly rounded transcendentals each port matches bit for bit. | abs=1e-6 |
