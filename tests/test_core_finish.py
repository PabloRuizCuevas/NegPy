"""negpy_core.apply_carrier against the NumPy carrier it replaces, with a correctly rounded
cube root where NumPy's own depends on the CPU."""

import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.finish.logic import CARRIER_SAMPLES, CARRIER_TONE_SAMPLES, apply_carrier, carrier_profiles, linear_carrier_tone

pytestmark = cv2_oracle

CARRIER_JITTER = 0.24
CARRIER_INNER_ROUGH = 0.2
CARRIER_OUTER_JITTER = 0.4
CARRIER_CORNER = 1.4
CARRIER_GATE_CORNER = 0.2
# The film never sits centered in the carrier.
CARRIER_OFFSET_X = -0.2
CARRIER_OFFSET_Y = -0.25
CARRIER_MARGIN = 0.7
# The gate prints soft; filed metal is a hard stop, which is what reads as filed.
CARRIER_SOFT = 0.22
CARRIER_FILED_SOFT = 0.06
CARRIER_FLARE_DEPTH = 0.25
CARRIER_FLARE_GAIN = 0.6
CARRIER_FLARE_BASE = 0.35
# A 1-D profile is a height field, so it cannot overhang or shed a fleck. The 2-D field
# displacing the distance field is what makes the edge read as torn metal. Hash noise
# rather than a library, because WGSL has to reproduce it bit for bit.
CARRIER_NOISE_SEED = 0x51ED270B
CARRIER_NOISE_CELL = 0.35
CARRIER_NOISE_OCTAVES = 4
CARRIER_NOISE_OUTER = 0.12
CARRIER_NOISE_INNER = 0.02
# Tone table sampled at t = u**POWER, dense in the toe.
_CARRIER_BLOCK_PIXELS = 1 << 19


def _hash_lattice(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Lattice value in [-1, 1). u32 wrap-around only, so WGSL reproduces it exactly."""
    h = (ix * np.uint32(0x27D4EB2D)) ^ (iy * np.uint32(0x165667B1)) ^ np.uint32(CARRIER_NOISE_SEED)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x297A2D39)
    h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 / 16777216.0) - np.float32(1.0)


def _value_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    fx, fy = np.floor(x), np.floor(y)
    ux, uy = (x - fx), (y - fy)
    ux = ux * ux * (3.0 - 2.0 * ux)
    uy = uy * uy * (3.0 - 2.0 * uy)
    i0, j0 = fx.astype(np.int64).astype(np.uint32), fy.astype(np.int64).astype(np.uint32)
    i1, j1 = i0 + np.uint32(1), j0 + np.uint32(1)
    lo = _hash_lattice(i0, j0)
    lo = lo + (_hash_lattice(i1, j0) - lo) * ux
    hi = _hash_lattice(i0, j1)
    hi = hi + (_hash_lattice(i1, j1) - hi) * ux
    return lo + (hi - lo) * uy


def carrier_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """fBm value noise in [-1, 1]; x/y in noise cells. Mirrored in finish.wgsl."""
    total = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float32)
    amp, freq, norm = np.float32(1.0), np.float32(1.0), np.float32(0.0)
    for _ in range(CARRIER_NOISE_OCTAVES):
        total += amp * _value_noise(x * freq, y * freq)
        norm += amp
        amp *= np.float32(0.5)
        freq *= np.float32(2.0)
    return total / norm


def carrier_tone_lookup(tone, t):
    u = np.cbrt(np.clip(t, 0.0, 1.0).astype(np.float64)).astype(np.float32) * np.float32(CARRIER_TONE_SAMPLES - 1)
    grid = np.arange(CARRIER_TONE_SAMPLES, dtype=np.float32)
    return np.stack([np.interp(u, grid, tone[:, c]).astype(np.float32) for c in range(3)], axis=-1)


def _carrier_reference(
    img: np.ndarray,
    width_px: float,
    rough: float,
    flare: float = 0.0,
    corner: float = 0.0,
    paper: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tone: np.ndarray | None = None,
) -> np.ndarray:
    """
    Filed-out negative carrier: the rebate prints between the film gate and the filed
    aperture, inside a margin of bare paper (scene-linear, matching the mat). The filed
    edge's penumbra and the flare are exposure, read through tone (rebate_tone).
    """
    if width_px <= 0.0:
        return img

    h, w = img.shape[:2]
    tone = linear_carrier_tone() if tone is None else np.asarray(tone, dtype=np.float32)
    band = (
        int(
            np.ceil(
                width_px
                * (
                    CARRIER_MARGIN
                    + CARRIER_CORNER * corner
                    + 1.0
                    + CARRIER_JITTER
                    + CARRIER_NOISE_OUTER
                    + CARRIER_NOISE_INNER
                    + max(abs(CARRIER_OFFSET_X), abs(CARRIER_OFFSET_Y))
                )
                + max(1.0, width_px * CARRIER_SOFT)
            )
        )
        + 1
    )
    out = img.copy()
    if 2 * band >= h or 2 * band >= w:
        regions = [(0, h, 0, w, (0, 1, 2, 3))]
    else:
        # An edge's effect ends within `band`.
        hb, wb = h - band, w - band
        regions = [
            (0, band, 0, band, (0, 2)),
            (0, band, wb, w, (0, 3)),
            (hb, h, 0, band, (1, 2)),
            (hb, h, wb, w, (1, 3)),
            (0, band, band, wb, (0,)),
            (hb, h, band, wb, (1,)),
            (band, hb, 0, band, (2,)),
            (band, hb, wb, w, (3,)),
        ]
    for y0, y1, x0, x1, sides in regions:
        rows = max(1, _CARRIER_BLOCK_PIXELS // max(1, x1 - x0))
        for b0 in range(y0, y1, rows):
            b1 = min(b0 + rows, y1)
            block = img[b0:b1, x0:x1]
            out[b0:b1, x0:x1] = _carrier_block(block, b0, x0, h, w, width_px, rough, flare, corner, paper, tone, sides)
    return np.clip(out, 0.0, 1.0)


def _carrier_block(
    img: np.ndarray,
    y0: int,
    x0: int,
    h: int,
    w: int,
    width_px: float,
    rough: float,
    flare: float,
    corner: float,
    paper: tuple[float, float, float],
    tone: np.ndarray,
    sides: tuple[int, ...],
) -> np.ndarray:
    profiles = carrier_profiles()
    soft = max(1.0, width_px * CARRIER_SOFT)
    soft_filed = max(1.0, width_px * CARRIER_FILED_SOFT)
    margin = width_px * CARRIER_MARGIN
    radius = width_px * CARRIER_CORNER * corner
    gate_radius = width_px * CARRIER_GATE_CORNER
    cell = max(1.0, width_px * CARRIER_NOISE_CELL)
    py = np.arange(y0, y0 + img.shape[0], dtype=np.float32)[:, None]
    px = np.arange(x0, x0 + img.shape[1], dtype=np.float32)[None, :]
    sx = (px + 0.5) / np.float32(w)
    sy = (py + 0.5) / np.float32(h)
    end_x = np.minimum(px, (w - 1.0) - px)
    end_y = np.minimum(py, (h - 1.0) - py)
    qx = px - np.float32(width_px * CARRIER_OFFSET_X)
    qy = py - np.float32(width_px * CARRIER_OFFSET_Y)
    fend_x = np.minimum(qx, (w - 1.0) - qx)
    fend_y = np.minimum(qy, (h - 1.0) - qy)
    n2 = carrier_noise(px / cell, py / cell)

    def prof(row: int, s: np.ndarray) -> np.ndarray:
        return profiles[row, np.minimum((s * CARRIER_SAMPLES).astype(np.int32), CARRIER_SAMPLES - 1)]

    def arc(r: float, at: float, end: np.ndarray) -> np.ndarray | np.float32:
        """Corner retreat of a boundary `at` px from the print edge, from its own corner."""
        if r <= 0.0:
            return np.float32(0.0)
        x = np.clip(r - (end - at), 0.0, r)
        return r - np.sqrt(np.maximum(r * r - x * x, 0.0))

    def bounds(edge: int, s: np.ndarray, end: np.ndarray, fend: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outer = margin + width_px * rough * (CARRIER_OUTER_JITTER * prof(edge + 4, s) + CARRIER_NOISE_OUTER * n2)
        wobble = CARRIER_JITTER * CARRIER_INNER_ROUGH * prof(edge, s) + CARRIER_NOISE_INNER * n2
        inner = margin + width_px * (1.0 + wobble) + arc(gate_radius, margin + width_px, end)
        return outer + arc(radius, margin, fend), inner

    # (edge, profile position, gate end, filed end, gate distance, filed distance)
    edges = (
        (0, sx, end_x, fend_x, py, qy),
        (1, sx, end_x, fend_x, (h - 1.0) - py, (h - 1.0) - qy),
        (2, sy, end_y, fend_y, px, qx),
        (3, sy, end_y, fend_y, (w - 1.0) - px, (w - 1.0) - qx),
    )
    reach = max(1.0, width_px * CARRIER_FLARE_DEPTH)
    # Floored |noise|, not a one-sided gate: that left whole edges with no flare.
    flare_amp = flare * CARRIER_FLARE_GAIN * (CARRIER_FLARE_BASE + (1.0 - CARRIER_FLARE_BASE) * np.abs(n2))
    a_in = np.ones(img.shape[:2], dtype=np.float32)
    a_out = np.ones(img.shape[:2], dtype=np.float32)
    peaks, gates = [], []
    for e, s, end, fend, d, fd in (edges[k] for k in sides):
        outer, inner = bounds(e, s, end, fend)
        a_in = a_in * np.clip((d - inner) / soft + 0.5, 0.0, 1.0)
        a_out = a_out * np.clip((fd - outer) / soft_filed + 0.5, 0.0, 1.0)
        peaks.append(np.clip(1.0 - np.abs(fd - outer) / reach, 0.0, 1.0) ** 2)
        gates.append(np.clip(1.0 + (fd - outer) / reach, 0.0, 1.0))
    lit = np.float32(0.0)
    if flare > 0.0:
        # The bevel spans only the aperture, so the other edges gate each edge's flare.
        for e in range(len(peaks)):
            lit = lit + peaks[e] * np.prod([gates[k] for k in range(len(gates)) if k != e] or [np.float32(1.0)], axis=0)
        lit = flare_amp * lit

    rebate = np.asarray(paper, dtype=np.float32) * carrier_tone_lookup(tone, a_out + lit)
    shown = (a_in * a_out)[..., None]
    return img * shown + rebate * (1.0 - shown)


_RNG = np.random.default_rng(29)
_TONES = [
    linear_carrier_tone(),
    np.ascontiguousarray(np.clip(linear_carrier_tone() * _RNG.uniform(0.8, 1.1, (64, 3)).astype(np.float32), 0, 1.2)),
]


@pytest.mark.parametrize("shape", [(300, 400), (120, 90), (40, 30)])
@pytest.mark.parametrize("width, rough, flare, corner, tone", [(12.0, 0.5, 0.0, 0.0, 0), (20.0, 1.0, 0.7, 0.5, 1), (7.3, 0.2, 1.0, 1.0, 1)])
def test_carrier_matches_numpy(shape, width, rough, flare, corner, tone):
    img = _RNG.random((*shape, 3), dtype=np.float32) * np.float32(1.2)
    paper = (0.98, 0.97, 0.95)
    want = _carrier_reference(img, width, rough, flare, corner, paper, _TONES[tone])
    assert apply_carrier(img, width, rough, flare, corner, paper, _TONES[tone]).tobytes() == want.tobytes()
