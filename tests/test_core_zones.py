"""Hash pins of the zone, loupe and vectorscope reads on fixed inputs."""

import numpy as np
from core_pin import digest, linux_pin

from negpy.features.exposure import analysis as A


def _frame(seed, shape=(900, 1300)):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 + 0.45 * np.sin(x / (w / 3.0) + seed) * np.cos(y / (h / 2.5))
    rgb = np.stack([base, base * 0.9, base * 1.05], axis=2) + rng.normal(0, 0.02, (h, w, 3))
    return np.clip(rgb, 0, 1).astype(np.float32)


@linux_pin
def test_zone_field_pin():
    assert digest([A.zone_field(_frame(seed)) for seed in range(3)]) == "4d24663a1fb8101b6d75beaba9136f1306e684b24efb3a49cf0790170c75008d"


@linux_pin
def test_zone_anchors_pin():
    assert (
        digest([A.zone_anchors(A.zone_field(_frame(seed))) for seed in range(4)])
        == "a3656e430a5519c33cf8227e0a9b595d7317ab8b0d825b696201c9159c78feb1"
    )


@linux_pin
def test_loupe_and_labels_pin():
    loupe = [A.loupe_acutance(_frame(seed, (60, 90))) for seed in range(3)]
    labels = [A.zone_region_labels(A.zone_grid(_frame(seed))) for seed in range(3)]
    assert digest(loupe, labels) == "bc377184b08546c822f0d3621ae2ee3a8298620d9042bd6d05ad45b0118905ce"


@linux_pin
def test_vectorscope_density_pin():
    counts = np.random.default_rng(2).integers(0, 50, (A.SCOPE_HUE_BINS, A.SCOPE_CHROMA_BINS)).astype(float)
    assert (
        digest([A.vectorscope_density(counts, size, log) for size, log in ((256, True), (180, False))])
        == "b8be1259a44f76609ce91135af8e94526f2dbc6e8bbe8b7eb966552220d9e5d3"
    )
