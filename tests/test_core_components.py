import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(4)


def _mask(h, w, density):
    return ((_RNG.random((h, w)) < density) * 255).astype(np.uint8)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("shape", [(1, 1), (1, 57), (57, 1), (2, 2), (31, 47), (128, 200), (257, 129)])
@pytest.mark.parametrize("density", [0.0, 0.1, 0.4, 0.7, 1.0])
def test_connected_components_with_stats_match_cv2(connectivity, shape, density):
    mask = _mask(*shape, density)
    want = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    count, labels, stats, centroids = negpy_core.connected_components_with_stats(mask, connectivity)
    assert count == want[0]
    assert np.array_equal(labels, want[1])
    assert np.array_equal(stats, want[2])
    assert centroids.tobytes() == want[3].tobytes()
