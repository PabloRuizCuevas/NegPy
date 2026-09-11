"""GPU engine's shared meter buffer: downsampled before fine rotation and keystone,
not after -- both are full-frame resamples whose cost scales with pixel count, and
only a meter reads the result, so warping the full-res crop first just to shrink it
away spends the expensive part on pixels the analysis never sees. Pure function, no GPU.
"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.services.rendering.gpu_engine import _build_analysis_source


class TestBuildAnalysisSource(unittest.TestCase):
    def setUp(self):
        self.geometry = WorkspaceConfig().geometry

    def test_fine_rotation_receives_the_downsampled_buffer(self):
        img = np.zeros((800, 800, 3), dtype=np.float32)
        geometry = replace(self.geometry, fine_rotation=2.0)
        seen = []
        with patch("negpy.services.rendering.gpu_engine.apply_fine_rotation", side_effect=lambda a, angle: (seen.append(a.shape), a)[1]):
            _build_analysis_source(img, geometry, None, 1.0, None, False, 200)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(max(seen[0][:2]), 200)

    def test_keystone_receives_the_downsampled_buffer(self):
        img = np.zeros((800, 800, 3), dtype=np.float32)
        geometry = replace(self.geometry, converge_v=5.0)
        seen = []
        with patch("negpy.services.rendering.gpu_engine.apply_keystone", side_effect=lambda a, v, h: (seen.append(a.shape), a)[1]):
            _build_analysis_source(img, geometry, None, 1.0, None, False, 200)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(max(seen[0][:2]), 200)

    def test_a_buffer_already_at_or_under_the_cap_is_unaffected(self):
        """No downsample needed: the warps still see the whole (cropped) buffer."""
        img = np.zeros((150, 150, 3), dtype=np.float32)
        geometry = replace(self.geometry, fine_rotation=2.0)
        seen = []
        with patch("negpy.services.rendering.gpu_engine.apply_fine_rotation", side_effect=lambda a, angle: (seen.append(a.shape), a)[1]):
            _build_analysis_source(img, geometry, None, 1.0, None, False, 200)
        self.assertEqual(seen[0][:2], (150, 150))

    def test_output_shape_matches_the_cap_regardless_of_warp_order(self):
        img = np.zeros((800, 400, 3), dtype=np.float32)
        geometry = replace(self.geometry, fine_rotation=3.0, converge_h=4.0)
        out, _ = _build_analysis_source(img, geometry, None, 1.0, None, False, 200)
        self.assertLessEqual(max(out.shape[:2]), 200)
