"""GPU/CPU parity for the crop tool's full-frame preview.

crop_preview_full shows the whole rotated frame, ignoring crop_rect, while the
crop tool is active. The GPU engine widens only its late-stage dispatch extent
(toning/finish/layout) to match -- the meter, the contrast mask and the
reported active_roi stay on the real crop, so this must render identically to
the CPU engine (which always computed the whole frame and only skips the
final CropProcessor slice) and to itself with the crop tool off.
"""

import unittest
from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.gpu.device import GPUDevice


def _cropped_and_warped_settings() -> WorkspaceConfig:
    s = WorkspaceConfig()
    return replace(
        s,
        geometry=replace(s.geometry, crop_rect=(0.15, 0.1, 0.8, 0.9), fine_rotation=1.5, converge_v=3.0),
    )


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestCropPreviewFullParity(unittest.TestCase):
    def _render(self, processor, settings, img, prefer_gpu, crop_preview_full):
        result, metrics = processor.run_pipeline(
            img,
            settings,
            "parity-src",
            render_size_ref=float(max(img.shape[:2])),
            prefer_gpu=prefer_gpu,
            readback_metrics=False,
            crop_preview_full=crop_preview_full,
        )
        arr = np.asarray(result.readback())[:, :, :3] if hasattr(result, "readback") else np.asarray(result)[:, :, :3]
        return arr.astype(np.float64), metrics

    def _img(self):
        rng = np.random.default_rng(0)
        h, w = 96, 128
        grad = np.linspace(0.05, 0.9, w, dtype=np.float32)
        img = np.repeat(grad[None, :], h, axis=0)
        img = np.stack([img, img * 0.95, img * 0.9], axis=-1)
        return np.ascontiguousarray(img + rng.uniform(0, 0.01, img.shape).astype(np.float32))

    def test_full_frame_matches_cpu_at_the_crop_tools_own_tolerance(self):
        """Same numerical gap the cropped (already-shipped) GPU path has against
        the CPU engine -- full_frame introduces nothing beyond that."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped_cpu, _ = self._render(processor, settings, img, prefer_gpu=False, crop_preview_full=False)
        cropped_gpu, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        cropped_tolerance = float(np.max(np.abs(cropped_cpu - cropped_gpu)))

        full_cpu, cpu_metrics = self._render(processor, settings, img, prefer_gpu=False, crop_preview_full=True)
        full_gpu, gpu_metrics = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=True)

        self.assertEqual(full_cpu.shape, img.shape)  # the whole frame, not the crop
        self.assertEqual(full_cpu.shape, full_gpu.shape)
        self.assertLessEqual(float(np.max(np.abs(full_cpu - full_gpu))), cropped_tolerance + 1e-9)
        # The overlay must still track the real crop, not the frame the render widened to.
        self.assertEqual(cpu_metrics["active_roi"], gpu_metrics["active_roi"])
        self.assertIsNotNone(cpu_metrics["active_roi"])

    def test_full_frame_off_still_crops_on_gpu(self):
        """full_frame is opt-in: nothing here reaches for it unasked."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped_gpu, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        self.assertNotEqual(cropped_gpu.shape, img.shape)

    def test_toggling_full_frame_is_picked_up_with_no_other_setting_change(self):
        """Entering/leaving the crop tool alone must resize the render -- this is the
        one case a bare WorkspaceConfig diff cannot see, since full_frame is a render
        parameter, not a config field."""
        from negpy.services.rendering.image_processor import ImageProcessor

        processor = ImageProcessor()
        if processor.engine_gpu is None:
            self.skipTest("GPU engine not initialised")
        settings = _cropped_and_warped_settings()
        img = self._img()

        cropped, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)
        full, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=True)
        back_to_cropped, _ = self._render(processor, settings, img, prefer_gpu=True, crop_preview_full=False)

        self.assertNotEqual(cropped.shape, full.shape)
        self.assertEqual(cropped.shape, back_to_cropped.shape)
