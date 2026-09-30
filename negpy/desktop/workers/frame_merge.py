import gc
import os
import threading
from dataclasses import dataclass
from typing import List

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from negpy.domain.models import TiffCompression, WorkspaceConfig
from negpy.kernel.image.logic import calculate_file_hash
from negpy.services.export.frame_merge import MergeCancelled, decode_params, write_merged_frame
from negpy.services.rendering.image_processor import ImageProcessor


@dataclass(frozen=True)
class FrameMergeTask:
    """One assembled frame to merge. ``asset`` is a copy of its Film Strip entry."""

    asset: dict
    params: WorkspaceConfig
    out_path: str
    compression: TiffCompression
    kind: str


@dataclass(frozen=True)
class FrameMergeResult:
    asset: dict
    out_path: str
    kind: str = ""
    new_hash: str = ""
    error: str = ""


class FrameMergeWorker(QObject):
    """Writes and verifies each merged TIFF. It deletes nothing: the controller moves the
    source files to the Trash once the edits have followed the frames."""

    progress = pyqtSignal(int, int, str)  # current, total, label
    finished = pyqtSignal(list, bool)  # [FrameMergeResult], aborted

    def __init__(self) -> None:
        super().__init__()
        self._processor = ImageProcessor()
        self._cancel = threading.Event()

    @pyqtSlot()
    def cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot(list)
    def run(self, tasks: List[FrameMergeTask]) -> None:
        self._cancel.clear()
        results: List[FrameMergeResult] = []
        aborted = False
        try:
            for i, task in enumerate(tasks):
                if self._cancel.is_set():
                    aborted = True
                    break
                self.progress.emit(i + 1, len(tasks), os.path.basename(task.out_path))
                try:
                    # _load_source_f32, not _decode_oriented_f32: a stitch is assembled here,
                    # and a single-file frame passes straight through it unchanged.
                    params = decode_params(task.params, task.kind)
                    f32, _ir, _cs = self._processor._load_source_f32(task.asset["path"], params)
                    # The decode cannot be interrupted, so Abort pressed during it is honoured
                    # here — before anything is written, while the frame still has its sources.
                    if self._cancel.is_set():
                        raise MergeCancelled(os.path.basename(task.out_path))
                    write_merged_frame(f32, task.asset["path"], task.out_path, params, task.compression, self._cancel.is_set)
                    del f32
                    new_hash = calculate_file_hash(task.out_path)
                    if new_hash.startswith("err_"):
                        raise OSError(f"Could not read {os.path.basename(task.out_path)} back")
                    results.append(FrameMergeResult(task.asset, task.out_path, kind=task.kind, new_hash=new_hash))
                except MergeCancelled:
                    # No result at all: a cancelled frame is neither merged nor failed, and the
                    # controller only trashes sources for a result it was handed.
                    aborted = True
                    self._processor.release_source_cache()
                    break
                except Exception as e:
                    results.append(FrameMergeResult(task.asset, task.out_path, kind=task.kind, error=str(e)))
                # _load_source_f32 keeps its result in a single-slot cache; a stitch canvas
                # held there across the batch is the largest buffer in the app.
                self._processor.release_source_cache()
                gc.collect()
        finally:
            self.finished.emit(results, aborted)
