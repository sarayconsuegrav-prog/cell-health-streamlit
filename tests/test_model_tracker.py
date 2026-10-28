import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from model_adapters import (
    CentroidIoUTracker,
    SegmentationAdapter,
    tracker_ids_by_detection,
)


class CentroidIoUTrackerTests(unittest.TestCase):
    def test_class_flicker_keeps_the_same_cell_identity(self):
        tracker = CentroidIoUTracker(min_iou=0.1)
        first_id = tracker.assign(
            np.asarray([[10, 10, 30, 30]], dtype=float),
            np.asarray([0]),
        )
        second_id = tracker.assign(
            np.asarray([[11, 10, 31, 30]], dtype=float),
            np.asarray([1]),
        )
        self.assertEqual(first_id.tolist(), [1])
        self.assertEqual(second_id.tolist(), [1])

    def test_one_track_cannot_claim_two_cells_in_the_same_frame(self):
        tracker = CentroidIoUTracker(min_iou=0.1)
        tracker.assign(np.asarray([[10, 10, 30, 30]], dtype=float), np.asarray([0]))
        ids = tracker.assign(
            np.asarray([[10, 10, 30, 30], [10, 10, 30, 30]], dtype=float),
            np.asarray([0, 0]),
        )
        self.assertEqual(len(set(ids.tolist())), 2)


class TrackerAdapterTests(unittest.TestCase):
    def test_tracker_output_indices_map_back_to_masks(self):
        ids = tracker_ids_by_detection(
            np.asarray([[1, 2, 11, 12, 42, 0.9, 1, 2]], dtype=float),
            detection_count=4,
        )
        self.assertEqual(ids.tolist(), [-1, -1, 42, -1])

    def test_empty_tracker_output_leaves_detections_untracked(self):
        ids = tracker_ids_by_detection(np.asarray([]), detection_count=3)
        self.assertEqual(ids.tolist(), [-1, -1, -1])

    def test_tracker_id_mapping_ignores_invalid_indices(self):
        ids = tracker_ids_by_detection(
            np.asarray(
                [
                    [1, 2, 11, 12, 8, 0.9, 1, 0],
                    [1, 2, 11, 12, 9, 0.8, 1, 4],
                ],
                dtype=float,
            ),
            detection_count=2,
        )
        self.assertEqual(ids.tolist(), [8, -1])

    def test_results_constructor_receives_raw_box_data_and_masks(self):
        adapter = SegmentationAdapter()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((1, 32, 32), dtype=bool)
        mask[0, 4:12, 5:14] = True
        result = adapter._results(
            image,
            mask,
            np.asarray([1]),
            np.asarray([0.9]),
        )
        self.assertEqual(result.boxes.xyxy.shape, (1, 4))
        self.assertEqual(result.boxes.cls.int().cpu().tolist(), [1])
        self.assertEqual(tuple(result.masks.data.shape), (1, 32, 32))
        tracked_result = adapter._results(
            image,
            mask,
            np.asarray([1]),
            np.asarray([0.9]),
            tracked=True,
            tracker_ids=np.asarray([42]),
        )
        self.assertEqual(tracked_result.boxes.id.int().cpu().tolist(), [42])

    def test_configured_botsort_preserves_ids_when_class_fluctuates(self):
        class DummyAdapter(SegmentationAdapter):
            def __init__(self):
                super().__init__()
                self.frame_index = 0

            def _infer(self, image, confidence):
                self.frame_index += 1
                masks = np.zeros((2, *image.shape[:2]), dtype=bool)
                masks[0, 10:30, 10 + self.frame_index : 30 + self.frame_index] = True
                masks[1, 50:70, 50:70] = True
                classes = np.asarray([self.frame_index % 2, 1])
                scores = np.asarray([0.9, 0.8])
                return masks, classes, scores

        repo_root = Path(__file__).resolve().parents[1]
        tracker_config = repo_root / "trackers" / "live_botsort.yaml"
        adapter = DummyAdapter()
        state = SimpleNamespace()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first = adapter.track(
            frame, tracker=tracker_config, tracker_state=state
        )[0]
        second = adapter.track(
            frame, tracker=tracker_config, tracker_state=state
        )[0]
        self.assertEqual(np.asarray(first.boxes.id).astype(int).tolist(), [1, 2])
        self.assertEqual(np.asarray(second.boxes.id).astype(int).tolist(), [1, 2])
        self.assertEqual(np.asarray(second.boxes.cls).astype(int).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
