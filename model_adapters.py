"""Inference adapters for the CEDIA WSSV v6 winning instance-segmentation models.

They expose Ultralytics-compatible ``Results`` so the existing Streamlit UI can
render masks and calculate the same per-image measurements and video reports.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.engine.results import Boxes, Masks, Results

from model_contract import validate_cell_model_contract


NAMES = {0: "celula_enferma", 1: "celula_sana"}
_INFERENCE_LOCK = Lock()


class CentroidIoUTracker:
    """Small greedy IoU tracker for models that do not expose Ultralytics tracking."""

    def __init__(self, max_age: int = 12, min_iou: float = 0.15) -> None:
        self.max_age = max_age
        self.min_iou = min_iou
        self.next_id = 1
        self.frame = 0
        self.tracks: dict[int, tuple[np.ndarray, int, int]] = {}

    @staticmethod
    def _iou(left: np.ndarray, right: np.ndarray) -> float:
        x1, y1 = np.maximum(left[:2], right[:2])
        x2, y2 = np.minimum(left[2:], right[2:])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(area_left + area_right - intersection, 1e-6)

    def assign(self, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        self.frame += 1
        ids = np.full(len(boxes), -1, dtype=np.int64)
        candidates = sorted(
            (
                (self._iou(box, old_box), track_id, index)
                for track_id, (old_box, old_class, last_seen) in self.tracks.items()
                if self.frame - last_seen <= self.max_age
                for index, (box, cls) in enumerate(zip(boxes, classes))
                if int(cls) == old_class
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for overlap, track_id, index in candidates:
            if overlap < self.min_iou:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            ids[index] = track_id
            used_tracks.add(track_id)
            used_detections.add(index)
        for index, box in enumerate(boxes):
            if ids[index] < 0:
                ids[index] = self.next_id
                self.next_id += 1
            self.tracks[int(ids[index])] = (box.copy(), int(classes[index]), self.frame)
        self.tracks = {
            track_id: item
            for track_id, item in self.tracks.items()
            if self.frame - item[2] <= self.max_age
        }
        return ids


class SegmentationAdapter:
    task = "segment"
    names = NAMES

    def __init__(self) -> None:
        self.tracker = CentroidIoUTracker()
        self._cell_health_backend = "PyTorch · CEDIA WSSV v6"
        self._cell_health_model_names = NAMES.copy()
        self._cell_health_model_path = ""
        self._cell_health_model_source = ""

    def reset_tracker(self) -> None:
        self.tracker = CentroidIoUTracker()

    def _results(
        self, image: np.ndarray, masks: np.ndarray, classes: np.ndarray, scores: np.ndarray,
        tracked: bool = False, tracker: CentroidIoUTracker | None = None,
    ) -> Results:
        h, w = image.shape[:2]
        rows: list[list[float]] = []
        valid_masks: list[np.ndarray] = []
        valid_classes: list[int] = []
        valid_scores: list[float] = []
        for mask, cls, score in zip(masks, classes, scores):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            valid_masks.append(mask)
            valid_classes.append(int(cls))
            valid_scores.append(float(score))
            rows.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

        boxes_xyxy = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
        classes_array = np.asarray(valid_classes, dtype=np.float32)
        scores_array = np.asarray(valid_scores, dtype=np.float32)
        if tracked:
            track_ids = (tracker or self.tracker).assign(boxes_xyxy, classes_array)
            box_data = np.column_stack((boxes_xyxy, track_ids, scores_array, classes_array))
        else:
            box_data = np.column_stack((boxes_xyxy, scores_array, classes_array))
        boxes = Boxes(torch.as_tensor(box_data, dtype=torch.float32), orig_shape=(h, w))
        mask_array = np.asarray(valid_masks, dtype=np.uint8).reshape(-1, h, w)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.uint8)
        return Results(orig_img=image, path="", names=self.names, boxes=boxes, masks=Masks(mask_tensor, (h, w)))

    def predict(self, image: np.ndarray, conf: float = 0.35, **_: Any) -> list[Results]:
        masks, classes, scores = self._infer(image, conf)
        return [self._results(image, masks, classes, scores)]

    def track(
        self, source: str | Path | np.ndarray, stream: bool = False, persist: bool = True,
        conf: float = 0.35, vid_stride: int = 1, tracker_state: Any = None, **kwargs: Any,
    ) -> Iterator[Results] | list[Results]:
        if not persist:
            self.reset_tracker()
        tracker = getattr(tracker_state, "_cedia_tracker", None) if tracker_state is not None else None
        if tracker is None:
            tracker = CentroidIoUTracker()
            if tracker_state is not None:
                setattr(tracker_state, "_cedia_tracker", tracker)

        def generate() -> Iterator[Results]:
            if isinstance(source, np.ndarray):
                frames = (source,)
            else:
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError(f"No se pudo abrir el video: {source}")

                def read_frames() -> Iterator[np.ndarray]:
                    index = 0
                    try:
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            if index % max(1, int(vid_stride)) == 0:
                                yield frame
                            index += 1
                    finally:
                        capture.release()
                frames = read_frames()
            for frame in frames:
                masks, classes, scores = self._infer(frame, conf)
                yield self._results(frame, masks, classes, scores, tracked=True, tracker=tracker)

        if stream or isinstance(source, (str, Path)):
            return generate()
        return list(generate())

    def _infer(self, image: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class Mask2FormerAdapter(SegmentationAdapter):
    """CEDIA's fine-tuned Mask2Former Swin-S (best validation checkpoint)."""

    CHECKPOINT_URL = "https://github.com/sarayconsuegrav-prog/cell-health-streamlit/releases/download/wssv-v6-models/mask2former-mejor.pth"
    HF_MODEL = "facebook/mask2former-swin-small-coco-instance"
    INPUT_SIZE = (960, 1280)

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

        config = Mask2FormerConfig.from_pretrained(
            self.HF_MODEL, num_labels=2, num_queries=400,
            id2label={0: "celula_enferma", 1: "celula_sana"},
            label2id={"celula_enferma": 0, "celula_sana": 1},
        )
        self.model = Mask2FormerForUniversalSegmentation(config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1) * 255
        self._cell_health_model_source = "Mask2Former Swin-S · CEDIA WSSV v6"
        self._cell_health_model_path = str(checkpoint)

    def _infer(self, image: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        pixels = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        pixels = (pixels - self.mean) / self.std
        with _INFERENCE_LOCK, torch.inference_mode():
            output = self.model(pixel_values=pixels)
            probabilities = output.class_queries_logits[0].float().softmax(-1)[:, :-1]
            num_classes = probabilities.shape[1]
            scores, indices = probabilities.flatten().topk(min(1000, probabilities.numel()))
            queries, classes = indices // num_classes, indices % num_classes
            probabilities_mask = F.interpolate(
                output.masks_queries_logits.float(), size=image.shape[:2],
                mode="bilinear", align_corners=False,
            )[0].sigmoid()
            masks = probabilities_mask > 0.5
            mask_conf = (probabilities_mask * masks).sum((1, 2)) / masks.sum((1, 2)).clamp(min=1)
            scores = scores * mask_conf[queries]
            keep = scores >= confidence
            queries, classes, scores = queries[keep], classes[keep], scores[keep]
            masks_np = masks[queries].cpu().numpy()
        return masks_np, classes.cpu().numpy().astype(int), scores.cpu().numpy()


class RFDETRSegAdapter(SegmentationAdapter):
    """CEDIA's fine-tuned RF-DETR-Seg 2XL, used by the live-camera mode."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from rfdetr import RFDETRSeg2XLarge

        self.model = RFDETRSeg2XLarge(
            num_queries=500, num_select=500, resolution=1008,
            pretrain_weights=str(checkpoint),
        )
        self._cell_health_model_source = "RF-DETR-Seg 2XL · CEDIA WSSV v6"
        self._cell_health_model_path = str(checkpoint)
        names = getattr(self.model, "class_names", ["celula_enferma", "celula_sana"])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        self.class_map = {
            index: 0 if "enferm" in str(name).lower() or "sick" in str(name).lower() else 1
            for index, name in enumerate(names)
        }
        validate_cell_model_contract("segment", self.names)

    def _infer(self, image: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _INFERENCE_LOCK:
            detection = self.model.predict(rgb, threshold=confidence)
        raw_masks = getattr(detection, "mask", None)
        if raw_masks is None:
            return np.zeros((0, *image.shape[:2]), dtype=bool), np.array([], dtype=int), np.array([], dtype=float)
        masks = np.asarray(raw_masks, dtype=bool)
        classes = np.asarray(detection.class_id, dtype=int)
        scores = np.asarray(detection.confidence, dtype=float)
        classes = np.asarray([self.class_map.get(int(value), -1) for value in classes], dtype=int)
        keep = classes >= 0
        return masks[keep], classes[keep], scores[keep]

