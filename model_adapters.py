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
from ultralytics.engine.results import Results

from model_contract import (
    build_class_id_map,
    select_mask2former_candidates,
    validate_cell_model_contract,
)


NAMES = {0: "celula_enferma", 1: "celula_sana"}
_INFERENCE_LOCK = Lock()


class CentroidIoUTracker:
    """Greedy box-IoU association; per-track class votes tolerate label flicker."""

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
                for track_id, (old_box, _, last_seen) in self.tracks.items()
                if self.frame - last_seen <= self.max_age
                for index, box in enumerate(boxes)
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


def tracker_ids_by_detection(tracker_rows: Any, detection_count: int) -> np.ndarray:
    """Map Ultralytics tracker rows back to the segmentation detections.

    BoT-SORT/ByteTrack return rows in ``xyxy, id, score, class, detection_idx``
    order.  Keeping that index mapping explicit is important because the UI
    renders masks in the model's original detection order.
    """
    ids = np.full(max(0, int(detection_count)), -1, dtype=np.int64)
    rows = np.asarray(tracker_rows, dtype=np.float32)
    if rows.size == 0:
        return ids
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    for row in rows:
        if len(row) < 8 or not np.isfinite(row[[4, -1]]).all():
            continue
        detection_index = int(row[-1])
        if 0 <= detection_index < len(ids):
            ids[detection_index] = int(row[4])
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
        tracker_ids: np.ndarray | None = None,
    ) -> Results:
        h, w = image.shape[:2]
        rows: list[list[float]] = []
        valid_masks: list[np.ndarray] = []
        valid_classes: list[int] = []
        valid_scores: list[float] = []
        valid_track_ids: list[int] = []
        for index, (mask, cls, score) in enumerate(zip(masks, classes, scores)):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            valid_masks.append(mask)
            valid_classes.append(int(cls))
            valid_scores.append(float(score))
            if tracker_ids is not None:
                valid_track_ids.append(int(tracker_ids[index]))
            rows.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

        boxes_xyxy = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
        classes_array = np.asarray(valid_classes, dtype=np.float32)
        scores_array = np.asarray(valid_scores, dtype=np.float32)
        if tracked:
            track_ids = (
                np.asarray(valid_track_ids, dtype=np.int64)
                if tracker_ids is not None
                else (tracker or self.tracker).assign(boxes_xyxy, classes_array)
            )
            box_data = np.column_stack((boxes_xyxy, track_ids, scores_array, classes_array))
        else:
            box_data = np.column_stack((boxes_xyxy, scores_array, classes_array))
        mask_array = np.asarray(valid_masks, dtype=np.uint8).reshape(-1, h, w)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.uint8)
        box_tensor = torch.as_tensor(box_data, dtype=torch.float32)
        return Results(orig_img=image, path="", names=self.names, boxes=box_tensor, masks=mask_tensor)

    def predict(self, image: np.ndarray, conf: float = 0.35, **_: Any) -> list[Results]:
        masks, classes, scores = self._infer(image, conf)
        return [self._results(image, masks, classes, scores)]

    def track(
        self, source: str | Path | np.ndarray, stream: bool = False, persist: bool = True,
        conf: float = 0.35, vid_stride: int = 1, tracker_state: Any = None,
        tracker: str | Path | None = None, **kwargs: Any,
    ) -> Iterator[Results] | list[Results]:
        if not persist:
            self.reset_tracker()
            if tracker_state is not None:
                for attribute in (
                    "_cedia_tracker",
                    "_cedia_tracker_args",
                    "_cedia_tracker_config",
                ):
                    if hasattr(tracker_state, attribute):
                        delattr(tracker_state, attribute)
        tracker_instance, tracker_args = self._resolve_tracker(tracker, tracker_state)

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
                inference_confidence = conf
                if tracker_args is not None:
                    # ByteTrack's second association stage needs detections below
                    # the high threshold; keep those only when the tracker links
                    # them to an existing identity.
                    inference_confidence = min(
                        conf, float(getattr(tracker_args, "track_low_thresh", conf))
                    )
                masks, classes, scores = self._infer(frame, inference_confidence)
                masks, classes, scores = self._nonempty_candidates(
                    frame, masks, classes, scores
                )
                if tracker_args is None:
                    yield self._results(
                        frame, masks, classes, scores, tracked=True,
                        tracker=tracker_instance,
                    )
                    continue

                detections = self._results(frame, masks, classes, scores)
                tracker_rows = tracker_instance.update(detections.boxes, img=frame)
                ids = tracker_ids_by_detection(tracker_rows, len(scores))
                keep = (scores >= conf) | (ids >= 0)
                yield self._results(
                    frame, masks[keep], classes[keep], scores[keep],
                    tracked=True, tracker_ids=ids[keep],
                )

        if stream or isinstance(source, (str, Path)):
            return generate()
        return list(generate())

    @staticmethod
    def _nonempty_candidates(
        image: np.ndarray, masks: np.ndarray, classes: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize masks and remove empty candidates before tracker indexing."""
        height, width = image.shape[:2]
        valid_masks: list[np.ndarray] = []
        valid_classes: list[int] = []
        valid_scores: list[float] = []
        for mask, class_id, score in zip(masks, classes, scores):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not mask.any():
                continue
            valid_masks.append(mask)
            valid_classes.append(int(class_id))
            valid_scores.append(float(score))
        return (
            np.asarray(valid_masks, dtype=bool).reshape(-1, height, width),
            np.asarray(valid_classes, dtype=int),
            np.asarray(valid_scores, dtype=float),
        )

    @staticmethod
    def _load_tracker(tracker_config: str | Path) -> tuple[Any, Any]:
        """Load the requested Ultralytics tracker from its YAML configuration."""
        from ultralytics.trackers import BOTSORT, BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        config_path = check_yaml(str(tracker_config))
        config = YAML.load(config_path)
        tracker_type = str(config.get("tracker_type", "")).lower()
        tracker_class = {"botsort": BOTSORT, "bytetrack": BYTETracker}.get(tracker_type)
        if tracker_class is None:
            raise ValueError(f"Tipo de tracker Ultralytics no compatible: {tracker_type!r}.")
        args = IterableSimpleNamespace(**config)
        return tracker_class(args), args

    def _resolve_tracker(
        self, tracker_config: str | Path | None, tracker_state: Any
    ) -> tuple[Any, Any | None]:
        if tracker_config is None:
            tracker = (
                getattr(tracker_state, "_cedia_tracker", None)
                if tracker_state is not None
                else None
            )
            if tracker is None or not isinstance(tracker, CentroidIoUTracker):
                tracker = CentroidIoUTracker()
                if tracker_state is not None:
                    setattr(tracker_state, "_cedia_tracker", tracker)
                    setattr(tracker_state, "_cedia_tracker_config", None)
            return tracker, None

        config_key = str(Path(tracker_config).expanduser())
        tracker = (
            getattr(tracker_state, "_cedia_tracker", None)
            if tracker_state is not None
            else None
        )
        tracker_args = (
            getattr(tracker_state, "_cedia_tracker_args", None)
            if tracker_state is not None
            else None
        )
        existing_key = (
            getattr(tracker_state, "_cedia_tracker_config", None)
            if tracker_state is not None
            else None
        )
        if tracker is None or existing_key != config_key or tracker_args is None:
            tracker, tracker_args = self._load_tracker(tracker_config)
            if tracker_state is not None:
                setattr(tracker_state, "_cedia_tracker", tracker)
                setattr(tracker_state, "_cedia_tracker_args", tracker_args)
                setattr(tracker_state, "_cedia_tracker_config", config_key)
        return tracker, tracker_args

    def _infer(self, image: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class Mask2FormerAdapter(SegmentationAdapter):
    """CEDIA's fine-tuned Mask2Former Swin-S (best validation checkpoint)."""

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
        self._cell_health_device = str(self.device)
        self._cell_health_input_size = f"{self.INPUT_SIZE[1]}×{self.INPUT_SIZE[0]}"
        self._cell_health_num_queries = int(config.num_queries)
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
            probabilities_mask = F.interpolate(
                output.masks_queries_logits.float(), size=image.shape[:2],
                mode="bilinear", align_corners=False,
            )[0].sigmoid()
            masks = probabilities_mask > 0.5
            mask_conf = (probabilities_mask * masks).sum((1, 2)) / masks.sum((1, 2)).clamp(min=1)
            candidates = select_mask2former_candidates(
                probabilities.detach().cpu().tolist(),
                mask_conf.detach().cpu().tolist(),
                confidence,
            )
            query_ids = [query_id for query_id, _, _ in candidates]
            classes = np.asarray([class_id for _, class_id, _ in candidates], dtype=int)
            scores = np.asarray([score for _, _, score in candidates], dtype=float)
            query_indices = torch.as_tensor(query_ids, dtype=torch.long, device=masks.device)
            masks_np = masks[query_indices].cpu().numpy()
        return masks_np, classes, scores


class RFDETRSegAdapter(SegmentationAdapter):
    """CEDIA's fine-tuned RF-DETR-Seg 2XL, used by the live-camera mode."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from rfdetr import RFDETRSeg2XLarge

        # Rebuild the architecture from the checkpoint's saved configuration.
        # Hardcoding num_queries/resolution here can partially load weights if
        # the training run used different values.
        self.model = RFDETRSeg2XLarge.from_checkpoint(str(checkpoint))
        self._cell_health_model_source = "RF-DETR-Seg 2XL · CEDIA WSSV v6"
        self._cell_health_model_path = str(checkpoint)
        model_context = getattr(self.model, "model", None)
        model_config = getattr(self.model, "model_config", None)
        runtime_device = getattr(model_context, "device", None)
        runtime_resolution = getattr(
            model_context,
            "resolution",
            getattr(model_config, "resolution", None),
        )
        runtime_queries = getattr(
            model_config,
            "num_queries",
            getattr(getattr(model_context, "args", None), "num_queries", None),
        )
        self._cell_health_device = str(runtime_device or "no expuesto por RF-DETR")
        self._cell_health_input_size = (
            f"{runtime_resolution}×{runtime_resolution}"
            if runtime_resolution
            else "configuración del checkpoint"
        )
        self._cell_health_num_queries = (
            str(runtime_queries) if runtime_queries is not None else "configuración del checkpoint"
        )
        names = getattr(model_context, "class_names", None)
        if names is None:
            names = getattr(self.model, "class_names", None)
        if names is None:
            # CEDIA v6 checkpoint's documented training order when metadata is absent.
            names = ["celula_enferma", "celula_sana"]
        self.class_map = build_class_id_map(names)
        validate_cell_model_contract(
            "segment",
            {source_id: self.names[target_id] for source_id, target_id in self.class_map.items()},
        )

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

