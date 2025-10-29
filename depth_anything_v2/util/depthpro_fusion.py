"""Utilities for fusing Depth Anything V2 predictions with DepthPro outputs.

This module provides a light-weight interface that can ingest DepthPro scale/
shift factors (either from a TorchScript checkpoint or from per-image metadata
files) and fuse them with DAV2 predictions.  It also exposes an evaluator that
computes metrics for agricultural/tractor scenarios so that the fusion formula
or maximum depth clipping can be tuned automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is available during runtime
    torch = None  # type: ignore


@dataclass
class ScaleShift:
    """Container storing scale/shift parameters and optional metadata."""

    scale: float
    shift: float
    depth_map: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)

    def with_updates(self, scale: Optional[float] = None, shift: Optional[float] = None) -> "ScaleShift":
        """Return a new instance with updated values while preserving metadata."""

        return ScaleShift(
            scale=self.scale if scale is None else scale,
            shift=self.shift if shift is None else shift,
            depth_map=self.depth_map,
            info=dict(self.info),
        )


class DepthProDistanceEstimator:
    """Loads DepthPro predictions and returns per-image scale/shift factors.

    The estimator supports two modes:

    1. TorchScript checkpoint: a lightweight wrapper around a scripted
       DepthPro distance estimator that outputs ``scale`` and ``shift``.
    2. Metadata directory: a folder containing ``.json``, ``.npz`` or
       ``.npy`` files with pre-computed scale/shift factors (optionally with a
       dense depth map prediction).

    Parameters
    ----------
    checkpoint_path:
        Optional path to a TorchScript model that accepts a normalised RGB
        image (shape ``1x3xHxW``) and returns scale/shift predictions.
    metadata_dir:
        Directory that stores per-image metadata files.  The files are matched
        by the base filename (without extension) of the input frame.
    default_scale / default_shift:
        Fallback parameters if no DepthPro output is available.  Defaults to a
        no-op transform (scale ``1.0`` / shift ``0.0``).
    device:
        Torch device string.  Falls back to ``cpu`` if Torch is unavailable or
        a checkpoint is not supplied.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        default_scale: float = 1.0,
        default_shift: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self.metadata_dir = metadata_dir
        self.default_scale = default_scale
        self.default_shift = default_shift

        self.model = None
        self.device = device or "cpu"

        if checkpoint_path:
            if torch is None:
                raise ImportError(
                    "Torch is required to load DepthPro checkpoints. "
                    "Install torch or provide metadata files instead."
                )

            self.model = torch.jit.load(checkpoint_path, map_location=self.device)
            self.model.eval()

    def estimate(self, image_bgr: np.ndarray, basename: str) -> ScaleShift:
        """Return DepthPro scale/shift factors for ``basename``.

        ``image_bgr`` is expected in OpenCV's BGR format so the method converts
        it to RGB before feeding into the TorchScript model (if available).
        """

        metadata = self._load_metadata(basename)
        if metadata is not None:
            return metadata

        if self.model is not None:
            image_tensor = self._image_to_tensor(image_bgr)
            with torch.no_grad():  # type: ignore[attr-defined]
                prediction = self.model(image_tensor)

            scale, shift, depth_map = self._parse_model_output(prediction)
            return ScaleShift(scale=scale, shift=shift, depth_map=depth_map, info={"source": "checkpoint"})

        return ScaleShift(scale=self.default_scale, shift=self.default_shift, info={"source": "default"})

    def _image_to_tensor(self, image_bgr: np.ndarray) -> "torch.Tensor":
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _parse_model_output(self, prediction) -> Tuple[float, float, Optional[np.ndarray]]:
        """Parse the flexible output formats of DepthPro checkpoints."""

        scale: Optional[float] = None
        shift: Optional[float] = None
        depth_map: Optional[np.ndarray] = None

        if isinstance(prediction, dict):
            for key in ("scale", "scales", "depth_scale"):
                if key in prediction:
                    value = prediction[key]
                    scale = float(value.item() if hasattr(value, "item") else value)
                    break

            for key in ("shift", "offset", "depth_shift"):
                if key in prediction:
                    value = prediction[key]
                    shift = float(value.item() if hasattr(value, "item") else value)
                    break

            for key in ("depth", "depth_map", "metric_depth"):
                if key in prediction:
                    value = prediction[key]
                    depth_map = self._tensor_to_numpy(value)
                    break

        elif isinstance(prediction, (tuple, list)):
            if len(prediction) >= 2:
                scale = float(self._maybe_to_scalar(prediction[0]))
                shift = float(self._maybe_to_scalar(prediction[1]))
            if len(prediction) >= 3:
                depth_map = self._tensor_to_numpy(prediction[2])

        else:  # single tensor
            tensor = prediction
            if tensor.ndim == 2:  # assume [2] => [scale, shift]
                scale = float(self._maybe_to_scalar(tensor[0]))
                shift = float(self._maybe_to_scalar(tensor[1]))
            elif tensor.ndim == 1 and tensor.numel() >= 2:  # pragma: no cover
                scale = float(self._maybe_to_scalar(tensor[0]))
                shift = float(self._maybe_to_scalar(tensor[1]))

        return (
            scale if scale is not None else self.default_scale,
            shift if shift is not None else self.default_shift,
            depth_map,
        )

    def _maybe_to_scalar(self, value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _tensor_to_numpy(self, value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).squeeze()

    def _load_metadata(self, basename: str) -> Optional[ScaleShift]:
        if not self.metadata_dir:
            return None

        for ext in (".json", ".npz", ".npy"):
            candidate = os.path.join(self.metadata_dir, basename + ext)
            if os.path.isfile(candidate):
                if ext == ".json":
                    with open(candidate, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    scale = float(payload.get("scale", payload.get("scales", self.default_scale)))
                    shift = float(payload.get("shift", payload.get("offset", self.default_shift)))
                    depth = payload.get("depth")
                    depth_map = np.asarray(depth).squeeze() if depth is not None else None
                elif ext == ".npz":
                    payload = np.load(candidate)
                    scale = float(payload.get("scale", payload.get("scales", self.default_scale)))
                    shift = float(payload.get("shift", payload.get("offset", self.default_shift)))
                    depth_map = payload.get("depth")
                    depth_map = np.asarray(depth_map).squeeze() if depth_map is not None else None
                else:  # .npy
                    arr = np.load(candidate, allow_pickle=False)
                    depth_map = None
                    if arr.ndim == 0:
                        scale = float(arr.item())
                        shift = self.default_shift
                    elif arr.ndim == 1:
                        scale = float(arr[0])
                        shift = float(arr[1]) if arr.size > 1 else self.default_shift
                    else:
                        scale = float(arr.flat[0])
                        shift = float(arr.flat[1]) if arr.size > 1 else self.default_shift

                info = {"source": "metadata", "path": candidate}
                return ScaleShift(scale=scale, shift=shift, depth_map=depth_map, info=info)

        return None


class DepthProFusion:
    """Fuse DAV2 depth with DepthPro predictions."""

    @staticmethod
    def apply(
        raw_depth: np.ndarray,
        scale_shift: ScaleShift,
        strategy: str = "scale_shift",
        hybrid_weight: float = 0.5,
    ) -> np.ndarray:
        """Apply the requested fusion strategy.

        Parameters
        ----------
        raw_depth:
            Depth Anything V2 raw prediction (float32 array).
        scale_shift:
            Scale/shift parameters obtained from ``DepthProDistanceEstimator``.
        strategy:
            ``"scale_shift"`` multiplies/adds the factors. ``"hybrid"`` blends
            the affine result with a dense DepthPro depth map (if available).
        hybrid_weight:
            Blend weight for the hybrid strategy.  ``0.0`` relies solely on DAV2
            while ``1.0`` returns the DepthPro depth map when available.
        """

        depth = raw_depth.astype(np.float32, copy=False)
        fused = depth * float(scale_shift.scale) + float(scale_shift.shift)

        if strategy == "hybrid":
            if scale_shift.depth_map is not None:
                depthpro_depth = np.asarray(scale_shift.depth_map, dtype=np.float32)
                depthpro_depth = _resize_if_needed(depthpro_depth, fused.shape)
                weight = np.clip(hybrid_weight, 0.0, 1.0)
                fused = (1.0 - weight) * fused + weight * depthpro_depth
        elif strategy not in ("scale_shift", "none"):
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

        return fused


class TractorScenarioEvaluator:
    """Evaluate absolute depth quality for agricultural/tractor scenes."""

    def __init__(
        self,
        gt_dir: Optional[str] = None,
        report_path: Optional[str] = None,
        percentile: float = 99.0,
        epsilon: float = 1e-6,
    ) -> None:
        self.gt_dir = gt_dir
        self.report_path = report_path
        self.percentile = percentile
        self.epsilon = epsilon

        self.metrics_history: Dict[str, list] = {"abs_rel": [], "rmse": [], "delta1": [], "max": []}
        self.gt_values: list = []

    def evaluate(
        self,
        basename: str,
        prediction: np.ndarray,
        max_depth: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        gt = self._load_ground_truth(basename)
        if gt is None:
            return None

        gt = _resize_if_needed(gt, prediction.shape)
        mask = np.isfinite(gt) & (gt > 0.0)
        if max_depth is not None:
            mask &= gt <= max_depth

        if not np.any(mask):
            return None

        pred = prediction[mask].astype(np.float32)
        truth = gt[mask].astype(np.float32)

        abs_rel = float(np.mean(np.abs(truth - pred) / np.maximum(truth, self.epsilon)))
        rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
        delta = float(np.mean(_delta_ratio(truth, pred, 1.25)))
        max_val = float(np.percentile(truth, self.percentile))

        self.metrics_history["abs_rel"].append(abs_rel)
        self.metrics_history["rmse"].append(rmse)
        self.metrics_history["delta1"].append(delta)
        self.metrics_history["max"].append(max_val)
        self.gt_values.append(truth)

        metrics = {"abs_rel": abs_rel, "rmse": rmse, "delta1": delta, "suggested_max_depth": max_val}
        return metrics

    def summarise(self) -> Optional[Dict[str, float]]:
        if not self.metrics_history["abs_rel"]:
            return None

        summary = {
            key: float(np.mean(values)) for key, values in self.metrics_history.items() if values
        }

        if self.gt_values:
            stacked = np.concatenate(self.gt_values)
            summary["global_suggested_max_depth"] = float(np.percentile(stacked, self.percentile))

        if self.report_path:
            directory = os.path.dirname(self.report_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary

    def solve_linear_regression(
        self,
        basename: str,
        raw_depth: np.ndarray,
        max_depth: Optional[float] = None,
    ) -> Optional[ScaleShift]:
        """Fit ``raw_depth * scale + shift`` to the ground truth using least squares."""

        gt = self._load_ground_truth(basename)
        if gt is None:
            return None

        gt = _resize_if_needed(gt, raw_depth.shape)
        mask = np.isfinite(gt) & (gt > 0)
        if max_depth is not None:
            mask &= gt <= max_depth

        if not np.any(mask):
            return None

        x = raw_depth[mask].astype(np.float64)
        y = gt[mask].astype(np.float64)

        A = np.stack([x, np.ones_like(x)], axis=1)
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        scale, shift = result

        return ScaleShift(scale=float(scale), shift=float(shift), info={"source": "linear_regression"})

    def _load_ground_truth(self, basename: str) -> Optional[np.ndarray]:
        if not self.gt_dir:
            return None

        for ext in (".npy", ".npz", ".png", ".exr"):
            candidate = os.path.join(self.gt_dir, basename + ext)
            if os.path.isfile(candidate):
                if ext == ".npy":
                    return np.load(candidate)
                if ext == ".npz":
                    payload = np.load(candidate)
                    if "depth" in payload:
                        return payload["depth"]
                    return payload[payload.files[0]]
                if ext == ".png":
                    image = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    if image.dtype == np.uint16:
                        return image.astype(np.float32) / 1000.0
                    return image.astype(np.float32)
                if ext == ".exr":
                    image = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    if image.ndim == 3:
                        image = image[..., 0]
                    return image.astype(np.float32)

        return None


def _delta_ratio(gt: np.ndarray, pred: np.ndarray, threshold: float) -> np.ndarray:
    ratio = np.maximum(gt / np.maximum(pred, 1e-6), pred / np.maximum(gt, 1e-6))
    return ratio < threshold


def _resize_if_needed(array: np.ndarray, target_shape: Iterable[int]) -> np.ndarray:
    target_h, target_w = target_shape[:2]
    if array.shape[0] == target_h and array.shape[1] == target_w:
        return array

    interpolation = cv2.INTER_NEAREST if array.dtype.kind in {"i", "u"} else cv2.INTER_LINEAR
    if array.ndim == 2:
        return cv2.resize(array, (target_w, target_h), interpolation=interpolation)

    channels = array.shape[2]
    resized = [
        cv2.resize(array[..., c], (target_w, target_h), interpolation=interpolation)
        for c in range(channels)
    ]
    return np.stack(resized, axis=2)

