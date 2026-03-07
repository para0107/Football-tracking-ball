"""
detector.py — Stage 1: 2D Ball Detection

Primary:  YOLOv8s pretrained on COCO (class 32 = sports ball)
Fallback: Hough Circle Transform when YOLO confidence < threshold

Why this design:
- YOLOv8 is the dominant method in sports ball detection literature
  (Moreira et al. 2025: 26/38 studies used or benchmarked YOLO)
- Hough circles are retained as fallback — geometrically precise for
  round objects, computationally lightweight, no model required
- Combining both is more robust than either alone

Localisation concern (Redmon et al. 2016):
  "YOLO struggles to precisely localise some objects, especially small ones."
  Mitigation: Hough fallback for low-confidence frames. radius_px smoothed
  downstream by Kalman filter. imgsz NOT increased (speed tradeoff rejected).

Output per frame:
  DetectionResult(cx, cy, radius_px, confidence, source)
  where source is 'yolo' | 'hough' | None (no detection)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_SPORTS_BALL_CLASS,
    YOLO_MODEL_VARIANT,
    HOUGH_DP,
    HOUGH_MIN_DIST,
    HOUGH_PARAM1,
    HOUGH_PARAM2,
    HOUGH_MIN_RADIUS,
    HOUGH_MAX_RADIUS,
    HOUGH_MAX_DIST_FROM_KALMAN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Single-frame ball detection output.

    cx, cy      : ball centre in pixel coordinates
    radius_px   : ball radius in pixels — critical for 3D depth estimation
                  Z = (fx * D_real) / (2 * radius_px)
    confidence  : YOLO confidence score, or 1.0 for Hough (no score available)
    source      : 'yolo' | 'hough' | None
    """
    cx:         float
    cy:         float
    radius_px:  float
    confidence: float
    source:     str

    @property
    def center(self) -> tuple:
        return (int(self.cx), int(self.cy))

    @property
    def bbox(self) -> tuple:
        """Return (x1, y1, x2, y2) bounding box."""
        r = self.radius_px
        return (
            int(self.cx - r), int(self.cy - r),
            int(self.cx + r), int(self.cy + r),
        )


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class BallDetector:
    """
    Two-stage ball detector:
      1. YOLOv8s — primary, deep learning based
      2. Hough Circle Transform — fallback for low-confidence frames

    Usage:
        detector = BallDetector()
        result = detector.detect(frame)   # returns DetectionResult or None
    """

    def __init__(self, model_path: str = YOLO_MODEL_VARIANT):
        self._model      = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model. Gracefully degrades to Hough-only if unavailable."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info("YOLOv8 model loaded: %s", self._model_path)
        except Exception as exc:
            logger.warning(
                "YOLOv8 unavailable (%s). Running Hough-only mode.", exc
            )
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray,
               kalman_pos: tuple = None,
               allow_hough: bool = True) -> Optional[DetectionResult]:
        """
        Detect ball in a single BGR frame.

        Args:
            frame:       BGR image from cv2.VideoCapture
            kalman_pos:  (cx, cy) of last Kalman estimate. When provided,
                         Hough detections far from this are rejected as noise.
            allow_hough: When False, skip Hough entirely. Used during long
                         miss streaks where Hough has no spatial anchor and
                         would latch onto background circles, corrupting
                         the Kalman with false velocities on re-acquisition.

        Returns DetectionResult if ball found, else None.
        """
        # --- Stage 1: YOLO ---
        yolo_result = self._detect_yolo(frame)

        if yolo_result is not None:
            logger.debug(
                "YOLO detection: center=(%.1f, %.1f) r=%.1f conf=%.2f",
                yolo_result.cx, yolo_result.cy,
                yolo_result.radius_px, yolo_result.confidence,
            )
            return yolo_result

        # --- Stage 2: Hough fallback ---
        if not allow_hough:
            logger.debug("Hough disabled (long miss streak — no spatial anchor).")
            return None

        hough_result = self._detect_hough(frame, kalman_pos=kalman_pos)

        if hough_result is not None:
            logger.debug(
                "Hough fallback: center=(%.1f, %.1f) r=%.1f",
                hough_result.cx, hough_result.cy, hough_result.radius_px,
            )
            return hough_result

        logger.debug("No detection in frame.")
        return None

    def detect_batch(self, frames: list) -> list:
        """Run detect() over a list of frames. Returns list of results."""
        return [self.detect(f) for f in frames]

    # ------------------------------------------------------------------
    # YOLO detection
    # ------------------------------------------------------------------

    def _detect_yolo(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Run YOLOv8 inference and extract the highest-confidence
        sports ball detection.

        Returns None if:
          - model is not loaded
          - no sports ball detected
          - best confidence < YOLO_CONFIDENCE_THRESHOLD
        """
        if self._model is None:
            return None

        try:
            results = self._model(
                frame,
                classes=[YOLO_SPORTS_BALL_CLASS],
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return None

        best_conf   = -1.0
        best_result = None

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])

                if cls != YOLO_SPORTS_BALL_CLASS:
                    continue
                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue
                if conf <= best_conf:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx        = (x1 + x2) / 2.0
                cy        = (y1 + y2) / 2.0
                radius_px = (x2 - x1) / 2.0   # width-based radius

                best_conf   = conf
                best_result = DetectionResult(
                    cx=cx,
                    cy=cy,
                    radius_px=radius_px,
                    confidence=conf,
                    source='yolo',
                )

        return best_result

    # ------------------------------------------------------------------
    # Hough Circle fallback
    # ------------------------------------------------------------------

    def _detect_hough(self, frame: np.ndarray,
                      kalman_pos: tuple = None) -> Optional[DetectionResult]:
        """
        Hough Circle Transform fallback.

        Advantages over YOLO for this fallback role:
          - Geometrically fits a circle rather than predicting a bbox
          - radius_px is more precise for round objects
          - Zero model inference overhead

        Disadvantages (why it's fallback not primary):
          - No semantic understanding — can match any circular object
          - Sensitive to lighting and edge quality
          - No confidence score

        Mitigation for false positives:
          - Raised param2 threshold (stricter accumulator)
          - Spatial gate: if kalman_pos provided, reject circles that are
            more than HOUGH_MAX_DIST_FROM_KALMAN pixels away.
            Rationale: if the ball was at position P in the last frame,
            it cannot have teleported 300px in one frame — so any circle
            far from P is background noise, not the ball.

        Pipeline:
          1. Convert to grayscale
          2. Gaussian blur to suppress noise
          3. HoughCircles detection
          4. Spatial gate filter (if Kalman position available)
          5. Return circle closest to Kalman position, or largest if no prior
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS,
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)

        # --- Spatial gate ---
        # If we have a Kalman position estimate, filter out any circle
        # that is too far away to plausibly be the ball.
        if kalman_pos is not None:
            kx, ky = kalman_pos
            filtered = []
            for c in circles:
                dist = float(np.sqrt((c[0] - kx)**2 + (c[1] - ky)**2))
                if dist <= HOUGH_MAX_DIST_FROM_KALMAN:
                    filtered.append((c, dist))

            if len(filtered) == 0:
                # All circles too far from Kalman — reject all as noise
                logger.debug(
                    "Hough: all %d circle(s) rejected by spatial gate "
                    "(max dist=%.0f px from Kalman pos (%.0f, %.0f))",
                    len(circles), HOUGH_MAX_DIST_FROM_KALMAN, kx, ky,
                )
                return None

            # Among spatially valid circles, pick the closest to Kalman
            best_c = min(filtered, key=lambda x: x[1])[0]

        else:
            # No Kalman position yet — pick the largest circle
            # (ball is typically the most prominent circle in early frames)
            best_c = max(circles, key=lambda c: c[2])

        cx, cy, radius_px = float(best_c[0]), float(best_c[1]), float(best_c[2])

        return DetectionResult(
            cx=cx,
            cy=cy,
            radius_px=radius_px,
            confidence=1.0,   # Hough has no confidence score
            source='hough',
        )