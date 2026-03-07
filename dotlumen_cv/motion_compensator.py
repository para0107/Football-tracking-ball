"""
motion_compensator.py — Camera Motion Compensation via Optical Flow

Problem:
    The pipeline tracks ball position in pixel space. When the camera
    moves (pan, tilt, shake), the ball's pixel coordinates change even
    if the ball is stationary in the world. This produces a false
    trajectory that is an artifact of camera motion, not ball motion.

    Observed pixel motion = Ball world motion + Camera motion + Detector noise

Solution:
    Estimate camera motion from background feature points between
    consecutive frames using Lucas-Kanade optical flow, then subtract
    the camera-induced displacement from the ball's detected position.

    This gives a camera-compensated ball position that represents
    true ball motion relative to the world, not the camera.

Method:
    1. Detect Shi-Tomasi corner features in the previous frame,
       masking out the region around the ball (ball features would
       corrupt the camera motion estimate).
    2. Track those features to the current frame using
       cv2.calcOpticalFlowPyrLK (pyramidal Lucas-Kanade).
    3. Filter out outliers using RANSAC homography estimation
       (cv2.findHomography). The homography H encodes the full
       camera motion including rotation, translation, and scale.
    4. Apply the inverse of H to transform the raw ball detection
       into a camera-compensated coordinate.

References:
    Lucas & Kanade (1981) — "An iterative image registration technique
        with an application to stereo vision." IJCAI.
    Shi & Tomasi (1994) — "Good features to track." CVPR.
    Hartley & Zisserman — "Multiple View Geometry" Ch. 4 (homography).
    OpenCV docs — calcOpticalFlowPyrLK, findHomography, goodFeaturesToTrack.

Known limitations:
    - Requires sufficient background texture for feature detection.
    - Fails if >50% of the frame is the moving ball or a person.
    - Assumes background is static (rigid scene). Dynamic backgrounds
      (crowd movement) introduce residual error.
    - At very high camera speed, features may be lost between frames.

Upgrade path:
    For more robust compensation in broadcast/drone footage, consider:
    - DeepFlow or FlowNet2 (dense optical flow)
    - Background subtraction + homography on the static layer
    - IMU-based compensation using the D435i's built-in IMU
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Shi-Tomasi corner detection parameters
_FEATURE_MAX_CORNERS:   int   = 200    # max background features to track
_FEATURE_QUALITY_LEVEL: float = 0.01   # minimum quality threshold
_FEATURE_MIN_DISTANCE:  int   = 10     # min distance between features (px)
_FEATURE_BLOCK_SIZE:    int   = 3      # neighbourhood for corner detection

# Lucas-Kanade optical flow parameters
_LK_WIN_SIZE:   Tuple = (21, 21)  # search window size
_LK_MAX_LEVEL:  int   = 3         # pyramid levels
_LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,    # max iterations
    0.01,  # epsilon
)

# RANSAC homography parameters
_RANSAC_REPROJ_THRESHOLD: float = 3.0   # pixels
_MIN_INLIERS:             int   = 10    # min inliers for valid homography

# Ball mask padding — exclude this radius around ball from feature detection
# Prevents ball motion from contaminating camera motion estimate
_BALL_MASK_PADDING: int = 40   # pixels beyond detected ball radius


# ---------------------------------------------------------------------------
# Main compensator class
# ---------------------------------------------------------------------------

class CameraMotionCompensator:
    """
    Compensates ball pixel positions for camera motion between frames.

    Usage:
        cmc = CameraMotionCompensator()
        cmc.initialise(first_frame)

        for frame in video:
            cx_raw, cy_raw = detector.detect(frame)

            # Compensate for camera motion
            cx_comp, cy_comp = cmc.compensate(frame, cx_raw, cy_raw,
                                               ball_radius_px)
            # cx_comp, cy_comp is now camera-motion-corrected

            cmc.update(frame)
    """

    def __init__(self):
        self._prev_gray:     Optional[np.ndarray] = None
        self._prev_features: Optional[np.ndarray] = None
        self._last_H:        Optional[np.ndarray] = None  # last valid homography
        self._frame_count:   int   = 0
        self._compensation_applied: int = 0
        self._compensation_failed:  int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(self, frame: np.ndarray) -> None:
        """
        Must be called on the first frame before any compensation.
        Detects initial background feature points.
        """
        gray = self._to_gray(frame)
        self._prev_gray     = gray
        self._prev_features = self._detect_features(gray, ball_cx=None,
                                                     ball_cy=None,
                                                     ball_r=None)
        self._frame_count   = 1
        logger.debug(
            "CMC initialised: %d background features detected.",
            len(self._prev_features) if self._prev_features is not None else 0,
        )

    def compensate(self,
                   curr_frame: np.ndarray,
                   ball_cx: float,
                   ball_cy: float,
                   ball_radius_px: float = 20.0
                   ) -> Tuple[float, float]:
        """
        Apply camera motion compensation to a raw ball detection.

        Args:
            curr_frame:    Current BGR video frame
            ball_cx:       Raw detected ball centre x (pixels)
            ball_cy:       Raw detected ball centre y (pixels)
            ball_radius_px: Detected ball radius — used to mask ball
                            region from background feature detection

        Returns:
            (cx_compensated, cy_compensated) — ball position with
            camera motion subtracted. If compensation fails, returns
            the original (ball_cx, ball_cy) unchanged.
        """
        if self._prev_gray is None:
            logger.warning("CMC not initialised — call initialise() first.")
            return ball_cx, ball_cy

        curr_gray = self._to_gray(curr_frame)

        # --- Estimate camera homography from background features ---
        H = self._estimate_homography(curr_gray, ball_cx, ball_cy,
                                      ball_radius_px)

        if H is not None:
            self._last_H = H
            # Apply inverse homography to ball position
            cx_comp, cy_comp = self._apply_inverse_homography(
                H, ball_cx, ball_cy
            )
            self._compensation_applied += 1

            displacement = np.sqrt(
                (cx_comp - ball_cx)**2 + (cy_comp - ball_cy)**2
            )
            logger.debug(
                "CMC applied: raw=(%.1f,%.1f) → comp=(%.1f,%.1f) "
                "Δ=%.1f px",
                ball_cx, ball_cy, cx_comp, cy_comp, displacement,
            )
            return cx_comp, cy_comp

        else:
            # Homography failed — return raw position unchanged
            self._compensation_failed += 1
            logger.debug(
                "CMC failed (insufficient inliers) — using raw position."
            )
            return ball_cx, ball_cy

    def update(self, curr_frame: np.ndarray,
               ball_cx: float = None,
               ball_cy: float = None,
               ball_radius_px: float = None) -> None:
        """
        Advance the compensator to the next frame.
        Must be called after compensate() each frame.

        Args:
            curr_frame:    Current BGR frame (becomes prev_frame next step)
            ball_cx/cy:    Ball position for masking (None = no mask)
            ball_radius_px: Ball radius for masking
        """
        curr_gray = self._to_gray(curr_frame)

        # Refresh feature points every frame, masking ball region
        self._prev_features = self._detect_features(
            curr_gray,
            ball_cx=ball_cx,
            ball_cy=ball_cy,
            ball_r=ball_radius_px,
        )
        self._prev_gray   = curr_gray
        self._frame_count += 1

    def summary(self) -> dict:
        """Return compensation statistics."""
        total = self._compensation_applied + self._compensation_failed
        return {
            "frames_processed":       self._frame_count,
            "compensation_applied":   self._compensation_applied,
            "compensation_failed":    self._compensation_failed,
            "success_rate":           (self._compensation_applied / total
                                       if total > 0 else 0.0),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _detect_features(self,
                         gray: np.ndarray,
                         ball_cx: Optional[float],
                         ball_cy: Optional[float],
                         ball_r:  Optional[float]) -> Optional[np.ndarray]:
        """
        Detect Shi-Tomasi corner features in background regions.

        Creates a mask that excludes the ball region to prevent ball
        motion from contaminating the camera motion estimate.
        """
        h, w = gray.shape

        # Build exclusion mask — all ones (include everything) by default
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Exclude ball region if position is known
        if ball_cx is not None and ball_cy is not None:
            r = int((ball_r or 20) + _BALL_MASK_PADDING)
            cv2.circle(
                mask,
                (int(ball_cx), int(ball_cy)),
                r,
                0,      # exclude (black)
                -1,     # filled
            )

        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=_FEATURE_MAX_CORNERS,
            qualityLevel=_FEATURE_QUALITY_LEVEL,
            minDistance=_FEATURE_MIN_DISTANCE,
            blockSize=_FEATURE_BLOCK_SIZE,
            mask=mask,
        )

        if features is None:
            logger.debug("CMC: no background features detected.")
            return None

        return features.reshape(-1, 1, 2).astype(np.float32)

    def _estimate_homography(self,
                             curr_gray: np.ndarray,
                             ball_cx: float,
                             ball_cy: float,
                             ball_radius_px: float
                             ) -> Optional[np.ndarray]:
        """
        Estimate camera motion homography from background feature tracking.

        Steps:
        1. Track previous background features to current frame (LK flow)
        2. Filter by tracking status (keep only successfully tracked)
        3. Estimate homography with RANSAC (rejects outlier features)
        4. Return H if enough inliers, else None

        The homography H maps points in the previous frame to points
        in the current frame:  p_curr = H * p_prev
        """
        if (self._prev_gray is None or
                self._prev_features is None or
                len(self._prev_features) < _MIN_INLIERS):
            return None

        # --- Track features forward using Lucas-Kanade ---
        curr_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            self._prev_features,
            None,
            winSize=_LK_WIN_SIZE,
            maxLevel=_LK_MAX_LEVEL,
            criteria=_LK_CRITERIA,
        )

        if curr_features is None:
            return None

        # Keep only successfully tracked features
        good_mask    = (status.ravel() == 1)
        prev_good    = self._prev_features[good_mask].reshape(-1, 2)
        curr_good    = curr_features[good_mask].reshape(-1, 2)

        if len(prev_good) < _MIN_INLIERS:
            logger.debug(
                "CMC: only %d features tracked (need %d)",
                len(prev_good), _MIN_INLIERS,
            )
            return None

        # --- Estimate homography with RANSAC ---
        H, inlier_mask = cv2.findHomography(
            prev_good,
            curr_good,
            cv2.RANSAC,
            _RANSAC_REPROJ_THRESHOLD,
        )

        if H is None:
            return None

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if n_inliers < _MIN_INLIERS:
            logger.debug(
                "CMC: homography has only %d inliers (need %d)",
                n_inliers, _MIN_INLIERS,
            )
            return None

        logger.debug(
            "CMC: homography estimated from %d/%d inliers",
            n_inliers, len(prev_good),
        )
        return H

    @staticmethod
    def _apply_inverse_homography(H: np.ndarray,
                                  cx: float,
                                  cy: float) -> Tuple[float, float]:
        """
        Apply the inverse of homography H to point (cx, cy).

        H maps prev_frame → curr_frame (forward camera motion).
        H_inv maps curr_frame → prev_frame coordinate system.

        Applying H_inv to the ball's current pixel position removes
        the camera-induced displacement, leaving only the ball's
        motion relative to the background.

        Homogeneous coordinate arithmetic:
            p = [cx, cy, 1]ᵀ
            p_comp = H_inv * p
            result = (p_comp[0]/p_comp[2], p_comp[1]/p_comp[2])
        """
        H_inv = np.linalg.inv(H)
        p     = np.array([cx, cy, 1.0], dtype=float)
        p_comp = H_inv @ p

        # Normalise homogeneous coordinates
        if abs(p_comp[2]) < 1e-8:
            return cx, cy   # degenerate case — return unchanged

        return float(p_comp[0] / p_comp[2]), float(p_comp[1] / p_comp[2])