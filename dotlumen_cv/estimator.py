"""
estimator.py — Stage 2: 3D Ball Position Estimation

Method: Monocular depth estimation from known object size (pinhole model).

Core insight:
    We cannot recover depth from a single RGB image in general — it is
    mathematically underdetermined (one pixel maps to an infinite ray).
    However, if we know the real physical size of the object, apparent
    size in pixels gives us depth directly via similar triangles.

    A football has a fixed diameter of 0.22m (FIFA size 5).
    If it appears as radius_px pixels wide, and the camera has focal
    length fx pixels, then by similar triangles:

        radius_px / fx  =  (D_real/2) / Z

    Solving for Z (depth):

        Z = (fx * D_real) / (2 * radius_px)

    Then lateral world coordinates via back-projection:

        X = (cx - ppx) * Z / fx
        Y = (cy - ppy) * Z / fy

    And Euclidean distance from camera:

        distance = sqrt(X² + Y² + Z²)

Coordinate system (camera-relative, right-hand):
    X: positive = right of camera
    Y: positive = below camera centre
    Z: positive = in front of camera (depth)

References:
    Hartley & Zisserman — "Multiple View Geometry in Computer Vision"
        Ch. 2 (projective geometry), Ch. 6 (camera model)
        The projection equation: x = K[R|t]X
        Back-projection: X = K⁻¹ * x * Z

    Intel RealSense SDK 2.0 Projection Documentation
        https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0
        Validates our formula against D435i manufacturer implementation.

    Van Zandycke & De Vleeschouwer, CVPRW 2022
        "3D Ball Localization From a Single Calibrated Image"
        Baseline method on DeepSportRadar — our MAE comparison target.

Known limitations:
    - radius_px error propagates directly into depth error (1:1 ratio)
    - Assumes ball is a perfect sphere (FIFA size 5 = 0.22m diameter)
    - Assumes lens has no distortion (valid for D435i, stated in SDK docs)
    - Assumes principal point at image centre (small but non-zero error)
    - Does not account for ball spin or deformation during kick
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Position3D:
    """
    3D position of the ball relative to the camera.

    Coordinate system: camera-relative right-hand:
        X: metres, positive = right
        Y: metres, positive = downward
        Z: metres, positive = in front of camera (depth)

    All values are None if estimation was not possible
    (e.g. radius_px = 0, or frame was a miss).
    """
    frame:       int
    time_s:      float
    X:           Optional[float]   # lateral  (metres)
    Y:           Optional[float]   # vertical (metres)
    Z:           Optional[float]   # depth    (metres)
    distance:    Optional[float]   # Euclidean distance from camera (metres)
    radius_px:   Optional[float]   # detected radius in pixels (for audit)
    cx_px:       Optional[float]   # compensated ball centre x (pixels)
    cy_px:       Optional[float]   # compensated ball centre y (pixels)
    source:      str               # 'yolo' | 'hough' | 'kalman' | 'none'

    @property
    def valid(self) -> bool:
        return self.Z is not None and self.Z > 0

    def to_row(self) -> list:
        """Return as CSV row."""
        fmt = lambda v: f"{v:.4f}" if v is not None else ""
        return [
            self.frame,
            f"{self.time_s:.4f}",
            fmt(self.X),
            fmt(self.Y),
            fmt(self.Z),
            fmt(self.distance),
            fmt(self.radius_px),
            fmt(self.cx_px),
            fmt(self.cy_px),
            self.source,
        ]

    @staticmethod
    def csv_header() -> list:
        return [
            "frame", "time_s",
            "X_m", "Y_m", "Z_m", "distance_m",
            "radius_px", "cx_px", "cy_px",
            "source",
        ]


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class BallEstimator3D:
    """
    Estimates 3D ball position from 2D pixel detections using
    pinhole camera back-projection and known ball diameter.

    The estimator is stateless — it processes one frame at a time.
    Smoothing is handled upstream by the Kalman filter.

    Usage:
        estimator = BallEstimator3D(camera)
        pos = estimator.estimate(
            frame=42, time_s=1.4,
            cx_px=640, cy_px=360, radius_px=25,
            source='yolo'
        )
        print(pos.Z, pos.distance)
    """

    def __init__(self, camera_config):
        """
        Args:
            camera_config: CameraConfig instance (from config.py)
                           Must be loaded (camera.load(video_path) called).
        """
        if not camera_config.loaded:
            raise RuntimeError(
                "CameraConfig not loaded. Call camera.load(video_path) first."
            )
        self._cam = camera_config
        logger.info(
            "BallEstimator3D initialised: fx=%.1f ppx=%.1f ppy=%.1f",
            self._cam.fx, self._cam.ppx, self._cam.ppy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self,
                 frame:      int,
                 time_s:     float,
                 cx_px:      float,
                 cy_px:      float,
                 radius_px:  float,
                 source:     str = 'unknown',
                 ball_diameter_m: float = None) -> Position3D:
        """
        Estimate 3D ball position from a single 2D detection.

        Args:
            frame:          Frame index
            time_s:         Time in seconds from start of video
            cx_px:          Ball centre x in pixels (CMC-corrected)
            cy_px:          Ball centre y in pixels (CMC-corrected)
            radius_px:      Ball radius in pixels (Kalman-smoothed)
            source:         Detection source string for audit trail
            ball_diameter_m: Override ball diameter (m). If None, uses
                            config.BALL_DIAMETER_M (0.22m for football).
                            Set to 0.24m when evaluating on DeepSportRadar
                            basketball dataset.

        Returns:
            Position3D with X, Y, Z, distance in metres.
            Returns Position3D with all None values if radius_px <= 0.
        """
        from config import BALL_DIAMETER_M
        D_real = ball_diameter_m if ball_diameter_m is not None else BALL_DIAMETER_M

        if radius_px is None or radius_px <= 0:
            logger.debug(
                "Frame %d: invalid radius_px=%.2f — skipping 3D estimation",
                frame, radius_px or 0,
            )
            return Position3D(
                frame=frame, time_s=time_s,
                X=None, Y=None, Z=None, distance=None,
                radius_px=radius_px, cx_px=cx_px, cy_px=cy_px,
                source=source,
            )

        # --- Step 1: Recover depth Z from apparent ball size ---
        # Similar triangles:
        #   radius_px / fx  =  (D_real/2) / Z
        #   → Z = (fx * D_real/2) / radius_px
        # Equivalently with diameter:
        #   Z = (fx * D_real) / (2 * radius_px)
        Z = (self._cam.fx * D_real) / (2.0 * radius_px)

        # --- Step 2: Back-project centre pixel to world X, Y ---
        # Derived from the projection equations:
        #   u = fx * (X/Z) + ppx  →  X = (u - ppx) * Z / fx
        #   v = fy * (Y/Z) + ppy  →  Y = (v - ppy) * Z / fy
        X = (cx_px - self._cam.ppx) * Z / self._cam.fx
        Y = (cy_px - self._cam.ppy) * Z / self._cam.fy

        # --- Step 3: Euclidean distance from camera origin ---
        distance = math.sqrt(X**2 + Y**2 + Z**2)

        logger.debug(
            "Frame %d: r=%.1fpx → Z=%.2fm X=%.2fm Y=%.2fm dist=%.2fm",
            frame, radius_px, Z, X, Y, distance,
        )

        return Position3D(
            frame=frame, time_s=time_s,
            X=X, Y=Y, Z=Z, distance=distance,
            radius_px=radius_px, cx_px=cx_px, cy_px=cy_px,
            source=source,
        )

    def estimate_null(self, frame: int, time_s: float) -> Position3D:
        """Return an empty Position3D for missed frames."""
        return Position3D(
            frame=frame, time_s=time_s,
            X=None, Y=None, Z=None, distance=None,
            radius_px=None, cx_px=None, cy_px=None,
            source='none',
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def depth_sensitivity(self, radius_px: float,
                          delta_px: float = 1.0,
                          ball_diameter_m: float = None) -> dict:
        """
        Compute how much depth Z changes per pixel error in radius_px.

        This is the fundamental error propagation formula:
            dZ/dr = -Z / radius_px   (from Z = fx*D/2r, differentiating)

        So a 1px error in radius_px at distance Z causes:
            ΔZ = Z / radius_px   metres of depth error

        This is why Kalman smoothing of radius_px matters —
        it directly reduces 3D estimation error.

        Args:
            radius_px:      Current detected radius
            delta_px:       Radius error to analyse (default 1px)

        Returns:
            dict with Z, dZ_dr, depth_error_per_px, relative_error_pct
        """
        from config import BALL_DIAMETER_M
        D_real = ball_diameter_m or BALL_DIAMETER_M

        Z     = (self._cam.fx * D_real) / (2.0 * radius_px)
        dZ_dr = -Z / radius_px   # derivative dZ/dr

        return {
            "radius_px":          radius_px,
            "Z_m":                Z,
            "dZ_dr":              dZ_dr,
            "depth_error_m":      abs(dZ_dr) * delta_px,
            "relative_error_pct": abs(dZ_dr) * delta_px / Z * 100,
        }