"""
config.py — Central configuration for dotLumen CV Challenge 3.

All camera intrinsics are derived at runtime from the actual video
metadata — nothing is hardcoded that depends on resolution or FPS.
This ensures consistency between the detector (which operates in pixel
space) and the 3D estimator (which uses pixel-space intrinsics).

Camera: Intel RealSense D435i (RGB sensor)
  - Horizontal FOV: 69 degrees
  - Pixel size: 3 μm × 3 μm (global shutter)
  - Focal length physical: 1.93 mm

Ball: Standard FIFA size-5 football
  - Diameter: 0.22 m
  - Radius:   0.11 m
"""

import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical constants — do not change per video
# ---------------------------------------------------------------------------

# Intel RealSense D435i RGB sensor horizontal field of view (degrees)
D435I_FOV_H_DEG: float = 69.0
D435I_FOV_H_RAD: float = math.radians(D435I_FOV_H_DEG)

# Standard FIFA size-5 football dimensions (metres)
BALL_DIAMETER_M: float = 0.22
BALL_RADIUS_M:   float = BALL_DIAMETER_M / 2.0   # 0.11 m

# Basketball dimensions — used when evaluating on DeepSportRadar dataset
BASKETBALL_DIAMETER_M: float = 0.24
BASKETBALL_RADIUS_M:   float = BASKETBALL_DIAMETER_M / 2.0

# Gravitational acceleration (m/s²) — used for parabola validation
GRAVITY_MS2: float = 9.81


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# YOLOv8 confidence threshold below which we fall back to Hough
YOLO_CONFIDENCE_THRESHOLD: float = 0.40

# COCO class index for "sports ball"
YOLO_SPORTS_BALL_CLASS: int = 32

# YOLOv8 model variant — 's' is small: best speed/accuracy tradeoff
YOLO_MODEL_VARIANT: str = "yolov8s.pt"

# Hough Circle parameters (cv2.HoughCircles)
HOUGH_DP:         float = 1.2
HOUGH_MIN_DIST:   int   = 50    # minimum distance between circle centres
HOUGH_PARAM1:     int   = 100   # upper Canny threshold
HOUGH_PARAM2:     int   = 45    # accumulator threshold — higher = stricter
HOUGH_MIN_RADIUS: int   = 10    # pixels
HOUGH_MAX_RADIUS: int   = 60    # pixels — ball at 2-8m range: ~15-55px radius

# Hough spatial gate: reject Hough detections further than this from
# last Kalman position. Prevents Hough from latching onto background
# circles when ball is absent. Only active after Kalman is initialised.
HOUGH_MAX_DIST_FROM_KALMAN: float = 150.0   # pixels

# Minimum YOLO confidence to accept a detection as real during recovery
# (when ball reappears after a long miss). Lower than normal threshold
# to help with re-acquisition.
YOLO_RECOVERY_CONFIDENCE: float = 0.35


# ---------------------------------------------------------------------------
# Camera motion compensation
# ---------------------------------------------------------------------------

# Enable/disable optical flow camera motion compensation.
# When True, ball pixel positions are corrected for camera movement
# before being fed to the Kalman filter.
# Rationale: pixel trajectory ≠ world trajectory when camera moves.
# Method: Lucas-Kanade optical flow on background features + RANSAC homography
# References: Lucas & Kanade 1981; Shi & Tomasi 1994; Hartley & Zisserman Ch.4
ENABLE_CAMERA_COMPENSATION: bool = True

# ---------------------------------------------------------------------------
# Kalman validity / reset
# ---------------------------------------------------------------------------

# After this many consecutive missed frames, the Kalman prediction is
# considered unreliable and is frozen (velocity set to zero).
# Rationale: at 30fps, 20 missed frames = 0.67s. A ball cannot travel
# more than ~20m in 0.67s, so any prediction beyond this is physically
# implausible and will corrupt the spatial gate.
KALMAN_MAX_MISSED_FRAMES: int = 20

# Pixel bounds — Kalman position outside the frame is physically impossible.
# Used to detect runaway predictions and trigger a soft reset.
# Set to frame dimensions at runtime in CameraConfig.load().
KALMAN_RESET_ON_OOB: bool = True   # reset velocity when KF goes out of bounds


# ---------------------------------------------------------------------------
# Kalman filter parameters
# ---------------------------------------------------------------------------

# State vector: [cx, cy, vx, vy]
# Process noise — velocity diagonal higher to allow sudden kicks
KALMAN_Q_POSITION: float = 1.0    # pixels²
KALMAN_Q_VELOCITY: float = 10.0   # pixels²/frame²

# Measurement noise — how much we trust YOLO detections
KALMAN_R_POSITION: float = 5.0    # pixels²

# Initial state covariance — large = high uncertainty at start
KALMAN_P_INITIAL: float = 100.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Number of past positions to draw in trajectory overlay
TRAJECTORY_DEQUE_LENGTH: int = 60

# Colour scheme (BGR for OpenCV)
COLOR_DETECTION:   tuple = (0,   255,   0)    # green  — YOLO detection
COLOR_PREDICTION:  tuple = (0,   165, 255)    # orange — Kalman prediction
COLOR_TRAJECTORY:  tuple = (255, 255,   0)    # cyan   — trajectory trail
COLOR_TEXT:        tuple = (255, 255, 255)    # white  — overlay text


# ---------------------------------------------------------------------------
# Runtime intrinsics — populated by load_video_config()
# ---------------------------------------------------------------------------

class CameraConfig:
    """
    Holds all camera intrinsics derived from a specific video file.
    Must call load_video_config(video_path) before using any other module.
    """

    def __init__(self):
        self.frame_width:  int   = None
        self.frame_height: int   = None
        self.fps:          float = None
        self.dt:           float = None   # seconds per frame = 1 / fps
        self.fx:           float = None   # focal length in pixels (x)
        self.fy:           float = None   # focal length in pixels (y)
        self.ppx:          float = None   # principal point x (image centre)
        self.ppy:          float = None   # principal point y (image centre)
        self.total_frames: int   = None
        self.loaded:       bool  = False

    def load(self, video_path: str) -> None:
        """
        Read video metadata via OpenCV and derive all intrinsics.
        Must be called before any other module uses camera parameters.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video file: {video_path}"
            )

        self.frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.fps <= 0:
            raise ValueError(
                f"Video FPS read as {self.fps} — invalid. "
                "Cannot derive dt or Kalman time step."
            )

        self.dt  = 1.0 / self.fps

        # Focal length in pixels from pinhole geometry:
        # fx = (W/2) / tan(FOV_H / 2)
        # This is consistent with the D435i datasheet at any resolution
        # because FOV is fixed and pixel count scales linearly.
        self.fx  = (self.frame_width  / 2.0) / math.tan(D435I_FOV_H_RAD / 2.0)
        self.fy  = self.fx   # square pixels — valid for D435i RGB sensor

        # Principal point assumed at image centre.
        # D435i true principal point deviates by a few pixels — negligible
        # for the distances and accuracies in this challenge.
        self.ppx = self.frame_width  / 2.0
        self.ppy = self.frame_height / 2.0

        self.loaded = True

        logger.info(
            "CameraConfig loaded from %s:\n"
            "  Resolution : %dx%d\n"
            "  FPS        : %.2f  (dt=%.4f s)\n"
            "  fx=fy      : %.2f px\n"
            "  Principal  : (%.1f, %.1f)",
            video_path,
            self.frame_width, self.frame_height,
            self.fps, self.dt,
            self.fx,
            self.ppx, self.ppy,
        )

    def __repr__(self) -> str:
        if not self.loaded:
            return "CameraConfig(not loaded)"
        return (
            f"CameraConfig("
            f"{self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps, "
            f"fx={self.fx:.1f}, ppx={self.ppx:.1f}, ppy={self.ppy:.1f})"
        )


# Singleton instance — import this everywhere
camera = CameraConfig()