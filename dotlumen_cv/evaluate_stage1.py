"""
evaluate_stage1.py — Stage 1 Evaluation: 2D Ball Detection

Runs the detector on rgb.avi and produces:
  1. Output video with bounding boxes drawn (output_stage1.avi)
  2. Per-frame detection log CSV (detections_stage1.csv)
  3. Detection rate summary printed to console
  4. Kalman state visualisation plots (kalman_state_frameXXXX.png)

Also supports evaluation against DFL Soccer Ball Detection dataset
if ground truth annotations are provided (--gt_dir argument).

Usage:
    # Run on rgb.avi only (visual inspection):
    python evaluate_stage1.py --video rgb.avi

    # Run on rgb.avi with Kalman state plots:
    python evaluate_stage1.py --video rgb.avi --plot_kalman

    # Evaluate against DFL dataset ground truth:
    python evaluate_stage1.py --video rgb.avi --gt_dir /path/to/dfl/labels

References:
    Redmon et al. CVPR 2016 — YOLO original
    Moreira et al. PeerJ CS 2025 — ball detection survey (Fig. 3)
    Kalman R.E. ASME 1960 — Kalman filter
"""

import argparse
import csv
import logging
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging setup — use logging, not print, throughout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1")


# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------
from config import (
    camera,
    COLOR_DETECTION,
    COLOR_PREDICTION,
    COLOR_TRAJECTORY,
    COLOR_TEXT,
    TRAJECTORY_DEQUE_LENGTH,
    KALMAN_Q_POSITION,
    KALMAN_Q_VELOCITY,
    KALMAN_R_POSITION,
    KALMAN_P_INITIAL,
)
from detector import BallDetector, DetectionResult


# ---------------------------------------------------------------------------
# Minimal Kalman Filter (pure numpy — no filterpy dependency for portability)
# State: [cx, cy, vx, vy]
# ---------------------------------------------------------------------------

class SimpleKalmanFilter:
    """
    Linear Kalman Filter for 2D ball tracking.
    State vector: [cx, cy, vx, vy]

    This is a self-contained implementation using only numpy,
    so the stage 1 script runs without filterpy installed.
    The full tracker.py will use filterpy for cleaner code.

    References:
        Kalman R.E. (1960) — original paper
        kalmanfilter.net   — intuition and examples
        Wan & Van der Merwe (2000) — UKF upgrade path
    """

    def __init__(self, dt: float):
        self.dt = dt
        n = 4   # state dimension
        m = 2   # measurement dimension (cx, cy only)

        # State transition matrix — constant velocity model
        # x(t+1) = Φ · x(t) + process_noise
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        # Measurement matrix — we observe position only, not velocity
        # YOLO gives (cx, cy); velocity is inferred
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance Q
        # Velocity diagonal is higher to allow for sudden kicks
        # (Mitigation for nonlinearity — see discussion in docs)
        self.Q = np.diag([
            KALMAN_Q_POSITION,
            KALMAN_Q_POSITION,
            KALMAN_Q_VELOCITY,
            KALMAN_Q_VELOCITY,
        ])

        # Measurement noise covariance R
        # Reflects YOLO localisation uncertainty
        # Tune based on DeepSportRadar oracle evaluation (post-implementation)
        self.R = np.diag([KALMAN_R_POSITION, KALMAN_R_POSITION])

        # State estimate and covariance — uninitialised
        self.x = None           # [cx, cy, vx, vy]
        self.P = None           # 4x4 covariance matrix
        self.initialised = False

        # Store prior (after predict, before update) for visualisation
        self.x_prior = None
        self.P_prior = None

    def initialise(self, cx: float, cy: float) -> None:
        """Initialise state from first detection."""
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4) * KALMAN_P_INITIAL
        self.initialised = True
        logger.info(
            "Kalman filter initialised at (%.1f, %.1f)", cx, cy
        )

    def predict(self) -> np.ndarray:
        """
        Predict step — project state forward using physics model.
        Uncertainty GROWS here (P increases).

        Corresponds to Kalman (1960) Eq. (24):
          P*(t+1) = Φ* · P*(t) · Φ*ᵀ + Q(t)
        """
        self.x_prior = self.F @ self.x
        self.P_prior = self.F @ self.P @ self.F.T + self.Q
        self.x = self.x_prior
        self.P = self.P_prior
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step — correct prediction with new measurement.
        Uncertainty SHRINKS here (P decreases).

        Kalman Gain K determines the blend ratio:
          K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹

        Innovation = measurement - H · x_predicted
        If innovation is large → kick detected → filter adapts

        Corresponds to Kalman (1960) Eq. (25) — ∆*(t)
        """
        z = np.array(measurement, dtype=float).reshape(2, 1)

        # Innovation (residual)
        innovation = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()

    @property
    def position(self) -> tuple:
        """Current estimated position (cx, cy)."""
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    @property
    def velocity(self) -> tuple:
        """Current estimated velocity (vx, vy) in pixels/frame."""
        return (float(self.x[2, 0]), float(self.x[3, 0]))

    @property
    def position_uncertainty(self) -> float:
        """Standard deviation of position estimate (pixels)."""
        return float(np.sqrt((self.P[0, 0] + self.P[1, 1]) / 2.0))


# ---------------------------------------------------------------------------
# Kalman state visualisation (heatmap)
# ---------------------------------------------------------------------------

def plot_kalman_heatmap(mean, cov_2x2, title, save_path):
    """
    Plot 2D Gaussian probability density over (position, velocity) space.
    Produces the hot-colormap visualisation as discussed in design docs.

    Shows:
    - Bright center = most likely state
    - Ellipse shape = uncertainty in position vs velocity
    - Tilt = correlation between position and velocity errors
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import multivariate_normal

        sx = np.sqrt(abs(cov_2x2[0, 0])) * 5
        sv = np.sqrt(abs(cov_2x2[1, 1])) * 5

        x_range = np.linspace(mean[0] - sx, mean[0] + sx, 150)
        v_range = np.linspace(mean[1] - sv, mean[1] + sv, 150)
        X, V    = np.meshgrid(x_range, v_range)
        pos     = np.dstack((X, V))

        # Regularise covariance to ensure positive definite
        cov_reg = cov_2x2 + np.eye(2) * 1e-6
        rv      = multivariate_normal(mean=mean, cov=cov_reg)
        Z       = rv.pdf(pos)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.contourf(X, V, Z, levels=20, cmap='hot')
        ax.set_xlabel("Position (cx) [px]", color='white')
        ax.set_ylabel("Velocity (vx) [px/frame]", color='white')
        ax.set_title(title, color='white', fontsize=10)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight',
                    facecolor='black')
        plt.close()

    except ImportError as e:
        logger.warning("Cannot plot Kalman heatmap: %s", e)


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_detection(frame, result: DetectionResult, kf=None) -> np.ndarray:
    """Draw detection box, center, source label, and Kalman prediction."""
    overlay = frame.copy()
    cx, cy  = int(result.cx), int(result.cy)
    r       = int(result.radius_px)

    # Detection circle
    color = COLOR_DETECTION if result.source == 'yolo' else (0, 165, 255)
    cv2.circle(overlay, (cx, cy), r, color, 2)
    cv2.circle(overlay, (cx, cy), 3, color, -1)

    # Label
    label = (f"{result.source.upper()} "
             f"conf={result.confidence:.2f} "
             f"r={result.radius_px:.1f}px")
    cv2.putText(overlay, label,
                (cx - r, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                cv2.LINE_AA)

    # Kalman predicted position
    if kf is not None and kf.initialised and kf.x_prior is not None:
        px = int(kf.x_prior[0, 0])
        py = int(kf.x_prior[1, 0])
        cv2.circle(overlay, (px, py), 6, COLOR_PREDICTION, 2)
        cv2.drawMarker(overlay, (px, py), COLOR_PREDICTION,
                       cv2.MARKER_CROSS, 12, 2)

    return overlay


def draw_trajectory(frame, trail: deque) -> np.ndarray:
    """Draw fading trajectory polyline."""
    if len(trail) < 2:
        return frame

    overlay = frame.copy()
    pts     = list(trail)
    n       = len(pts)

    for i in range(1, n):
        alpha = i / n   # 0 = oldest (transparent), 1 = newest (opaque)
        color = tuple(int(c * alpha) for c in COLOR_TRAJECTORY)
        thickness = max(1, int(3 * alpha))
        cv2.line(overlay,
                 (int(pts[i-1][0]), int(pts[i-1][1])),
                 (int(pts[i][0]),   int(pts[i][1])),
                 color, thickness, cv2.LINE_AA)

    return overlay


def draw_hud(frame, frame_idx, fps, detected, source, kf=None) -> np.ndarray:
    """Draw heads-up display with frame stats."""
    overlay = frame.copy()
    h, w    = frame.shape[:2]

    lines = [
        f"Frame: {frame_idx}",
        f"Detected: {'YES' if detected else 'NO'} [{source}]",
    ]
    if kf is not None and kf.initialised:
        vx, vy = kf.velocity
        pu     = kf.position_uncertainty
        lines += [
            f"vel: ({vx:+.1f}, {vy:+.1f}) px/frame",
            f"pos uncertainty: {pu:.1f} px",
        ]

    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1,
                    cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Ground truth evaluation (DFL dataset)
# ---------------------------------------------------------------------------

def load_yolo_label(label_path: str, img_w: int, img_h: int):
    """
    Load YOLO-format label file.
    Format per line: class_id cx_norm cy_norm w_norm h_norm
    Returns list of (cx_px, cy_px, w_px, h_px) for sports ball class.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != 0:   # DFL dataset uses class 0 for ball
                continue
            cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
            boxes.append((
                cx_n * img_w, cy_n * img_h,
                w_n  * img_w, h_n  * img_h,
            ))
    return boxes


def compute_iou(pred_cx, pred_cy, pred_r, gt_cx, gt_cy, gt_w, gt_h):
    """Compute IoU between predicted circle bbox and GT bbox."""
    px1, py1 = pred_cx - pred_r, pred_cy - pred_r
    px2, py2 = pred_cx + pred_r, pred_cy + pred_r
    gx1 = gt_cx - gt_w / 2
    gy1 = gt_cy - gt_h / 2
    gx2 = gt_cx + gt_w / 2
    gy2 = gt_cy + gt_h / 2

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area   = gt_w * gt_h
    union     = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(video_path: str,
               output_path: str,
               csv_path: str,
               gt_dir: str = None,
               plot_kalman: bool = False,
               plot_dir: str = "kalman_plots") -> dict:
    """
    Run Stage 1 pipeline on a video file.

    Returns dict of summary metrics.
    """
    # --- Load camera config ---
    camera.load(video_path)
    logger.info("Camera config: %s", camera)

    # --- Initialise detector ---
    detector = BallDetector()

    # --- Initialise Kalman filter ---
    kf = SimpleKalmanFilter(dt=camera.dt)

    # --- Video I/O ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out    = cv2.VideoWriter(
        output_path, fourcc, camera.fps,
        (camera.frame_width, camera.frame_height)
    )

    # --- Trajectory trail ---
    trail = deque(maxlen=TRAJECTORY_DEQUE_LENGTH)

    # --- Missed frame counter for Kalman reset ---
    consecutive_misses = 0

    # --- CSV writer ---
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame', 'time_s',
        'det_cx', 'det_cy', 'det_radius_px', 'det_confidence', 'det_source',
        'kf_cx',  'kf_cy',  'kf_vx', 'kf_vy', 'kf_uncertainty',
    ])

    # --- Kalman plot directory ---
    if plot_kalman:
        os.makedirs(plot_dir, exist_ok=True)

    # --- Metrics counters ---
    total_frames     = 0
    detected_frames  = 0
    yolo_frames      = 0
    hough_frames     = 0
    missed_frames    = 0

    # GT evaluation (DFL)
    gt_tp = gt_fp = gt_fn = 0

    logger.info("Starting Stage 1 processing...")
    logger.info("Video: %s | Output: %s", video_path, output_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        time_s        = (total_frames - 1) / camera.fps

        # ----------------------------------------------------------------
        # Detection — pass Kalman position for Hough spatial gate.
        # Strategy:
        #   - Kalman reliable (< MAX_MISSED frames): gate Hough to KF position
        #   - Kalman uncertain (>= MAX_MISSED, < 2x MAX_MISSED): gate loosely
        #   - Kalman long-lost (>= 2x MAX_MISSED): disable Hough entirely.
        #     Rationale: without a spatial anchor, Hough will latch onto random
        #     circles (heads, logos) and corrupt Kalman with false velocities
        #     when it re-acquires. Better to wait for YOLO to rediscover.
        # ----------------------------------------------------------------
        from config import KALMAN_MAX_MISSED_FRAMES
        if not kf.initialised:
            # No prior — run YOLO only, Hough ungated
            kalman_pos  = None
            allow_hough = True
        elif consecutive_misses < KALMAN_MAX_MISSED_FRAMES:
            # Kalman reliable — gate Hough tightly to KF position
            kalman_pos  = kf.position
            allow_hough = True
        elif consecutive_misses < KALMAN_MAX_MISSED_FRAMES * 2:
            # Kalman uncertain — use last frozen position as loose gate
            kalman_pos  = kf.position
            allow_hough = True
        else:
            # Kalman long-lost — disable Hough, trust YOLO only
            kalman_pos  = None
            allow_hough = False

        result = detector.detect(
            frame,
            kalman_pos=kalman_pos,
            allow_hough=allow_hough,
        )

        if result is not None:
            detected_frames += 1
            if result.source == 'yolo':
                yolo_frames += 1
            else:
                hough_frames += 1

        # ----------------------------------------------------------------
        # Kalman filter — with missed frame tracking and reset logic
        # ----------------------------------------------------------------
        if result is not None and not kf.initialised:
            kf.initialise(result.cx, result.cy)

        if kf.initialised:
            kf.predict()

            if result is not None:
                # Good detection — update and reset miss counter
                kf.update([result.cx, result.cy])
                consecutive_misses = 0
            else:
                # No detection this frame — predict only
                consecutive_misses += 1
                missed_frames += 1

                # --- Kalman soft reset ---
                # After KALMAN_MAX_MISSED_FRAMES consecutive misses,
                # zero the velocity. The position prediction becomes
                # stationary — much better than drifting off-screen.
                # Rationale: we don't know where the ball went, so
                # holding the last known position is more honest than
                # extrapolating a stale velocity indefinitely.
                from config import KALMAN_MAX_MISSED_FRAMES, KALMAN_RESET_ON_OOB
                if consecutive_misses >= KALMAN_MAX_MISSED_FRAMES:
                    if consecutive_misses == KALMAN_MAX_MISSED_FRAMES:
                        logger.info(
                            "Frame %d: %d consecutive misses — "
                            "freezing Kalman velocity to prevent drift. "
                            "Last known position: (%.0f, %.0f)",
                            total_frames, consecutive_misses,
                            kf.x[0, 0], kf.x[1, 0],
                        )
                    # Zero velocity — hold last position
                    kf.x[2, 0] = 0.0
                    kf.x[3, 0] = 0.0

                # --- Out-of-bounds check ---
                # If Kalman drifted outside the frame despite the above,
                # clamp position to frame boundaries.
                if KALMAN_RESET_ON_OOB:
                    kf.x[0, 0] = float(np.clip(
                        kf.x[0, 0], 0, camera.frame_width
                    ))
                    kf.x[1, 0] = float(np.clip(
                        kf.x[1, 0], 0, camera.frame_height
                    ))

            kf_cx, kf_cy = kf.position
            kf_vx, kf_vy = kf.velocity
            kf_unc       = kf.position_uncertainty

            # Only add to trail when Kalman is reliable
            # (not during long miss streaks)
            if consecutive_misses < KALMAN_MAX_MISSED_FRAMES:
                trail.append((kf_cx, kf_cy))
        else:
            kf_cx = kf_cy = kf_vx = kf_vy = kf_unc = None

        # ----------------------------------------------------------------
        # Kalman state visualisation
        # ----------------------------------------------------------------
        if plot_kalman and kf.initialised and total_frames % 10 == 0:
            mean   = [kf.x[0, 0], kf.x[2, 0]]         # cx, vx slice
            cov_2x2 = kf.P[np.ix_([0, 2], [0, 2])]

            plot_kalman_heatmap(
                mean=mean,
                cov_2x2=cov_2x2,
                title=f"State at t={total_frames} after update",
                save_path=os.path.join(
                    plot_dir, f"kalman_{total_frames:04d}.png"
                ),
            )

        # ----------------------------------------------------------------
        # Ground truth evaluation (DFL dataset)
        # ----------------------------------------------------------------
        if gt_dir is not None and result is not None:
            frame_name = f"{total_frames:06d}.txt"
            label_path = os.path.join(gt_dir, frame_name)
            gt_boxes   = load_yolo_label(
                label_path, camera.frame_width, camera.frame_height
            )

            if len(gt_boxes) == 0:
                gt_fp += 1
            else:
                matched = False
                for gt in gt_boxes:
                    iou = compute_iou(
                        result.cx, result.cy, result.radius_px,
                        gt[0], gt[1], gt[2], gt[3]
                    )
                    if iou >= 0.5:
                        matched = True
                        break
                if matched:
                    gt_tp += 1
                else:
                    gt_fp += 1

        elif gt_dir is not None and result is None:
            frame_name = f"{total_frames:06d}.txt"
            label_path = os.path.join(gt_dir, frame_name)
            gt_boxes   = load_yolo_label(
                label_path, camera.frame_width, camera.frame_height
            )
            if len(gt_boxes) > 0:
                gt_fn += 1

        # ----------------------------------------------------------------
        # CSV logging
        # ----------------------------------------------------------------
        csv_writer.writerow([
            total_frames,
            f"{time_s:.4f}",
            f"{result.cx:.2f}"         if result else "",
            f"{result.cy:.2f}"         if result else "",
            f"{result.radius_px:.2f}"  if result else "",
            f"{result.confidence:.3f}" if result else "",
            result.source              if result else "none",
            f"{kf_cx:.2f}"  if kf_cx is not None else "",
            f"{kf_cy:.2f}"  if kf_cy is not None else "",
            f"{kf_vx:.2f}"  if kf_vx is not None else "",
            f"{kf_vy:.2f}"  if kf_vy is not None else "",
            f"{kf_unc:.2f}" if kf_unc is not None else "",
        ])

        # ----------------------------------------------------------------
        # Visualisation — draw on frame
        # ----------------------------------------------------------------
        vis = frame.copy()

        # Trajectory
        vis = draw_trajectory(vis, trail)

        # Detection
        if result is not None:
            vis = draw_detection(vis, result, kf=kf)

        # HUD
        vis = draw_hud(
            vis, total_frames, camera.fps,
            detected=result is not None,
            source=result.source if result else "none",
            kf=kf,
        )

        out.write(vis)

        if total_frames % 50 == 0:
            logger.info(
                "Frame %d/%d | detected=%d | missed=%d",
                total_frames, camera.total_frames,
                detected_frames, missed_frames,
            )

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    cap.release()
    out.release()
    csv_file.close()

    # ----------------------------------------------------------------
    # Summary metrics
    # ----------------------------------------------------------------
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0

    metrics = {
        "total_frames":     total_frames,
        "detected_frames":  detected_frames,
        "yolo_frames":      yolo_frames,
        "hough_frames":     hough_frames,
        "missed_frames":    missed_frames,
        "detection_rate":   detection_rate,
    }

    if gt_dir is not None and (gt_tp + gt_fp + gt_fn) > 0:
        precision = gt_tp / (gt_tp + gt_fp) if (gt_tp + gt_fp) > 0 else 0
        recall    = gt_tp / (gt_tp + gt_fn) if (gt_tp + gt_fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)
        metrics.update({
            "gt_tp": gt_tp, "gt_fp": gt_fp, "gt_fn": gt_fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    # Print summary
    logger.info("=" * 50)
    logger.info("STAGE 1 SUMMARY")
    logger.info("=" * 50)
    logger.info("Total frames     : %d", total_frames)
    logger.info("Detected frames  : %d", detected_frames)
    logger.info("  - YOLO         : %d", yolo_frames)
    logger.info("  - Hough        : %d", hough_frames)
    logger.info("  - Missed       : %d", missed_frames)
    logger.info("Detection rate   : %.1f%%", detection_rate * 100)
    if "precision" in metrics:
        logger.info("Precision        : %.3f", metrics["precision"])
        logger.info("Recall           : %.3f", metrics["recall"])
        logger.info("F1               : %.3f", metrics["f1"])
    logger.info("Output video     : %s", output_path)
    logger.info("Detection CSV    : %s", csv_path)
    if plot_kalman:
        logger.info("Kalman plots     : %s/", plot_dir)
    logger.info("=" * 50)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Ball Detection + Kalman Tracking"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video (e.g. rgb.avi)"
    )
    parser.add_argument(
        "--output", type=str, default="output_stage1.avi",
        help="Path for output annotated video"
    )
    parser.add_argument(
        "--csv", type=str, default="detections_stage1.csv",
        help="Path for per-frame detection CSV"
    )
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Directory of YOLO-format label files for GT evaluation (DFL)"
    )
    parser.add_argument(
        "--plot_kalman", action="store_true",
        help="Save Kalman state heatmap plots every 10 frames"
    )
    parser.add_argument(
        "--plot_dir", type=str, default="kalman_plots",
        help="Directory for Kalman plot output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = run_stage1(
        video_path=args.video,
        output_path=args.output,
        csv_path=args.csv,
        gt_dir=args.gt_dir,
        plot_kalman=args.plot_kalman,
        plot_dir=args.plot_dir,
    )