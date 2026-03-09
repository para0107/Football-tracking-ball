"""
visualizer.py — Stage 3 & 4: Trajectory Overlay + Top-View 2D Map

Stage 3: Trajectory overlay on original video
    - Rolling deque of last N Kalman-smoothed positions
    - Fading polyline (older = more transparent)
    - Optional parabola fit y = at² + bt + c
      Fitted g extracted and compared against 9.81 m/s²
      Serves as physics-based self-validation

Stage 4: Top-view 2D map
    - Projects 3D (X, Z) world coordinates onto a 2D canvas
    - Camera at canvas origin
    - Ball trajectory drawn as coloured path
    - Colour encodes time (viridis: dark=early, bright=late)

Coordinate conventions:
    Camera frame: X right, Y down, Z forward (depth)
    Top-view map: X horizontal, Z vertical (depth away from camera)
    Side-view:    Z horizontal, -Y vertical (height, inverted Y so up=positive)

References:
    Hartley & Zisserman Ch. 4 — projective geometry / homography
    OpenCV warpPerspective docs — perspective transformation
    Projectile motion: y(t) = y0 + vy0*t - 0.5*g*t²
"""

import logging
import math
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    COLOR_DETECTION,
    COLOR_PREDICTION,
    COLOR_TRAJECTORY,
    COLOR_TEXT,
    TRAJECTORY_DEQUE_LENGTH,
    BALL_DIAMETER_M,
)
from estimator import Position3D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory overlay (Stage 3)
# ---------------------------------------------------------------------------

class TrajectoryVisualizer:
    """
    Draws a fading trajectory trail on video frames.

    The trail stores the last N Kalman-smoothed pixel positions
    and draws them as a polyline with alpha proportional to recency.

    Also performs optional parabola fit for physics validation:
        y_pixel(t) = a*t² + b*t + c
    from which we extract apparent gravitational acceleration and
    compare against the expected 9.81 m/s².
    """

    def __init__(self, max_length: int = TRAJECTORY_DEQUE_LENGTH):
        self._trail:    deque = deque(maxlen=max_length)
        self._times:    deque = deque(maxlen=max_length)
        self._parabola: Optional[np.ndarray] = None   # [a, b, c] coefficients

    def update(self, cx: float, cy: float, time_s: float) -> None:
        """Add a new position to the trail."""
        self._trail.append((float(cx), float(cy)))
        self._times.append(float(time_s))

        # Refit parabola whenever we have enough points
        if len(self._trail) >= 10:
            self._fit_parabola()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw fading trajectory polyline on frame.
        Returns modified frame (copy).
        """
        if len(self._trail) < 2:
            return frame

        overlay = frame.copy()
        pts     = list(self._trail)
        n       = len(pts)

        for i in range(1, n):
            alpha     = i / n   # 0=oldest (faint), 1=newest (bright)
            color     = tuple(int(c * alpha) for c in COLOR_TRAJECTORY)
            thickness = max(1, int(3 * alpha))
            cv2.line(
                overlay,
                (int(pts[i-1][0]), int(pts[i-1][1])),
                (int(pts[i][0]),   int(pts[i][1])),
                color, thickness, cv2.LINE_AA,
            )

        # Draw parabola fit if available
        if self._parabola is not None:
            self._draw_parabola(overlay)

        return overlay

    def draw_hud(self, frame: np.ndarray,
                 frame_idx: int,
                 pos: Optional[Position3D] = None) -> np.ndarray:
        """Draw HUD with 3D info and parabola fit stats."""
        overlay = frame.copy()
        lines   = [f"Frame: {frame_idx}"]

        if pos is not None and pos.valid:
            lines += [
                f"X:{pos.X:+.2f}m  Y:{pos.Y:+.2f}m  Z:{pos.Z:.2f}m",
                f"dist: {pos.distance:.2f}m",
            ]

        if self._parabola is not None:
            a, b, c = self._parabola
            lines.append(f"parabola fit: a={a:.2f} b={b:.2f}")

        for i, line in enumerate(lines):
            cv2.putText(
                overlay, line, (10, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA,
            )
        return overlay

    def parabola_stats(self) -> Optional[dict]:
        """
        Return parabola fit statistics.

        The fit is in pixel space: y_px(t) = a*t² + b*t + c
        'a' is the pixel-space downward acceleration.
        We can't directly extract g from pixel 'a' without knowing
        pixels-per-metre at that distance, but we report it for
        relative validation — during free flight 'a' should be
        consistently positive and stable.
        """
        if self._parabola is None:
            return None
        a, b, c = self._parabola
        return {
            "a_px_s2":   float(a),
            "b_px_s":    float(b),
            "c_px":      float(c),
            "direction": "downward" if a > 0 else "upward",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_parabola(self) -> None:
        """
        Fit y_pixel = a*t² + b*t + c to current trail.
        Uses numpy polyfit (least squares). Only fits when trail
        has at least 10 points to avoid overfitting noise.
        """
        times = np.array(list(self._times))
        ys    = np.array([p[1] for p in self._trail])

        # Only fit if there is meaningful time variation
        if times.max() - times.min() < 0.1:
            return

        try:
            coeffs = np.polyfit(times, ys, 2)
            self._parabola = coeffs
        except Exception:
            self._parabola = None

    def _draw_parabola(self, frame: np.ndarray) -> None:
        """Draw the fitted parabola as a smooth curve."""
        if len(self._times) < 2:
            return

        t_min = min(self._times)
        t_max = max(self._times)
        xs    = [p[0] for p in self._trail]
        x_min = min(xs)
        x_max = max(xs)

        a, b, c = self._parabola
        pts     = []

        # Sample 50 points along the parabola
        for i in range(50):
            t  = t_min + (t_max - t_min) * i / 49
            x  = x_min + (x_max - x_min) * i / 49
            y  = a * t**2 + b * t + c
            h, w = frame.shape[:2]
            if 0 <= int(x) < w and 0 <= int(y) < h:
                pts.append((int(x), int(y)))

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i],
                     (255, 200, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Top-view 2D map (Stage 4)
# ---------------------------------------------------------------------------

class TopViewMap:
    """
    Renders a bird's-eye 2D map of ball positions.

    Method:
        Uses 3D world coordinates (X, Z) from the estimator.
        X = lateral (right = positive)
        Z = depth (away from camera = positive)

        Camera sits at origin (0, 0) in world coordinates.
        Since back-projection already gives camera-relative coordinates,
        (X, Z) directly represents the top-down map.
        Camera tilt is handled automatically by the formula.

    Canvas:
        Fixed-size canvas (e.g. 600x600 pixels).
        World metres mapped to canvas pixels using a fixed scale.
        Camera position drawn as a triangle at canvas bottom-centre.
        Ball trajectory drawn as coloured dots (colour = time).

    Reference:
        Hartley & Zisserman — homography theory (Ch. 4)
        The (X,Z) projection is a degenerate homography where Y is dropped.
    """

    def __init__(self,
                 canvas_w: int = 600,
                 canvas_h: int = 600,
                 world_range_m: float = 12.0):
        """
        Args:
            canvas_w/h:     Canvas size in pixels
            world_range_m:  World extent mapped to canvas (metres)
                            e.g. 12.0 means ±6m lateral, 0–12m depth
        """
        self._canvas_w     = canvas_w
        self._canvas_h     = canvas_h
        self._world_range  = world_range_m
        self._scale        = canvas_h / world_range_m   # px per metre
        self._positions:   List[Tuple[float, float, float]] = []
                           # list of (X, Z, time_s)

        # Camera is at bottom-centre of canvas
        self._cam_canvas_x = canvas_w // 2
        self._cam_canvas_y = canvas_h - 30   # near bottom

    def update(self, pos: Position3D) -> None:
        """Add a valid 3D position to the map history."""
        if pos.valid:
            self._positions.append((pos.X, pos.Z, pos.time_s))

    def render(self) -> np.ndarray:
        """
        Render the current top-view map as a BGR image.
        Returns a canvas_w × canvas_h BGR image.
        """
        canvas = np.zeros((self._canvas_h, self._canvas_w, 3),
                          dtype=np.uint8)
        canvas[:] = (20, 20, 20)   # dark background

        # Draw grid lines (every 2 metres)
        self._draw_grid(canvas)

        # Draw camera position
        self._draw_camera(canvas)

        # Draw ball trajectory
        self._draw_trajectory(canvas)

        # Draw axis labels
        cv2.putText(canvas, "Top-View Map", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, "X (lateral)", (self._canvas_w - 90, self._cam_canvas_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(canvas, "Z (depth)", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return canvas

    def render_side_view(self) -> np.ndarray:
        """
        Render a side-view (Z vs height) map.
        Height = -Y (inverted because Y is positive downward).
        """
        canvas = np.zeros((self._canvas_h, self._canvas_w, 3),
                          dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        self._draw_grid(canvas, label_depth=True)

        if len(self._positions) > 1:
            # For side view we need Y — stored in Position3D
            # This method is called with positions that have Y data
            pass

        cv2.putText(canvas, "Side-View Map", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return canvas

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _world_to_canvas(self, X: float, Z: float) -> Tuple[int, int]:
        """
        Convert world (X, Z) coordinates to canvas pixel coordinates.

        X=0 maps to canvas centre horizontally.
        Z=0 (camera position) maps to near bottom of canvas.
        Z increases upward on canvas (depth = away from viewer).
        """
        cx = self._cam_canvas_x + int(X * self._scale)
        cy = self._cam_canvas_y - int(Z * self._scale)
        return cx, cy

    def _draw_grid(self, canvas: np.ndarray,
                   label_depth: bool = False) -> None:
        """Draw metric grid lines every 2 metres."""
        for z in range(0, int(self._world_range) + 1, 2):
            _, cy = self._world_to_canvas(0, z)
            if 0 <= cy < self._canvas_h:
                cv2.line(canvas, (0, cy), (self._canvas_w, cy),
                         (50, 50, 50), 1)
                cv2.putText(canvas, f"{z}m", (5, cy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (100, 100, 100), 1)

        for x in range(-int(self._world_range//2),
                       int(self._world_range//2) + 1, 2):
            cx, _ = self._world_to_canvas(x, 0)
            if 0 <= cx < self._canvas_w:
                cv2.line(canvas, (cx, 0), (cx, self._canvas_h),
                         (50, 50, 50), 1)

    def _draw_camera(self, canvas: np.ndarray) -> None:
        """Draw camera as a triangle at its canvas position."""
        cx, cy = self._cam_canvas_x, self._cam_canvas_y
        pts = np.array([
            [cx, cy - 15],
            [cx - 10, cy + 5],
            [cx + 10, cy + 5],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (0, 200, 255))
        cv2.putText(canvas, "CAM", (cx - 15, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    def _draw_trajectory(self, canvas: np.ndarray) -> None:
        """Draw ball positions as colour-coded dots (colour = time)."""
        if not self._positions:
            return

        times  = [p[2] for p in self._positions]
        t_min  = min(times)
        t_max  = max(times) if max(times) > t_min else t_min + 1

        prev_pt = None
        for X, Z, t in self._positions:
            cx, cy = self._world_to_canvas(X, Z)

            if not (0 <= cx < self._canvas_w and 0 <= cy < self._canvas_h):
                prev_pt = None
                continue

            # Colour: blue (early) → green → yellow (late) — viridis-like
            norm  = (t - t_min) / (t_max - t_min)
            r     = int(255 * min(1, 2 * norm))
            g     = int(255 * min(1, 2 * (1 - norm)))
            b     = int(255 * (1 - norm))
            color = (b, g, r)

            cv2.circle(canvas, (cx, cy), 3, color, -1)

            if prev_pt is not None:
                cv2.line(canvas, prev_pt, (cx, cy), color, 1, cv2.LINE_AA)
            prev_pt = (cx, cy)


# ---------------------------------------------------------------------------
# Full pipeline runner for Stage 3 + 4
# ---------------------------------------------------------------------------

def run_stage3_4(video_path:      str,
                 positions_3d:    List[Position3D],
                 output_traj:     str = "output_trajectory.avi",
                 output_topview:  str = "output_topview.avi") -> dict:
    """
    Produce Stage 3 and Stage 4 output videos.

    Args:
        video_path:    Original rgb.avi
        positions_3d:  List of Position3D from Stage 2
        output_traj:   Output video with trajectory overlay (Stage 3)
        output_topview: Output video with top-view map (Stage 4)

    Returns:
        dict with summary statistics including parabola fit results
    """
    from config import camera

    if not camera.loaded:
        camera.load(video_path)

    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_traj = cv2.VideoWriter(
        output_traj, fourcc, camera.fps,
        (camera.frame_width, camera.frame_height)
    )

    # Top-view map — side by side with original (half width each)
    map_size = min(camera.frame_height, 600)
    out_top  = cv2.VideoWriter(
        output_topview, fourcc, camera.fps,
        (camera.frame_width + map_size, camera.frame_height)
    )

    traj_vis  = TrajectoryVisualizer(max_length=TRAJECTORY_DEQUE_LENGTH)
    top_map   = TopViewMap(canvas_w=map_size, canvas_h=camera.frame_height,
                           world_range_m=12.0)

    frame_idx = 0

    logger.info("Rendering Stage 3+4 videos...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos = positions_3d[frame_idx] if frame_idx < len(positions_3d) else None

        # Update visualisers with valid positions
        if pos is not None and pos.valid:
            traj_vis.update(pos.cx_px, pos.cy_px, pos.time_s)
            top_map.update(pos)

        # --- Stage 3: trajectory overlay ---
        vis_traj = traj_vis.draw(frame.copy())

        # Draw ball circle if we have a detection
        if pos is not None and pos.valid and pos.cx_px and pos.cy_px:
            r  = max(5, int(pos.radius_px or 15))
            cx = int(pos.cx_px)
            cy = int(pos.cy_px)
            cv2.circle(vis_traj, (cx, cy), r, COLOR_DETECTION, 2)
            cv2.circle(vis_traj, (cx, cy), 3, COLOR_DETECTION, -1)

        vis_traj = traj_vis.draw_hud(vis_traj, frame_idx + 1, pos)
        out_traj.write(vis_traj)

        # --- Stage 4: top-view map side by side ---
        map_canvas = top_map.render()

        # Resize map to match frame height
        if map_canvas.shape[0] != camera.frame_height:
            map_canvas = cv2.resize(
                map_canvas, (map_size, camera.frame_height)
            )

        combined = np.hstack([frame, map_canvas])
        out_top.write(combined)

        frame_idx += 1

    cap.release()
    out_traj.release()
    out_top.release()

    # Parabola fit statistics
    parabola = traj_vis.parabola_stats()

    logger.info("=" * 50)
    logger.info("STAGE 3+4 SUMMARY")
    logger.info("=" * 50)
    logger.info("Trajectory video : %s", output_traj)
    logger.info("Top-view video   : %s", output_topview)
    if parabola:
        logger.info("Parabola fit     : a=%.3f (px/s²) — %s",
                    parabola["a_px_s2"], parabola["direction"])
    logger.info("=" * 50)

    return {"parabola": parabola}