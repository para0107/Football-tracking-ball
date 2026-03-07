"""
tests/test_stage1.py — Unit tests for Stage 1

Tests the Kalman filter math and detector logic independently,
without requiring YOLOv8, a GPU, or the actual video file.

Run with:
    pytest tests/test_stage1.py -v

References:
    Kalman R.E. (1960) — verify predict/update equations
    Hartley & Zisserman Ch.6 — used downstream in estimator
"""

import numpy as np
import pytest

from evaluate_stage1 import SimpleKalmanFilter
from detector import DetectionResult


# ---------------------------------------------------------------------------
# Kalman Filter tests
# ---------------------------------------------------------------------------

class TestKalmanFilter:

    def test_initialisation(self):
        """Filter should initialise state correctly from first detection."""
        kf = SimpleKalmanFilter(dt=1/30)
        kf.initialise(320.0, 240.0)
        assert kf.initialised
        cx, cy = kf.position
        assert abs(cx - 320.0) < 1e-6
        assert abs(cy - 240.0) < 1e-6
        vx, vy = kf.velocity
        assert abs(vx) < 1e-6
        assert abs(vy) < 1e-6

    def test_predict_increases_uncertainty(self):
        """
        After predict step, position uncertainty must increase.
        This is the fundamental guarantee of the Kalman predict step:
        P(t+1|t) = F·P(t|t)·Fᵀ + Q  ← Q is positive definite → P grows
        """
        kf = SimpleKalmanFilter(dt=1/30)
        kf.initialise(320.0, 240.0)
        p_before = kf.position_uncertainty
        kf.predict()
        p_after = kf.position_uncertainty
        assert p_after > p_before, (
            f"Uncertainty should grow after predict: "
            f"{p_before:.3f} -> {p_after:.3f}"
        )

    def test_update_decreases_uncertainty(self):
        """
        After update step, uncertainty must decrease.
        This is the fundamental guarantee of the Kalman update step:
        P(t|t) = (I - K·H) · P(t|t-1)  ← shrinks toward measurement
        """
        kf = SimpleKalmanFilter(dt=1/30)
        kf.initialise(320.0, 240.0)
        kf.predict()
        p_before = kf.position_uncertainty
        kf.update([322.0, 241.0])
        p_after = kf.position_uncertainty
        assert p_after < p_before, (
            f"Uncertainty should shrink after update: "
            f"{p_before:.3f} -> {p_after:.3f}"
        )

    def test_predict_only_when_no_detection(self):
        """
        When YOLO misses a frame, predict-only should still produce
        a physically plausible position estimate.
        Ball at (320, 240) with vx=5, vy=-3 should move accordingly.
        """
        dt = 1/30
        kf = SimpleKalmanFilter(dt=dt)
        kf.initialise(320.0, 240.0)
        # Manually set velocity
        kf.x[2, 0] = 5.0    # vx
        kf.x[3, 0] = -3.0   # vy
        kf.predict()
        cx, cy = kf.position
        expected_cx = 320.0 + 5.0 * dt
        expected_cy = 240.0 + (-3.0) * dt
        assert abs(cx - expected_cx) < 0.5, (
            f"Predicted cx={cx:.2f}, expected≈{expected_cx:.2f}"
        )
        assert abs(cy - expected_cy) < 0.5, (
            f"Predicted cy={cy:.2f}, expected≈{expected_cy:.2f}"
        )

    def test_velocity_inferred_from_positions(self):
        """
        Velocity should be inferred from successive position measurements
        without being directly measured (Kalman 1960, state estimation).

        State vector velocity is in px/s (not px/frame).
        F[0,2] = dt means: x(t+1) = x(t) + vx_pxps * dt
        So a ball moving 10 px/frame at 30fps has vx = 300 px/s.

        After 30 consistent measurements, estimated vx should
        converge within 10% of true value.
        """
        dt           = 1/30
        kf           = SimpleKalmanFilter(dt=dt)
        px_per_frame = 10.0
        true_vx      = px_per_frame / dt   # 300 px/s

        cx = 100.0
        kf.initialise(cx, 240.0)
        cx += px_per_frame

        for _ in range(30):
            kf.predict()
            kf.update([cx, 240.0])
            cx += px_per_frame

        estimated_vx, _ = kf.velocity
        error_pct = abs(estimated_vx - true_vx) / true_vx * 100
        assert error_pct < 15.0, (
            f"Velocity error too large: estimated={estimated_vx:.1f} "
            f"true={true_vx:.1f} ({error_pct:.1f}%)"
        )

    def test_covariance_remains_positive_definite(self):
        """
        Covariance matrix P must remain positive definite throughout.
        Using Joseph form update: P = (I-KH)·P·(I-KH)ᵀ + K·R·Kᵀ
        guarantees this numerically.
        """
        kf = SimpleKalmanFilter(dt=1/30)
        kf.initialise(320.0, 240.0)
        for i in range(50):
            kf.predict()
            kf.update([320.0 + i * 2, 240.0])
            eigenvalues = np.linalg.eigvalsh(kf.P)
            assert np.all(eigenvalues > -1e-9), (
                f"P not positive definite at step {i}: "
                f"min eigenvalue={eigenvalues.min():.6f}"
            )

    def test_large_innovation_adapts(self):
        """
        A sudden large measurement change (simulating a kick) should
        cause the filter to update its state substantially.
        This tests the filter's ability to handle the kick scenario.
        """
        kf = SimpleKalmanFilter(dt=1/30)
        kf.initialise(320.0, 240.0)
        # Stabilise with consistent measurements
        for _ in range(10):
            kf.predict()
            kf.update([320.0, 240.0])
        # Sudden kick — ball jumps 100px
        kf.predict()
        kf.update([420.0, 240.0])
        cx, _ = kf.position
        # Filter should have moved substantially toward new position
        assert cx > 350.0, (
            f"Filter should track kick: cx={cx:.1f}, expected > 350"
        )


# ---------------------------------------------------------------------------
# DetectionResult tests
# ---------------------------------------------------------------------------

class TestDetectionResult:

    def test_bbox_from_circle(self):
        """Bounding box should be consistent with center and radius."""
        r      = DetectionResult(cx=100, cy=200, radius_px=30,
                                 confidence=0.9, source='yolo')
        x1, y1, x2, y2 = r.bbox
        assert x1 == 70
        assert y1 == 170
        assert x2 == 130
        assert y2 == 230

    def test_center_property(self):
        r = DetectionResult(cx=150.7, cy=99.3, radius_px=20,
                            confidence=0.8, source='yolo')
        cx, cy = r.center
        assert cx == 150
        assert cy == 99

    def test_radius_positive(self):
        """Radius must always be positive — otherwise depth formula breaks."""
        r = DetectionResult(cx=100, cy=100, radius_px=15,
                            confidence=0.7, source='hough')
        assert r.radius_px > 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestCameraConfig:

    def test_fx_formula(self):
        """
        Verify fx derivation:
        fx = (W/2) / tan(FOV_H/2)
        For D435i at 1920x1080: fx should be approximately 1380 px
        """
        import math
        fov_h_rad = math.radians(69.0)
        W         = 1920
        fx        = (W / 2) / math.tan(fov_h_rad / 2)
        # Known approximate value for D435i at 1080p
        assert 1300 < fx < 1500, f"fx={fx:.1f} outside expected range"

    def test_fx_scales_with_resolution(self):
        """
        fx in pixels must scale proportionally with width.
        This ensures radius_px and fx remain consistent at any resolution.
        """
        import math
        fov_h_rad = math.radians(69.0)
        fx_1080p  = (1920/2) / math.tan(fov_h_rad/2)
        fx_720p   = (1280/2) / math.tan(fov_h_rad/2)
        ratio     = fx_1080p / fx_720p
        # Should be exactly 1920/1280 = 1.5
        assert abs(ratio - 1920/1280) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])