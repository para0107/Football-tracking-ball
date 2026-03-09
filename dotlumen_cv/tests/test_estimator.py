"""
tests/test_estimator.py — Unit tests for Stage 2: 3D Estimator

Tests the pinhole back-projection formula independently,
without requiring a video file, GPU, or YOLO.

All tests use analytically derived expected values so results
can be verified by hand against Hartley & Zisserman Ch. 6.

Run with:
    pytest tests/test_estimator.py -v
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from estimator import BallEstimator3D, Position3D


# ---------------------------------------------------------------------------
# Minimal camera config for testing (no video file needed)
# ---------------------------------------------------------------------------

class MockCamera:
    """Minimal camera config for unit testing."""
    def __init__(self, width=1280, height=720, fov_h_deg=69.0):
        self.frame_width  = width
        self.frame_height = height
        self.fps          = 30.0
        self.dt           = 1/30.0
        fov_h_rad         = math.radians(fov_h_deg)
        self.fx           = (width / 2) / math.tan(fov_h_rad / 2)
        self.fy           = self.fx
        self.ppx          = width  / 2.0
        self.ppy          = height / 2.0
        self.total_frames = 100
        self.loaded       = True

    def __repr__(self):
        return f"MockCamera({self.frame_width}x{self.frame_height})"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def camera():
    return MockCamera(width=1280, height=720)

@pytest.fixture
def estimator(camera):
    return BallEstimator3D(camera)


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------

class TestDepthFormula:

    def test_ball_at_image_centre_has_zero_lateral(self, estimator, camera):
        """
        Ball at principal point (ppx, ppy) should have X=0, Y=0.
        Only depth Z is non-zero. This is a direct consequence of:
            X = (cx - ppx) * Z / fx  →  X = 0 when cx = ppx
        """
        pos = estimator.estimate(
            frame=1, time_s=0.0,
            cx_px=camera.ppx,
            cy_px=camera.ppy,
            radius_px=25.0,
        )
        assert pos.valid
        assert abs(pos.X) < 1e-6, f"X should be 0, got {pos.X}"
        assert abs(pos.Y) < 1e-6, f"Y should be 0, got {pos.Y}"

    def test_depth_formula_manual(self, estimator, camera):
        """
        Manually verify: Z = (fx * D_real) / (2 * radius_px)
        With fx=931.21 (D435i at 720p), D_real=0.22m, radius_px=25px:
            Z = (931.21 * 0.22) / (2 * 25) = 204.87 / 50 = 4.097m
        """
        radius_px = 25.0
        D_real    = 0.22
        expected_Z = (camera.fx * D_real) / (2.0 * radius_px)

        pos = estimator.estimate(
            frame=1, time_s=0.0,
            cx_px=camera.ppx,
            cy_px=camera.ppy,
            radius_px=radius_px,
            ball_diameter_m=D_real,
        )
        assert abs(pos.Z - expected_Z) < 1e-6, (
            f"Z={pos.Z:.4f}, expected={expected_Z:.4f}"
        )

    def test_larger_radius_means_closer(self, estimator, camera):
        """
        Ball appears larger when closer to camera.
        Z = (fx * D) / (2r) — Z is inversely proportional to radius_px.
        """
        pos_close = estimator.estimate(1, 0.0, camera.ppx, camera.ppy, 50.0)
        pos_far   = estimator.estimate(1, 0.0, camera.ppx, camera.ppy, 10.0)
        assert pos_close.Z < pos_far.Z, (
            f"Larger radius should mean closer: "
            f"r=50→Z={pos_close.Z:.2f}m, r=10→Z={pos_far.Z:.2f}m"
        )

    def test_ball_right_of_centre_has_positive_X(self, estimator, camera):
        """
        Ball to the right of principal point: cx > ppx → X > 0.
        X = (cx - ppx) * Z / fx
        """
        pos = estimator.estimate(
            frame=1, time_s=0.0,
            cx_px=camera.ppx + 100,   # 100px right of centre
            cy_px=camera.ppy,
            radius_px=25.0,
        )
        assert pos.X > 0, f"Ball right of centre should have X>0, got {pos.X}"

    def test_ball_below_centre_has_positive_Y(self, estimator, camera):
        """
        Ball below principal point: cy > ppy → Y > 0.
        Y = (cy - ppy) * Z / fy
        (Y positive = downward in camera convention)
        """
        pos = estimator.estimate(
            frame=1, time_s=0.0,
            cx_px=camera.ppx,
            cy_px=camera.ppy + 80,    # 80px below centre
            radius_px=25.0,
        )
        assert pos.Y > 0, f"Ball below centre should have Y>0, got {pos.Y}"

    def test_distance_geq_depth(self, estimator, camera):
        """
        Euclidean distance >= depth Z always.
        distance = sqrt(X²+Y²+Z²) >= Z
        """
        pos = estimator.estimate(
            frame=1, time_s=0.0,
            cx_px=camera.ppx + 50,
            cy_px=camera.ppy + 50,
            radius_px=20.0,
        )
        assert pos.distance >= pos.Z, (
            f"distance={pos.distance:.3f} should be >= Z={pos.Z:.3f}"
        )

    def test_zero_radius_returns_invalid(self, estimator):
        """
        radius_px=0 would cause division by zero in depth formula.
        Estimator should return invalid Position3D, not raise.
        """
        pos = estimator.estimate(1, 0.0, 640, 360, radius_px=0.0)
        assert not pos.valid, "Zero radius should return invalid position"

    def test_negative_radius_returns_invalid(self, estimator):
        """Negative radius is physically impossible."""
        pos = estimator.estimate(1, 0.0, 640, 360, radius_px=-5.0)
        assert not pos.valid

    def test_basketball_diameter_override(self, estimator, camera):
        """
        When evaluating on DeepSportRadar, ball_diameter_m=0.24
        should give different Z than football (0.22).
        Z scales linearly with D_real.
        """
        pos_football   = estimator.estimate(1, 0.0, camera.ppx, camera.ppy,
                                            25.0, ball_diameter_m=0.22)
        pos_basketball = estimator.estimate(1, 0.0, camera.ppx, camera.ppy,
                                            25.0, ball_diameter_m=0.24)
        ratio = pos_basketball.Z / pos_football.Z
        expected_ratio = 0.24 / 0.22
        assert abs(ratio - expected_ratio) < 1e-6, (
            f"Z ratio={ratio:.4f}, expected {expected_ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# Sensitivity / error propagation tests
# ---------------------------------------------------------------------------

class TestDepthSensitivity:

    def test_sensitivity_formula(self, estimator):
        """
        Verify dZ/dr = -Z/r analytically.
        For r=25px, Z≈4.1m: sensitivity ≈ 4.1/25 ≈ 0.164 m/px
        """
        r = 25.0
        s = estimator.depth_sensitivity(r, delta_px=1.0)

        # Analytical: ΔZ = Z/r for 1px error
        expected_error = s["Z_m"] / r
        assert abs(s["depth_error_m"] - expected_error) < 1e-6, (
            f"Sensitivity error: {s['depth_error_m']:.4f} "
            f"expected {expected_error:.4f}"
        )

    def test_sensitivity_increases_at_distance(self, estimator):
        """
        Depth error per pixel is worse at larger distances (smaller radius).
        A ball far away (small r) has higher sensitivity than close (large r).
        """
        s_close = estimator.depth_sensitivity(radius_px=50.0)  # close
        s_far   = estimator.depth_sensitivity(radius_px=10.0)  # far
        assert s_far["depth_error_m"] > s_close["depth_error_m"], (
            "Far ball should have larger depth error per pixel"
        )

    def test_relative_error_is_constant(self, estimator):
        """
        Relative depth error = ΔZ/Z = delta_px/radius_px = constant.
        This means relative error depends only on relative pixel error,
        not on absolute distance. A key property of similar triangles.
        """
        # For 1px error, relative error = 1/radius_px regardless of distance
        for r in [10.0, 25.0, 50.0]:
            s = estimator.depth_sensitivity(radius_px=r, delta_px=1.0)
            expected_rel = 1.0 / r * 100  # percent
            assert abs(s["relative_error_pct"] - expected_rel) < 0.01, (
                f"r={r}: relative error={s['relative_error_pct']:.2f}%, "
                f"expected={expected_rel:.2f}%"
            )


# ---------------------------------------------------------------------------
# Position3D data container tests
# ---------------------------------------------------------------------------

class TestPosition3D:

    def test_valid_position(self):
        pos = Position3D(
            frame=1, time_s=0.5,
            X=0.5, Y=-0.3, Z=3.2, distance=3.25,
            radius_px=20.0, cx_px=650.0, cy_px=340.0,
            source='yolo',
        )
        assert pos.valid

    def test_invalid_position_none_Z(self):
        pos = Position3D(
            frame=1, time_s=0.5,
            X=None, Y=None, Z=None, distance=None,
            radius_px=None, cx_px=None, cy_px=None,
            source='none',
        )
        assert not pos.valid

    def test_invalid_position_zero_Z(self):
        pos = Position3D(
            frame=1, time_s=0.5,
            X=0.0, Y=0.0, Z=0.0, distance=0.0,
            radius_px=0.0, cx_px=640.0, cy_px=360.0,
            source='yolo',
        )
        assert not pos.valid

    def test_csv_row_length(self):
        pos = Position3D(
            frame=5, time_s=1.5,
            X=0.1, Y=-0.2, Z=4.0, distance=4.01,
            radius_px=18.5, cx_px=660.0, cy_px=350.0,
            source='yolo',
        )
        assert len(pos.to_row()) == len(Position3D.csv_header())

    def test_csv_header_matches_row(self):
        """CSV header and row must have same length for Excel output."""
        header = Position3D.csv_header()
        pos    = Position3D(1, 0.0, 1.0, 2.0, 3.0, 4.0, 20.0, 640.0, 360.0, 'yolo')
        assert len(header) == len(pos.to_row())


# ---------------------------------------------------------------------------
# Resolution independence test
# ---------------------------------------------------------------------------

class TestResolutionIndependence:

    def test_same_world_position_at_different_resolutions(self):
        """
        A ball at the same world position should give the same Z
        regardless of resolution, as long as intrinsics are consistent.

        This test verifies that fx and radius_px scale together —
        a critical property for multi-resolution robustness.

        At 1080p: fx≈1397, ball at 2m appears as r≈48px
        At 720p:  fx≈931,  ball at 2m appears as r≈32px
        Both should give Z≈2m.
        """
        D_real = 0.22
        Z_true = 2.0  # metres

        for width, height in [(1920, 1080), (1280, 720), (640, 480)]:
            cam = MockCamera(width=width, height=height)
            est = BallEstimator3D(cam)

            # Compute what radius_px should be at this distance
            # From Z = (fx*D)/(2r) → r = (fx*D)/(2Z)
            r_px = (cam.fx * D_real) / (2.0 * Z_true)

            pos = est.estimate(1, 0.0, cam.ppx, cam.ppy, r_px,
                               ball_diameter_m=D_real)

            assert abs(pos.Z - Z_true) < 0.001, (
                f"Resolution {width}x{height}: Z={pos.Z:.4f}m, "
                f"expected {Z_true:.4f}m"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])