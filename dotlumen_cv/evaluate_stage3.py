"""
evaluate_stage3.py — Stage 3 & 4: Trajectory Overlay + Top-View 2D Map

Reads the 3D positions CSV from Stage 2 and produces:
  1. output_trajectory.avi  — original video with fading trajectory trail
                               + parabola fit overlay (Stage 3 deliverable)
  2. output_topview.avi     — original video + side-by-side top-view 2D map
                               (Stage 4 deliverable)

Usage:
    # Run Stages 3+4 using pre-computed CSV from Stage 2:
    python evaluate_stage3.py --video ../rgb.avi \
                               --positions 3D_positions_stage2.csv

    # Run full pipeline from scratch (Stages 1+2+3+4):
    python evaluate_stage3.py --video ../rgb.avi --full_pipeline
"""

import argparse
import csv
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage3")

from config import camera, ENABLE_CAMERA_COMPENSATION, KALMAN_MAX_MISSED_FRAMES
from estimator import Position3D
from visualizer import TrajectoryVisualizer, TopViewMap, run_stage3_4


# ---------------------------------------------------------------------------
# Load positions from CSV
# ---------------------------------------------------------------------------

def load_positions_csv(path: str) -> list:
    """
    Load 3D positions from Stage 2 CSV output.
    Returns list of Position3D objects, one per frame.
    """
    positions = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            def parse(key):
                v = row.get(key, '')
                return float(v) if v else None

            pos = Position3D(
                frame=int(row['frame']),
                time_s=float(row['time_s']),
                X=parse('X_m'),
                Y=parse('Y_m'),
                Z=parse('Z_m'),
                distance=parse('distance_m'),
                radius_px=parse('radius_px'),
                cx_px=parse('cx_px'),
                cy_px=parse('cy_px'),
                source=row.get('source', 'unknown'),
            )
            positions.append(pos)

    valid = sum(1 for p in positions if p.valid)
    logger.info(
        "Loaded %d positions from %s (%d valid)",
        len(positions), path, valid,
    )
    return positions


# ---------------------------------------------------------------------------
# Full pipeline runner (Stages 1+2+3+4 combined)
# ---------------------------------------------------------------------------

def run_full_pipeline(video_path: str,
                      output_traj:    str = "output_trajectory.avi",
                      output_topview: str = "output_topview.avi",
                      csv_3d:         str = "3D_positions_stage2.csv",
                      excel_3d:       str = "3D_positions_stage2.xlsx") -> None:
    """
    Run the complete pipeline end-to-end:
    Stage 1: Detection + Kalman + CMC
    Stage 2: 3D Estimation
    Stage 3: Trajectory overlay
    Stage 4: Top-view map
    """
    from detector import BallDetector, DetectionResult
    from estimator import BallEstimator3D
    from motion_compensator import CameraMotionCompensator
    from evaluate_stage1 import SimpleKalmanFilter
    from evaluate_stage2 import save_excel
    import csv as csv_module

    camera.load(video_path)

    detector    = BallDetector()
    estimator   = BallEstimator3D(camera)
    kf          = SimpleKalmanFilter(dt=camera.dt)
    compensator = (CameraMotionCompensator()
                   if ENABLE_CAMERA_COMPENSATION else None)
    traj_vis    = TrajectoryVisualizer()
    top_map     = TopViewMap(world_range_m=12.0)

    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    map_size = min(camera.frame_height, 600)
    out_traj = cv2.VideoWriter(
        output_traj, fourcc, camera.fps,
        (camera.frame_width, camera.frame_height)
    )
    out_top = cv2.VideoWriter(
        output_topview, fourcc, camera.fps,
        (camera.frame_width + map_size, camera.frame_height)
    )

    positions          = []
    consecutive_misses = 0
    frame_idx          = 0

    # CSV writer for 3D positions
    csv_f      = open(csv_3d, 'w', newline='')
    csv_writer = csv_module.writer(csv_f)
    csv_writer.writerow(Position3D.csv_header())

    logger.info("Full pipeline: processing %d frames...", camera.total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_s     = (frame_idx - 1) / camera.fps

        # --- Stage 1: Detection ---
        kalman_pos = allow_hough = None
        allow_hough = True
        if kf.initialised:
            if consecutive_misses < KALMAN_MAX_MISSED_FRAMES:
                kalman_pos = kf.position
            elif consecutive_misses >= KALMAN_MAX_MISSED_FRAMES * 2:
                allow_hough = False

        result = detector.detect(frame, kalman_pos=kalman_pos,
                                  allow_hough=allow_hough)

        # --- CMC ---
        if compensator is not None:
            if frame_idx == 1:
                compensator.initialise(frame)
            else:
                if result is not None:
                    cx_c, cy_c = compensator.compensate(
                        frame, result.cx, result.cy, result.radius_px
                    )
                    result = DetectionResult(
                        cx=cx_c, cy=cy_c,
                        radius_px=result.radius_px,
                        confidence=result.confidence,
                        source=result.source,
                    )
                compensator.update(
                    frame,
                    ball_cx=result.cx if result else None,
                    ball_cy=result.cy if result else None,
                    ball_radius_px=result.radius_px if result else None,
                )

        # --- Kalman ---
        if result is not None and not kf.initialised:
            kf.initialise(result.cx, result.cy, result.radius_px)

        if kf.initialised:
            kf.predict()
            if result is not None:
                kf.update([result.cx, result.cy, result.radius_px])
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if consecutive_misses >= KALMAN_MAX_MISSED_FRAMES:
                    kf.x[2, 0] = 0.0
                    kf.x[3, 0] = 0.0
                kf.x[0, 0] = float(np.clip(kf.x[0, 0], 0, camera.frame_width))
                kf.x[1, 0] = float(np.clip(kf.x[1, 0], 0, camera.frame_height))

        # --- Stage 2: 3D Estimation ---
        if kf.initialised and result is not None:
            kf_cx, kf_cy = kf.position
            pos = estimator.estimate(
                frame=frame_idx, time_s=time_s,
                cx_px=kf_cx, cy_px=kf_cy,
                radius_px=result.radius_px,
                source=result.source,
            )
        elif kf.initialised and consecutive_misses < KALMAN_MAX_MISSED_FRAMES:
            kf_cx, kf_cy = kf.position
            last_r = next(
                (p.radius_px for p in reversed(positions) if p.radius_px),
                None
            )
            if last_r:
                pos = estimator.estimate(
                    frame=frame_idx, time_s=time_s,
                    cx_px=kf_cx, cy_px=kf_cy,
                    radius_px=last_r, source='kalman',
                )
            else:
                pos = estimator.estimate_null(frame_idx, time_s)
        else:
            pos = estimator.estimate_null(frame_idx, time_s)

        positions.append(pos)
        csv_writer.writerow(pos.to_row())

        # --- Stage 3: Update visualisers ---
        if pos.valid and pos.cx_px and pos.cy_px:
            traj_vis.update(pos.cx_px, pos.cy_px, pos.time_s, pos.source)
            top_map.update(pos)

        # --- Stage 3: Draw trajectory overlay ---
        vis_traj = traj_vis.draw(frame.copy())
        if pos.valid and pos.cx_px:
            r  = max(5, int(pos.radius_px or 15))
            cv2.circle(vis_traj, (int(pos.cx_px), int(pos.cy_px)),
                       r, (0, 255, 0), 2)
        vis_traj = traj_vis.draw_hud(vis_traj, frame_idx, pos)
        out_traj.write(vis_traj)

        # --- Stage 4: Top-view composite ---
        map_canvas = top_map.render()
        map_canvas = cv2.resize(map_canvas, (map_size, camera.frame_height))
        combined   = np.hstack([frame, map_canvas])
        out_top.write(combined)

        if frame_idx % 50 == 0:
            logger.info("Frame %d/%d", frame_idx, camera.total_frames)

    cap.release()
    out_traj.release()
    out_top.release()
    csv_f.close()

    save_excel(positions, excel_3d)

    parabola = traj_vis.parabola_stats()
    logger.info("=" * 50)
    logger.info("FULL PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info("Trajectory video : %s", output_traj)
    logger.info("Top-view video   : %s", output_topview)
    logger.info("3D CSV           : %s", csv_3d)
    logger.info("3D Excel         : %s", excel_3d)
    if parabola:
        logger.info("Parabola a       : %.3f px/s² (%s)",
                    parabola["a_px_s2"], parabola["direction"])
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 3+4: Trajectory Overlay + Top-View Map"
    )
    p.add_argument("--video",       type=str, required=True)
    p.add_argument("--positions",   type=str, default=None,
                   help="Path to 3D_positions_stage2.csv (skip stages 1+2)")
    p.add_argument("--full_pipeline", action="store_true",
                   help="Run full pipeline from scratch")
    p.add_argument("--output_traj",   type=str,
                   default="output_trajectory.avi")
    p.add_argument("--output_topview", type=str,
                   default="output_topview.avi")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.full_pipeline:
        run_full_pipeline(
            video_path=args.video,
            output_traj=args.output_traj,
            output_topview=args.output_topview,
        )
    elif args.positions:
        # Fast path: load pre-computed 3D positions, skip stages 1+2
        camera.load(args.video)
        positions = load_positions_csv(args.positions)
        run_stage3_4(
            video_path=args.video,
            positions_3d=positions,
            output_traj=args.output_traj,
            output_topview=args.output_topview,
        )
    else:
        logger.error(
            "Provide either --positions (CSV from stage 2) "
            "or --full_pipeline flag."
        )
        sys.exit(1)