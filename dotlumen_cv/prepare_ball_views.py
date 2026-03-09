"""
prepare_ball_views.py — Generate ball_views.pickle from DeepSportRadar dataset

Each entry in the pickle has:
    image_path  : str       — path to the .png image
    cx_px       : float     — ball centre x in image pixels  (projected)
    cy_px       : float     — ball centre y in image pixels  (projected)
    ball_center : [X, Y, Z] — 3D world coordinates in mm
    visible     : bool
    arena, game_id, timestamp, camera_idx, raw

Usage:
    python prepare_ball_views.py --dataset-folder data/DeepSportRadar
"""

import os
import sys
import pickle
import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Camera calibration helpers
# ---------------------------------------------------------------------------

def load_calibration(game_folder: Path, timestamp: int) -> dict | None:
    """
    Load camcourt1_<timestamp>.json from the game folder.
    Returns the 'calibration' dict, or None if not found.
    """
    cal_path = game_folder / f"camcourt1_{timestamp}.json"
    if not cal_path.exists():
        return None
    with open(cal_path, 'r') as f:
        data = json.load(f)
    return data.get("calibration")


def project_world_to_image(center_mm: list, cal: dict) -> tuple[float, float] | None:
    """
    Project a 3D world point (mm) to 2D image pixel coordinates using
    the pinhole model with radial/tangential distortion.

    KK  — 3×3 intrinsic matrix (row-major flat list of 9)
    R   — 3×3 rotation matrix  (row-major flat list of 9)
    T   — translation vector   [Tx, Ty, Tz] in mm
    kc  — distortion coefficients [k1, k2, p1, p2, k3]

    Pipeline: X_cam = R @ X_world + T
              xn = X_cam[0]/X_cam[2], yn = X_cam[1]/X_cam[2]
              apply distortion
              u = fx*xn_d + cx,  v = fy*yn_d + cy
    """
    KK = np.array(cal["KK"], dtype=np.float64).reshape(3, 3)
    R  = np.array(cal["R"],  dtype=np.float64).reshape(3, 3)
    T  = np.array(cal["T"],  dtype=np.float64)          # shape (3,)
    kc = cal.get("kc", [0, 0, 0, 0, 0])

    X_world = np.array(center_mm, dtype=np.float64)     # (3,)
    X_cam   = R @ X_world + T                            # (3,)

    if X_cam[2] <= 0:
        return None   # behind camera

    # Normalised image coordinates
    xn = X_cam[0] / X_cam[2]
    yn = X_cam[1] / X_cam[2]

    # Radial + tangential distortion  (OpenCV model)
    k1, k2, p1, p2, k3 = (kc + [0]*5)[:5]
    r2 = xn*xn + yn*yn
    r4 = r2 * r2
    r6 = r4 * r2

    radial  = 1 + k1*r2 + k2*r4 + k3*r6
    xd = xn*radial + 2*p1*xn*yn       + p2*(r2 + 2*xn*xn)
    yd = yn*radial + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn

    fx, fy = KK[0, 0], KK[1, 1]
    cx, cy = KK[0, 2], KK[1, 2]

    u = fx * xd + cx
    v = fy * yd + cy

    return float(u), float(v)


def estimate_radius_px(center_mm: list, cal: dict,
                       ball_diameter_mm: float = 240.0) -> float | None:
    """
    Estimate the projected ball radius in pixels using the focal length
    and the distance from the camera to the ball centre.

    r_px = (fx * (D/2)) / Z_cam
    where Z_cam is the depth along the camera optical axis.
    """
    R_mat = np.array(cal["R"], dtype=np.float64).reshape(3, 3)
    T_vec = np.array(cal["T"], dtype=np.float64)
    KK    = np.array(cal["KK"], dtype=np.float64).reshape(3, 3)

    X_world = np.array(center_mm, dtype=np.float64)
    X_cam   = R_mat @ X_world + T_vec

    Z_cam = X_cam[2]
    if Z_cam <= 0:
        return None

    fx = KK[0, 0]
    radius_px = fx * (ball_diameter_mm / 2.0) / Z_cam
    return float(radius_px)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_image_and_calib(arena_folder: Path, game_id: int, timestamp: int,
                         offsets: list, camera_idx: int):
    """
    Returns (image_path, calibration_dict) or (None, None).
    """
    game_folder = arena_folder / str(game_id)
    if not game_folder.exists():
        return None, None

    # Image file: camcourt1_<timestamp>_<offset>.png
    image_path = None
    if offsets and camera_idx < len(offsets):
        offset    = offsets[camera_idx]
        candidate = game_folder / f"camcourt1_{timestamp}_{offset}.png"
        if candidate.exists():
            image_path = str(candidate)

    # Fallback scan
    if image_path is None:
        for f in game_folder.iterdir():
            if str(timestamp) in f.name and f.suffix == ".png":
                if offsets and camera_idx < len(offsets):
                    if f"_{offsets[camera_idx]}." in f.name:
                        image_path = str(f)
                        break
                else:
                    image_path = str(f)
                    break

    if image_path is None:
        return None, None

    cal = load_calibration(game_folder, timestamp)
    return image_path, cal


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------

def parse_dataset(dataset_folder: str):
    dataset_path = Path(dataset_folder)
    json_path    = dataset_path / "basketball-instants-dataset.json"

    if not json_path.exists():
        print(f"[ERROR] basketball-instants-dataset.json not found at {json_path}")
        sys.exit(1)

    print(f"Parsing {json_path} ...")
    with open(json_path, 'r') as f:
        records = json.load(f)

    print(f"Total records in JSON: {len(records)}")

    ball_views           = []
    skipped_no_image     = 0
    skipped_no_ball      = 0
    skipped_no_calib     = 0
    skipped_behind_cam   = 0

    for record in records:
        arena_label  = record.get("arena_label", "")
        game_id      = record.get("game_id")
        timestamp    = record.get("timestamp")
        offsets      = record.get("offsets", [0])
        annotations  = record.get("annotations", [])

        arena_folder = dataset_path / arena_label

        ball_anns = [a for a in annotations if a.get("type") == "ball"]
        if not ball_anns:
            skipped_no_ball += 1
            continue

        for ball_ann in ball_anns:
            camera_idx = ball_ann.get("image", 0)
            center_3d  = ball_ann.get("center")   # [X, Y, Z] world mm
            visible    = ball_ann.get("visible", True)

            image_path, cal = find_image_and_calib(
                arena_folder, game_id, timestamp, offsets, camera_idx
            )

            if image_path is None:
                skipped_no_image += 1
                continue

            if cal is None:
                skipped_no_calib += 1
                # Still keep the view, but without pixel projection
                view = {
                    "image_path":  image_path,
                    "arena":       arena_label,
                    "game_id":     game_id,
                    "timestamp":   timestamp,
                    "camera_idx":  camera_idx,
                    "ball_center": center_3d,
                    "cx_px":       None,
                    "cy_px":       None,
                    "radius_px":   None,
                    "visible":     visible,
                    "calibration": None,
                    "raw":         record,
                }
                ball_views.append(view)
                continue

            proj = project_world_to_image(center_3d, cal)
            if proj is None:
                skipped_behind_cam += 1
                continue

            cx_px, cy_px = proj
            radius_px    = estimate_radius_px(center_3d, cal,
                                              ball_diameter_mm=240.0)

            view = {
                "image_path":  image_path,
                "arena":       arena_label,
                "game_id":     game_id,
                "timestamp":   timestamp,
                "camera_idx":  camera_idx,
                "ball_center": center_3d,   # world mm
                "cx_px":       cx_px,
                "cy_px":       cy_px,
                "radius_px":   radius_px,
                "visible":     visible,
                "calibration": cal,
                "raw":         record,
            }
            ball_views.append(view)

    print(f"  Skipped (no ball annotation):  {skipped_no_ball}")
    print(f"  Skipped (image not found):     {skipped_no_image}")
    print(f"  Skipped (no calibration):      {skipped_no_calib}")
    print(f"  Skipped (ball behind camera):  {skipped_behind_cam}")
    print(f"  Collected: {len(ball_views)} ball views")

    print("\n  First 3 projected ball positions:")
    for v in ball_views[:3]:
        r_str = f"{v['radius_px']:.1f}" if v['radius_px'] else 'N/A'
        print(f"    {Path(v['image_path']).name}")
        print(f"      cx_px={v['cx_px']:.1f}, cy_px={v['cy_px']:.1f}, radius_px={r_str}")
        print(f"      ball_center_mm={v['ball_center']}, visible={v['visible']}")

    return ball_views


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ball_views.pickle from DeepSportRadar dataset"
    )
    parser.add_argument("--dataset-folder", required=True,
                        help="Path to DeepSportRadar root folder")
    args = parser.parse_args()

    dataset_folder = os.path.abspath(args.dataset_folder)

    if not os.path.isdir(dataset_folder):
        print(f"[ERROR] Folder not found: {dataset_folder}")
        sys.exit(1)

    print(f"Scanning: {dataset_folder}\n")

    ball_views = parse_dataset(dataset_folder)

    output_path = os.path.join(dataset_folder, "ball_views.pickle")
    with open(output_path, 'wb') as f:
        pickle.dump(ball_views, f)

    print(f"\n[OK] Saved {len(ball_views)} ball views → {output_path}")


if __name__ == "__main__":
    main()
