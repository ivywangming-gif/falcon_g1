"""Asset and posture contracts for the half-meter experiment.

This file is intentionally a small, auditable registry.  It does not import
Isaac Lab at module import time, so the response-selection unit tests remain
simulator independent.  The V2 asset is an overlay on the immutable Natural
asset; no old C6 path is used by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .half_meter_executor import (
    FORMAL_EE_VARIANTS,
    OFFICIAL_FALCON_SHA256,
    PALM_DOWN_V2_SHA256,
    Q_UPPER_SHA256,
    RUBBER_HAND_MASS_PER_SIDE_KG,
)


NATURAL_SHA256 = "1c0d553c934c709c721128173d1ee9860ed28753fd685c036144fb976b3cecaa"
WRIST_ONLY_SHA256 = "f1f689012b0cd3af02959e13602d5ae6a422cdd273e75f98bd42f9ebcb19b3df"
# Historical C6 provenance only.  This is the observed immutable file hash;
# the old value used by an earlier registry described a different byte
# snapshot and must not be silently presented as evidence for this task.
OLD_C6_SHA256 = "88b582839a28025888eeab322d02d6aadeb26e45bfc3188212488c266e6b5f83"
NATURAL_RELATIVE_ASSET = "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda"
WRIST_ONLY_RELATIVE_ASSET = "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_wrist_only.usd"
V2_RELATIVE_ASSET = "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_down_v2.usda"
OLD_C6_RELATIVE_ASSET = "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_fingers_down_c6.usda"
FALCON_ONNX_ABSOLUTE = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER_RELATIVE = "configs/push_feedback/old_sphere_reference.json"
HAND_MESH_DIR = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/humanoidverse/data/robots/g1/meshes")
SIDES = ("left", "right")


@dataclass(frozen=True)
class AssetSpec:
    formal_name: str
    relative_path: str
    sha256: str
    contact_body_expected: tuple[str, str]
    has_rubber_hand: bool
    contact_class: str


ASSET_SPECS: Mapping[str, AssetSpec] = {
    "WRIST_ONLY": AssetSpec(
        "WRIST_ONLY", WRIST_ONLY_RELATIVE_ASSET, WRIST_ONLY_SHA256,
        ("left_wrist_yaw_link", "right_wrist_yaw_link"), False,
        "WRIST_ONLY_WRIST_CONTACT",
    ),
    "RUBBER_HAND_NATURAL": AssetSpec(
        "RUBBER_HAND_NATURAL", NATURAL_RELATIVE_ASSET, NATURAL_SHA256,
        ("left_rubber_hand", "right_rubber_hand"), True,
        "NATURAL_RUBBER_HAND_CONTACT",
    ),
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": AssetSpec(
        "RUBBER_HAND_PALM_FORWARD_DOWN_V2", V2_RELATIVE_ASSET, PALM_DOWN_V2_SHA256,
        ("left_rubber_hand", "right_rubber_hand"), True,
        "VISUAL_HAND_WITH_WRIST_DOMINANT_PUSHING",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(repo: Path, formal_name: str) -> Path:
    if formal_name not in ASSET_SPECS:
        raise ValueError(f"unknown formal EE: {formal_name}")
    return (Path(repo) / ASSET_SPECS[formal_name].relative_path).resolve()


def validate_frozen_files(repo: Path) -> dict[str, Any]:
    """Resolve and hash every immutable input before a simulator is started."""

    repo = Path(repo).resolve()
    onnx = FALCON_ONNX_ABSOLUTE
    q_upper = repo / Q_UPPER_RELATIVE
    files: dict[str, Any] = {
        "official_falcon": {
            "path": str(onnx), "present": onnx.is_file(),
            "observed_sha256": sha256_file(onnx) if onnx.is_file() else None,
            "expected_sha256": OFFICIAL_FALCON_SHA256,
        },
        "q_upper": {
            "path": str(q_upper), "present": q_upper.is_file(),
            "observed_sha256": sha256_file(q_upper) if q_upper.is_file() else None,
            "expected_sha256": Q_UPPER_SHA256,
        },
        "variants": {},
    }
    for name in FORMAL_EE_VARIANTS:
        spec = ASSET_SPECS[name]
        path = asset_path(repo, name)
        files["variants"][name] = {
            "path": str(path), "present": path.is_file(),
            "observed_sha256": sha256_file(path) if path.is_file() else None,
            "expected_sha256": spec.sha256,
            "contact_body_expected": list(spec.contact_body_expected),
            "has_rubber_hand": spec.has_rubber_hand,
            "contact_class": spec.contact_class,
        }
    failed = []
    for key in ("official_falcon", "q_upper"):
        item = files[key]
        if not item["present"] or item["observed_sha256"] != item["expected_sha256"]:
            failed.append(key)
    for name, item in files["variants"].items():
        if not item["present"] or item["observed_sha256"] != item["expected_sha256"]:
            failed.append(name)
    files["pass"] = not failed
    files["failed"] = failed
    if failed:
        raise RuntimeError(f"FROZEN_ASSET_INPUT_HASH_FAIL:{failed}:{files}")
    return files


def _normalize(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not np.isfinite(array).all() or norm <= 1.0e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return array / norm


def _orthogonal(value: Any, against: Any, name: str) -> np.ndarray:
    base = _normalize(value, name)
    ref = _normalize(against, f"{name}.reference")
    return _normalize(base - ref * float(base @ ref), name)


def fit_hand_landmarks(mesh: Any, side: str) -> dict[str, Any]:
    """Fit palm/finger axes from the actual visible STL, without fixed axes."""

    from sklearn.cluster import DBSCAN

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = vertices[faces].mean(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    wrist_region = vertices[vertices[:, 0] <= 0.015]
    distal = vertices[vertices[:, 0] >= 0.105]
    if len(wrist_region) < 100 or len(distal) < 100:
        raise RuntimeError(f"{side}: hand mesh landmark regions are too small")
    wrist_center = wrist_region.mean(axis=0)
    labels = DBSCAN(eps=0.003, min_samples=30).fit_predict(distal[:, 1:3])
    clusters = [distal[labels == label] for label in sorted(set(labels)) if label >= 0]
    clusters = [cluster for cluster in clusters if len(cluster) >= 500]
    if len(clusters) != 4:
        raise RuntimeError(f"{side}: expected four distal branches, got {len(clusters)}")
    clusters.sort(key=lambda cluster: float(cluster[:, 2].mean()), reverse=True)
    band = vertices[(vertices[:, 0] >= 0.045) & (vertices[:, 0] <= 0.075)]
    tips: list[np.ndarray] = []
    knuckles: list[np.ndarray] = []
    for cluster in clusters:
        tip_region = cluster[cluster[:, 0] >= np.quantile(cluster[:, 0], 0.98)]
        tip = tip_region.mean(axis=0)
        distances = np.linalg.norm(band[:, 1:3] - tip[1:3], axis=1)
        near = band[distances <= np.quantile(distances, 0.12)]
        tips.append(tip)
        knuckles.append(near.mean(axis=0))
    finger_axis = _normalize(tips[1] - wrist_center, f"{side}.finger_axis")
    palm_span = _orthogonal(knuckles[0] - knuckles[3], finger_axis, f"{side}.palm_span")
    raw_normal = _normalize(np.cross(finger_axis, palm_span), f"{side}.raw_palm_normal")
    region = (
        (centers[:, 0] >= 0.02) & (centers[:, 0] <= 0.085)
        & (centers[:, 2] >= knuckles[3][2] - 0.006)
        & (centers[:, 2] <= knuckles[0][2] + 0.006)
    )
    candidates = []
    for sign in (-1.0, 1.0):
        candidate = sign * raw_normal
        mask = region & (normals @ candidate > 0.65)
        candidates.append((float(areas[mask].sum()), sign, mask))
    _, dorsal_sign, _ = max(candidates, key=lambda item: item[0])
    palm_sign = -dorsal_sign
    palm_mask = region & (normals @ (palm_sign * raw_normal) > 0.65)
    area = float(areas[palm_mask].sum())
    if area <= 1.0e-4:
        raise RuntimeError(f"{side}: measured palm surface is empty")
    surface = _normalize((normals[palm_mask] * areas[palm_mask, None]).sum(axis=0), f"{side}.surface_normal")
    if float(surface @ (palm_sign * raw_normal)) < 0.0:
        surface *= -1.0
    palm_normal = _orthogonal(surface, finger_axis, f"{side}.palm_normal")
    if float(palm_normal @ (palm_sign * raw_normal)) < 0.0:
        palm_normal *= -1.0
    return {
        "wrist_center": wrist_center,
        "palm_center": (centers[palm_mask] * areas[palm_mask, None]).sum(axis=0) / area,
        "finger_axis": finger_axis,
        "palm_normal": palm_normal,
        "surface_normal": surface,
        "surface_area_m2": area,
        "surface_face_count": int(palm_mask.sum()),
    }


def quat_matrix_wxyz(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ), dtype=np.float64)


def body_pose_map(robot: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = [str(name).rsplit("/", 1)[-1] for name in robot.body_names]
    positions = np.asarray(robot.data.body_pos_w[0].detach().cpu().numpy(), dtype=np.float64)
    quaternions = np.asarray(robot.data.body_quat_w[0].detach().cpu().numpy(), dtype=np.float64)
    return {name: (positions[index], quaternions[index]) for index, name in enumerate(names)}


def runtime_posture_metrics(robot: Any, formal_name: str, landmarks: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Measure endpoint symmetry and (for V2) anatomical directions at runtime."""

    poses = body_pose_map(robot)
    endpoint_positions = []
    endpoint_orientations: dict[str, Any] = {}
    for side in SIDES:
        if formal_name == "WRIST_ONLY":
            body = f"{side}_wrist_yaw_link"
            position, quat = poses[body]
        else:
            body = f"{side}_rubber_hand"
            position, quat = poses[body]
        endpoint_positions.append(position)
        endpoint_orientations[side] = {"body": body, "position_world_m": position, "quaternion_wxyz": quat}
        if landmarks and side in landmarks:
            rotation = quat_matrix_wxyz(quat)
            endpoint_orientations[side].update({
                "palm_center_world_m": position + rotation @ np.asarray(landmarks[side]["palm_center"], dtype=float),
                "palm_normal_world": rotation @ np.asarray(landmarks[side]["palm_normal"], dtype=float),
                "finger_axis_world": rotation @ np.asarray(landmarks[side]["finger_axis"], dtype=float),
            })
    left, right = endpoint_positions
    result: dict[str, Any] = {
        "left_right_height_difference_m": abs(float(left[2] - right[2])),
        "left_right_forward_reach_difference_m": abs(float(left[0] - right[0])),
        "left_right_lateral_mirror_error_m": abs(float(left[1] + right[1])),
        "finite": bool(np.isfinite(np.concatenate(endpoint_positions)).all()),
        "endpoints": endpoint_orientations,
    }
    result["symmetry_pass"] = bool(
        result["finite"]
        and result["left_right_height_difference_m"] <= 0.01
        and result["left_right_forward_reach_difference_m"] <= 0.01
        and result["left_right_lateral_mirror_error_m"] <= 0.01
    )
    if landmarks and formal_name == "RUBBER_HAND_PALM_FORWARD_DOWN_V2":
        forward = np.asarray((1.0, 0.0, 0.0))
        down = np.asarray((0.0, 0.0, -1.0))
        normals = [np.asarray(endpoint_orientations[s]["palm_normal_world"], dtype=float) for s in SIDES]
        fingers = [np.asarray(endpoint_orientations[s]["finger_axis_world"], dtype=float) for s in SIDES]
        result["palm_forward_dots"] = [float(item @ forward) for item in normals]
        result["finger_down_dots"] = [float(item @ down) for item in fingers]
        result["orientation_pass"] = bool(
            all(item >= math.cos(math.radians(5.0)) for item in result["palm_forward_dots"])
            and all(item >= math.cos(math.radians(5.0)) for item in result["finger_down_dots"])
        )
    else:
        result["orientation_pass"] = True
    result["pass"] = bool(result["symmetry_pass"] and result["orientation_pass"])
    return result


def collision_body_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_wrist_yaw_link", f"{side}_wrist_pitch_link", f"{side}_wrist_roll_link"
    )


def _pxr_matrix(value: Any) -> np.ndarray:
    return np.asarray([[float(value[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def _pxr_quat_matrix(value: Any) -> np.ndarray:
    q = np.asarray((float(value.GetReal()), *[float(item) for item in value.GetImaginary()]), dtype=np.float64)
    return quat_matrix_wxyz(q)


def composed_fixed_joint_closure(asset: Path, side: str) -> dict[str, Any]:
    """Evaluate the composed USD fixed-joint closure at the authored pose.

    The equation and matrix convention intentionally match the previously
    qualified V2 audit: ``T_parent*T_parent_joint`` and
    ``T_child*T_child_joint`` are compared in the composed stage.  This is a
    read-only hard gate; it never derives a joint rotation by copying a child
    orientation.
    """

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD_OPEN_FAILED:{asset}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = "/g1_29dof"
    parent_path = f"{root}/{side}_wrist_yaw_link"
    child_path = f"{root}/{side}_rubber_hand"
    joint_path = f"{root}/joints/{side}_hand_palm_joint"
    parent_prim = stage.GetPrimAtPath(parent_path)
    child_prim = stage.GetPrimAtPath(child_path)
    joint = stage.GetPrimAtPath(joint_path)
    if not parent_prim.IsValid() or not child_prim.IsValid() or not joint.IsValid():
        raise RuntimeError(f"FIXED_JOINT_PRIMS_MISSING:{side}:{asset}")
    body0 = [str(item) for item in joint.GetRelationship("physics:body0").GetTargets()]
    body1 = [str(item) for item in joint.GetRelationship("physics:body1").GetTargets()]
    if body0 != [parent_path] or body1 != [child_path]:
        raise RuntimeError(f"FIXED_JOINT_BODY_CONTRACT_FAIL:{side}:{body0}:{body1}")
    parent = _pxr_matrix(cache.GetLocalToWorldTransform(parent_prim))
    child = _pxr_matrix(cache.GetLocalToWorldTransform(child_prim))
    p0 = np.asarray([float(item) for item in joint.GetAttribute("physics:localPos0").Get()], dtype=np.float64)
    p1 = np.asarray([float(item) for item in joint.GetAttribute("physics:localPos1").Get()], dtype=np.float64)
    q0_attr = joint.GetAttribute("physics:localRot0").Get()
    q1_attr = joint.GetAttribute("physics:localRot1").Get()
    q0 = _pxr_quat_matrix(q0_attr)
    q1 = _pxr_quat_matrix(q1_attr)
    parent_r = parent[:3, :3]
    child_r = child[:3, :3]
    parent_point = p0 @ parent_r.T + parent[3, :3]
    child_point = p1 @ child_r.T + child[3, :3]
    parent_frame = parent_r @ q0
    child_frame = child_r @ q1
    relative = parent_frame.T @ child_frame
    angle = float(math.acos(np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)))
    position = float(np.linalg.norm(parent_point - child_point))
    return {
        "body0": parent_path,
        "body1": child_path,
        "localPos0_m": p0,
        "localPos1_m": p1,
        "localRot0_wxyz": [float(q0_attr.GetReal()), *[float(item) for item in q0_attr.GetImaginary()]],
        "localRot1_wxyz": [float(q1_attr.GetReal()), *[float(item) for item in q1_attr.GetImaginary()],],
        "parent_joint_frame_world_m": parent_point,
        "child_joint_frame_world_m": child_point,
        "position_residual_m": position,
        "rotation_residual_rad": angle,
        "position_tolerance_m": 1.0e-5,
        "rotation_tolerance_rad": 1.0e-5,
        "pass": bool(position <= 1.0e-5 and angle <= 1.0e-5),
    }


def composed_rubber_hand_mass(asset: Path) -> dict[str, Any]:
    """Read the composed mass/COM/inertia properties of both rubber hands."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD_OPEN_FAILED:{asset}")
    result: dict[str, Any] = {"asset": str(asset), "sides": {}}
    for side in SIDES:
        prim = stage.GetPrimAtPath(f"/g1_29dof/{side}_rubber_hand")
        if not prim.IsValid():
            raise RuntimeError(f"RUBBER_HAND_PRIM_MISSING:{side}:{asset}")
        values: dict[str, Any] = {}
        for name in ("physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"):
            attr = prim.GetAttribute(name)
            if not attr or not attr.HasAuthoredValueOpinion():
                raise RuntimeError(f"RUBBER_HAND_PROPERTY_MISSING:{name}:{side}:{asset}")
            value = attr.Get()
            if hasattr(value, "GetReal"):
                values[name] = [float(value.GetReal()), *[float(item) for item in value.GetImaginary()]]
            elif isinstance(value, (float, int)):
                values[name] = float(value)
            else:
                values[name] = [float(item) for item in value]
        values["mass_pass"] = bool(abs(float(values["physics:mass"]) - RUBBER_HAND_MASS_PER_SIDE_KG) <= 1.0e-7)
        result["sides"][side] = values
    result["mass_per_side_kg"] = [float(result["sides"][side]["physics:mass"]) for side in SIDES]
    result["mass_pass"] = bool(all(result["sides"][side]["mass_pass"] for side in SIDES))
    return result


__all__ = [
    "AssetSpec", "ASSET_SPECS", "NATURAL_SHA256", "WRIST_ONLY_SHA256", "OLD_C6_SHA256",
    "NATURAL_RELATIVE_ASSET", "WRIST_ONLY_RELATIVE_ASSET", "V2_RELATIVE_ASSET",
    "OLD_C6_RELATIVE_ASSET", "FALCON_ONNX_ABSOLUTE", "Q_UPPER_RELATIVE", "HAND_MESH_DIR",
    "SIDES", "sha256_file", "asset_path", "validate_frozen_files", "fit_hand_landmarks",
    "quat_matrix_wxyz", "body_pose_map", "runtime_posture_metrics", "collision_body_names",
    "composed_fixed_joint_closure", "composed_rubber_hand_mass",
]
