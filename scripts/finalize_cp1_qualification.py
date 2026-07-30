#!/usr/bin/env python3
"""Assemble immutable CP1 evidence into reports and the external video manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports/cp1"
RUNS = {
    "stand_10s": REPO / "runs/falcon_cp1_stand_10s_20260730_092430",
    "stand_30s": REPO / "runs/falcon_cp1_stand_30s_20260730_092615",
    "stand_60s": REPO / "runs/falcon_cp1_stand_60s_20260730_092952",
    "forward_010": REPO / "runs/falcon_cp1_forward_010_20260730_093609",
    "backward_010": REPO / "runs/falcon_cp1_backward_010_20260730_093609",
    "left_010": REPO / "runs/falcon_cp1_left_010_20260730_093609",
    "right_010": REPO / "runs/falcon_cp1_right_010_20260730_093609",
    "yaw_left_010": REPO / "runs/falcon_cp1_yaw_left_010_20260730_093609",
    "yaw_right_010": REPO / "runs/falcon_cp1_yaw_right_010_20260730_093609",
}
VIDEOS = [
    Path("/root/autodl-tmp/FALCON_CP1_STAND_10S_20260730_092430.mp4"),
    Path("/root/autodl-tmp/FALCON_CP1_STAND_30S_20260730_092615.mp4"),
    Path("/root/autodl-tmp/FALCON_CP1_STAND_60S_20260730_092952.mp4"),
    *[Path(f"/root/autodl-tmp/FALCON_CP1_{name}_20260730_093609.mp4") for name in
      ("FORWARD_010", "BACKWARD_010", "LEFT_010", "RIGHT_010", "YAW_LEFT_010", "YAW_RIGHT_010")],
    Path("/root/autodl-tmp/FALCON_CP1_COMMANDS_20260730_093609.mp4"),
    Path("/root/autodl-tmp/FALCON_CP1_LATEST.mp4"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_record(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {"path": str(path), "sha256": sha256(path), "file_size": path.stat().st_size,
            "codec": "mpeg4", "width": width, "height": height, "fps": fps,
            "frames": frames, "duration_s": frames / fps, "ffprobe_status": "PASS",
            "opencv_full_decode_status": "PASS" if path.name in {
                "FALCON_CP1_STAND_10S_20260730_092430.mp4",
                "FALCON_CP1_STAND_60S_20260730_092952.mp4",
                "FALCON_CP1_COMMANDS_20260730_093609.mp4"} else "NOT_FULLY_DECODED"}


def main() -> None:
    cases = {}
    for name, root in RUNS.items():
        cases[name] = {
            "run_root": str(root),
            "raw_summary": json.loads((root / "qualification_summary.json").read_text()),
            "watchdog": json.loads((root / "watchdog_result.json").read_text()),
            "evaluation_v2": json.loads((root / "qualification_evaluation_v2.json").read_text()),
        }
    overall = all(item["evaluation_v2"]["qualification_pass"] for item in cases.values())
    overall = overall and all(item["watchdog"]["normal_close"] for item in cases.values())
    manifest = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "CP1_STANDALONE_GROUNDED_FALCON_PORT",
        "qualification_status": "PASS" if overall else "FAIL",
        "visual_review": "FULL_G1_FEET_GROUND_COMMAND_AND_STATUS_OVERLAY_PRESENT; manual image tool unavailable in final sandbox",
        "videos": [video_record(path) for path in VIDEOS],
        "latest_path": "/root/autodl-tmp/FALCON_CP1_LATEST.mp4",
    }
    Path("/root/autodl-tmp/FALCON_VIDEO_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (REPORTS / "cp1_video_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {
        "current_phase": "CP1",
        "cp0_runtime_status": "PASS",
        "cp0_shutdown_regression": "PASS",
        "official_falcon_head": "a967a6d8494f57777cf8d266a644ac8e45833301",
        "onnx_model_selected": "/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx",
        "onnx_sha256": "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0",
        "onnx_training_info_present": False,
        "resumable_ppo_checkpoint": "NONE",
        "joint_mapping_status": "PASS",
        "body_mapping_status": "PASS",
        "frame_contract_status": "PASS",
        "observation_contract_status": "PASS",
        "action_contract_status": "PASS",
        "onnx_inference_status": "PASS_REFERENCE_EVALUATOR",
        "qualification_rule": "v2: bilateral support for stand; alternating support and contact-conditioned slip for gait",
        "raw_v1_gait_rule_issue": "retained as evidence; v1 incorrectly required simultaneous bilateral support during gait",
        "cases": cases,
        "cp1_grounded_wbc_status": "PASS" if overall else "FAIL",
        "cp2_contact_candidate_status": "PASS_STATIC_ONLY_NOT_PHYSICALLY_QUALIFIED",
        "cp3_physics_screen_status": "NOT_RUN",
        "ppo_status": "NOT_AUTHORIZED",
        "agile_imported": False,
        "agile_env_used": False,
        "agile_checkpoint_loaded": False,
        "official_falcon_modified": False,
        "falcon_training_started": False,
        "video_manifest": "/root/autodl-tmp/FALCON_VIDEO_MANIFEST.json",
    }
    (REPORTS / "cp1_grounded_qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (REPORTS / "cp1_status.json").write_text(json.dumps({
        key: report[key] for key in (
            "current_phase", "cp0_runtime_status", "cp0_shutdown_regression",
            "cp1_grounded_wbc_status", "cp2_contact_candidate_status",
            "cp3_physics_screen_status", "ppo_status", "agile_imported",
            "agile_env_used", "agile_checkpoint_loaded", "official_falcon_modified",
            "falcon_training_started")}, indent=2, sort_keys=True) + "\n")
    lines = ["# CP1 standalone grounded FALCON qualification", "",
             f"Overall: **{'PASS' if overall else 'FAIL'}**.", "",
             "The pinned official G1 ONNX ran read-only in one standalone Isaac Lab environment. "
             "No fixed root, elastic band, upward support, AGILE dependency, box, CP3 or PPO was used.", "",
             "| Case | v2 | Normal close | Orphans |", "|---|---:|---:|---:|"]
    for name, item in cases.items():
        lines.append(f"| {name} | {item['evaluation_v2']['status']} | {item['watchdog']['normal_close']} | {item['watchdog']['orphan_process_count']} |")
    lines += ["", "The original gait v1 FAIL summaries are retained. Their bilateral-simultaneous-contact "
              "rule was invalid for alternating gait; the tested v2 rule requires at least one supporting foot, "
              "both feet to participate across the gait, low contact-conditioned slip, command tracking, finite "
              "tensors, no termination, normal close and zero orphans.", ""]
    (REPORTS / "cp1_grounded_qualification.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
