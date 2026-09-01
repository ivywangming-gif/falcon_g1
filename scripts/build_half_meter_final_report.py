#!/usr/bin/env python3
"""Assemble the measured-response/blockwise experiment audit.

This is a read-only collector over immutable raw runs and generated audit
files.  It does not start Isaac Sim, alter an asset, or schedule another
trial.  The report deliberately keeps exploratory single-side collision
filter runs separate from the frozen-physics response evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping


RUN_ROOT = Path(
    "/root/autodl-tmp/robotics/runs/"
    "falcon_half_meter_measured_response_blockwise_20260831"
)
REPO = Path(__file__).resolve().parents[1]
FORMAL = (
    "WRIST_ONLY",
    "RUBBER_HAND_NATURAL",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2",
)
ASSETS = {
    "WRIST_ONLY": REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_wrist_only.usd",
    "RUBBER_HAND_NATURAL": REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_back_current_filtered.usda",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": REPO / "artifacts/ee_ablation_sixway/g1_usd/g1_29dof_rubberhand_palm_forward_down_v2.usda",
}
EXPECTED_ASSET_SHA = {
    "WRIST_ONLY": "f1f689012b0cd3af02959e13602d5ae6a422cdd273e75f98bd42f9ebcb19b3df",
    "RUBBER_HAND_NATURAL": "1c0d553c934c709c721128173d1ee9860ed28753fd685c036144fb976b3cecaa",
    "RUBBER_HAND_PALM_FORWARD_DOWN_V2": "539f5818df16b43c34a45989706967a2e01c888d48af314522f3bd3ea056b7db",
}
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON/sim2real/models/falcon/g1_29dof.onnx")
Q_UPPER = REPO / "configs/push_feedback/old_sphere_reference.json"
EXPECTED_FALCON_SHA = "8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0"
EXPECTED_Q_SHA = "35a1078c9b72aed52dbe33764dd63f5834d62cfed369e1155271fee7fdae1453"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RUN_ROOT.resolve()))
    except ValueError:
        return str(path)


def command_output(*args: str, cwd: Path = REPO) -> str:
    try:
        return subprocess.run(
            list(args), cwd=str(cwd), check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def video_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "relative_to_run_root": rel(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path),
        "readable": False,
        "frame_count": None,
        "fps": None,
        "width": None,
        "height": None,
    }
    if not path.is_file() or record["bytes"] <= 0:
        return record
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        record["readable"] = bool(capture.isOpened())
        if record["readable"]:
            record["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            record["fps"] = float(capture.get(cv2.CAP_PROP_FPS))
            record["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            record["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Decode one frame as an additional integrity check.
            ok, _ = capture.read()
            record["readable"] = bool(ok)
        capture.release()
    except Exception as exc:  # pragma: no cover - environment-dependent
        record["video_probe_error"] = f"{type(exc).__name__}: {exc}"
    return record


def load_response_tables() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table_root = RUN_ROOT / "response_tables_corrected_active"
    tables: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for formal in FORMAL:
        path = table_root / f"{formal}.json"
        table = load(path, {})
        tables[formal] = table
        diagnostics = table.get("candidate_diagnostics", {})
        for item in table.get("responses", []):
            wz = float(item.get("wz_radps", 0.0))
            diagnostic = diagnostics.get(f"{wz:+.2f}", {})
            first = diagnostic.get("first_illegal_contact") or {}
            rows.append(
                {
                    "formal_ee": formal,
                    "wz_radps": wz,
                    "delta_s_m": item.get("delta_s_m"),
                    "delta_y_m": item.get("delta_y_m"),
                    "delta_yaw_rad": item.get("delta_yaw_rad"),
                    "effective_bilateral_fraction": item.get("effective_bilateral_fraction"),
                    "longest_effective_bilateral_s": diagnostic.get("longest_effective_bilateral_s"),
                    "cross_track_max_abs_m": item.get("cross_track_max_abs_m"),
                    "yaw_max_abs_rad": item.get("yaw_max_abs_rad"),
                    "completed": item.get("completed"),
                    "valid": item.get("valid"),
                    "fall": item.get("fall"),
                    "robot_leaves_box": item.get("robot_leaves_box"),
                    "first_illegal_sensor_body": first.get("sensor_body"),
                    "first_illegal_classification": first.get("classification"),
                    "first_illegal_time_s": first.get("time_s"),
                    "first_illegal_force_N": first.get("force_N"),
                    "source_dir": (diagnostic.get("source", {}) or {}).get("source_dir"),
                }
            )
    return tables, rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    seen: set[str] = set()
    for row in values:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: clean(value) for key, value in row.items()})


def response_video_dirs(tables: Mapping[str, Any]) -> list[tuple[str, Path]]:
    root = RUN_ROOT / "response_video_corrected"
    result: list[tuple[str, Path]] = []
    for formal, table in tables.items():
        for action in ("STRAIGHT", "LEFT_CORRECT", "RIGHT_CORRECT"):
            if table.get(action) is None:
                continue
            candidate = root / formal / action
            result.append((f"response_representative:{formal}:{action}", candidate))
    return result


def collect_videos(tables: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    directories: list[tuple[str, Path]] = response_video_dirs(tables)
    fallback_root = RUN_ROOT / "hand_only_fallback_opposite_endpoint"
    for formal in FORMAL:
        for side in ("LEFT", "RIGHT"):
            directories.append((f"hand_only_opposite_endpoint:{formal}:{side}", fallback_root / formal / f"{side}_HAND_ONLY"))
    # Keep the two control diagnostics explicitly separate from formal
    # endpoint-filter evidence.
    sensor_canary = RUN_ROOT / "hand_only_fallback_sensor_only" / "RUBBER_HAND_NATURAL" / "RIGHT_HAND_ONLY"
    directories.append(("hand_only_sensor_only_canary:NATURAL:RIGHT", sensor_canary))
    all_filter_root = RUN_ROOT / "hand_only_fallback"
    for formal in FORMAL:
        for side in ("LEFT", "RIGHT"):
            directories.append((f"hand_only_all_body_filter:{formal}:{side}", all_filter_root / formal / f"{side}_HAND_ONLY"))

    records: list[dict[str, Any]] = []
    for label, directory in directories:
        for name in ("top_local", "side_close", "front_contact"):
            path = directory / "videos" / f"{name}.mp4"
            record = video_record(path)
            record.update({"group": label, "camera": name})
            records.append(record)
    representative = [item for item in records if item["group"].startswith("response_representative:")]
    pass_gate = bool(representative) and all(
        item["present"] and item["bytes"] > 0 and item["readable"]
        for item in representative
    )
    return records, pass_gate


def fallback_record(path: Path, label: str) -> dict[str, Any]:
    measurement = load(path / "response_measurement.json", {}) or {}
    filter_audit = load(path / "single_side_collision_filter.json", {}) or {}
    first = measurement.get("first_illegal_contact") or {}
    return {
        "label": label,
        "path": str(path),
        "formal_ee": measurement.get("formal_ee"),
        "selected_side": measurement.get("selected_side"),
        "attached": measurement.get("attached"),
        "completed": measurement.get("completed"),
        "termination_reason": measurement.get("termination_reason"),
        "delta_s_m": measurement.get("delta_s_m"),
        "delta_y_m": measurement.get("delta_y_m"),
        "delta_yaw_rad": measurement.get("delta_yaw_rad"),
        "selected_endpoint_contact_fraction": measurement.get("selected_endpoint_contact_fraction"),
        "opposite_endpoint_contact_fraction": measurement.get("opposite_endpoint_contact_fraction"),
        "effective_bilateral_fraction": measurement.get("effective_bilateral_fraction"),
        "effective_contact_class": measurement.get("effective_contact_class"),
        "first_illegal_classification": first.get("classification"),
        "first_illegal_sensor_body": first.get("sensor_body"),
        "first_illegal_time_s": first.get("time_s"),
        "first_illegal_force_N": first.get("force_N"),
        "first_illegal_prim_paths": first.get("prim_paths"),
        "illegal_contact_event_count": measurement.get("illegal_contact_event_count"),
        "fall": measurement.get("fall"),
        "robot_leaves_box": measurement.get("robot_leaves_box"),
        "physical_filter_enabled": filter_audit.get("enabled"),
        "physical_filter_scope": filter_audit.get("scope"),
        "filtered_body_count": filter_audit.get("filtered_body_count"),
        "runtime_body_count": filter_audit.get("runtime_body_count"),
        "source_asset_modified": filter_audit.get("source_asset_modified"),
        "physics_unchanged": filter_audit.get("physics_unchanged", not bool(filter_audit.get("enabled"))),
        "measurement_file": str(path / "response_measurement.json"),
        "telemetry_file": str(path / "telemetry.csv"),
        "contact_events_file": str(path / "contact_events.json"),
        "timeline_file": str(path / "state_transition_timeline.json"),
    }


def collect_fallbacks() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = RUN_ROOT / "hand_only_fallback_opposite_endpoint"
    for formal in FORMAL:
        for side in ("LEFT", "RIGHT"):
            path = root / formal / f"{side}_HAND_ONLY"
            if (path / "response_measurement.json").is_file():
                records.append(fallback_record(path, f"opposite_endpoint_filter:{formal}:{side}"))
    sensor = RUN_ROOT / "hand_only_fallback_sensor_only" / "RUBBER_HAND_NATURAL" / "RIGHT_HAND_ONLY"
    if (sensor / "response_measurement.json").is_file():
        records.append(fallback_record(sensor, "sensor_only_canary:RUBBER_HAND_NATURAL:RIGHT"))
    all_filter = RUN_ROOT / "hand_only_fallback"
    for formal in FORMAL:
        for side in ("LEFT", "RIGHT"):
            path = all_filter / formal / f"{side}_HAND_ONLY"
            if (path / "response_measurement.json").is_file():
                records.append(fallback_record(path, f"all_body_filter:{formal}:{side}"))
    return records


def timeline_index(tables: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots: list[tuple[str, Path]] = []
    raw = RUN_ROOT / "response_campaign" / "response"
    for formal in FORMAL:
        for directory in sorted((raw / formal).glob("wz_*")):
            roots.append((f"raw_response:{formal}:{directory.name}", directory))
    for label, directory in response_video_dirs(tables):
        roots.append((label, directory))
    for record in collect_fallbacks():
        roots.append((record["label"], Path(record["path"])))
    result: list[dict[str, Any]] = []
    for label, directory in roots:
        path = directory / "state_transition_timeline.json"
        transitions = load(path, [])
        result.append({
            "label": label,
            "timeline_path": str(path),
            "present": path.is_file(),
            "transitions": transitions,
        })
    return result


def source_audit() -> dict[str, Any]:
    files = [
        REPO / "scripts/run_half_meter_response_trial.py",
        REPO / "scripts/run_half_meter_blockwise_trial.py",
        REPO / "scripts/run_half_meter_response_campaign.py",
        REPO / "src/falcon_g1/half_meter_executor.py",
        REPO / "src/falcon_g1/half_meter_assets.py",
    ]
    hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in files}
    runner = files[0].read_text(encoding="utf-8") if files[0].is_file() else ""
    blockwise = files[1].read_text(encoding="utf-8") if files[1].is_file() else ""
    # These are source-level guard checks, not a claim that prohibited words
    # can never occur in documentation strings.  Active construction is
    # checked through the actual contract fields and the no-training fields.
    active_contract_checks = {
        "response_path_controller_false": '"path_controller": False' in runner,
        "response_time_indexed_path_false": '"time_indexed_robot_path": False' in runner,
        "blockwise_continuous_path_controller_false": '"continuous_path_controller": False' in blockwise,
        "blockwise_E2_QP_false": '"E2_QP": False' in blockwise,
        "blockwise_integral_false": '"integral": False' in blockwise,
        "response_training_disabled": '"training_started": False' in runner,
        "response_ppo_zero": '"ppo_updates": 0' in runner,
        "blockwise_training_disabled": '"training_started": False' in blockwise,
        "blockwise_ppo_zero": '"ppo_updates": 0' in blockwise,
        "no_force_api_in_response_runner": not bool(re.search(r"apply_.*force|set_.*force|torque_controller", runner, re.I)),
    }
    return {
        "source_hashes": hashes,
        "active_contract_checks": active_contract_checks,
        "active_contract_checks_pass": all(active_contract_checks.values()),
        "note": "PPO/E2_QP/force/planner names retained only as prohibited-contract labels where present; no active training path was invoked.",
    }


def supervision_snapshot() -> dict[str, Any]:
    """Record whether any owned Isaac/scheduler process is still alive."""

    heartbeat_records: list[dict[str, Any]] = []
    for path in sorted(RUN_ROOT.rglob("heartbeat.json")):
        payload = load(path, {}) or {}
        pid = payload.get("pid")
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            pid_int = None
        heartbeat_records.append({
            "path": str(path),
            "payload": payload,
            "pid": pid_int,
            "pid_alive": bool(pid_int is not None and Path(f"/proc/{pid_int}").exists()),
        })
    process_rows: list[dict[str, Any]] = []
    ignored_pids: set[int] = set()
    current_pid = os.getpid()
    while current_pid > 1:
        ignored_pids.add(current_pid)
        try:
            current_pid = int(Path(f"/proc/{current_pid}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            break
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,cmd="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        pattern = re.compile(r"isaaclab|isaac.?sim|omni\.kit|run_half_meter", re.I)
        for line in output.splitlines():
            match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)", line)
            pid = int(match.group(1)) if match else -1
            if match and pid not in ignored_pids and pattern.search(match.group(4)):
                process_rows.append({
                    "pid": int(match.group(1)),
                    "ppid": int(match.group(2)),
                    "state": match.group(3),
                    "command": match.group(4),
                })
    except OSError:
        pass
    tmux = command_output("tmux", "ls")
    return {
        "active_isaac_or_half_meter_processes": process_rows,
        "active_process_count": len(process_rows),
        "campaign_processes_active": bool(process_rows),
        "heartbeat_records": heartbeat_records,
        "stale_heartbeat_count": sum(1 for item in heartbeat_records if not item["pid_alive"]),
        "tmux_listing": tmux,
        "snapshot_source": "ps and /proc; no process was started by this collector",
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    tables = report["response_tables"]
    requested = report["requested_fields"]
    yes_no = lambda value: "YES" if bool(value) else "NO"
    lines = [
        "# FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "",
        "## 结论",
        "",
        f"- `FINAL_STATUS={summary['final_status']}`",
        f"- `SELECTED_EE={summary['selected_ee']}`",
        f"- `NO_NATIVE_BIDIRECTIONAL_AUTHORITY={summary['no_native_bidirectional_authority']}`",
        f"- `READY_TO_RUN_BLOCKWISE={summary['ready_to_run_blockwise']}`",
        f"- `PALM_CONTACT_SCIENTIFIC_CLAIM_ALLOWED={summary['palm_contact_scientific_claim_allowed']}`",
        "",
        "## Requested machine-readable fields",
        "",
        f"`PALM_DOWN_V2_CONTACT_CLASS={requested['PALM_DOWN_V2_CONTACT_CLASS']}`",
        f"`PALM_FIRST_CONTACT_MARGIN_LEFT={requested['PALM_FIRST_CONTACT_MARGIN_LEFT']}`",
        f"`PALM_FIRST_CONTACT_MARGIN_RIGHT={requested['PALM_FIRST_CONTACT_MARGIN_RIGHT']}`",
        f"`WRIST_ONLY_RESPONSE_TABLE={requested['WRIST_ONLY_RESPONSE_TABLE']}`",
        f"`RUBBER_NATURAL_RESPONSE_TABLE={requested['RUBBER_NATURAL_RESPONSE_TABLE']}`",
        f"`PALM_DOWN_V2_RESPONSE_TABLE={requested['PALM_DOWN_V2_RESPONSE_TABLE']}`",
        f"`WRIST_ONLY_BIDIRECTIONAL_AUTHORITY={yes_no(requested['WRIST_ONLY_BIDIRECTIONAL_AUTHORITY'])}`",
        f"`RUBBER_NATURAL_BIDIRECTIONAL_AUTHORITY={yes_no(requested['RUBBER_NATURAL_BIDIRECTIONAL_AUTHORITY'])}`",
        f"`PALM_DOWN_V2_BIDIRECTIONAL_AUTHORITY={yes_no(requested['PALM_DOWN_V2_BIDIRECTIONAL_AUTHORITY'])}`",
        f"`BEST_EE={requested['BEST_EE']}`",
        f"`5M_PASS={requested['5M_PASS']}` `10M_PASS={requested['10M_PASS']}` `DOORWAY_PASS={requested['DOORWAY_PASS']}`",
        f"`FIG3B_PLAN_GENERATED={requested['FIG3B_PLAN_GENERATED']}` `FIG3B_EXECUTION_PASS={requested['FIG3B_EXECUTION_PASS']}`",
        f"`NO_SAFE_BIDIRECTIONAL_STEERING_AUTHORITY={yes_no(requested['NO_SAFE_BIDIRECTIONAL_STEERING_AUTHORITY'])}`",
        "",
        "native response calibration 的三种 EE 都只有一个可接受的 correction sign，另一方向没有通过 0.5 m validity/authority 门。因此没有启动 1 m mirror validation、5 m、10 m、doorway 或 Fig.3(b)，也没有启动 PPO。",
        "",
        "## 冻结与资产审计",
        "",
        f"- FALCON SHA256: `{report['frozen_inputs']['falcon_sha256']}`",
        f"- q_upper SHA256: `{report['frozen_inputs']['q_upper_sha256']}`",
        f"- V2 SHA256: `{report['assets']['RUBBER_HAND_PALM_FORWARD_DOWN_V2']['sha256']}`",
        f"- V2 mass: `{report['palm_down_v2']['mass_per_side_kg']} kg/side`",
        f"- V2 contact class: `{report['palm_down_v2']['contact_class']}`",
        f"- V2 palm-only claim: `{report['palm_down_v2']['palm_contact_scientific_claim_allowed']}`",
        f"- composed asset provenance: `{report['palm_down_v2']['provenance_pass']}`; unexpected stage differences: `{report['palm_down_v2']['unexpected_stage_difference_count']}`",
        "",
        "Palm-first support margins（静态、保守 composed-body bounds）:",
        "",
        f"- left: `{report['palm_down_v2']['support_margin_left_m']:.9f} m`",
        f"- right: `{report['palm_down_v2']['support_margin_right_m']:.9f} m`",
        "- 几何审计的 geometry child paths 在当前 composed traversal 中为空，因此 margin 来源标记为 `composed_body_collision_fallback`；该 caveat 没有被隐藏。",
        "",
        "## 0.5 m corrected response tables",
        "",
        "| EE | STRAIGHT wz | LEFT_CORRECT wz | RIGHT_CORRECT | native bidirectional | table SHA256 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for formal in FORMAL:
        table = tables[formal]
        fmt = lambda item: "NONE" if not item else f"{float(item['wz_radps']):+.2f}"
        lines.append(
            f"| {formal} | {fmt(table.get('STRAIGHT'))} | {fmt(table.get('LEFT_CORRECT'))} | {fmt(table.get('RIGHT_CORRECT'))} | {table.get('BIDIRECTIONAL_AUTHORITY')} | `{table.get('sha256')}` |"
        )
    lines += [
        "",
        "说明：V2 的 effective contact 是 wrist-dominant；所有表格都没有被解释为 palm-only contact。完整 21 candidate 数据和 active-interval 重算来源在 CSV/JSON 中。",
        "",
        "## 单侧 fallback 诊断",
        "",
        "全刚体过滤版使 selected endpoint force 全为 0，已标作 invalid exploratory evidence。sensor-only canary 中 selected 与 opposite 同时接触，不能归因于单侧。仅过滤对侧 endpoint 的六次对照能建立 selected contact，但每次都记录到相邻 wrist pitch/yaw 的真实非法 Box 接触，因此没有安全单侧 primitive。",
        "",
        "| label | selected fraction | opposite fraction | progress (m) | yaw (deg) | first illegal body | force (N) | usable authority |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for item in report["fallback_diagnostics"]:
        if not item["label"].startswith("opposite_endpoint_filter"):
            continue
        yaw = item.get("delta_yaw_rad")
        yaw_deg = "" if yaw is None else f"{math.degrees(float(yaw)):.3f}"
        force = "" if item.get("first_illegal_force_N") is None else f"{float(item['first_illegal_force_N']):.2f}"
        usable = bool(
            item.get("attached")
            and float(item.get("selected_endpoint_contact_fraction") or 0.0) >= 0.70
            and float(item.get("opposite_endpoint_contact_fraction") or 0.0) == 0.0
            and not item.get("first_illegal_sensor_body")
        )
        lines.append(
            f"| {item['label']} | {float(item.get('selected_endpoint_contact_fraction') or 0):.3f} | {float(item.get('opposite_endpoint_contact_fraction') or 0):.3f} | {float(item.get('delta_s_m') or 0):.4f} | {yaw_deg} | {item.get('first_illegal_sensor_body') or 'NONE'} | {force} | {usable} |"
        )
    lines += [
        "",
        "## Engineering gates",
        "",
        "- `1M_VALIDATION=NOT_RUN`（三动作表缺少 RIGHT_CORRECT）。",
        "- `5M_PASS=NOT_RUN`, `10M_PASS=NOT_RUN`, `DOORWAY_PASS=NOT_RUN`。",
        "- `FIG3B_PLAN_GENERATED=NO`, `FIG3B_EXECUTION_PASS=NOT_RUN`。",
        "- `training_started=false`, `ppo_updates=0`。",
        "",
        "## 监督收口",
        "",
        f"- 当前活跃 Isaac/half-meter 进程数: `{report['supervision']['active_process_count']}`。",
        f"- 旧 heartbeat 中已确认 PID 消失的记录数: `{report['supervision']['stale_heartbeat_count']}`。",
        "旧 scheduler 的 `response_campaign`/`response_campaign_v2` 目录保留原始证据，但其 `EXIT=143/137` 和 stale heartbeat 不被当作新的有效运行；最终候选数据只引用 21 个 raw probe 及其 active-interval 重算表。旧 `response_video` 目录也不作为代表性视频证据，代表性视频只引用 `response_video_corrected`。",
        "",
        "## 证据目录",
        "",
        f"- corrected response CSV: `{report['artifacts']['corrected_response_csv']}`",
        f"- fallback CSV: `{report['artifacts']['fallback_csv']}`",
        f"- video manifest: `{report['artifacts']['video_manifest_csv']}`",
        f"- timeline index: `{report['artifacts']['timeline_index_json']}`",
        f"- supervision status: `{report['artifacts']['supervision_status_json']}`",
        f"- asset provenance audit: `{report['artifacts']['asset_provenance_audit']}`",
        f"- variant equivalence audit: `{report['artifacts']['variant_equivalence_audit']}`",
        "",
        f"代表性 response 视频完整性门: `{report['video_evidence']['representative_video_evidence_pass']}`（共 {report['video_evidence']['representative_video_count']} 个视频）。",
        "",
        "本轮未 commit/push；隔离 worktree 和原 dirty branch 的 provenance 见 JSON。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    global RUN_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    RUN_ROOT = args.run_root.resolve()
    output = RUN_ROOT / "final_report"
    output.mkdir(parents=True, exist_ok=True)

    tables, response_rows = load_response_tables()
    corrected_csv = output / "corrected_response_metrics.csv"
    write_csv(corrected_csv, response_rows)
    fallbacks = collect_fallbacks()
    fallback_csv = output / "fallback_diagnostics.csv"
    write_csv(fallback_csv, fallbacks)
    videos, representative_pass = collect_videos(tables)
    video_csv = output / "video_manifest.csv"
    write_csv(video_csv, videos)
    video_json = output / "video_manifest.json"
    write_json(video_json, videos)
    timelines = timeline_index(tables)
    timeline_json = output / "state_transition_timeline_index.json"
    write_json(timeline_json, timelines)
    supervision = supervision_snapshot()
    supervision_json = output / "supervision_status.json"
    write_json(supervision_json, supervision)

    support = load(RUN_ROOT / "support_audit_runtime_v2" / "PALM_FIRST_SUPPORT_AUDIT_RUNTIME.json", {}) or {}
    provenance = load(RUN_ROOT / "asset_provenance_audit_v3" / "audit.json", {}) or load(RUN_ROOT / "asset_provenance_audit_v2" / "audit.json", {}) or {}
    equivalence_path = next(
        (
            RUN_ROOT / name
            for name in (
                "variant_equivalence_audit_v6.json",
                "variant_equivalence_audit_v5.json",
                "variant_equivalence_audit_v4.json",
                "variant_equivalence_audit_v3.json",
                "variant_equivalence_audit_v2.json",
            )
            if (RUN_ROOT / name).is_file()
        ),
        None,
    )
    equivalence = load(equivalence_path, {}) if equivalence_path else {}
    v2_support = support.get("variants", {}).get("RUBBER_HAND_PALM_FORWARD_DOWN_V2", {})
    v2_provenance = provenance.get("palm_down_v2", {})
    stage_diff = provenance.get("composed_stage_diff", {})

    assets: dict[str, Any] = {}
    for formal, path in ASSETS.items():
        observed = sha256_file(path)
        assets[formal] = {
            "path": str(path),
            "sha256": observed,
            "expected_sha256": EXPECTED_ASSET_SHA[formal],
            "sha_pass": observed == EXPECTED_ASSET_SHA[formal],
            "mass_per_side_kg": 0.170 if formal != "WRIST_ONLY" else None,
        }
    falcon_sha = sha256_file(FALCON)
    q_sha = sha256_file(Q_UPPER)

    response_table_summary: dict[str, Any] = {}
    for formal, table in tables.items():
        response_table_summary[formal] = {
            "path": str(RUN_ROOT / "response_tables_corrected_active" / f"{formal}.json"),
            "sha256": table.get("response_table_sha256"),
            "STRAIGHT": table.get("STRAIGHT"),
            "LEFT_CORRECT": table.get("LEFT_CORRECT"),
            "RIGHT_CORRECT": table.get("RIGHT_CORRECT"),
            "BIDIRECTIONAL_AUTHORITY": bool(table.get("BIDIRECTIONAL_AUTHORITY")),
        }
    no_native = all(not item["BIDIRECTIONAL_AUTHORITY"] for item in response_table_summary.values())
    endpoint_fallback = [item for item in fallbacks if item["label"].startswith("opposite_endpoint_filter")]
    safe_fallback = [
        item for item in endpoint_fallback
        if item.get("attached")
        and float(item.get("selected_endpoint_contact_fraction") or 0.0) >= 0.70
        and float(item.get("opposite_endpoint_contact_fraction") or 0.0) == 0.0
        and not item.get("first_illegal_sensor_body")
        and not item.get("fall")
    ]
    opposite_yaw_signs = {
        formal: {
            side: next((item.get("delta_yaw_rad") for item in endpoint_fallback if item.get("formal_ee") == formal and item.get("selected_side") == side.lower()), None)
            for side in ("LEFT", "RIGHT")
        }
        for formal in FORMAL
    }

    source = source_audit()
    git_branch = command_output("git", "branch", "--show-current")
    git_head = command_output("git", "rev-parse", "HEAD")
    git_status = command_output("git", "status", "--short", "--branch")
    original_branch = command_output("git", "-C", "/root/autodl-tmp/robotics/falcon-g1-access-push", "branch", "--show-current")
    original_head = command_output("git", "-C", "/root/autodl-tmp/robotics/falcon-g1-access-push", "rev-parse", "HEAD")

    report: dict[str, Any] = {
        "schema": "FALCON_HALF_METER_FINAL_REPORT.v1",
        "task": "FALCON_HALF_METER_MEASURED_RESPONSE_AND_BLOCKWISE_EXECUTOR",
        "generated_from": str(REPO),
        "summary": {
            "final_status": "NO_BIDIRECTIONAL_AUTHORITY" if no_native and not safe_fallback else "HARD_INFRASTRUCTURE_BLOCK",
            "selected_ee": "UNRESOLVED",
            "best_ee_for_blockwise": "NONE",
            "no_native_bidirectional_authority": no_native,
            "ready_to_run_blockwise": bool(not no_native),
            "one_meter_validation": "NOT_RUN",
            "5M_PASS": "NOT_RUN",
            "10M_PASS": "NOT_RUN",
            "DOORWAY_PASS": "NOT_RUN",
            "doorway_pilot_run": "NO",
            "fig3b_plan_generated": "NO",
            "fig3b_execution_pass": "NOT_RUN",
            "palm_contact_scientific_claim_allowed": False,
            "training_started": False,
            "ppo_updates": 0,
            "no_safe_bidirectional_steering_authority": len(safe_fallback) == 0,
        },
        "frozen_inputs": {
            "falcon_path": str(FALCON),
            "falcon_sha256": falcon_sha,
            "falcon_expected_sha256": EXPECTED_FALCON_SHA,
            "falcon_sha_pass": falcon_sha == EXPECTED_FALCON_SHA,
            "q_upper_path": str(Q_UPPER),
            "q_upper_sha256": q_sha,
            "q_upper_expected_sha256": EXPECTED_Q_SHA,
            "q_upper_sha_pass": q_sha == EXPECTED_Q_SHA,
            "nominal_speed_mps": 0.30,
            "physics_dt_s": 0.005,
            "control_decimation": 4,
            "response_timeout_s": 10.0,
            "fixed_path_length_m": 10.0,
            "progress_source": "actual_box_pose_projection",
            "elapsed_time_speed_product_forbidden": True,
        },
        "assets": assets,
        "palm_down_v2": {
            "geometry_qualified": True,
            "nobox_qualified": True,
            "short_behavioral_push": True,
            "true_palm_contact": False,
            "effective_contact": "WRIST_DOMINANT",
            "contact_class": "VISUAL_HAND_WITH_WRIST_DOMINANT_PUSHING",
            "mass_per_side_kg": 0.170,
            "palm_contact_scientific_claim_allowed": False,
            "provenance_pass": provenance.get("PROVENANCE_PASS"),
            "unexpected_stage_difference_count": len(stage_diff.get("unexpected_differences", [])),
            "support_margin_left_m": v2_support.get("LEFT_HAND_MINUS_WRIST_FORWARD_MARGIN"),
            "support_margin_right_m": v2_support.get("RIGHT_HAND_MINUS_WRIST_FORWARD_MARGIN"),
            "support_source": "composed_body_collision_fallback",
            "support_geometric_available": v2_support.get("PALM_FIRST_CONTACT_GEOMETRICALLY_AVAILABLE"),
        },
        "response_tables": response_table_summary,
        "requested_fields": {
            "PALM_DOWN_V2_CONTACT_CLASS": "VISUAL_HAND_WITH_WRIST_DOMINANT_PUSHING",
            "PALM_FIRST_CONTACT_MARGIN_LEFT": v2_support.get("LEFT_HAND_MINUS_WRIST_FORWARD_MARGIN"),
            "PALM_FIRST_CONTACT_MARGIN_RIGHT": v2_support.get("RIGHT_HAND_MINUS_WRIST_FORWARD_MARGIN"),
            "WRIST_ONLY_RESPONSE_TABLE": response_table_summary["WRIST_ONLY"]["path"],
            "RUBBER_NATURAL_RESPONSE_TABLE": response_table_summary["RUBBER_HAND_NATURAL"]["path"],
            "PALM_DOWN_V2_RESPONSE_TABLE": response_table_summary["RUBBER_HAND_PALM_FORWARD_DOWN_V2"]["path"],
            "WRIST_ONLY_BIDIRECTIONAL_AUTHORITY": response_table_summary["WRIST_ONLY"]["BIDIRECTIONAL_AUTHORITY"],
            "RUBBER_NATURAL_BIDIRECTIONAL_AUTHORITY": response_table_summary["RUBBER_HAND_NATURAL"]["BIDIRECTIONAL_AUTHORITY"],
            "PALM_DOWN_V2_BIDIRECTIONAL_AUTHORITY": response_table_summary["RUBBER_HAND_PALM_FORWARD_DOWN_V2"]["BIDIRECTIONAL_AUTHORITY"],
            "BEST_EE": "NONE",
            "5M_PASS": "NOT_RUN",
            "10M_PASS": "NOT_RUN",
            "DOORWAY_PASS": "NOT_RUN",
            "FIG3B_PLAN_GENERATED": "NO",
            "FIG3B_EXECUTION_PASS": "NOT_RUN",
            "NO_SAFE_BIDIRECTIONAL_STEERING_AUTHORITY": len(safe_fallback) == 0,
            "FINAL_STATUS": "NO_BIDIRECTIONAL_AUTHORITY" if no_native and not safe_fallback else "HARD_INFRASTRUCTURE_BLOCK",
        },
        "response_measurement": {
            "raw_candidate_count": len(response_rows),
            "corrected_interval_csv": str(corrected_csv),
            "correction_method": "ACTIVE/BRAKE/terminal SETTLE/DONE rows from recorded ATTACH->SETTLE->ACTIVE transition",
            "native_bidirectional_authority_by_ee": {formal: item["BIDIRECTIONAL_AUTHORITY"] for formal, item in response_table_summary.items()},
            "opposite_yaw_signs_in_single_side_diagnostics": opposite_yaw_signs,
        },
        "fallback_diagnostics": fallbacks,
        "fallback_conclusion": {
            "endpoint_filter_runs": len(endpoint_fallback),
            "safe_endpoint_filter_authority_count": len(safe_fallback),
            "all_body_filter_is_invalid_exploratory": True,
            "sensor_only_canary_is_not_isolated": True,
            "physical_filter_changes_collision_pair_set": True,
            "no_safe_bidirectional_steering_authority": len(safe_fallback) == 0,
            "reason": "Every endpoint-filter run recorded an adjacent wrist/forearm Box contact before/at short push completion; no run is clean authority evidence.",
        },
        "video_evidence": {
            "representative_video_evidence_pass": representative_pass,
            "representative_video_count": sum(1 for item in videos if item["group"].startswith("response_representative:")),
            "all_collected_video_count": len(videos),
            "manifest_json": str(video_json),
        },
        "supervision": supervision,
        "audits": {
            "asset_provenance": provenance,
            "variant_equivalence": equivalence,
            "variant_equivalence_pass": equivalence.get("ABC_OTHER_THAN_EE_DIFFERENCE_PASS"),
            "source_audit": source,
        },
        "git_provenance": {
            "isolated_worktree": str(REPO),
            "isolated_branch": git_branch,
            "isolated_head": git_head,
            "isolated_status_at_report": git_status,
            "original_worktree": "/root/autodl-tmp/robotics/falcon-g1-access-push",
            "original_branch": original_branch,
            "original_head": original_head,
            "commit_or_push_performed": False,
        },
        "artifacts": {
            "corrected_response_csv": str(corrected_csv),
            "fallback_csv": str(fallback_csv),
            "video_manifest_csv": str(video_csv),
            "video_manifest_json": str(video_json),
            "timeline_index_json": str(timeline_json),
            "supervision_status_json": str(supervision_json),
            "final_report_sha256": str(output / "FINAL_REPORT.sha256"),
            "asset_provenance_audit": str(RUN_ROOT / "asset_provenance_audit_v3" / "audit.json"),
            "variant_equivalence_audit": str(equivalence_path or (RUN_ROOT / "variant_equivalence_audit_v4.json")),
            "support_audit": str(RUN_ROOT / "support_audit_runtime_v2" / "PALM_FIRST_SUPPORT_AUDIT_RUNTIME.json"),
            "raw_response_root": str(RUN_ROOT / "response_campaign"),
            "representative_video_root": str(RUN_ROOT / "response_video_corrected"),
            "fallback_root": str(RUN_ROOT / "hand_only_fallback_opposite_endpoint"),
        },
    }
    report_json = output / "FINAL_REPORT.json"
    report["artifacts"]["final_report_json"] = str(report_json)
    report["artifacts"]["final_report_md"] = str(output / "FINAL_REPORT.md")
    # Hash the finalized payload excluding the hash field itself, then write
    # exactly that payload once so external provenance checks can reproduce it.
    report.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(clean({key: value for key, value in report.items() if key != "report_sha256"}), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    write_json(report_json, report)
    report_json.with_suffix(".sha256").write_text(sha256_file(report_json) + "\n", encoding="utf-8")
    (output / "FINAL_REPORT.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(clean(report["summary"]), indent=2, sort_keys=True))
    print(f"FINAL_REPORT_JSON={report_json}")
    print(f"FINAL_REPORT_MD={output / 'FINAL_REPORT.md'}")
    return 0 if report["summary"]["final_status"] == "NO_BIDIRECTIONAL_AUTHORITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
