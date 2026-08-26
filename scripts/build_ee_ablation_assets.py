#!/usr/bin/env python3
"""Build the non-destructive three-EE ablation asset registry.

WRIST_ONLY is converted from the pinned official ``g1_29dof.urdf``.  The two
rubber-hand entries are explicit USD layers over the already validated current
side-video asset (B) and the committed palm-forward mount (C).  No sphere,
capsule, pad, or hand geometry is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
FALCON = Path("/root/autodl-tmp/robotics/falcon_sandbox/FALCON")
WRIST_URDF = FALCON / "humanoidverse/data/robots/g1/g1_29dof.urdf"
CURRENT_RUBBER_USD = Path("/root/autodl-tmp/robotics/falcon-g1-access-push/.cache/cp1_13r/g1_usd/g1_29dof_fakehand.usd")
PALM_FORWARD_ASSET = REPO / "artifacts/s2x_v22b0_palm_forward/g1_usd/g1_29dof_rubberhand_palm_forward.usda"
PALM_FORWARD_SHA = "8d06902ed918b1738eb0d0eefc09ad30851f12461af8c2a6c03e56f4a175872a"
CURRENT_RUBBER_SHA = "86135447c01f5cf6ace8afec763c3543677fb9b1932e3d8e241ae3d2f59c8750"
WRIST_URDF_SHA = "56d0c7b4e44351ee8e89e7b1f1fcc8b02d5ec5cd191e43239d2b06f039e75860"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_wrist_only_urdf(output: Path) -> Path:
    """Remove only the two rubber-hand links and fixed joints from official URDF."""

    tree = ET.parse(WRIST_URDF)
    root = tree.getroot()
    removed = {"left_rubber_hand", "right_rubber_hand", "left_hand_palm_joint", "right_hand_palm_joint"}
    for element in list(root):
        if element.tag == "link" and element.attrib.get("name") in removed:
            root.remove(element)
        elif element.tag == "joint" and element.attrib.get("name") in removed:
            root.remove(element)
    mesh_root = WRIST_URDF.parent / "meshes"
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("meshes/"):
            mesh.set("filename", str(mesh_root / filename.removeprefix("meshes/")))
    remaining = {element.attrib.get("name") for element in root if element.tag in {"link", "joint"}}
    if removed & remaining:
        raise RuntimeError(f"WRIST_ONLY_STRIP_FAILED:{sorted(removed & remaining)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/tmp/falcon_push_ee_ablation_assets"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not WRIST_URDF.is_file():
        raise FileNotFoundError(WRIST_URDF)
    if sha256(WRIST_URDF) != WRIST_URDF_SHA:
        raise RuntimeError(f"WRIST_ONLY_URDF_SHA_MISMATCH:{sha256(WRIST_URDF)}")
    if not CURRENT_RUBBER_USD.is_file() or sha256(CURRENT_RUBBER_USD) != CURRENT_RUBBER_SHA:
        raise RuntimeError("RUBBER_BACK_CURRENT_SOURCE_CONTRACT_FAILED")
    if not PALM_FORWARD_ASSET.is_file() or sha256(PALM_FORWARD_ASSET) != PALM_FORWARD_SHA:
        raise RuntimeError("RUBBER_PALM_FORWARD_ASSET_CONTRACT_FAILED")

    # IsaacLab must be imported only after AppLauncher construction.
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        enable_extension("isaacsim.asset.importer.urdf")
        wrist_root = output_root / "wrist_only"
        wrist_root.mkdir(parents=True, exist_ok=True)
        stripped_urdf = make_wrist_only_urdf(wrist_root / "g1_29dof_wrist_only.urdf")
        config = UrdfConverterCfg(
            asset_path=str(stripped_urdf),
            usd_dir=str(wrist_root),
            usd_file_name="g1_29dof_wrist_only.usd",
            fix_base=False,
            merge_fixed_joints=True,
            force_usd_conversion=bool(args.force),
            make_instanceable=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
        converter = UrdfConverter(config)
        wrist_usd = Path(converter.usd_path).resolve()
        if not wrist_usd.is_file():
            raise RuntimeError(f"WRIST_ONLY_USD_MISSING:{wrist_usd}")
        wrist_manifest = {
            "variant": "WRIST_ONLY",
            "source_urdf": str(WRIST_URDF),
            "source_urdf_sha256": WRIST_URDF_SHA,
            "conversion_urdf": str(stripped_urdf),
            "conversion_urdf_sha256": sha256(stripped_urdf),
            "asset": str(wrist_usd),
            "asset_sha256": sha256(wrist_usd),
            "converter": "isaaclab.sim.converters.UrdfConverter",
            "merge_fixed_joints": True,
            "no_extra_geometry": True,
            "wrist_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "contact_sensor_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (wrist_root / "WRIST_ONLY_ASSET_CONTRACT.json").write_text(json.dumps(wrist_manifest, indent=2, sort_keys=True) + "\n")
    finally:
        # Keep the registry write below reachable even when IsaacSim emits
        # cleanup warnings.  The app is closed after the durable provenance
        # files have been written.
        pass

    back = REPO / "artifacts/ee_ablation/g1_usd/g1_29dof_rubberhand_back_current.usda"
    registry = {
        "campaign": "FALCON_PUSH_PATH_FEEDBACK_EE_ABLATION",
        "variants": {
            "WRIST_ONLY": {
                "asset": str(wrist_usd),
                "asset_sha256": sha256(wrist_usd),
                "source": str(WRIST_URDF),
                "source_sha256": WRIST_URDF_SHA,
                "contact_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
                "has_rubber_hand": False,
                "has_added_sphere_capsule_pad": False,
            },
            "RUBBER_BACK_CURRENT": {
                "asset": str(back),
                "asset_sha256": sha256(back),
                "source": str(CURRENT_RUBBER_USD),
                "source_sha256": CURRENT_RUBBER_SHA,
                "contact_bodies": ["left_rubber_hand", "right_rubber_hand"],
                "has_rubber_hand": True,
                "mount_change": "none",
            },
            "RUBBER_PALM_FORWARD": {
                "asset": str(PALM_FORWARD_ASSET),
                "asset_sha256": PALM_FORWARD_SHA,
                "source_sha256": CURRENT_RUBBER_SHA,
                "contact_bodies": ["left_rubber_hand", "right_rubber_hand"],
                "has_rubber_hand": True,
                "mount_change": "fixed wrist-to-hand rotation only",
                "hand_mass_kg_per_side": 0.17,
                "palm_normal_world": [1.0, 0.0, 0.0],
            },
        },
        "falcon_contract": {"dof": 29, "policy_input": [1, 575], "policy_output": [1, 29]},
    }
    registry_path = REPO / "artifacts/ee_ablation/EE_VARIANTS.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps(registry, indent=2, sort_keys=True))
    app.close(wait_for_replicator=False, skip_cleanup=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
