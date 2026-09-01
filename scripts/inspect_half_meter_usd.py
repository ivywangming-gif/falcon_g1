#!/usr/bin/env python3
"""Temporary read-only USD composition inspection for the half-meter assets."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    args = parser.parse_args()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False).app
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(str(args.asset), load=Usd.Stage.LoadAll)
        print("STAGE", stage)
        print("LAYERS", [layer.identifier for layer in stage.GetLayerStack()])
        print("PRIM_COUNT", sum(1 for _ in stage.TraverseAll()))
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for prim in stage.TraverseAll():
            path = str(prim.GetPath())
            if any(token in path.lower() for token in ("wrist", "rubber_hand", "collision", "collider", "mesh", "joints")):
                print("PRIM", path, "type=", prim.GetTypeName(), "apis=", prim.GetAppliedSchemas())
                for name in ("physics:rigidBodyEnabled", "physics:collisionEnabled", "physics:approximation", "physics:localPos0", "physics:localRot0", "physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes", "xformOp:translate", "xformOp:orient"):
                    attr = prim.GetAttribute(name)
                    if attr and attr.HasAuthoredValueOpinion():
                        print("  ATTR", name, "=", attr.Get())
                if prim.IsA(UsdGeom.Gprim):
                    try:
                        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        print("  BOUND", bound.GetMin(), bound.GetMax())
                    except Exception as exc:
                        print("  BOUND_ERROR", repr(exc))
        for q in ("/g1_29dof/left_wrist_yaw_link", "/g1_29dof/left_wrist_pitch_link", "/g1_29dof/left_wrist_roll_link", "/g1_29dof/left_rubber_hand"):
            prim = stage.GetPrimAtPath(q)
            print("BODY", q, prim.IsValid(), prim.GetTypeName(), prim.GetAppliedSchemas())
            if prim.IsValid():
                print(" CHILDREN", [str(child.GetPath()) for child in prim.GetChildren()])
                print(" STACK", [layer.identifier for spec in prim.GetPrimStack() for layer in [spec.layer]])
                try:
                    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    print(" BODY_BOUND", bound.GetMin(), bound.GetMax())
                except Exception as exc:
                    print(" BODY_BOUND_ERROR", repr(exc))
        return 0
    finally:
        try:
            app.close(wait_for_replicator=False, skip_cleanup=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
