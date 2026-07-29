#!/usr/bin/env python3
"""Read-only RTX 5090/Torch/Isaac Gym compatibility gate.

This intentionally never starts PPO and never mutates an external environment.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def classify(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "no kernel image" in msg or "sm_120" in msg or "invalid device function" in msg:
        return "TORCH_NO_SM120"
    if "launch" in msg and "kernel" in msg:
        return "TORCH_KERNEL_LAUNCH_FAILED"
    if "physx" in msg and ("binary" in msg or "kernel" in msg or "sm_120" in msg):
        return "ISAAC_GYM_PHYSX_NO_SM120"
    return "OTHER_EXACT_REASON"


def main() -> int:
    out = Path(os.environ.get("T0_OUTPUT", "t0_gate.json"))
    result: dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": {},
        "isaac_gym": {},
        "official_training_started": False,
    }
    try:
        import torch

        t = result["torch"]
        assert isinstance(t, dict)
        t.update(
            {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
            }
        )
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available()=False")
        device = torch.device("cuda:0")
        t["device_name"] = torch.cuda.get_device_name(device)
        t["compute_capability"] = list(torch.cuda.get_device_capability(device))
        x = torch.arange(256, device=device, dtype=torch.float32)
        y = x * 2.0 + 1.0
        z = torch.matmul(torch.ones((32, 32), device=device), torch.eye(32, device=device))
        red = z.sum()
        torch.cuda.synchronize(device)
        t.update(
            {
                "cuda_tensor": True,
                "elementwise": bool(torch.equal(y, x * 2.0 + 1.0)),
                "matrix_multiply": bool(torch.allclose(z, torch.ones_like(z))),
                "reduction": float(red.detach().cpu()),
                "finite": bool(torch.isfinite(y).all() and torch.isfinite(z).all()),
                "synchronize": True,
                "subprocess_exit_code": 0,
            }
        )
        t["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001 - gate must record exact failure.
        t = result["torch"]
        assert isinstance(t, dict)
        t.update({"status": "FAIL", "failure_class": classify(exc), "error": repr(exc)})

    try:
        isaacgym = importlib.import_module("isaacgym")
        gymapi = importlib.import_module("isaacgym.gymapi")
        result["isaac_gym"] = {
            "status": "IMPORT_PASS",
            "module": getattr(isaacgym, "__file__", None),
            "gymapi": getattr(gymapi, "__file__", None),
        }
        gym = gymapi.acquire_gym()
        result["isaac_gym"]["acquire_gym"] = bool(gym)
    except Exception as exc:  # noqa: BLE001 - exact gate record.
        result["isaac_gym"] = {
            "status": "FAIL",
            "failure_class": "ISAAC_GYM_IMPORT_FAILED" if isinstance(exc, ModuleNotFoundError) else classify(exc),
            "error": repr(exc),
        }

    result["official_falcon_isaac_gym_rtx5090_compatibility"] = (
        "FAIL" if result["isaac_gym"].get("status") == "FAIL" else "RUNNING"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
