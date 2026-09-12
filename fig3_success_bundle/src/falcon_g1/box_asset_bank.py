"""Contracts for the pre-built box geometry bank used by General-SE(2) DR.

The important boundary in this module is intentional: dimensions are selected
when an environment is constructed, not when an episode is reset.  Isaac Sim
5.1 does not expose a supported way to resize a collision shape after the
GPU physics view has been initialized.  The simulator worker therefore uses
``MultiUsdFileCfg`` to assign one immutable USD asset to each environment and
only changes mass, material friction and center of mass at episode reset.

This module is simulator-free so that the assignment, dimension membership and
held-out provenance rules can be tested without starting Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


BOX_ASSET_BANK_SCHEMA = "BOX_ASSET_BANK_V1"
BOX_ASSET_BANK_VERSION = "V1"
BOX_LENGTHS_M = (1.35, 1.40, 1.45)
BOX_WIDTHS_M = (0.65, 0.70, 0.75)
BOX_HEIGHTS_M = (0.75, 0.80, 0.85)
NOMINAL_DIMS_M = np.asarray((1.40, 0.70, 0.80), dtype=np.float64)
RUNTIME_GEOMETRY_MUTATION = False


def _number_label(value: float) -> str:
    """Use a filesystem-safe, reversible label for a dimension in metres."""
    return f"{value:.2f}".replace(".", "p")


@dataclass(frozen=True)
class BoxAssetSpec:
    """One immutable visual/collision box asset."""

    asset_id: str
    dimensions_m: tuple[float, float, float]
    filename: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "dimensions_m": list(self.dimensions_m),
            "filename": self.filename,
            "geometry_mutable_at_runtime": False,
        }


def _make_specs() -> tuple[BoxAssetSpec, ...]:
    specs: list[BoxAssetSpec] = []
    for length in BOX_LENGTHS_M:
        for width in BOX_WIDTHS_M:
            for height in BOX_HEIGHTS_M:
                label = f"L{_number_label(length)}_W{_number_label(width)}_H{_number_label(height)}"
                specs.append(
                    BoxAssetSpec(
                        asset_id=f"box_{label}",
                        dimensions_m=(float(length), float(width), float(height)),
                        filename=f"{label}.usda",
                    )
                )
    return tuple(specs)


BOX_ASSET_SPECS = _make_specs()
BOX_ASSET_IDS = tuple(spec.asset_id for spec in BOX_ASSET_SPECS)
BOX_ASSET_BY_ID = {spec.asset_id: spec for spec in BOX_ASSET_SPECS}
NOMINAL_ASSET_ID = next(
    spec.asset_id for spec in BOX_ASSET_SPECS if np.allclose(spec.dimensions_m, NOMINAL_DIMS_M, atol=1e-12, rtol=0.0)
)


def asset_spec(asset_id: str) -> BoxAssetSpec:
    try:
        return BOX_ASSET_BY_ID[str(asset_id)]
    except KeyError as exc:
        raise ValueError(f"unknown BOX_ASSET_BANK_V1 asset id: {asset_id!r}") from exc


def dimensions_for_asset_id(asset_id: str) -> np.ndarray:
    return np.asarray(asset_spec(asset_id).dimensions_m, dtype=np.float64)


def asset_id_for_dimensions(dimensions_m: Iterable[float], atol: float = 1e-8) -> str:
    dimensions = np.asarray(tuple(dimensions_m), dtype=np.float64).reshape(3)
    matches = [
        spec.asset_id
        for spec in BOX_ASSET_SPECS
        if np.allclose(dimensions, np.asarray(spec.dimensions_m), atol=float(atol), rtol=0.0)
    ]
    if len(matches) != 1:
        raise ValueError(f"dimensions are not one exact BOX_ASSET_BANK_V1 member: {dimensions.tolist()}")
    return matches[0]


def dimensions_for_asset_ids(asset_ids: Iterable[str]) -> np.ndarray:
    ids = tuple(str(value) for value in asset_ids)
    if not ids:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray([asset_spec(value).dimensions_m for value in ids], dtype=np.float32)


def fixed_asset_assignment(num_envs: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic, balanced env-level assignment.

    The permutation is chosen once per simulator process.  Reusing an
    environment across episodes never changes its asset id.  Every complete
    block of 27 environments contains all bank members exactly once; a
    non-multiple has the prefix of the same seeded permutation.
    """
    count = int(num_envs)
    if count <= 0:
        raise ValueError("num_envs must be positive")
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(len(BOX_ASSET_IDS))
    ordered = np.asarray(BOX_ASSET_IDS, dtype=object)[permutation]
    return np.resize(ordered, count).astype(object)


def assignment_indices(asset_ids: Iterable[str]) -> np.ndarray:
    index = {asset_id: i for i, asset_id in enumerate(BOX_ASSET_IDS)}
    result = []
    for value in asset_ids:
        if str(value) not in index:
            raise ValueError(f"unknown BOX_ASSET_BANK_V1 asset id: {value!r}")
        result.append(index[str(value)])
    return np.asarray(result, dtype=np.int64)


def asset_paths(bank_root: Path, asset_ids: Iterable[str]) -> tuple[Path, ...]:
    root = Path(bank_root).resolve()
    return tuple(root / asset_spec(asset_id).filename for asset_id in asset_ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_manifest(bank_root: Path, asset_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the canonical manifest payload for the bank generator/auditor."""
    root = Path(bank_root).resolve()
    records = []
    for spec in BOX_ASSET_SPECS:
        record = spec.as_dict()
        record["path"] = str((root / spec.filename).resolve())
        if asset_hashes is not None and spec.asset_id in asset_hashes:
            record["sha256"] = str(asset_hashes[spec.asset_id])
        records.append(record)
    return {
        "schema": BOX_ASSET_BANK_SCHEMA,
        "version": BOX_ASSET_BANK_VERSION,
        "asset_count": len(records),
        "dimension_grid_m": {
            "length_m": list(BOX_LENGTHS_M),
            "width_m": list(BOX_WIDTHS_M),
            "height_m": list(BOX_HEIGHTS_M),
        },
        "nominal_asset_id": NOMINAL_ASSET_ID,
        "nominal_dimensions_m": NOMINAL_DIMS_M.tolist(),
        "visual_mesh_and_collision_geometry_prebuilt": True,
        "rigid_body_per_asset": True,
        "mass_inertia_metadata_per_asset": True,
        "runtime_geometry_mutation": False,
        "assets": records,
    }


def validate_manifest(
    manifest_or_path: Mapping[str, Any] | Path,
    bank_root: Path | None = None,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate structure and (optionally) on-disk files of a bank manifest."""
    if isinstance(manifest_or_path, (str, Path)):
        manifest_path = Path(manifest_or_path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = Path(bank_root).resolve() if bank_root is not None else manifest_path.parent
    else:
        payload = dict(manifest_or_path)
        root = Path(bank_root).resolve() if bank_root is not None else Path.cwd()

    errors: list[str] = []
    if payload.get("schema") != BOX_ASSET_BANK_SCHEMA:
        errors.append("schema")
    if payload.get("version") != BOX_ASSET_BANK_VERSION:
        errors.append("version")
    if int(payload.get("asset_count", -1)) != len(BOX_ASSET_SPECS):
        errors.append("asset_count")
    if payload.get("runtime_geometry_mutation") is not False:
        errors.append("runtime_geometry_mutation")
    if payload.get("visual_mesh_and_collision_geometry_prebuilt") is not True:
        errors.append("prebuilt_geometry")
    if payload.get("rigid_body_per_asset") is not True:
        errors.append("rigid_body_per_asset")
    if payload.get("mass_inertia_metadata_per_asset") is not True:
        errors.append("mass_inertia_metadata_per_asset")
    records = payload.get("assets")
    if not isinstance(records, list) or len(records) != len(BOX_ASSET_SPECS):
        errors.append("assets")
        records = []

    expected = {spec.asset_id: spec for spec in BOX_ASSET_SPECS}
    seen: set[str] = set()
    file_pass = True
    hash_pass = True
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("asset_record_type")
            file_pass = False
            continue
        asset_id = str(record.get("asset_id", ""))
        spec = expected.get(asset_id)
        if spec is None or asset_id in seen:
            errors.append(f"asset_id:{asset_id}")
            file_pass = False
            continue
        seen.add(asset_id)
        try:
            record_dims = np.asarray(record.get("dimensions_m", []), dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            record_dims = np.full(3, np.nan, dtype=np.float64)
        if not np.allclose(record_dims, spec.dimensions_m, atol=1e-8, rtol=0.0):
            errors.append(f"dimensions:{asset_id}")
        if record.get("geometry_mutable_at_runtime") is not False:
            errors.append(f"mutable:{asset_id}")
        if str(record.get("filename", "")) != spec.filename:
            errors.append(f"filename:{asset_id}")
        path = Path(str(record.get("path", root / spec.filename)))
        if not path.is_absolute():
            path = root / path
        if verify_files and not path.is_file():
            errors.append(f"missing:{asset_id}")
            file_pass = False
        if verify_hashes:
            if not record.get("sha256"):
                errors.append(f"sha256_missing:{asset_id}")
                hash_pass = False
            elif path.is_file():
                actual = sha256_file(path)
                if actual != str(record["sha256"]):
                    errors.append(f"sha256:{asset_id}")
                    hash_pass = False
        
    audited = payload.get("assets_audited")
    if not isinstance(audited, list) or len(audited) != len(BOX_ASSET_SPECS):
        errors.append("assets_audited")
    else:
        audit_by_id = {str(row.get("asset_id")): row for row in audited if isinstance(row, Mapping)}
        for spec in BOX_ASSET_SPECS:
            row = audit_by_id.get(spec.asset_id)
            if row is None:
                errors.append(f"audit_missing:{spec.asset_id}")
                continue
            required = (
                "visual_mesh_present", "collision_api_present", "rigid_body_api_present",
                "physx_rigid_body_api_present", "contact_report_api_present", "mass_api_present",
                "dimension_geometry_pass", "asset_schema_pass",
            )
            if any(row.get(key) is not True for key in required):
                errors.append(f"audit_schema:{spec.asset_id}")
            if row.get("runtime_geometry_mutation") is not False:
                errors.append(f"audit_mutable:{spec.asset_id}")

    complete = len(seen) == len(expected)
    if not complete:
        errors.append("asset_set_incomplete")
    structure_pass = not errors or all(not item.startswith(("missing:", "sha256:")) for item in errors)
    pass_value = bool(structure_pass and (file_pass if verify_files else True) and (hash_pass if verify_hashes else True))
    return {
        "schema": BOX_ASSET_BANK_SCHEMA,
        "asset_count": len(records),
        "expected_asset_count": len(expected),
        "all_expected_ids_present": complete,
        "structure_pass": bool(structure_pass),
        "files_pass": bool(file_pass if verify_files else True),
        "hashes_pass": bool(hash_pass if verify_hashes else True),
        "errors": errors,
        "BOX_ASSET_BANK_PASS": pass_value,
    }
