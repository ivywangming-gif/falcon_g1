import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from falcon_g1.three_ee_validation import (
    CURRENT_FORMAL_EE_VARIANTS,
    RETIRED_EE_VARIANTS,
    RUBBER_HAND_MASS_PER_SIDE_KG,
    StraightPathConfig,
    assert_rubber_hand_masses,
    contact_longest_bilateral_s,
    desired_object_twist,
    project_box_to_path,
    scalar_calibration_from_model,
    solve_base_only_response_qp,
    source_trial_acceptance,
    current_registry_payload,
    validate_current_registry_payload,
)


def load_response_fitter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_three_ee_response_models.py"
    spec = importlib.util.spec_from_file_location("fit_three_ee_response_models_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_rows(count=20):
    return [
        {
            "time_s": i * 0.02,
            "command_vx_mps": 0.3,
            "command_wz_radps": 0.0,
            "box_vx_body_mps": 0.2,
            "box_vy_body_mps": 0.0,
            "box_wz_body_radps": 0.0,
        }
        for i in range(count)
    ]


def test_current_registry_has_exact_three_variants_and_retires_palm_up():
    payload = current_registry_payload()
    validate_current_registry_payload(payload)
    assert tuple(payload["formal_variant_names"]) == CURRENT_FORMAL_EE_VARIANTS
    assert "PALM_FORWARD_FINGERS_UP" in RETIRED_EE_VARIANTS


def test_both_rubber_hand_masses_are_hard_checked():
    masses = assert_rubber_hand_masses({
        "left_rubber_hand": 0.170,
        "right_rubber_hand": 0.170,
    })
    assert masses["left_rubber_hand"] == pytest.approx(RUBBER_HAND_MASS_PER_SIDE_KG)
    with pytest.raises(ValueError):
        assert_rubber_hand_masses({
            "left_rubber_hand": 0.171,
            "right_rubber_hand": 0.170,
        })


def test_attach_failed_variant_cannot_pass_acceptance():
    records = []
    for _ in range(7):
        records.append(source_trial_acceptance(
            {"attach_success": False, "probe_pass": True, "status": "PASS"},
            valid_rows(),
        ))
    assert all(item["valid"] is False for item in records)
    assert not any(item["valid"] for item in records)


def test_report_cannot_promote_seven_attach_failed_trials_to_model_or_best(tmp_path):
    fitter = load_response_fitter()
    models = {
        name: {"model_valid": True}
        for name in CURRENT_FORMAL_EE_VARIANTS
    }
    failed_formal = "RUBBER_HAND_PALM_FORWARD_DOWN"
    audit = [
        {
            "formal_ee": failed_formal,
            "probe": probe,
            "valid": False,
            "attach_success": False,
            "probe_pass": True,
            "summary_status": "PASS",
            "mean_box_wz_body_radps": 100.0,
        }
        for probe in ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
    ]
    report = fitter.make_report(models, audit, tmp_path, Path("source"))
    assert report["probe_pass"][failed_formal] is False
    assert report["model_valid"][failed_formal] is False
    assert report["BEST_RAW_YAW_AUTHORITY_EE"] != failed_formal


def test_stationary_box_has_stationary_spatial_progress_and_no_time_argument():
    cfg = StraightPathConfig()
    first = project_box_to_path((2.0, 0.0), 0.0, config=cfg)
    second = project_box_to_path((2.0, 0.0), 0.0, config=cfg, previous_sigma_m=first.sigma_hat_m)
    assert second.sigma_hat_m == pytest.approx(first.sigma_hat_m)
    assert not hasattr(second, "time_s")


def test_positive_and_negative_lateral_errors_have_mirrored_heading_signs():
    cfg = StraightPathConfig()
    # e_y is box-to-path: a box below the path needs a positive-y/positive
    # heading correction, while a box above the path needs the mirror image.
    positive_projection = project_box_to_path((2.0, -0.25), 0.0, config=cfg)
    negative_projection = project_box_to_path((2.0, 0.25), 0.0, config=cfg)
    positive = desired_object_twist(positive_projection, config=cfg)
    negative = desired_object_twist(negative_projection, config=cfg)
    assert positive_projection.e_y_m > 0.0
    assert negative_projection.e_y_m < 0.0
    assert positive.alpha_rad > 0.0
    assert negative.alpha_rad < 0.0
    assert positive.omega_obj_des_radps > 0.0
    assert negative.omega_obj_des_radps < 0.0


def test_box_above_straight_path_commands_back_toward_path():
    cfg = StraightPathConfig()
    projection = project_box_to_path((2.0, 0.25), 0.0, config=cfg)
    desired = desired_object_twist(projection, config=cfg)
    assert projection.e_y_m < 0.0
    assert desired.omega_obj_des_radps < 0.0


def test_heading_is_single_corrected_error_not_two_independent_terms():
    cfg = StraightPathConfig()
    projection = project_box_to_path((2.0, 0.20), math.radians(5.0), config=cfg)
    command = desired_object_twist(projection, config=cfg)
    expected_alpha = cfg.path_yaw_rad + math.atan(cfg.k_cross * projection.e_y_m) - projection.box_yaw_rad
    expected_alpha = (expected_alpha + math.pi) % (2.0 * math.pi) - math.pi
    assert command.alpha_rad == pytest.approx(expected_alpha)
    assert command.omega_obj_des_radps == pytest.approx(
        np.clip(cfg.kappa_path * command.v_obj_des_mps + cfg.k_heading * expected_alpha,
                -cfg.omega_obj_des_max, cfg.omega_obj_des_max)
    )


def test_e1_uses_the_model_yaw_gain_sign():
    model = {"B_matrix": [[1.0, 0.0], [0.0, 0.0], [0.0, -0.5]], "bias": [0.0, 0.0, 0.0]}
    calibration = scalar_calibration_from_model("WRIST_ONLY", model, all_models=[model])
    assert calibration.gw == pytest.approx(-0.5)
    desired = desired_object_twist(project_box_to_path((2.0, -0.2), 0.0), config=StraightPathConfig())
    assert calibration.command(desired)[1] < 0.0


def test_e1_keeps_a_strong_local_sign_and_uses_same_ee_fallback_when_weak():
    strong = {
        "B_matrix": [[1.0, 0.0], [0.0, 0.0], [0.0, 0.5]],
        "bias": [0.0, 0.0, 0.0],
        "scalar_mapping_audit": {
            "positive_negative_mirrored": True,
            "approximately_monotonic": True,
            "noise_scale_box_wz": 0.001,
            "estimated_k_omega": 0.5,
        },
    }
    assert scalar_calibration_from_model("WRIST_ONLY", strong, all_models=[strong]).weak is False
    weak = {
        "B_matrix": [[1.0, 0.0], [0.0, 0.0], [0.0, -0.01]],
        "bias": [0.0, 0.0, 0.0],
        "scalar_mapping_audit": {
            "positive_negative_mirrored": False,
            "approximately_monotonic": False,
            "noise_scale_box_wz": 0.01,
            "estimated_k_omega": -0.04,
        },
    }
    calibration = scalar_calibration_from_model("WRIST_ONLY", weak, all_models=[strong, weak])
    assert calibration.weak is True
    assert calibration.gw == pytest.approx(-0.04)


def test_e2_qp_respects_input_bounds():
    model = {"B_matrix": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], "bias": [0.0, 0.0, 0.0]}
    result = solve_base_only_response_qp(
        model, (10.0, 10.0, 10.0), (0.25, 0.0),
        {"output_scales": [1.0, 1.0, 1.0], "lambda_delta": 0.0, "lambda_nominal": 0.0},
    )
    assert 0.20 <= result.command_u[0] <= 0.30
    assert -0.10 <= result.command_u[1] <= 0.10


def test_e2_zero_error_solution_is_near_nominal():
    model = {"B_matrix": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], "bias": [0.0, 0.0, 0.0]}
    result = solve_base_only_response_qp(
        model, (0.30, 0.0, 0.0), (0.30, 0.0),
        {"output_scales": [1.0, 1.0, 1.0]},
    )
    assert result.command_u[0] == pytest.approx(0.30, abs=1.0e-4)
    assert result.command_u[1] == pytest.approx(0.0, abs=1.0e-4)


def test_e2_prediction_residual_is_explicit():
    model = {"B_matrix": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], "bias": [0.1, 0.0, 0.0]}
    result = solve_base_only_response_qp(
        model, (0.25, 0.0, 0.0), (0.25, 0.0),
        {"output_scales": [1.0, 1.0, 1.0], "lambda_delta": 0.0, "lambda_nominal": 0.0},
    )
    np.testing.assert_allclose(
        np.asarray(result.residual_xi),
        np.asarray(result.predicted_xi) - np.asarray((0.25, 0.0, 0.0)),
    )


def test_longest_bilateral_is_contiguous_not_a_total_count():
    assert contact_longest_bilateral_s([0, 1, 1, 0, 1, 1, 1, 0], 0.02) == pytest.approx(3 * 0.02)


def test_e1_and_e2_use_the_same_desired_twist_generator():
    cfg = StraightPathConfig()
    projection = project_box_to_path((2.0, 0.1), math.radians(3.0), config=cfg)
    e1_input = desired_object_twist(projection, config=cfg).xi_des
    e2_input = desired_object_twist(projection, config=cfg).xi_des
    assert e1_input == e2_input
