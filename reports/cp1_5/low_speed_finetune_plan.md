# Low-speed fine-tuning plan (design only)

No training is authorized or started. If CP1.5 evidence selects fine-tuning, the sampler must explicitly represent 0.00–0.30 m/s in 0.05 m/s increments and distinguish stand, walking, and low-speed transitions. The original `norm(command_xy) <= 0.2 -> 0` behavior cannot remain the only low-speed treatment.

The reward contract must separately cover longitudinal tracking, cross-axis suppression, zero-yaw straightness, heading/cross-track drift, low-speed foot slip, action smoothness, push-ready upper-body tracking, and recovery under small external load. Exact weights remain intentionally unset until the audit is reviewed.
