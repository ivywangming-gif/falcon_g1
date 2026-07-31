# CP1.8 reward/PPO audit

The implementation audit found an observability/control defect: aggregate metrics are persisted, but per-term reward statistics and explained variance are not, and the runner does not stop when KL exceeds the registered desired value. This is an audit finding only; no new PPO is authorized in this campaign.

The affected implementation is `scripts/cp1_7_worker.py`: KL is computed at
line 616 but is not used as a stop gate, and the aggregate-only metric payload
is built at lines 624-633 and written at line 637. No fix commit or retraining
was produced in this campaign. Consequently this finding cannot be classified
as `TRAINING_IMPLEMENTATION_BUG_FIXED`.
