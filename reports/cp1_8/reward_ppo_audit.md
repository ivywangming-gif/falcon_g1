# CP1.8 reward/PPO audit

The implementation audit found an observability/control defect: aggregate metrics are persisted, but per-term reward statistics and explained variance are not, and the runner does not stop when KL exceeds the registered desired value. This is an audit finding only; no new PPO is authorized in this campaign.
