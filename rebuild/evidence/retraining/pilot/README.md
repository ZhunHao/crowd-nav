# EC2 pilot — 4 October 2026

This is infrastructure and throughput evidence only. The training configuration and held-out protocol are not frozen; the ten full training runs and held-out evaluation have not run.

The approved pilot used one c6a.4xlarge (16 vCPUs, 32 GiB RAM) in ap-southeast-1, with a 60 GiB encrypted gp3 root volume, IMDSv2, no inbound security-group rules, SSM and prefix-scoped S3 permissions. Both an EventBridge stop schedule and on-host shutdown were armed. The schedule was re-armed after each instance restart without extending the original three-hour cap.

## Measurements

| CPU environments | Environment steps/s during PPO learning |
| --- | ---: |
| 1 | 64.47 |
| 2 | 104.40 |
| 4 | 166.94 |
| 8 | 307.29 |

The separate environment-only pilot measured 592.96 steps/s without the safety filter and 64.94 with it. These are short sizing measurements, not convergence or policy-quality results. The throughput source was `0a4747506852867bdcec5fe97931d55dd67ef689`; the successful recovery source was `be1cd8c1ee73d30c9f875d9731a45388e6b80d0e`, which only changes termination handling relative to it.

At the measured eight-environment rate, 100 chunks (2,048,000 steps) take approximately 1.85 learning hours per run, or 18.51 hours for ten runs. Process startup and checkpoint publication add overhead. A proposed 24-hour training allowance costs approximately US$19.17 using `cloud/budget.py`, including minimum one-day EBS storage, one month of 2 GiB S3 storage and a US$2 ancillary allowance. Selection and held-out evaluation are excluded; this proposal is not approval to launch full training.

## Failures and fixes

1. The first Linux smoke command disabled `pytestqt`, while the installed entry point is `pytest-qt`. Commit `ff18e33` corrects the flag and checks installed plugin metadata.
2. Repeating setup exposed SIGPIPE from `ldd --version | head -1` under `pipefail`. Commit `0a47475` removes the truncating pipe; a regression reproduces exit 141 before the fix.
3. An instance stop at a chunk boundary retained the first complete checkpoint but interrupted environment startup. Two subsequent mid-chunk systemd TERM checks saved no partial checkpoint and required the 120-second kill timeout. Diagnostic stack traces proved overlapping signals re-entered `threading.Event.set()` while holding its non-reentrant lock. Commit `be1cd8c` ignores further TERM before setting the stop flag. A deterministic subprocess regression reproduces the old deadlock and verifies handler restoration after the fix. An independent review found no blocking issue.

The failed attempt's two complete checkpoints remain in S3 and local ignored results; its status and curve are retained here. They are not candidates for full-training selection. No fixture or preregistration file was changed to fix these failures.

## Successful recovery

A fresh three-chunk run on the corrected source saved `chunk-0000` at 20,480 steps. A real EC2 stop during the next chunk published `partial-0001-26624` and `terminated`/143. After restart, an inactive-unit check preceded the resume command. The same logical chunk completed at 40,960, and the final chunk completed at 61,440. S3 then recorded `complete`/0.

Every downloaded manifest and payload SHA256 was verified. The final checkpoint ran successfully through the local marine service on the first generated selection scenario. This single episode confirms loading and execution; it is not a held-out evaluation or quality claim. The recorded source identity matched across all successful-run checkpoints. The Linux smoke suite passed 39 tests; the final local suite passed 534 tests (248 existing Qt deprecation warnings).

The pilot's conservative launch-to-final-stop-request duration was 0.646 hours, including intervening stopped time. Estimated compute is US$0.46; the storage-inclusive estimate with a US$1 ancillary allowance is US$1.67. This is an estimate, not an AWS invoice. The approved allowance was US$5.

`report.json` records hashes, curves, environment versions, SSM command identifiers, cost inputs and verification. Operational discovery, billing calculations and cleanup evidence are under ignored `cloud/local/`; checkpoint payloads are under ignored `results/pilot-recovery/` and S3. Pilot checkpoints must not be reused as one of the ten preregistered training runs.

Cleanup verified: the pilot instance is terminated and its stop schedule is deleted. Tag-filtered checks found no remaining volumes, snapshots, Elastic IPs, endpoints or NAT gateways. Temporary S3 source archives were removed; checkpoint payloads, curves and logs remain. IAM roles/profile and the security group are retained for the next approved training stage.
