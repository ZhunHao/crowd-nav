# Plan 04 native acceptance — 29 September 2026

**DONE_WITH_CONCERNS.** Functional native-window checks and export/smoke audits completed. A replay clock defect was fixed in `65aa0c2`. Smooth responsiveness while resizing Ubin during MPC remains **FAIL**: the full run had a 4.537 s heartbeat gap, and bounded probes localized 2.159–2.917 s main-thread canvas renders after resize. This is not an unqualified native acceptance pass.

Product base: `d0a3a5e`; replay correction: `65aa0c2`. Service/checkpoint results obtained before that correction remain applicable because it changes only GUI replay. Evidence is retained under ignored `rebuild/results/plan04-native/`, including failed attempts. The supplied `rl_model.pth` SHA-256 is `728cf759d3acf11990ed0b46f7dfe22beefffc6e8912b606d850ee5999608044`.

Method: actual displayed Cocoa Qt `Window`, production service/controller/checkpoint, QTest widget and Retina-corrected canvas clicks, 50 ms heartbeat, native move/resize, native window grabs and visual image inspection. File chooser responses were injected; exact Ubin metre coordinates used the production click handler to avoid pixel quantization. This was automated native acceptance, not a human manual session. The initial desktop capture establishes foreground visibility; later desktop captures can contain other apps and are not used to establish Ubin foreground visibility. No policy/draw mocks were used. Offscreen tests are separate evidence.

| Task 10 requirement | Outcome | Evidence / actual result |
|---|---|---|
| Complete suite once after product correction | PASS | `full-after-replay-fix.log`: 284 passed, 46.64 s, exit 0. |
| Default native harbour, supplied SARL, five ships, predictive filter | PASS | Initial native capture; `(2,2)` → `(22,22)`, seed 0; timeout at 100 s, 401 frames (`harbour-filtered.json`). Timeout is the recorded policy outcome. |
| Same-seed unfiltered collision and last diagnostic | PASS | Collision at 4.5 s; `land_collision=true`, `ship_collision=false`. Terminal `(0.469969,5.693821)` is 0.469969 m from map edge and 8.53003 m from island: **map edge**. Replayed terminal. |
| Run locks, move/resize, active Stop and rerun | PASS | Scene/config/duplicate-run locks checked; Stop cancelled at 2.8 simulated s, about 0.235 s after click; rerun completed with timeout. Playback-rate selector remains enabled; it is not a scene/config edit control. |
| Destination and two-click land edit change route; invalid land/endpoints rejected | PASS | `scene-checks.json`: destination near `(21,20)`, two land polygons; route changed to southern corridor. Messages: “New land blocks start or goal”; “Choose a point clear of land and map edges”. |
| Ubin harbour profile/corridor, toolbar zoom, exact metre endpoints, SARL run | PASS | Prescribed `(-4449.8,1383.9)` → `(4451.7,3038.1)` m. Fresh actual GUI SARL: timeout at 6268.409 physical s / 3134.205 model s, 12538 frames, 352.753 wall s, 175 resizes, maximum heartbeat gap 0.510 s. |
| Ubin MPC + marine run and export | PASS | Fresh actual GUI run: success at 2095 physical s / 1047.5 model s, 4191 frames, 40.936 wall s. It ran faster than the brief's descriptive expectation of over one minute. |
| Ubin MPC window stays responsive during resize | **FAIL** | Full-run maximum heartbeat gap 4.537 s. Separate actual bounded MPC probes reproduced 2.397 / 3.066 s gaps; main-thread resize paint is the observed blocking stage. See below. |
| Ubin 50× replay and terminal frame | PASS after correction | Fresh production-window probes: SARL 49.747× over 20.011 s; MPC 49.814× over 20.115 s. Full 200× replay reached terminal naturally in 31.511 / 10.657 s versus nominal 31.342 / 10.475 s. Original failed timing retained. |
| Coastal valid endpoint scaling; model profile refusal | PASS | Original start has 45.337 m shore clearance and is correctly cleared for coastal's 50 m clearance. Valid replacement `(-2000,500)` and original goal retain physical coordinates after rescaling. `model` refused: 101,379,210 cells exceed 250,000. |
| Southern Islands independent second geography | PASS | Independent non-heldout channel route, Direct + five ships: success at 337 physical s, 675 frames; actual run 1.712 s, replay 6.647 s at 50×. |
| Empty model error, controls restored, correct-model recovery | PASS | Missing `policy.config` error shown, controls restored; correct supplied model runs again. |
| Theta*, marine, noisy delayed observations; every-frame overlays/titles | PASS | Timeout at 100 s. All 401 frames checked against diagnostics: nominal/executed arrow geometry, prediction coordinates, override/no-feasible-action fields; terminal has no stale diagnostic. |
| JSON / CSV / PNG / MP4, coordinates and 10-second timing | PASS | Distinct `identity-timing.*` files; all 80 CSV rows match trace coordinates/times. Native 1× replay reached 10 simulated s in 10.224 wall s. MP4 frame 40 has PTS 10.000 s; all 80 video PTS values match trace timestamps. PNG and video key frames visually inspected. |
| ffprobe 4 FPS, every initial/terminal frame | PASS | Short identity trace: H.264, 4 FPS, 80/80 frames, 20.000 s file duration; terminal PTS 19.750 s. Real-map scaling follows recorded physical scale/default speedup, not this identity-scale expectation. |
| Impossible boundary-to-boundary wall | PASS | Corrected native setup preserves `(2,12)` → `(22,12)` and shows “Failed: No route at this grid resolution”; controls restored. |
| Eight-trace and 160-episode smoke, summaries inspected | PASS | Every JSON and summary row audited; paired traffic equal. 8: 2 success / 5 collision / 1 timeout. 160: 41 success / 110 collision / 9 timeout. All failures retained, no seed/count removal. |
| Retain placement error rows; documented reduced-count follow-up if needed | PASS / NOT-SUPPORTED | All rows retained; no placement error occurred in these 168 episodes, so the conditional reduced-count experiment was not applicable. |
| Native evidence beyond offscreen tests | PASS | Displayed Cocoa runs, heartbeat/move/resize, actual controls, screenshots and video inspections above. Task 12 heldout benchmark was not accessed. |

Replay previously counted timer callbacks, stretching time when rendering exceeded the timer interval: SARL's old bounded 50× segment achieved only 17.966×, MPC 18.965×; a full SARL attempt exceeded its driver timeout. The correction snapshots rate and physical frame times at Replay and selects by monotonic elapsed time, skipping stale frames. Eight deterministic fake-clock regressions cover delayed callbacks at 1×/10×/50×/200×, two scales, fractional terminal time, and rate snapshot. RED: eight failures; GREEN: eight passes. Old replay captures are superseded by `ubin-*-corrected-replay-checks.json` and fresh native screenshots.

Responsiveness localization used two actual MPC+marine episodes bounded to 200 model seconds solely for diagnosis, with fresh versus retained prior SARL trace. No screenshot, subprocess or file write occurred inside the measured worker interval. Fresh: resize at 2.012 s, canvas render 2.144–4.304 s, heartbeat gap 1.974–4.371 s. Retained: resize at 2.099 s, canvas render 2.211–5.128 s, heartbeat gap 2.091–5.156 s. Actual MPC calls peaked at 46/33 ms, GUI artist setup at 72/71 ms, GC at 99/1.1 ms. Canvas resize redraw is the reproduced main-thread pause; retained-trace GC does not explain it. No rendering overhaul or GC tuning was applied. `mpc-responsiveness-*-events.json` retains stage/thread timestamps.

| MP4 | H.264 frames | FPS | File duration | Default speedup |
|---|---:|---:|---:|---:|
| Identity | 80 | 4 | 20.000 s | 1× |
| Ubin SARL | 3136 (stride 4, terminal included) | 26.5 | 118.340 s | 53× |
| Ubin MPC | 2096 (stride 2, terminal included) | 18 | 116.444 s | 18× |

Failed harness attempts are preserved, not deleted or counted as product passes: early QTest waits starved the Python worker; Retina conversion caused initial scene-click misses; the coastal probe incorrectly assumed an invalid start survived; a fast planning error completed before the driver's blocked-edit probe, which then changed the goal and launched a valid same-side episode; fast JSON export finished before its busy sample. Corrected scene/profile/wall evidence supersedes those checks. Raw `checks.json` originally marked heartbeat activity as “responsive”; this table explicitly supersedes that weak MPC result with **FAIL**, and the maintained driver now rejects gaps of one second or more.

Commands (from `rebuild`, modern venv; Torch CPU threads bounded to 2):

```sh
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-modern/bin/python -m pytest -v
QT_QPA_PLATFORM=cocoa OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-modern/bin/python tools/native_acceptance.py
# Recovery flags preserve the original attempts and reuse only this acceptance's fresh trace:
# --resume-ubin, --resume-after-mpc, --final-checks
QT_QPA_PLATFORM=cocoa OMP_NUM_THREADS=2 .venv-modern/bin/python tools/native_scene_acceptance.py
QT_QPA_PLATFORM=cocoa OMP_NUM_THREADS=2 .venv-modern/bin/python tools/native_replay_timing.py results/plan04-native/ubin-sarl.json results/plan04-native/ubin-mpc.json
QT_QPA_PLATFORM=cocoa OMP_NUM_THREADS=2 .venv-modern/bin/python tools/native_responsiveness_probe.py
.venv-modern/bin/python tools/audit_acceptance.py results/plan04-native/evaluation-8 --expected 8
.venv-modern/bin/python tools/audit_acceptance.py results/plan04-native/evaluation-160 --expected 160
.venv-modern/bin/python tools/audit_acceptance.py results/plan04-native --exports
```

Smoke commands used `python -c` with `torch.set_num_threads(2)` and `runpy.run_module('shipnav.evaluate', run_name='__main__')`, passing `--seeds 2 --counts 5 --output results/plan04-native/evaluation-8` or default seeds/counts with `--output results/plan04-native/evaluation-160`. Latencies include cold first inference and concurrent acceptance work; they are descriptive only. Full suite and all artifact audits returned exit 0. A focused GUI/export suite also passed 30 tests before the narrow replay correction.
