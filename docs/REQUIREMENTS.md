# Requirements & Work-Package Breakdown

> Source-of-truth mapping: [slides.md](../slides.md) → this doc → [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) → implementation plan at `~/.claude/plans/added-slides-md-which-is-logical-grove.md`.

## Requirement index

| ID | Description | Slide | Status |
|----|-------------|-------|--------|
| R1 | Install environment, re-run given DRL code, generate simulation videos | 13 | Baseline runs today; video export via `--video_file` |
| R2 | Non-overlapping local goals; per-phase dynamic-obstacle distribution | 13 | Not implemented |
| R3 | Land map loaded + rendered as static obstacles | 13 | Scaffolded but disabled (`static_obs=false`) |
| R4 | Global path planner (Theta\*) emits local goals | 13 | Not implemented |
| R5 | GUI demo with 7 buttons (slide 14) | 14 | Implemented (WP-5) |
| R6 | Literature reading, final report, final presentation | 15 | Out of scope for code plan |
| R7 | Support existing trained model without retraining | 9–11 | Model at [crowd_nav/data/output_trained/](../crowd_nav/data/output_trained) |

## Work packages

| WP | Covers | Depends on | Owner hint |
|----|--------|-----------|-----------|
| **WP-1** Baseline reproduction | R1, R7 | — | Everyone |
| **WP-2** Non-overlapping goal allocator | R2 | WP-4 (waypoints needed) | Group B |
| **WP-3** Land map + rendering | R3 | WP-1 | Group A |
| **WP-4** Theta\* global planner | R4 | WP-3 (`StaticMap`) | Group B |
| **WP-5** PyQt GUI | R5 | WP-3, WP-4 | Group A |
| **WP-6** Packaging + smoke tests | cross-cutting | WP-1…5 | Group C |
| **WP-7** Literature, docs, presentation | R6 | WP-5 for demo clips | Group C |

See the implementation plan for file-level details and the verification plan.

## Group allocation (proposal)

Per slide 16, teams have sizes **A=3, B=3, C=2**. Suggested split:

- **Group A (3)** — WP-3 + WP-5. Visual/UX track. Owns map format, rendering, PyQt.
- **Group B (3)** — WP-4 + WP-2. Algorithms track. Owns Theta\* and goal allocation.
- **Group C (2)** — WP-1 + WP-6 + WP-7. Integration track. Owns baseline correctness, tests, packaging, documentation.

This mirrors the natural dependency graph (visual + algo run in parallel; integration glues them).

## Out of scope (deliberate)

- Retraining the DRL policy on maritime data — we freeze weights per R7.
- Full COLREGs rule-engine — slide 2 mentions it but slide 13 tasks do not require it.
- Real AIS data replay — slide 3 describes the dataset but slide 13 tasks operate on synthetic ships. Noted as a future extension in [SYSTEM_DESIGN.md §5.5](SYSTEM_DESIGN.md).
- Ship dynamics (turn-rate / acceleration constraints beyond the existing `unicycle` kinematics) — slide 2 notes these; an advanced group can opt in.
