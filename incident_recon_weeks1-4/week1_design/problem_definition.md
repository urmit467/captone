# Problem Definition (Week 1)

## 1. One-sentence problem
Given a network of paths where cameras cover every path except one (P4), and an incident at a known time T inside P4,
estimate for every person the probability P(location_p(T) = P4 | all observations before and after T, path graph,
travel-time model), then flag people above a threshold and reconstruct where they went afterwards.

## 2. Frozen MVP scope
| Item | Decision |
|---|---|
| Network | 7 paths (P1..P7), see `path_graph.png` / `path_graph.json` |
| Hidden path | P4 permanently unobserved, no camera |
| Cameras | C1, C2, C3, C5, C6, C7 (one per observable path) |
| People | 10-20 consenting participants (real footage); ~3000 simulated (simulator) |
| Incidents | Artificial, at a known timestamp inside P4 (no real incident is ever recorded) |
| Identity | Anonymised IDs only (Person_001...). No face recognition. Appearance Re-ID only. |
| Compute | Local machine, no deployment |

## 3. Out of scope
Face recognition, real incident detection, real-time streaming, multi-hidden-path networks, deployment, crowd-scale (>50) tracking.

## 4. Assumptions
1. Cameras are time-synchronised (offset measured and stored in `cameras.clock_offset_s`).
2. Every exit of P4 leads to an observable path (so a person inside P4 is eventually seen again).
3. Walking speed is bounded (0.4 - 2.5 m/s) and people do not teleport between paths.
4. Movement between paths follows the directed edges in `path_graph.json`.

## 5. Data flow
Video -> YOLO -> tracker -> Re-ID -> global person timeline -> path graph -> incident(T, P4)
-> look backward/forward in timelines -> P(inside P4 at T) -> threat list -> post-incident trajectory.

## 6. Evaluation metrics
| Metric | Definition |
|---|---|
| Affected-person precision | flagged & truly in P4 at T / flagged |
| Affected-person recall | flagged & truly in P4 at T / truly in P4 at T |
| F1 | harmonic mean |
| PR-AUC, Brier score | ranking quality and probability calibration |
| Incident-time localisation accuracy | fraction of persons whose predicted path at T equals the true path |
| Path reconstruction accuracy | reconstructed path sequence equals ground truth |
| Travel-time error | |estimated - true| entry/exit time of P4 |
| Post-incident tracking accuracy | reconstructed/observed path after T matches truth |
| Re-ID (Week 4) | pairwise link precision / recall / F1, identity fragmentation, identity merges |

Threat levels: HIGH >= 0.80, MEDIUM >= 0.50, else LOW (thresholds chosen on the PR curve in Week 7).

## 7. Ground-truth protocol
The true route/time of each participant is recorded separately (route sheet + stopwatch log or a phone timestamp app)
and stored ONLY in `ground_truth_presence`. Inference code must never read that table.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Re-ID weak | Distinct clothing colours for participants; small group; topology/time gating |
| Footage collection slips | Simulator gives data from Week 2; real footage is validation only |
| Camera clocks drift | Clap/flash sync at start and end of each recording; store offsets |
| Hidden ground truth wrong | Two independent timing sources per participant |

## 9. Deliverables for Week 1
- [x] Path graph diagram + JSON
- [x] PostgreSQL schema + docker-compose
- [x] Problem definition and metrics (this file)
- [x] Recording plan and consent template
- [ ] Environment set up on your machine (see README "Setup")
