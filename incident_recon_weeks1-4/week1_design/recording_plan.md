# Real-Footage Recording Plan (for Weeks 3-4 and the Week 9 validation)

## Layout
- Use 5-7 connected corridors/paths on campus or in a building. One path is P4: point NO camera at it.
- One fixed camera per observable path, mounted 2.5-3 m high, covering the full width of the path, 1080p, 25-30 fps.
- Overlap between camera views is not needed and not desired: the gap is what the model reasons about.

## Participants
- 10-20 consenting adults (see `consent_form_template.md`), assigned IDs Person_001... Do not record names next to footage.
- Ask for visibly different clothing (e.g. distinct top colours). This alone makes appearance Re-ID workable.

## Clock synchronisation
- Before and after each session, hold a phone showing a running clock (or clap) in view of every camera.
  Record the offset of each camera to a reference clock in `cameras.clock_offset_s`.
- Name files `C3_2026-10-14T10-00-00.mp4` (camera id + wall-clock start time).

## Scripted runs
Each run: participants walk pre-planned routes, e.g. P1->P2->P3->P4->P5->P6. Keep a route sheet with planned start times.
Vary: walking speed, one participant pausing inside P4, someone who takes the bypass P3->P5, people entering close together.
Record at least 10 runs. Hold out 3-4 runs as a test set that you never tune on.

## Ground truth
For each participant and run, log (path, enter time, exit time) for every path INCLUDING P4. Use a second person with a stopwatch
or a phone at the P4 entrance/exit (not a camera). Store in `ground_truth_presence`.

## Artificial incident
Pick a timestamp T inside the run after recording. The system never records an actual event.

## Data handling
Store footage on a local encrypted drive; delete raw footage after the project if the consent form says so; only anonymised
IDs, track coordinates and embeddings are kept in the database.
