# Video Processing — How the Math Formulas Connect to Real Video
   
This file explains, in simple English, how the mathematical formulas from
section 6 of `demo_dataset.md` actually work inside the video processing
pipeline of this project. It assumes you are not a math expert — think of it as
a story: video → numbers → probability.

---
 
## 1. What video processing does (the short version)

A CCTV camera records a video. The system watches that video and answers one
question:

> **"Which person was where, and at what time?"**

The pipeline is:

1. **YOLO** looks at every frame and says "I see a person here" (a bounding box).
2. **ByteTrack** gives each person an ID number and follows them across frames
   so the same person keeps the same ID while they are on camera.
3. The system saves an **observation** to MongoDB for each person:
   - who (`person_id` / tracking ID)
   - where (`path_id`, `camera_id`)
   - when (`timestamp`)
4. Later, the **reconstruction engine** reads those observations and uses the
   math formulas to guess whether that person was on the hidden path (P4) at
   the exact moment of the incident.

The math formulas do not run inside the video itself. They run **after** the
video has been turned into a list of time-stamped observations.

---

## 2. From video to the first formula: travel time per path

**Formula:** `expected_minutes = length / speed`

### What happens in the video

Each path (P1, P2, P3 …) has a known length (in meters) and an average walking
speed (84 meters/minute). The video does not measure speed directly — the system
already knows:

- Path P1 is 90 meters long
- A person walks about 84 meters per minute

So:

```
expected_minutes = 90 / 84 ≈ 1.07 minutes
```

That means "a normal person needs about 1 minute and 4 seconds to walk all of
path P1."

The system also keeps a minimum and maximum to be safe:

```
min_minutes  = 1.07 × 0.75 ≈ 0.80   (fast walker)
max_minutes  = 1.07 × 1.25 ≈ 1.34   (slow walker)
```

### Simple example

From the video, Person_007 was last seen on path P2 at 09:15:00, and the next
time they were seen on path P3 at 09:16:19. The gap is 79 seconds ≈ 1.32
minutes.

The system knows path P2 is 110 meters, so it should take about 1.31 minutes to
walk. Because the observed gap (1.32 min) is close to the expected time (1.31
min), the system believes Person_007 walked straight from P2 to P3 without
stopping.

---

## 3. Route travel time: adding up a whole trip

**Formula:** add up the travel time of every path in the route.

### What happens in the video

A person rarely stays on one path. The video gives us observations like:

```
09:15  on P2
09:16  on P3
09:18  on P5   (P4 is hidden, so we never see them here)
```

To get from P2 to P5, the person must have passed through P3 and P4. The system
adds up the expected times:

```
P2 (1.31 min) + P3 (0.95 min) + P4 (0.83 min) = 3.09 minutes total
```

So if they left P2 at 09:15, they could have arrived at P5 around 09:18:05 —
which matches the video observation at 09:18.

This tells the system: **"Yes, this route through P4 is physically possible."**

---

## 4. Presence window: when could they have been on the hidden path?

**Formula:** a time window `[start, end]` around a "mode" (most likely) time.

### What happens in the video

P4 has no camera, so we never directly see anyone on it. But we can guess when
they might have been there by looking at the observations before and after.

**Case: seen on P2 before the incident, seen on P5 after the incident.**

- Last seen on P2 at 09:45 (before incident at 10:30)
- First seen on P5 at 10:50 (after incident)

They must have walked P2 → P3 → P4 → P5. The system calculates:

- Earliest they could step onto P4: after finishing P2 and P3
- Latest they could step off P4: early enough to reach P5 by 10:50

This gives a **presence window** — a range of times during which they could
have been on P4. For example: `[10:05, 10:20]`.

If the incident happened at 10:30, and the window is `[10:05, 10:20]`, then the
person was probably **already off P4** by the time of the incident → low
probability.

If the window is `[10:25, 10:35]`, then the incident happened right in the
middle of it → high probability.

---

## 5. Triangular membership μ: how likely were they there at the exact moment?

**Formula:** a triangle shape. It peaks at the "mode" time and falls to zero at
the edges of the window.

```
μ(T) = (T − low) / (mode − low)      if T is before the mode
μ(T) = (high − T) / (high − mode)    if T is after the mode
μ(T) = 0                             if T is outside the window
```

### What this means in plain English

Imagine the presence window is `[10:20, 10:40]` and the most likely time
(mode) is `10:30`.

- At exactly 10:30 → score is **1.0** (100% likely they were on P4 then)
- At 10:25 → score is **0.5** (halfway to the edge, so 50% likely)
- At 10:20 or 10:40 → score is **0.0** (at the edge, could have just arrived
  or just left)
- At 10:10 → score is **0.0** (outside the window, impossible)

The incident time is plugged into this triangle. If the incident happened at
10:30 (the peak), μ = 1.0. If it happened near the edge, μ is small.

This is the **core probability**: "Given the time window, how likely was the
person on P4 at the exact moment of the incident?"

---

## 6. Route coverage: how many possible routes go through the hidden path?

**Formula:** `coverage = (routes through P4) / (all possible routes)`

### What happens in the video

Between two observed paths, there might be more than one way to walk through
the network.

Example: from P2 to P5
- Route A: P2 → P3 → P4 → P5 (goes through the hidden path P4)
- Route B: P2 → P6 → P5 (does NOT go through P4, if such a connection existed)

If both routes are physically possible in the time available:

```
coverage = 1 route through P4 / 2 total routes = 0.5
```

This reduces the final probability because the person might have taken the route
that avoids P4 entirely. If **all** feasible routes go through P4, coverage is
1.0 (no discount). If **none** go through P4, coverage is 0 (probability
becomes 0).

---

## 7. Recency decay: newer observations matter more

**Formula:** `recency = e^(−gap / 90)`

### What happens in the video

If you saw someone on P3 at 10:29, that is strong evidence of where they were
near the 10:30 incident. If you last saw them on P1 at 09:00, that is weak
evidence — they could have gone anywhere after that.

The "gap" is minutes between the last observation and the incident time:

| Gap | Recency score | Meaning |
|---|---|---|
| 1 min | e^(−1/90) ≈ 0.99 | Very fresh observation |
| 30 min | e^(−30/90) ≈ 0.72 | Still useful |
| 90 min | e^(−90/90) ≈ 0.37 | Getting stale |
| 180 min | e^(−180/90) ≈ 0.14 | Very stale |

The final probability is multiplied by this number, so old observations pull the
score down.

---

## 8. History factor: have they walked through P4 before?

**Formula:** `history = clamp(1.0 + 0.2 × (through / total), 1.0, 1.2)`

### What happens in the video

The system looks at the person's past camera observations (not the current
incident). If they were previously seen walking from a path before P4 to a path
after P4, and the shortest route between those past observations passed through
P4, that counts as "they have been through P4 before."

Example:
- Total past transitions between different paths: 5
- Of those, routes that passed through P4: 2

```
history = 1.0 + 0.2 × (2 / 5) = 1.0 + 0.08 = 1.08
```

This boosts the final probability by up to 20% (from 1.0 to 1.2 max). It is a
small bonus — it never creates probability out of nowhere, it just slightly
increases suspicion for people with a history of using that hidden path.

---

## 9. Final probability: putting everything together

**Formula:** `probability = clamp( μ × 100 × coverage × recency × history, 0, 100 )`

### Simple English recipe

Take all the pieces from the video:

1. **μ** — The triangle score from step 5 (were they on P4 at the exact time?)
2. **× 100** — Turn it into a percentage
3. **× coverage** — Discount if they could have avoided P4
4. **× recency** — Discount if your last sighting was long ago
5. **× history** — Slight bonus if they used P4 before
6. **clamp** — Make sure the result stays between 0 and 100

### A full mini-example

Person_007, incident at 10:30 on path P4:

| Step | Value | Where it comes from |
|---|---|---|
| μ (triangle) | 0.70 | Incident time sits inside their presence window, closer to the middle |
| coverage | 1.0 | The only feasible route from their last sighting to their next sighting went through P4 |
| recency | 0.85 | Last seen 15 minutes before the incident |
| history | 1.0 | No previous sightings through P4 |

```
probability = 0.70 × 100 × 1.0 × 0.85 × 1.0 = 59.5%
```

59.5% → classified as **MEDIUM** (because it is ≥ 50 but < 80).

---

## 10. What the video processor actually saves

So you know where the numbers come from, here is what the video processing
step writes to the database for each person:

| Field | Example value | Source |
|---|---|---|
| person_id | "Person_007" | Assigned from tracking ID |
| camera_id | "CAM_P2" | Which camera saw them |
| path_id | "P2" | Which path the camera is on |
| timestamp | "2026-11-19 10:28:00" | From video frame time |
| confidence | 0.92 | YOLO detection confidence |
| source | "camera" | Direct observation (not inferred) |

Later, the timeline generator may add **inferred** events (marked `source =
"inferred"`) to fill the gaps between observed paths — for example, inserting
P4 into the timeline even though no camera saw the person there. These inferred
events come from the route math in sections 3 and 4, not from the video itself.

---

## Summary in one paragraph

The video gives us **when and where** a person was seen. The math formulas turn
those sightings into a probability: travel-time formulas tell us how long it
takes to walk between sightings, the presence window finds when they could have
been on the hidden path, the triangle (μ) scores how likely they were there at
the exact incident moment, route coverage discounts alternative routes, recency
decay discounts old sightings, and history gives a tiny bonus for repeat
visitors. Multiply them all, clamp to 0–100, and you get the person's
probability of being on the hidden path when the incident happened.
