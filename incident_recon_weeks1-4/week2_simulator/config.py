"""Path network + simulation settings. Edit here; everything else reads from this file."""

# length in metres, speed in m/s (mean walking speed on that path), camera: is the path observed?
PATHS = {
    "P1": dict(length=120, speed=1.3, camera=True),
    "P2": dict(length=150, speed=1.3, camera=True),
    "P3": dict(length=200, speed=1.3, camera=True),
    "P4": dict(length=300, speed=1.2, camera=False),   # <- the unobservable path
    "P5": dict(length=180, speed=1.3, camera=True),
    "P6": dict(length=160, speed=1.3, camera=True),
    "P7": dict(length=140, speed=1.3, camera=True),
}

# Directed movement probabilities. "EXIT" = person leaves the monitored network.
TRANSITIONS = {
    "P1": {"P2": 1.0},
    "P2": {"P3": 1.0},
    "P3": {"P4": 0.6, "P5": 0.4},
    "P4": {"P5": 0.35, "P6": 0.35, "P7": 0.30},
    "P5": {"P6": 0.6, "EXIT": 0.4},
    "P6": {"P7": 0.2, "EXIT": 0.8},
    "P7": {"EXIT": 1.0},
}

# Where people enter the network
ENTRY = {"P1": 0.5, "P2": 0.2, "P3": 0.3}

HIDDEN_PATH = "P4"

# Simulation
N_PERSONS = 3000
SIM_DURATION_S = 4 * 3600        # arrivals spread over 4 hours
SPEED_SIGMA = 0.20               # lognormal sigma of per-person speed multiplier
SEGMENT_SIGMA = 0.10             # per-segment speed variation
DWELL_PROB = 0.15                # chance of pausing on a path
DWELL_MEAN_S = 30

# Camera / noise model
OBS_PERIOD_S = 3.0               # a person is sampled roughly every 3 s
P_DETECT = 0.85                  # per-sample detection probability
TIME_JITTER_S = 0.5
REID_SWAP_PROB = 0.0             # ID errors (used later for robustness tests; 0 for now)

N_INCIDENTS = 300
CONTEXT_WINDOW_S = 900           # persons considered around incident time (+-15 min)
SEED = 42

# Incident engine
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50
