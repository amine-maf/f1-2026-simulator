"""
Monte Carlo F1 Race Simulator
=============================
Lap-by-lap vectorised simulation of a Grand Prix:
- Driver pace derived from Free Practice long-run data
- Tyre compound degradation
- Pit stop strategy
- DNF probability per team
- Safety Car (gap compression)
- Wet-weather upset variance

Output: probability of each driver winning, finishing on the podium, scoring
points, plus the full distribution of finish positions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Physical constants (rough but realistic) ────────────────────────────────
# Linear tyre degradation in seconds per lap of stint life
DEG_RATE = {"SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.03}

# Clean-lap offset vs the medium compound (negative = faster)
COMPOUND_OFFSET = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.3}

PIT_LOSS_DEFAULT = 22.0          # seconds lost during a pit stop
LAP_NOISE_SIGMA = 0.30           # per-lap pace noise (s)
GRID_PENALTY_PER_POS = 0.30      # turn 1 chaos penalty per grid slot

# Race pace = blend of two clean signals (long run is too noisy by itself).
QUALI_RACE_RATIO = 0.35          # race deltas are ~35% of quali deltas (fuel mgmt)
W_QUALI = 0.7                    # weight on GapToPole-derived pace
W_SHORT = 0.3                    # weight on FP short-run-derived pace
MAX_PACE_DELTA = 1.2             # clip per-driver pace delta (s/lap)
DNF_BASE_RATE = 0.06             # per driver per race (≈ historical avg)
SC_PROBABILITY = 0.55            # at least one safety car per race
SC_COMPRESSION = 0.55            # gap multiplier when SC strikes
RAIN_NOISE_SIGMA = 8.0           # extra wet-race lottery (s, total race)

# Approximate scheduled lap counts (used as the dashboard default)
RACE_LAPS = {
    "Australian Grand Prix": 58,
    "Chinese Grand Prix": 56,
    "Japanese Grand Prix": 53,
    "Bahrain Grand Prix": 57,
    "Saudi Arabian Grand Prix": 50,
    "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63,
    "Monaco Grand Prix": 78,
    "Spanish Grand Prix": 66,
    "Canadian Grand Prix": 70,
    "Austrian Grand Prix": 71,
    "British Grand Prix": 52,
    "Hungarian Grand Prix": 70,
    "Belgian Grand Prix": 44,
    "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51,
    "Singapore Grand Prix": 62,
    "United States Grand Prix": 56,
    "Mexico City Grand Prix": 71,
    "São Paulo Grand Prix": 71,
    "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57,
    "Abu Dhabi Grand Prix": 58,
}


def simulate_race(
    grid_df: pd.DataFrame,
    n_laps: int = 60,
    n_sims: int = 10_000,
    pit_lap: int = 25,
    stint1: str = "MEDIUM",
    stint2: str = "HARD",
    pit_loss: float = PIT_LOSS_DEFAULT,
    weather: str = "dry",
    sc_probability: float = SC_PROBABILITY,
    dnf_scale: float = 1.0,
    seed: int | None = 42,
):
    """Run a Monte-Carlo simulation for one race.

    grid_df must contain: Driver, Team, GridPosition, FP_LongRunPace.
    Returns (summary_df, positions_matrix) where positions[sim, driver] is rank.
    """
    rng = np.random.default_rng(seed)

    drivers = grid_df["Driver"].to_numpy()
    teams = grid_df["Team"].to_numpy()
    grid_pos = grid_df["GridPosition"].to_numpy(dtype=float)
    n = len(drivers)

    # Build a race-pace delta from two cleaner signals than FP long run:
    #   1) GapToPole (%): cleanest single-lap signal, converted to s/lap
    #   2) FP short-run (s): backup signal, normalised against the field
    # FP long-run is intentionally ignored — fuel/engine modes corrupt it.
    quali_lap = grid_df["FP_ShortRunPace"].to_numpy(dtype=float)
    ref_lap = np.nanmedian(quali_lap)
    pace_from_quali = grid_df["GapToPole"].to_numpy(dtype=float) / 100.0 * ref_lap
    pace_from_short = quali_lap - np.nanmin(quali_lap)
    base_pace = (W_QUALI * pace_from_quali + W_SHORT * pace_from_short) * QUALI_RACE_RATIO
    base_pace = np.clip(base_pace - np.nanmedian(base_pace), -MAX_PACE_DELTA, MAX_PACE_DELTA)

    # ── 1. Grid penalty (lap-1 chaos) ───────────────────────────────────────
    total = np.tile(grid_pos * GRID_PENALTY_PER_POS, (n_sims, 1)).astype(float)

    # ── 2. Lap-by-lap pace integration ──────────────────────────────────────
    for lap in range(n_laps):
        if lap < pit_lap:
            compound, tire_age = stint1, lap
        else:
            compound, tire_age = stint2, lap - pit_lap
        lap_pace = base_pace + COMPOUND_OFFSET[compound] + DEG_RATE[compound] * tire_age
        noise = rng.normal(0.0, LAP_NOISE_SIGMA, size=(n_sims, n))
        total += lap_pace[None, :] + noise

    # ── 3. Pit stop (one stop per driver) ───────────────────────────────────
    total += pit_loss

    # ── 4. Safety Car: with prob p, compress all gaps to the leader ─────────
    sc_hits = rng.random(n_sims) < sc_probability
    if sc_hits.any():
        leader = total[sc_hits].min(axis=1, keepdims=True)
        total[sc_hits] = leader + (total[sc_hits] - leader) * SC_COMPRESSION

    # ── 5. DNF: random retirement penalises driver hugely ───────────────────
    dnf_rate = np.clip(DNF_BASE_RATE * dnf_scale, 0.0, 0.5)
    dnf_mask = rng.random((n_sims, n)) < dnf_rate
    total[dnf_mask] += 10_000.0

    # ── 6. Wet weather lottery ──────────────────────────────────────────────
    if weather.lower() == "wet":
        total += rng.normal(0.0, RAIN_NOISE_SIGMA, size=(n_sims, n))

    # ── 7. Ranking ──────────────────────────────────────────────────────────
    order = np.argsort(total, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(1, n + 1)

    summary = pd.DataFrame({
        "Driver": drivers,
        "Team": teams,
        "GridPosition": grid_pos.astype(int),
        "P_Win": (positions == 1).mean(axis=0),
        "P_Podium": (positions <= 3).mean(axis=0),
        "P_Points": (positions <= 10).mean(axis=0),
        "P_DNF": dnf_mask.mean(axis=0),
        "MeanPosition": positions.mean(axis=0),
        "StdPosition": positions.std(axis=0),
    }).sort_values("P_Win", ascending=False).reset_index(drop=True)

    return summary, positions


def prepare_grid(df: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """Filter the master 2026 dataset down to one race's grid."""
    grid = df[df["EventName"] == event_name].copy()
    grid = grid.dropna(subset=["FP_LongRunPace", "GridPosition"])
    return grid.sort_values("GridPosition").reset_index(drop=True)


if __name__ == "__main__":
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_parquet(os.path.join(here, "data", "f1_2026_dataset.parquet"))
    event = sys.argv[1] if len(sys.argv) > 1 else "Canadian Grand Prix"
    grid = prepare_grid(df, event)
    laps = RACE_LAPS.get(event, 60)
    print(f"Simulating {event} — {laps} laps × 10 000 sims …")
    summary, _ = simulate_race(grid, n_laps=laps, n_sims=10_000)
    print(summary.to_string(index=False))
