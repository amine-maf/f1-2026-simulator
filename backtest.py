"""
Backtest the Monte-Carlo race simulator against the 2026 races already run.

For every completed Grand Prix in the dataset we:
  1. Run the simulator on its starting grid
  2. Compare the predicted top-3 to the actual top-3
  3. Compute three industry-standard metrics

Outputs a tidy DataFrame ready to display in the dashboard.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from race_simulator import simulate_race, prepare_grid, RACE_LAPS


def _brier_score(p_win: np.ndarray, is_winner: np.ndarray) -> float:
    """Multi-class Brier on a single race: sum((p_i - y_i)^2)."""
    return float(np.sum((p_win - is_winner) ** 2))


def _normalised_entropy(p_win: np.ndarray) -> float:
    """0 = certain, 1 = uniform — a sanity check that probas aren't degenerate."""
    p = np.clip(p_win, 1e-12, 1.0)
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


def backtest(df: pd.DataFrame, n_sims: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Run a leave-one-race backtest.

    Each row of the returned DataFrame describes one race:
        Event, predicted_winner, actual_winner, top1_hit, top3_hit,
        brier, entropy, podium_predicted, podium_actual
    """
    races = (
        df.dropna(subset=["IsWinner"])
          .groupby("EventName")["Round"]
          .first()
          .sort_values()
          .index.tolist()
    )

    rows = []
    for event in races:
        grid = prepare_grid(df, event)
        if grid.empty:
            continue

        # Use the race result to derive the actual winner / podium
        race = df[df.EventName == event].dropna(subset=["FinishPosition"])
        if race.empty:
            continue
        actual_winner = race.sort_values("FinishPosition").iloc[0]["Driver"]
        actual_podium = race.sort_values("FinishPosition").head(3)["Driver"].tolist()

        n_laps = RACE_LAPS.get(event, 60)
        summary, _ = simulate_race(grid, n_laps=n_laps, n_sims=n_sims, seed=seed)

        predicted_winner = summary.iloc[0]["Driver"]
        predicted_podium = summary.head(3)["Driver"].tolist()

        # Build aligned p_win / is_winner vectors over the grid
        p_win = summary.set_index("Driver")["P_Win"].reindex(grid["Driver"]).fillna(0).to_numpy()
        is_winner = (grid["Driver"] == actual_winner).to_numpy().astype(float)

        rows.append({
            "Event": event,
            "PredictedWinner": predicted_winner,
            "ActualWinner": actual_winner,
            "Top1Hit": int(predicted_winner == actual_winner),
            "Top3Hit": len(set(predicted_podium) & set(actual_podium)),
            "PredictedPodium": " / ".join(predicted_podium),
            "ActualPodium": " / ".join(actual_podium),
            "P_Win_OfActualWinner": float(
                summary.set_index("Driver").loc[actual_winner, "P_Win"]
                if actual_winner in summary["Driver"].values else 0.0
            ),
            "Brier": _brier_score(p_win, is_winner),
            "Entropy": _normalised_entropy(p_win),
        })

    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame) -> dict:
    """Headline metrics over all backtested races."""
    if bt.empty:
        return {"n": 0}
    return {
        "n": len(bt),
        "top1_accuracy": float(bt["Top1Hit"].mean()),
        "top3_hit_avg": float(bt["Top3Hit"].mean()),       # of 3 — how many we caught
        "top3_hit_pct": float(bt["Top3Hit"].mean() / 3.0), # same, normalised
        "mean_brier": float(bt["Brier"].mean()),
        "mean_p_win_actual": float(bt["P_Win_OfActualWinner"].mean()),
    }


if __name__ == "__main__":
    import os
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__),
                                       "data", "f1_2026_dataset.parquet"))
    bt = backtest(df)
    print(bt.to_string(index=False))
    print("\nSUMMARY:")
    for k, v in summarise(bt).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
