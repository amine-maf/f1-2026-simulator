"""
F1 2026 — Generic Grand Prix Winner Predictor
=============================================
Auto-detects the next/current GP from the FastF1 schedule and the local
dataset, then predicts the winner based on Free Practice pace and Qualifying.

Usage:
    python src/data_collector_2026.py   # refresh the parquet first
    python predict.py                   # auto-detect
    python predict.py --gp monaco       # force a specific GP (substring match)
    python predict.py --list            # show available events
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "f1_2026_dataset.parquet")
FEATURES = ["FP_ShortRunPace", "FP_LongRunPace", "GapToPole", "GridPosition"]


def resolve_target(df: pd.DataFrame, forced: str | None) -> str:
    """Pick the GP to predict.

    Priority:
      1. --gp <substring> override (case-insensitive).
      2. Latest event with FP+Quali features but no race result yet
         (i.e. the weekend is mid-flight — the typical live use case).
      3. If everything already has a result, fall back to the latest event
         in the dataset (useful for replay / validation).
    """
    events_in_order = (
        df.sort_values("Round")
          .drop_duplicates("EventName", keep="last")[["Round", "EventName"]]
    )

    if forced:
        needle = forced.lower()
        matches = [e for e in events_in_order["EventName"] if needle in e.lower()]
        if not matches:
            sys.exit(f"No event matching '{forced}'. Try --list.")
        return matches[-1]

    # Pending = has features for at least one driver, but no winner yet
    feat_mask = df[FEATURES].notna().all(axis=1)
    no_result = df["IsWinner"].isna() | (df["FinishPosition"].isna())
    pending = df[feat_mask & no_result]
    if not pending.empty:
        return pending.sort_values("Round").iloc[-1]["EventName"]

    # Replay mode
    return events_in_order.iloc[-1]["EventName"]


def main():
    parser = argparse.ArgumentParser(description="F1 2026 GP winner predictor")
    parser.add_argument("--gp", help="Force a specific GP (substring match)")
    parser.add_argument("--list", action="store_true", help="List events and exit")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found: {DATA_PATH}\nRun src/data_collector_2026.py first.")

    df = pd.read_parquet(DATA_PATH)

    if args.list:
        print(f"{'Round':<8}{'Event':<35}{'Has result':<12}")
        print("-" * 55)
        for _, row in df.sort_values("Round").drop_duplicates("EventName").iterrows():
            has_result = df[(df.EventName == row.EventName) & df.IsWinner.notna()].shape[0] > 0
            print(f"{int(row.Round):<8}{row.EventName:<35}{'yes' if has_result else 'no':<12}")
        return

    target_event = resolve_target(df, args.gp)
    df_clean = df.dropna(subset=FEATURES).copy()

    train_df = df_clean[(df_clean.EventName != target_event) & df_clean.IsWinner.notna()]
    test_df = df_clean[df_clean.EventName == target_event].copy()

    if test_df.empty:
        sys.exit(f"No FP/Quali features for '{target_event}'. Run the data collector first.")

    if train_df.empty:
        print("Warning: no past races with results. Fitting on the target weekend (zero-shot).")
        train_df = test_df.copy()

    X_train, y_train = train_df[FEATURES], train_df["IsWinner"]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", random_state=42))
    model.fit(X_train, y_train)

    test_df["WinProbability"] = model.predict_proba(test_df[FEATURES])[:, 1]
    results = test_df.sort_values("WinProbability", ascending=False)

    print(f"\n{'=' * 70}")
    print(f"  PREDICTION: {target_event.upper()}")
    print(f"{'=' * 70}")
    print(f"  Training races: {train_df['EventName'].nunique()} | Drivers: {len(test_df)}\n")
    print(f"{'Rank':<6}{'Driver':<8}{'Team':<25}{'Grid':<8}{'Win Probability'}")
    print("-" * 70)
    for i, (_, row) in enumerate(results.iterrows(), 1):
        bar = "#" * int(row.WinProbability * 50)
        grid = int(row.GridPosition) if pd.notna(row.GridPosition) else "-"
        print(f"{i:<6}{row.Driver:<8}{row.Team:<25}{str(grid):<8}{row.WinProbability:.4f}  {bar}")

    print(f"\n  PREDICTED PODIUM:")
    for i, (_, row) in enumerate(results.head(3).iterrows()):
        medal = ["1st", "2nd", "3rd"][i]
        print(f"  {medal}  {row.Driver} ({row.Team}) — {row.WinProbability:.2%}")

    if results["IsWinner"].notna().any() and results["FinishPosition"].notna().any():
        print(f"\n  ACTUAL FINISH (race already run — validation):")
        actual = results.sort_values("FinishPosition").head(3)
        for _, row in actual.iterrows():
            print(f"   P{int(row.FinishPosition)}  {row.Driver} ({row.Team})")
    print()


if __name__ == "__main__":
    main()
