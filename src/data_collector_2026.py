"""
data_collector_2026.py
======================
Fetches 2026 F1 session telemetry (Free Practice, Qualifying, Race) via FastF1 API.
Calculates unique features for the 2026 regulation era:
- FP_ShortRunPace (absolute fastest lap in FP1/FP2/FP3)
- FP_LongRunPace (average pace over the longest stint >= 6 laps, usually in FP2)
- GapToPole (Qualifying gap)
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CACHE_DIR = os.path.join(os.path.dirname(_ROOT), "cache")
OUTPUT_PATH = os.path.join(_ROOT, "data", "f1_2026_dataset.parquet")

STREET_CIRCUITS = ["Monaco", "Azerbaijan", "Singapore", "Saudi Arabia", "Las Vegas", "Miami", "Baku", "Albert Park", "Australian"]

def _safe_load(year, event, session_id):
    try:
        session = fastf1.get_session(year, event, session_id)
        session.load(telemetry=False, weather=False, messages=False)
        return session
    except BaseException:
        return None

def process_fp_pace(year, event_name):
    """
    Extracts the best single lap and the best long-run average from Free Practice.
    We check FP1, FP2, FP3.
    """
    fp_best_lap = {}
    fp_long_run = {}

    for fp_id in ["FP1", "FP2", "FP3"]:
        session = _safe_load(year, event_name, fp_id)
        if not session or session.laps.empty:
            continue

        for driver in session.results["Abbreviation"]:
            laps = session.laps.pick_driver(driver).pick_accurate()
            if laps.empty:
                continue

            # Short run pace (best absolute lap time in seconds)
            best_t = laps["LapTime"].min()
            if pd.notna(best_t):
                best_sec = best_t.total_seconds()
                if driver not in fp_best_lap or best_sec < fp_best_lap[driver]:
                    fp_best_lap[driver] = best_sec

            # Long run pace (find longest stint)
            stints = laps.groupby("Stint")
            for stint_num, stint_laps in stints:
                # Minimum 6 flying, accurate laps to be considered a long run
                if len(stint_laps) >= 6:
                    avg_pace = stint_laps["LapTime"].mean().total_seconds()
                    # Keep the fastest long run representation if multiple exist
                    if driver not in fp_long_run or avg_pace < fp_long_run[driver]:
                        fp_long_run[driver] = avg_pace

    return fp_best_lap, fp_long_run

def process_quali(year, event_name):
    session = _safe_load(year, event_name, "Q")
    dict_q = {}
    if not session or session.results.empty:
        return dict_q

    results = session.results
    
    # Calculate best time per driver across Q1/Q2/Q3
    def get_best_q(row):
        times = []
        for col in ["Q1", "Q2", "Q3"]:
            if pd.notna(row.get(col)):
                times.append(row[col].total_seconds())
        return min(times) if times else np.nan

    times_list = results.apply(get_best_q, axis=1)
    results["BestQTime"] = times_list
    
    pole_time = results["BestQTime"].min()

    for _, row in results.iterrows():
        driver = row["Abbreviation"]
        best_q = row["BestQTime"]
        gap = (best_q - pole_time) / (pole_time + 1e-9) * 100 if pd.notna(best_q) else np.nan
        dict_q[driver] = {
            "GapToPole": gap,
            "GridPositionNorm": row.get("Position", np.nan), # We use Quali position as Grid proxy until Sunday
            "Team": row.get("TeamName", "Unknown")
        }
    return dict_q

def process_race(year, event_name):
    session = _safe_load(year, event_name, "R")
    dict_r = {}
    if not session or session.results.empty:
        return dict_r

    for _, row in session.results.iterrows():
        driver = row["Abbreviation"]
        pos = pd.to_numeric(row.get("Position"), errors="coerce")
        dict_r[driver] = {
            "FinishPosition": pos,
            "Points": pd.to_numeric(row.get("Points"), errors="coerce"),
            "IsWinner": 1 if pos == 1 else 0,
            "IsDNF": 0 if ("Finished" in str(row.get("Status", "")) or "+" in str(row.get("Status", ""))) else 1,
            "Team": row.get("TeamName", "Unknown")
        }
    return dict_r

def build_2026_dataset():
    fastf1.Cache.enable_cache(os.path.abspath(CACHE_DIR))
    year = 2026
    
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    target_formats = ["conventional", "sprint", "sprint_shootout", "sprint_qualifying"]
    race_events = schedule[schedule["EventFormat"].isin(target_formats)]

    all_records = []
    
    for _, event in race_events.iterrows():
        event_name = event["EventName"]
        round_num = event["RoundNumber"]
        
        # Determine if we even have any running yet
        # Let's check FP1 quickly
        temp_session = fastf1.get_session(year, event_name, "FP1")
        try:
            temp_session.load(telemetry=False, weather=False, messages=False)
            if temp_session.results.empty:
                continue # Event hasn't started
        except:
            continue
            
        print(f"Processing 2026 Round {round_num}: {event_name}...")

        fp_best, fp_long = process_fp_pace(year, event_name)
        q_data = process_quali(year, event_name)
        r_data = process_race(year, event_name)

        # We collect all drivers who appeared in either FP, Q, or R
        all_drivers = set(fp_best.keys()).union(q_data.keys()).union(r_data.keys())

        for drv in all_drivers:
            record = {
                "Year": year,
                "Round": round_num,
                "EventName": event_name,
                "IsStreetCircuit": int(any(s.lower() in event_name.lower() for s in STREET_CIRCUITS)),
                "Driver": drv,
                "Team": q_data.get(drv, {}).get("Team") or r_data.get(drv, {}).get("Team", "Unknown"),
                
                "FP_ShortRunPace": fp_best.get(drv, np.nan),
                "FP_LongRunPace": fp_long.get(drv, np.nan),
                
                "GapToPole": q_data.get(drv, {}).get("GapToPole", np.nan),
                "GridPosition": q_data.get(drv, {}).get("GridPositionNorm", np.nan),
                
                "FinishPosition": r_data.get(drv, {}).get("FinishPosition", np.nan),
                "Points": r_data.get(drv, {}).get("Points", 0.0),
                "IsWinner": r_data.get(drv, {}).get("IsWinner", np.nan),
                "IsDNF": r_data.get(drv, {}).get("IsDNF", np.nan)
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    else:
        print("No valid 2026 data found yet.")

if __name__ == "__main__":
    build_2026_dataset()
