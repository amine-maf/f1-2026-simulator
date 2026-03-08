import pandas as pd
import numpy as np
import os
import fastf1

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(_HERE), "cache")
OUTPUT_PATH = os.path.join(_HERE, "data", "f1_2026_dataset.parquet")
fastf1.Cache.enable_cache(os.path.abspath(CACHE_DIR))

df = pd.read_parquet(OUTPUT_PATH)

def process_race(year, event_name):
    session = fastf1.get_session(year, event_name, "R")
    try:
        session.load(telemetry=False, weather=False, messages=False)
    except:
        return {}
    dict_r = {}
    if session.results.empty:
        return dict_r
    for _, row in session.results.iterrows():
        driver = row["Abbreviation"]
        pos = pd.to_numeric(row.get("Position"), errors="coerce")
        dict_r[driver] = {
            "FinishPosition": pos,
            "Points": pd.to_numeric(row.get("Points"), errors="coerce"),
            "IsWinner": 1 if pos == 1 else 0,
            "IsDNF": 0 if ("Finished" in str(row.get("Status", "")) or "+" in str(row.get("Status", ""))) else 1
        }
    return dict_r

r_data = process_race(2026, "Australian Grand Prix")

for drv, data in r_data.items():
    idx = (df["Driver"] == drv) & (df["EventName"] == "Australian Grand Prix")
    if idx.any():
        df.loc[idx, "FinishPosition"] = data["FinishPosition"]
        df.loc[idx, "Points"] = data["Points"]
        df.loc[idx, "IsWinner"] = data["IsWinner"]
        df.loc[idx, "IsDNF"] = data["IsDNF"]

df.to_parquet(OUTPUT_PATH, index=False)
