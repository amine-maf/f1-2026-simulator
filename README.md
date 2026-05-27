# 🏎️  F1 2026 Race Simulator

A real-time **Monte Carlo race simulator** built for the 2026 Formula 1 regulation reset.
From a Friday Free Practice session to 10 000 plausible race outcomes — with live weather,
backtested predictions and an interactive dashboard.

![python](https://img.shields.io/badge/python-3.10+-blue)
![streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B)
![status](https://img.shields.io/badge/status-live-success)

---

## ✨ What it does

| Feature | Detail |
|---|---|
| 🏁  **Race prediction** | Probability of win / podium / points for every driver |
| 🎲  **Monte Carlo** | 10 000 lap-by-lap simulations with tyre deg, pit stops, Safety Cars, DNFs |
| 🌦  **Live weather** | Race-day forecast pulled from open-meteo.com per circuit |
| 📊  **Backtest** | Top-1 accuracy, top-3 hit rate and Brier score on every completed 2026 race |
| 🔍  **Driver deep-dive** | Position distribution histogram + 6-metric summary per driver |
| 🧭  **How-it-works tab** | Pipeline schema + engineering rationale |

## 🎯 Why a new 2026-only project?

The 2026 regulations represent the biggest technical reset in modern F1 history
(50/50 ICE/ERS power split, active aerodynamics, smaller cars). Historical data from
2018–2025 is structurally irrelevant. The simulator is therefore **trained and validated
exclusively on live 2026 data** as the season unfolds.

## 🧠 The model in one paragraph

Race pace is built from a blend of **GapToPole (70%)** and **best FP short-run lap (30%)** —
ignoring FP long runs because fuel loads and engine modes pollute them. The quali deltas
are scaled by a `QualiRaceRatio = 0.35` to reflect race-day fuel-saving and tyre
management. The simulator then runs a vectorised lap-by-lap loop with linear tyre
degradation per compound (SOFT 0.10 s/lap, MEDIUM 0.05, HARD 0.03), a single pit stop,
a Bernoulli Safety Car (compresses gaps × 0.55), per-driver DNF rolls, and an optional
wet-weather variance bump. Backtest: **60% top-1 accuracy across the first 5 races of 2026**.

## 🚀 Quick start

```bash
# 1) Set up
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Fetch the latest weekend data (after qualifying)
python src/data_collector_2026.py

# 3a) Run the dashboard
streamlit run app.py
# → http://localhost:8501

# 3b) Or a quick CLI prediction (auto-detects the upcoming GP)
python predict.py
python predict.py --gp monaco   # force a specific GP
python predict.py --list        # show available events
```

## 📂 Project layout

```
07_f1_2026_prediction/
├── app.py                       Streamlit dashboard (3 tabs)
├── predict.py                   One-shot CLI predictor with auto-detect
├── race_simulator.py            Vectorised lap-by-lap Monte Carlo engine
├── weather_api.py               open-meteo client + circuit coordinates
├── backtest.py                  Leave-one-race scoring (top-1, top-3, Brier)
├── src/data_collector_2026.py   FastF1 → parquet feature pipeline
├── 02_feature_engineering.ipynb Exploratory analysis of FP / Quali signals
├── data/f1_2026_dataset.parquet Master dataset (rebuilt each weekend)
└── .streamlit/config.toml       Dark theme
```

## 🌐 Deploy on Streamlit Cloud

1. Push this folder to a public GitHub repo.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → pick your repo, branch `main`, main file path `app.py`.
4. Done — you get a public URL like `https://<your-name>-f1-2026.streamlit.app`.

The `requirements.txt` and `.streamlit/config.toml` in this folder are enough for a
zero-config deploy.

## 🛠 How a race weekend flows

```
Friday  → FP1, FP2 happen
Saturday morning → FP3 then Qualifying happens
                ↓
        python src/data_collector_2026.py   (refresh parquet)
                ↓
                streamlit run app.py        (interact + screenshot)
                ↓
Sunday  → race result auto-feeds the next backtest cycle
```

## ⚠️ Caveats

- The pace model is calibrated for a 2-compound, 1-stop race. Bahrain and the like will
  drift if the field runs 2-stops; the strategy sliders let you compensate manually.
- The simulator ignores DRS gain, tyre warm-up, and overtake difficulty — these are
  approximated through lap noise rather than modelled explicitly.
- Backtest sample is currently **5 races** — interpret accuracy figures with that in mind.

---

*Built with FastF1 · NumPy · open-meteo · Plotly · Streamlit. May the best algorithm win.*
