# 🏎️ F1 2026 Season Predictor: New Regulations Era

A real-time Machine Learning prediction pipeline designed specifically for the massive **2026 Formula 1 Technical Regulations** overhaul.

## ⚠️ Why a New 2026-Only Project?
The 2026 regulations represent the biggest technical reset in modern F1 history (50/50 ICE/ERS power split, active aerodynamics, smaller cars). Because of this, historical data from 2018-2025 is structurally irrelevant. 

To prevent "data leakage" from past eras (e.g., Red Bull's previous ground-effect dominance), this model is trained **exclusively on live 2026 data**, utilizing a dynamic Temporal Split architecture.

## 🧠 Machine Learning Architecture

The pipeline uses a **Leave-One-Out Temporal Split** via `scikit-learn` Logistic Regression:
1. **Feature Engineering**: Extracts Free Practice Pace (Short & Long runs) and Qualifying gap-to-pole directly from the FastF1 telemetry API.
2. **Dynamic Training**: To predict Race $N$, the model trains on the relationship between Practice/Quali and Race Results from Races $1$ to $N-1$.
3. **Live Inference**: The model then applies these learned weights to the Practice/Quali data of Race $N$ to output probabilistic win percentages before the lights go out.

## 🚀 How to Run Predictions on a Race Weekend

This pipeline is built for **Live MLOps**. Here is exactly how to use it during a race weekend:

### For a Standard Weekend
1. **Wait for Qualifying to finish** (Saturday afternoon).
2. Run data collection to fetch FP1, FP2, FP3, and Quali telemetry:
   ```bash
   python src/data_collector_2026.py
   ```
3. Run the prediction notebook:
   ```bash
   jupyter nbconvert --to notebook --execute 03_race_predictor.ipynb
   ```
   *This will output the probability of each driver winning Sunday's Grand Prix.*

### For a Sprint Weekend (e.g., China 🇨🇳)
Sprint weekends have two races, meaning two prediction windows:

**Prediction 1: The Sprint Race**
1. Wait for **Sprint Qualifying (SQ)** to finish (Friday afternoon).
2. Run the pipeline (fetches FP1 + SQ data).
3. The model will predict the winner of the Saturday Sprint Race.

**Prediction 2: The Sunday Grand Prix**
1. Wait for **Main Qualifying** to finish (Saturday afternoon).
2. Run the pipeline again (fetches FP1, SQ, Sprint Race result, and Main Quali).
3. The model trains on the Sprint Race result and predicts the Sunday Grand Prix winner!

## 📂 Project Structure

- `src/data_collector_2026.py` - Custom FastF1 parser that extracts FP long-run pace degradation slopes and Qualifying deltas.
- `01_season_overview.ipynb` - High-level championship tracking.
- `02_feature_engineering.ipynb` - Correlation heatmaps proving the predictive power of FP Pace under 2026 regulations.
- `03_race_predictor.ipynb` - The core ML inference notebook.

*Built for the 2026 F1 Season. May the best algorithm win.*
