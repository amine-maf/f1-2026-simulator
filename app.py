"""
F1 2026 — Race Simulator Dashboard
==================================
Streamlit app: pick a Grand Prix, tweak strategy & conditions, and watch
10 000 Monte Carlo simulations roll out before your eyes.

Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from race_simulator import simulate_race, prepare_grid, RACE_LAPS
from weather_api import fetch_weather, suggest_weather_setting, CIRCUIT_COORDS
from backtest import backtest, summarise

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "f1_2026_dataset.parquet")

TEAM_COLORS = {
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Ferrari": "#E80020",
    "Red Bull Racing": "#3671C6",
    "Racing Bulls": "#6692FF",
    "Audi": "#52E252",
    "Aston Martin": "#229971",
    "Williams": "#64C4FF",
    "Alpine": "#FF87BC",
    "Haas F1 Team": "#B6BABD",
    "Cadillac": "#FFFFFF",
}

st.set_page_config(
    page_title="F1 2026 Race Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%); }
    .podium-card {
        background: linear-gradient(135deg, #1f1f1f 0%, #2a2a2a 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border-top: 4px solid var(--team-color);
    }
    .podium-driver { font-size: 2.2rem; font-weight: 800; letter-spacing: 2px; }
    .podium-team { font-size: 0.85rem; opacity: 0.7; }
    .podium-prob { font-size: 1.6rem; font-weight: 700; margin-top: 0.5rem; }
    .weather-pill {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
        background: #1f1f1f; margin-right: 0.5rem; font-size: 0.9rem;
    }
    h1 { color: #fff; font-weight: 800; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_data(show_spinner=False)
def run_sim(grid_dict, n_laps, n_sims, pit_lap, stint1, stint2,
            pit_loss, weather, sc_prob, dnf_scale, seed):
    grid = pd.DataFrame(grid_dict)
    return simulate_race(
        grid, n_laps=n_laps, n_sims=n_sims, pit_lap=pit_lap,
        stint1=stint1, stint2=stint2, pit_loss=pit_loss,
        weather=weather, sc_probability=sc_prob, dnf_scale=dnf_scale,
        seed=seed,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_weather(event_name):
    return fetch_weather(event_name)


@st.cache_data(show_spinner="Backtesting all completed 2026 races …")
def run_backtest_cached():
    df = load_data()
    return backtest(df, n_sims=5_000, seed=42)


df = load_data()
events = sorted(df["EventName"].unique(),
                key=lambda e: df[df.EventName == e]["Round"].iloc[0])

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️  Simulation Controls")
event = st.sidebar.selectbox("🏁 Grand Prix", events, index=len(events) - 1)
default_laps = RACE_LAPS.get(event, 60)

# Live weather fetch
wx = cached_weather(event)
weather_default_idx = 0
if wx and wx.get("rain_prob") is not None:
    weather_default_idx = 0 if suggest_weather_setting(wx["rain_prob"]) == "Dry" else 1
    rain_p = wx["rain_prob"]
    t = wx["temp_c"]
    wind = wx["wind_kmh"]
    st.sidebar.markdown(
        f"**☁️  Live forecast — {wx['date']}**  \n"
        f"<span class='weather-pill'>🌧 {rain_p:.0f}%</span>"
        f"<span class='weather-pill'>🌡 {t:.0f}°C</span>"
        f"<span class='weather-pill'>💨 {wind:.0f} km/h</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Forecast via open-meteo.com · race window 13–17h local")
else:
    st.sidebar.info("Live forecast unavailable — using defaults.")

with st.sidebar.expander("Race conditions", expanded=True):
    weather = st.radio("Weather", ["Dry", "Wet"], index=weather_default_idx, horizontal=True)
    sc_prob = st.slider("Safety Car probability", 0.0, 1.0, 0.55, 0.05)
    n_laps = st.slider("Race laps", 30, 80, default_laps)

with st.sidebar.expander("Strategy", expanded=False):
    stint1 = st.selectbox("First stint compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    stint2 = st.selectbox("Second stint compound", ["SOFT", "MEDIUM", "HARD"], index=2)
    pit_lap = st.slider("Pit stop lap", 5, n_laps - 5, min(25, n_laps // 2))
    pit_loss = st.slider("Pit loss (s)", 15.0, 30.0, 22.0, 0.5)

with st.sidebar.expander("Monte Carlo", expanded=False):
    n_sims = st.select_slider("Iterations", options=[1_000, 5_000, 10_000, 20_000, 50_000], value=10_000)
    dnf_scale = st.slider("DNF rate multiplier", 0.0, 3.0, 1.0, 0.1)
    seed = st.number_input("Random seed", value=42, step=1)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_sim, tab_how, tab_bt = st.tabs(["🏁 Simulation", "🧭 How it works", "📊 Backtest"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    grid = prepare_grid(df, event)
    if grid.empty:
        st.error(f"No FP/Quali data available for {event} yet.")
        st.stop()

    grid_dict = {col: grid[col].tolist() for col in grid.columns}
    summary, positions = run_sim(
        grid_dict=grid_dict, n_laps=n_laps, n_sims=n_sims,
        pit_lap=pit_lap, stint1=stint1, stint2=stint2,
        pit_loss=pit_loss, weather=weather.lower(),
        sc_prob=sc_prob, dnf_scale=dnf_scale, seed=int(seed),
    )

    col_title, col_metric = st.columns([3, 2])
    with col_title:
        st.title(f"🏎️  {event}")
        st.caption(f"Lap-by-lap Monte Carlo · {n_sims:,} simulations · {weather} · "
                   f"{stint1[:1]}→{stint2[:1]} strategy, pit lap {pit_lap}")
    with col_metric:
        favourite = summary.iloc[0]
        st.metric(
            label="🏆  Predicted winner",
            value=f"{favourite['Driver']} ({favourite['Team']})",
            delta=f"{favourite['P_Win']*100:.1f}% win probability",
        )

    # Podium
    st.markdown("### 🥇 Predicted podium")
    podium_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for col, (_, row), medal in zip(podium_cols, summary.head(3).iterrows(), medals):
        team_color = TEAM_COLORS.get(row["Team"], "#888")
        col.markdown(
            f"""
            <div class="podium-card" style="--team-color: {team_color};">
                <div style="font-size: 2.5rem;">{medal}</div>
                <div class="podium-driver" style="color: {team_color};">{row['Driver']}</div>
                <div class="podium-team">{row['Team']} · started P{int(row['GridPosition'])}</div>
                <div class="podium-prob">{row['P_Win']*100:.1f}%</div>
                <div style="font-size: 0.85rem; opacity: 0.6;">to win · {row['P_Podium']*100:.0f}% podium</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Bar chart
    st.markdown("### 📊 Win / Podium / Points probability")
    chart_df = summary.head(15)
    fig = go.Figure()
    fig.add_bar(name="Win",    x=chart_df["Driver"], y=chart_df["P_Win"],    marker_color="#FFD700")
    fig.add_bar(name="Podium", x=chart_df["Driver"], y=chart_df["P_Podium"], marker_color="#C0C0C0")
    fig.add_bar(name="Points", x=chart_df["Driver"], y=chart_df["P_Points"], marker_color="#3671C6")
    fig.update_layout(
        barmode="group", template="plotly_dark", height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, width="stretch")

    # Heatmap + driver detail
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🎯 Position distribution heatmap")
        n = positions.shape[1]
        drivers = grid["Driver"].tolist()
        heat = np.zeros((n, n))
        for i in range(n):
            for p in range(1, n + 1):
                heat[i, p - 1] = (positions[:, i] == p).mean()
        order = summary["Driver"].tolist()
        idx_map = [drivers.index(d) for d in order]
        heat = heat[idx_map]
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat,
            x=[f"P{p}" for p in range(1, n + 1)],
            y=order,
            colorscale="Viridis",
            zmin=0, zmax=heat.max(),
            colorbar=dict(title="prob", tickformat=".0%"),
        ))
        fig_heat.update_layout(template="plotly_dark", height=550,
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, width="stretch")

    with col_right:
        st.markdown("### 🔍 Driver deep-dive")
        selected = st.selectbox("Pick a driver", summary["Driver"].tolist())
        sel_idx = grid["Driver"].tolist().index(selected)
        sel_team = grid["Team"].iloc[sel_idx]
        sel_color = TEAM_COLORS.get(sel_team, "#888")
        sel_row = summary[summary["Driver"] == selected].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("P(Win)",    f"{sel_row['P_Win']*100:.1f}%")
        m2.metric("P(Podium)", f"{sel_row['P_Podium']*100:.1f}%")
        m3.metric("P(Points)", f"{sel_row['P_Points']*100:.1f}%")
        m1.metric("Mean pos", f"{sel_row['MeanPosition']:.1f}")
        m2.metric("Std pos",  f"±{sel_row['StdPosition']:.1f}")
        m3.metric("P(DNF)",   f"{sel_row['P_DNF']*100:.1f}%")

        fig_hist = go.Figure(data=[go.Histogram(
            x=positions[:, sel_idx],
            xbins=dict(start=0.5, end=positions.shape[1] + 0.5, size=1),
            marker_color=sel_color,
        )])
        fig_hist.update_layout(
            title=f"{selected} — finish position over {n_sims:,} sims",
            xaxis_title="Final position", yaxis_title="Frequency",
            template="plotly_dark", height=320,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_hist, width="stretch")

    # Full table
    st.markdown("### 📋 Full results")
    display = summary.copy()
    for col in ["P_Win", "P_Podium", "P_Points", "P_DNF"]:
        display[col] = display[col].apply(lambda v: f"{v*100:.2f}%")
    display["MeanPosition"] = display["MeanPosition"].round(2)
    display["StdPosition"]  = display["StdPosition"].round(2)
    st.dataframe(display, width="stretch", hide_index=True)

    # Validation
    race_finished = df[(df.EventName == event) & df.IsWinner.notna()]
    if not race_finished.empty and race_finished["FinishPosition"].notna().any():
        st.markdown("### ✅ Validation vs actual race result")
        actual = race_finished.dropna(subset=["FinishPosition"]).sort_values("FinishPosition").head(5)
        cols = st.columns(5)
        for i, ((_, row), col) in enumerate(zip(actual.iterrows(), cols)):
            col.metric(f"P{i+1}  (actual)", row["Driver"], delta=row["Team"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════
with tab_how:
    st.title("🧭 How the simulator works")
    st.caption("From a Friday Free Practice session to 10 000 plausible race outcomes")

    st.markdown("""
    ### Pipeline overview
    ```
    ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐
    │  FastF1 API    │───▶│ Feature Engine  │───▶│ Race-Pace Model  │
    │  FP1/FP2/FP3   │    │  short-run pace │    │ blend(quali 70%, │
    │  Qualifying    │    │  long-run pace  │    │ short-run 30%) × │
    │  (live timing) │    │  GapToPole      │    │ QualiRaceRatio   │
    └────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                          │
    ┌─────────────────────────────────────────────────────▼─────────┐
    │                  Monte-Carlo lap-by-lap loop                  │
    │                                                               │
    │   for sim in 1 … N_SIMS:                                      │
    │     for lap in 1 … N_LAPS:                                    │
    │       pace = base_pace + offset[compound] + deg × tire_age    │
    │       pace += Normal(0, σ)         ◀── lap-to-lap noise       │
    │     total += pit_loss (one stop)                              │
    │     if rng < SC_prob:  total = leader + 0.55·(total − leader) │
    │     if rng < DNF_rate:   total += 10 000   ◀── retirement     │
    │     if weather == wet:  total += Normal(0, 8)  ◀── rain lottery│
    │     positions[sim] = argsort(total)                           │
    └───────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       P(win), P(podium), P(points), full distribution
    ```
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🧮 Pace model
        - **Why blended?** FP long-run laps are heavily polluted by fuel load and engine
          modes — they can differ by 9 s between teammates. We use **GapToPole (70%)** as the
          cleanest single-lap signal and the **best FP short-run lap (30%)** as a backup.
        - Quali deltas are then **scaled to race-relevant deltas** (`QualiRaceRatio = 0.35`),
          because fuel saving and tire management compress real gaps.
        - Per-driver deltas are clipped to ±1.2 s/lap to avoid 100-second runaways.
        """)
        st.markdown("""
        #### 🏎 Tyre model
        - Linear degradation per compound: **SOFT 0.10 s/lap · MEDIUM 0.05 · HARD 0.03**
        - Clean-lap offset: SOFT −0.5 s, MEDIUM ±0, HARD +0.3 s
        - One pit stop per driver at a configurable lap
        """)
    with c2:
        st.markdown("""
        #### 🚨 Race incidents
        - **Safety Car** — Bernoulli(p) per race; when triggered, all gaps to the leader
          are compressed by ×0.55 (the field bunches up).
        - **DNF** — Bernoulli(6%) per driver per race; retirement = +10 000 s penalty.
        - **Rain** — adds Normal(0, 8 s) total-race noise → upset variance balloons.
        """)
        st.markdown("""
        #### 🌍 Live weather
        We hit **open-meteo.com** (free, no key) for the race-day forecast at each circuit's
        exact lat/lon and average the **13–17h local race window**. Rain ≥ 40% auto-sets the
        simulator to Wet mode, but you can always override.
        """)

    st.markdown("---")
    st.markdown("""
    ### Engineering choices that matter for an F1 team
    - **Vectorised NumPy** — 10 000 sims × 70 laps × 20 drivers in ~1.5 s (no Python loops over sims).
    - **Reproducible** — fixed RNG seed, exposed in the sidebar.
    - **Compositional** — each effect (pace, deg, pit, SC, DNF, weather) is a separate term
      so you can disable or recalibrate any of them with one number.
    - **Backtest-driven** — see the next tab; we report top-1 accuracy, top-3 hit rate and
      multiclass Brier score so the model can be tuned against ground truth rather than vibes.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.title("📊 Backtest")
    st.caption("Replay every completed 2026 race through the simulator and score the predictions")

    bt = run_backtest_cached()
    if bt.empty:
        st.warning("No completed races yet to backtest.")
    else:
        s = summarise(bt)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Races scored", s["n"])
        k2.metric("Top-1 accuracy",     f"{s['top1_accuracy']*100:.0f}%",
                  help="Share of races where the favourite was the actual winner.")
        k3.metric("Top-3 hit rate",     f"{s['top3_hit_pct']*100:.0f}%",
                  help="Avg # of predicted podium drivers that were in the actual podium / 3.")
        k4.metric("Mean P(actual win)", f"{s['mean_p_win_actual']*100:.0f}%",
                  help="How much probability the model put on whoever actually won.")

        st.markdown(f"**Mean Brier score:** `{s['mean_brier']:.3f}`  "
                    "*(lower is better — uniform field ≈ 1.0, perfect call = 0)*")

        st.markdown("### Race-by-race")
        display_bt = bt.copy()
        display_bt["P_Win_OfActualWinner"] = display_bt["P_Win_OfActualWinner"].apply(lambda v: f"{v*100:.1f}%")
        display_bt["Brier"]   = display_bt["Brier"].round(3)
        display_bt["Entropy"] = display_bt["Entropy"].round(3)
        display_bt["Top1Hit"] = display_bt["Top1Hit"].map({1: "✅", 0: "❌"})
        display_bt["Top3Hit"] = display_bt["Top3Hit"].astype(str) + " / 3"
        st.dataframe(display_bt, width="stretch", hide_index=True)

        # Visual: predicted P(win) for the actual winner, race by race
        st.markdown("### Confidence on the actual winner, race by race")
        fig_bt = go.Figure()
        fig_bt.add_bar(
            x=bt["Event"], y=bt["P_Win_OfActualWinner"],
            marker_color=["#FFD700" if h else "#888" for h in bt["Top1Hit"]],
            text=[f"{p*100:.0f}%" for p in bt["P_Win_OfActualWinner"]],
            textposition="outside",
        )
        fig_bt.update_layout(
            template="plotly_dark", height=360,
            yaxis_title="P(win) assigned to the eventual winner",
            yaxis_tickformat=".0%", yaxis_range=[0, 1.05],
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bt, width="stretch")

        st.markdown(
            "**Reading note.** Yellow = the model also picked them as favourite (top-1 hit). "
            "Grey = the actual winner wasn't our #1 favourite, but we still want to see how much "
            "probability they were given. A grey bar at 35% means the model 'knew it was possible'; "
            "a grey bar near 0 would be a real miss."
        )

st.markdown("---")
st.caption("Built with FastF1 · NumPy · open-meteo · Plotly · Streamlit  |  "
           "Lap-by-lap Monte Carlo: tyre deg · pit · Safety Car · DNF · weather")
