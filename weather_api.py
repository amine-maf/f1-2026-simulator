"""
Live weather data for F1 circuits via open-meteo.com (no API key required).

We fetch precipitation probability and air/track-relevant temperatures for the
race-day window of each Grand Prix and expose a single helper that returns a
small dictionary the rest of the app can consume.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from urllib.request import urlopen
from urllib.error import URLError
import json

# Circuit coordinates (decimal degrees) — sufficient precision for weather grid
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "Australian Grand Prix":      (-37.8497, 144.9680),
    "Chinese Grand Prix":         ( 31.3389, 121.2200),
    "Japanese Grand Prix":        ( 34.8431, 136.5410),
    "Bahrain Grand Prix":         ( 26.0325,  50.5106),
    "Saudi Arabian Grand Prix":   ( 21.6319,  39.1044),
    "Miami Grand Prix":           ( 25.9581, -80.2389),
    "Emilia Romagna Grand Prix":  ( 44.3439,  11.7167),
    "Monaco Grand Prix":          ( 43.7347,   7.4206),
    "Spanish Grand Prix":         ( 41.5700,   2.2611),
    "Canadian Grand Prix":        ( 45.5000, -73.5228),
    "Austrian Grand Prix":        ( 47.2197,  14.7647),
    "British Grand Prix":         ( 52.0786,  -1.0169),
    "Hungarian Grand Prix":       ( 47.5789,  19.2486),
    "Belgian Grand Prix":         ( 50.4372,   5.9714),
    "Dutch Grand Prix":           ( 52.3888,   4.5409),
    "Italian Grand Prix":         ( 45.6156,   9.2811),
    "Azerbaijan Grand Prix":      ( 40.3725,  49.8533),
    "Singapore Grand Prix":       (  1.2914, 103.8642),
    "United States Grand Prix":   ( 30.1328, -97.6411),
    "Mexico City Grand Prix":     ( 19.4042, -99.0907),
    "São Paulo Grand Prix":       (-23.7036, -46.6997),
    "Las Vegas Grand Prix":       ( 36.1147, -115.1728),
    "Qatar Grand Prix":           ( 25.4900,  51.4542),
    "Abu Dhabi Grand Prix":       ( 24.4672,  54.6031),
}


def fetch_weather(event_name: str, race_date: datetime | None = None,
                  timeout: float = 5.0) -> dict | None:
    """Pull a race-day forecast from open-meteo for the given GP.

    Returns a dict with keys: rain_prob (0–100), temp_c, wind_kmh, source, date
    or None on failure / unknown circuit.
    """
    coords = CIRCUIT_COORDS.get(event_name)
    if coords is None:
        return None

    lat, lon = coords
    if race_date is None:
        race_date = datetime.utcnow() + timedelta(days=2)
    target = race_date.strftime("%Y-%m-%d")

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation_probability,temperature_2m,wind_speed_10m"
        f"&start_date={target}&end_date={target}"
        "&timezone=auto"
    )

    try:
        with urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
    except (URLError, TimeoutError, json.JSONDecodeError, Exception):
        return None

    hourly = data.get("hourly", {})
    rain = hourly.get("precipitation_probability") or []
    temp = hourly.get("temperature_2m") or []
    wind = hourly.get("wind_speed_10m") or []
    if not rain:
        return None

    # Race usually runs in the local afternoon window (13:00–17:00).
    window = slice(13, 17)
    def _avg(arr): return sum(arr[window]) / max(1, len(arr[window])) if arr else None

    return {
        "rain_prob": _avg(rain),
        "temp_c": _avg(temp),
        "wind_kmh": _avg(wind),
        "source": "open-meteo.com",
        "date": target,
        "lat": lat,
        "lon": lon,
    }


def suggest_weather_setting(rain_prob: float | None,
                            wet_threshold: float = 40.0) -> str:
    """Map rain probability to the simulator's weather mode."""
    if rain_prob is None:
        return "Dry"
    return "Wet" if rain_prob >= wet_threshold else "Dry"


if __name__ == "__main__":
    import sys
    event = sys.argv[1] if len(sys.argv) > 1 else "Canadian Grand Prix"
    wx = fetch_weather(event)
    print(json.dumps(wx, indent=2))
