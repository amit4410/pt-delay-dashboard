"""Generate synthetic GTFS-Realtime-like data for the dashboard."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import (
    DATA_RAW, NUM_ROUTES, NUM_VEHICLES, DAYS_OF_DATA,
    TRIPS_PER_DAY, CITY_NAME, BUS_CAPACITY,
)


def _route_table(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "route_id": [f"R{i:03d}" for i in range(1, n + 1)],
        "route_name": [f"Route {i}" for i in range(1, n + 1)],
        "mode": np.random.choice(["Bus", "Bus", "Bus", "Metro"], n),
        "length_km": np.round(np.random.uniform(8, 35, n), 1),
    })


def _vehicle_table(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "vehicle_id": [f"V{i:04d}" for i in range(1, n + 1)],
        "depot": np.random.choice(["North", "South", "East", "West"], n),
        "capacity": BUS_CAPACITY,
        "fuel_type": np.random.choice(["Diesel", "CNG", "Electric"], n, p=[0.5, 0.3, 0.2]),
    })


def _trip_events(routes, vehicles, trips_per_day, days):
    rng = np.random.default_rng(42)
    start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=days)
    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        for _ in range(trips_per_day):
            route = routes.sample(1, random_state=rng.integers(1e9)).iloc[0]
            vehicle = vehicles.sample(1, random_state=rng.integers(1e9)).iloc[0]
            hour = int(np.clip(rng.normal(9, 4), 5, 23))
            minute = int(rng.integers(0, 60))
            sched_dep = day.replace(hour=hour, minute=minute, second=0)
            scheduled_dur = float(route["length_km"]) * rng.uniform(1.8, 2.6)
            peak = 1.6 if hour in (8, 9, 17, 18, 19) else 1.0
            delay = float(np.clip(rng.normal(2.5, 6.0) * peak, -3, 45))
            actual_dep = sched_dep + timedelta(minutes=delay)
            actual_dur = scheduled_dur + rng.normal(0, 2)
            actual_arr = actual_dep + timedelta(minutes=max(actual_dur, 1))
            passengers = int(np.clip(rng.normal(35, 18), 0, vehicle["capacity"]))

            rows.append({
                "trip_id": f"T{len(rows):07d}",
                "route_id": route["route_id"],
                "route_name": route["route_name"],
                "mode": route["mode"],
                "vehicle_id": vehicle["vehicle_id"],
                "depot": vehicle["depot"],
                "fuel_type": vehicle["fuel_type"],
                "scheduled_departure": sched_dep,
                "actual_departure": actual_dep,
                "scheduled_arrival": sched_dep + timedelta(minutes=scheduled_dur),
                "actual_arrival": actual_arr,
                "delay_min": round(delay, 2),
                "scheduled_duration_min": round(scheduled_dur, 2),
                "actual_duration_min": round(max(actual_dur, 1), 2),
                "passengers": passengers,
                "capacity": int(vehicle["capacity"]),
                "length_km": float(route["length_km"]),
            })
    df = pd.DataFrame(rows)
    df["hour"] = df["scheduled_departure"].dt.hour
    df["date"] = df["scheduled_departure"].dt.date
    df["weekday"] = df["scheduled_departure"].dt.day_name()
    df["on_time"] = df["delay_min"].between(-1, 5)
    df["load_factor"] = df["passengers"] / df["capacity"]
    return df


def generate_all():
    print(f"Generating synthetic data for {CITY_NAME}...")
    routes = _route_table(NUM_ROUTES)
    vehicles = _vehicle_table(NUM_VEHICLES)
    trips = _trip_events(routes, vehicles, TRIPS_PER_DAY, DAYS_OF_DATA)

    live = trips.sort_values("scheduled_departure").tail(NUM_VEHICLES).copy()
    live["lat"] = 28.55 + np.random.uniform(-0.25, 0.25, len(live))
    live["lon"] = 77.20 + np.random.uniform(-0.25, 0.25, len(live))
    live["speed_kmph"] = np.round(np.random.uniform(0, 55, len(live)), 1)

    routes.to_csv(DATA_RAW / "routes.csv", index=False)
    vehicles.to_csv(DATA_RAW / "vehicles.csv", index=False)
    trips.to_csv(DATA_RAW / "trips.csv", index=False)
    live[["vehicle_id", "route_id", "route_name", "mode", "lat", "lon",
          "speed_kmph", "delay_min", "passengers", "capacity"]].to_csv(
        DATA_RAW / "live_positions.csv", index=False)

    print(f"  routes.csv    : {len(routes):>6} rows")
    print(f"  vehicles.csv  : {len(vehicles):>6} rows")
    print(f"  trips.csv     : {len(trips):>6} rows")
    print(f"  live_positions: {len(live):>6} rows")


if __name__ == "__main__":
    generate_all()
