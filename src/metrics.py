"""KPI computation layer."""
import pandas as pd
import numpy as np

from src.config import OTP_EARLY, OTP_LATE, DATA_RAW


def load_trips() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "trips.csv", parse_dates=[
        "scheduled_departure", "actual_departure",
        "scheduled_arrival", "actual_arrival",
    ])


def load_live() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "live_positions.csv")


def kpi_summary(df: pd.DataFrame) -> dict:
    otp = df["delay_min"].between(OTP_EARLY, OTP_LATE).mean() * 100
    avg_delay = df["delay_min"].mean()
    p90_delay = df["delay_min"].quantile(0.9)
    active_vehicles = df["vehicle_id"].nunique()
    load_factor = (df["passengers"].sum() / df["capacity"].sum()) * 100
    revenue_km = (df["length_km"] * df["passengers"].gt(0)).sum()
    total_km = df["length_km"].sum()
    utilization = revenue_km / total_km * 100 if total_km else 0
    return {
        "otp": round(otp, 2),
        "avg_delay": round(avg_delay, 2),
        "p90_delay": round(p90_delay, 2),
        "active_vehicles": int(active_vehicles),
        "load_factor": round(load_factor, 1),
        "utilization": round(utilization, 1),
    }


def route_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["route_id", "route_name", "mode"]).agg(
        trips=("trip_id", "count"),
        avg_delay=("delay_min", "mean"),
        otp=("on_time", "mean"),
        load_factor=("load_factor", "mean"),
    ).reset_index()
    g["otp"] = (g["otp"] * 100).round(1)
    g["avg_delay"] = g["avg_delay"].round(2)
    g["load_factor"] = (g["load_factor"] * 100).round(1)
    return g.sort_values("avg_delay", ascending=False)


def delay_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("hour").agg(
        avg_delay=("delay_min", "mean"),
        trips=("trip_id", "count"),
    ).reset_index().round(2)


def delay_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="route_name", columns="hour",
        values="delay_min", aggfunc="mean",
    ).round(1)


def depot_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("depot").agg(
        vehicles=("vehicle_id", "nunique"),
        avg_delay=("delay_min", "mean"),
        otp=("on_time", "mean"),
        load_factor=("load_factor", "mean"),
    ).reset_index().round(2)


def fuel_mix(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("fuel_type").agg(
        trips=("trip_id", "count"),
        avg_delay=("delay_min", "mean"),
    ).reset_index().round(2)