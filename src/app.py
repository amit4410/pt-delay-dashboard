"""Streamlit dashboard - Public Transport Fleet Efficiency & Delay Intelligence."""
import sys
from pathlib import Path

# Make "src" importable when running via `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Auto-generate synthetic data on first run (Streamlit Cloud)
from src.config import DATA_RAW
if not (DATA_RAW / "trips.csv").exists():
    from src.generate_data import generate_all
    generate_all()

# Train model on first run if not present
from src.config import MODELS_DIR
if not (MODELS_DIR / "delay_rf.joblib").exists():
    from src.model import train
    train()

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CITY_NAME, OTP_EARLY, OTP_LATE
from src.metrics import (
    load_trips, load_live, kpi_summary, route_performance,
    delay_by_hour, delay_heatmap, depot_efficiency, fuel_mix,
)
from src import model as delay_model

st.set_page_config(
    page_title="PT Fleet & Delay Intelligence",
    page_icon="bus",
    layout="wide",
)


@st.cache_data(ttl=60)
def get_data():
    return load_trips(), load_live()


trips, live = get_data()

st.sidebar.title("PT Delay Intelligence")
st.sidebar.caption(f"{CITY_NAME} | {len(trips):,} trips")

modes = st.sidebar.multiselect("Mode", sorted(trips["mode"].unique()),
                               default=sorted(trips["mode"].unique()))
depots = st.sidebar.multiselect("Depot", sorted(trips["depot"].unique()),
                                default=sorted(trips["depot"].unique()))
hour_range = st.sidebar.slider("Hour range", 0, 23, (5, 23))

mask = trips["mode"].isin(modes) & trips["depot"].isin(depots) & \
       trips["hour"].between(hour_range[0], hour_range[1])
t = trips[mask]

st.title("Public Transport - Fleet Efficiency & Delay Intelligence")
kpi = kpi_summary(t)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("On-Time %", f"{kpi['otp']}%", delta=f"{kpi['otp'] - 85:.1f} vs 85%")
c2.metric("Avg Delay", f"{kpi['avg_delay']} min")
c3.metric("P90 Delay", f"{kpi['p90_delay']} min")
c4.metric("Active Vehicles", kpi["active_vehicles"])
c5.metric("Load Factor", f"{kpi['load_factor']}%")
c6.metric("Fleet Utilization", f"{kpi['utilization']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive", "Real-Time", "Fleet Efficiency",
    "Delay Analytics", "Predictive"
])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Average Delay by Hour")
        h = delay_by_hour(t)
        fig = px.line(h, x="hour", y="avg_delay", markers=True)
        fig.update_layout(height=350, xaxis_title="Hour", yaxis_title="Avg delay (min)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Fuel Type Mix")
        fm = fuel_mix(t)
        fig = px.pie(fm, names="fuel_type", values="trips", hole=0.5)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 10 Delayed Routes")
    rp = route_performance(t).head(10)
    fig = px.bar(rp, x="avg_delay", y="route_name", orientation="h",
                 color="otp", color_continuous_scale="RdYlGn")
    fig.update_layout(height=400, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Live Vehicle Positions (color = delay)")
    col1, col2 = st.columns([3, 1])
    with col2:
        st.metric("Vehicles on map", len(live))
        st.metric("Vehicles > 15 min late", int((live["delay_min"] > 15).sum()))
    with col1:
        fig = px.scatter_map(
            live, lat="lat", lon="lon",
            color="delay_min", size="passengers",
            hover_name="route_name",
            hover_data=["vehicle_id", "mode", "speed_kmph", "delay_min"],
            color_continuous_scale="RdYlGn_r",
            range_color=[-5, 30], zoom=10, height=560,
            map_style="open-street-map",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Active Alerts (delay > 15 min)")
    alerts = live[live["delay_min"] > 15][
        ["vehicle_id", "route_name", "mode", "delay_min", "speed_kmph"]
    ].sort_values("delay_min", ascending=False)
    st.dataframe(alerts, use_container_width=True, height=250)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Depot Efficiency")
        dep = depot_efficiency(t)
        st.dataframe(dep, use_container_width=True)
        fig = px.bar(dep, x="depot", y="otp", color="avg_delay",
                     color_continuous_scale="RdYlGn")
        fig.update_layout(height=320, yaxis_title="OTP %")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Load Factor Distribution")
        fig = px.histogram(t, x="load_factor", nbins=40,
                           color="mode", barmode="overlay")
        fig.update_layout(height=320, xaxis_title="Load factor")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Fuel Type Performance")
        st.dataframe(fuel_mix(t), use_container_width=True)
    st.subheader("Route-level Performance")
    st.dataframe(route_performance(t), use_container_width=True, height=350)

with tab4:
    st.subheader("Delay Heatmap: Route x Hour")
    hm = delay_heatmap(t)
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="RdYlGn_r",
                    labels=dict(color="Avg delay (min)"))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delay Distribution")
        fig = px.histogram(t, x="delay_min", nbins=50, color="mode",
                           barmode="overlay")
        fig.add_vrect(x0=OTP_EARLY, x1=OTP_LATE, fillcolor="green",
                      opacity=0.15, annotation_text="OTP window")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Delay by Weekday")
        wd = t.groupby("weekday")["delay_min"].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]).reset_index()
        fig = px.bar(wd, x="weekday", y="delay_min")
        fig.update_layout(height=350, yaxis_title="Avg delay (min)")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Delay Prediction (Random Forest)")
    try:
        pred_input = live.copy()
        pred_input["scheduled_departure"] = pd.Timestamp.now()
        preds = delay_model.predict(pred_input)
        live_pred = live.copy()
        live_pred["predicted_delay_min"] = np.round(preds, 2)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter(live_pred,
                             x="delay_min", y="predicted_delay_min",
                             color="mode", hover_name="route_name",
                             labels={"delay_min": "Actual delay (min)",
                                     "predicted_delay_min": "Predicted (min)"})
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mae = np.abs(live_pred.delay_min - live_pred.predicted_delay_min).mean()
            st.metric("MAE (live set)", f"{mae:.2f} min")
            st.metric("Predicted > 15 min",
                      int((live_pred.predicted_delay_min > 15).sum()))
            st.dataframe(
                live_pred[["vehicle_id", "route_name",
                           "delay_min", "predicted_delay_min"]]
                .sort_values("predicted_delay_min", ascending=False)
                .head(15),
                use_container_width=True, height=400)
    except FileNotFoundError:
        st.warning("No trained model found. Run: python -m src.model")
        st.code("python -m src.model", language="bash")

