import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="FuelSense AI", layout="wide")

# ---------------- SESSION STATE ----------------
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

# ---------------- TITLE ----------------
st.title("🔥 FuelSense AI")
st.subheader("Smart LPG Demand Prediction & Distribution Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Simulation Controls")

num_houses = st.sidebar.slider("Number of Houses", 100, 1000, 500)
demand_window = st.sidebar.slider("Demand Forecast Window (Days)", 5, 30, 15)
threshold_days = st.sidebar.slider("Auto Booking Threshold", 5, 15, 10)

run_button = st.sidebar.button("Run Prediction", key="run_btn")
reset_button = st.sidebar.button("Reset", key="reset_btn")

# ---------------- RESET LOGIC ----------------
if reset_button:
    st.session_state.run_prediction = False
    st.rerun()

# ---------------- RUN PREDICTION ----------------
if run_button:
    st.session_state.run_prediction = True

# ---------------- MAIN LOGIC ----------------
if st.session_state.run_prediction:

    # -------- DATA GENERATION --------
    data = {
        "House_ID": [f"H{str(i).zfill(3)}" for i in range(1, num_houses + 1)],
        "Family_Size": np.random.randint(2, 9, num_houses),
        "Cooking_Hours_Per_Day": np.round(np.random.uniform(2.0, 5.0, num_houses), 2),
        "Days_Since_Last_Refill": np.random.randint(1, 25, num_houses),
        "Avg_Booking_Interval_Days": np.random.randint(20, 35, num_houses)
    }

    df = pd.DataFrame(data)

    # -------- LPG LOGIC --------
    cylinder_capacity_days = 30

    df["Usage_Factor"] = df["Family_Size"] * df["Cooking_Hours_Per_Day"]
    df["Normalized_Usage"] = df["Usage_Factor"] / df["Usage_Factor"].max()

    df["Estimated_Cylinder_Days"] = cylinder_capacity_days / df["Normalized_Usage"]
    df["Days_Remaining"] = df["Estimated_Cylinder_Days"] - df["Days_Since_Last_Refill"]

    df["Days_Remaining"] = df["Days_Remaining"].clip(0, 40)
    df["Days_Remaining"] = df["Days_Remaining"].apply(lambda x: max(0, round(x)))

    df["Auto_Booking"] = df["Days_Remaining"].apply(
        lambda x: "YES" if x <= threshold_days else "NO"
    )

    # -------- FORECAST --------
    expected_demand = df[df["Days_Remaining"] <= demand_window]

    cylinders_needed = len(expected_demand)
    auto_count = len(df[df["Auto_Booking"] == "YES"])
    avg_days = round(df["Days_Remaining"].mean(), 2)

    # -------- METRICS --------
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🏠 Total Houses", num_houses)
    col2.metric("⚡ Auto Bookings", auto_count)
    col3.metric("🚚 Cylinders Needed", cylinders_needed)
    col4.metric("⏳ Avg Days Remaining", avg_days)

    st.divider()

    # -------- ALERTS --------
    if auto_count > num_houses * 0.3:
        st.warning("⚠ High LPG demand expected soon!")

    if cylinders_needed > num_houses * 0.4:
        st.error("🚨 Distribution trucks should prepare extra cylinders!")

    if cylinders_needed < num_houses * 0.1:
        st.success("✅ LPG supply stable in the area.")

    st.divider()

    # -------- HEATMAP --------
    st.subheader("🗺 Anna Nagar LPG Demand Heatmap")

    if st.button("View Heatmap", key="heatmap_btn"):

        lat_center = 13.0850
        lon_center = 80.2101

        heatmap_data = pd.DataFrame({
            "lat": lat_center + np.random.normal(0, 0.01, demand_window),
            "lon": lon_center + np.random.normal(0, 0.01, demand_window),
            "demand": np.random.randint(1, 10, demand_window)
        })

        st.map(heatmap_data)

    st.divider()

    # -------- DATASET PREVIEW --------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="lpg_prediction_data.csv",
        mime="text/csv"
    )

    st.divider()

    # -------- MACHINE LEARNING --------
    X = df[[
        "Family_Size",
        "Cooking_Hours_Per_Day",
        "Days_Since_Last_Refill",
        "Avg_Booking_Interval_Days"
    ]]

    y = df["Days_Remaining"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error = mean_absolute_error(y_test, predictions)

    st.subheader("🤖 AI Model Results")
    st.success(f"Mean Absolute Error: {round(error,2)} days")

    comparison = X_test.copy()
    comparison["Actual"] = y_test
    comparison["Predicted"] = predictions

    error_val = abs(comparison["Actual"] - comparison["Predicted"])

    # -------- ACCURACY METRICS --------
    high = sum(error_val <= 3)
    moderate = sum((error_val > 3) & (error_val <= 7))
    low = sum(error_val > 7)

    colA, colB, colC = st.columns(3)

    colA.metric("🟢 High Accuracy", high)
    colB.metric("🟠 Moderate Accuracy", moderate)
    colC.metric("🔴 Low Accuracy", low)

    st.divider()

    # -------- PREDICTION GRAPH --------
    colors = []

    for e in error_val:
        if e <= 3:
            colors.append("green")
        elif e <= 7:
            colors.append("orange")
        else:
            colors.append("red")

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(
        comparison["Actual"],
        comparison["Predicted"],
        c=colors,
        s=120,
        alpha=0.8
    )

    ax.plot([10, 40], [10, 40], linestyle="--")

    ax.set_xlabel("Actual Days Remaining")
    ax.set_ylabel("Predicted Days Remaining")
    ax.set_title("AI Prediction Accuracy")
    ax.grid(True)

    st.pyplot(fig)

    st.divider()

    # -------- DEMAND FORECAST --------
    st.subheader("📈 LPG Demand Forecast")

    demand_chart = expected_demand["Days_Remaining"].value_counts().sort_index()

    st.bar_chart(demand_chart)