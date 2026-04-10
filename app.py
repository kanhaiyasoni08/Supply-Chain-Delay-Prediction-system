from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"

# ---------------- ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    try:
        # We assume these exist based on model.py run
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")
        feature_order = joblib.load("feature_order.pkl")
        return model, encoders, feature_order
    except Exception as e:
        return None, None, None

model, encoders, feature_order = load_assets()

# ---------------- CUSTOM UI (Restored from your original app.py) ----------------
st.markdown("""
<style>
.main {background-color: #f4f6f9;}

.card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.08);
}

.metric-green {
    color: #28a745;
    font-size: 2.5rem;
    font-weight: bold;
}

.metric-red {
    color: #dc3545;
    font-size: 2.5rem;
    font-weight: bold;
}

.highlight-box {
    background-color: #fff8e1;
    padding: 15px;
    border-left: 5px solid #ffc107;
    border-radius: 8px;
}

.stButton>button {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def safe_encode(le, series):
    """Safely encode a pandas series, handling unseen labels."""
    labels = [str(c) for c in le.classes_]
    return series.astype(str).apply(lambda x: le.transform([x])[0] if x in labels else 0)

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE, on_bad_lines='warn')
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def append_prediction(record):
    if os.path.exists(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE)
        
        # ID Generation
        if "Shipment_ID" in existing_df.columns and len(existing_df) > 0:
            last_id = str(existing_df["Shipment_ID"].iloc[-1])
            try:
                last_num = int(''.join(filter(str.isdigit, last_id)))
                new_id = f"SHP{last_num + 1:07d}"
            except:
                new_id = f"SHP{len(existing_df) + 1:07d}"
        else:
            new_id = "SHP0000001"

        record["Shipment_ID"] = new_id
        new_row = pd.DataFrame([record])
        new_row = new_row.reindex(columns=existing_df.columns)
        new_row.to_csv(DATA_FILE, mode='a', header=False, index=False)

# ---------------- RETRAIN FUNCTION (Updated Logic) ----------------
def retrain_model(df):
    df = df.dropna(subset=["Is_Delayed"])
    if df.empty:
        return None, 0, None, None

    # Logic from model.py
    traffic_map = {"Low": 1, "Medium": 2, "High": 3}
    weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Fog": 3, "Snow": 3}
    
    df["Traffic_Distance"] = df["Distance_km"] * df["Traffic_Level"].map(traffic_map)
    df["Weather_Risk"] = df["Weather_Condition"].map(weather_map)
    
    y = df["Is_Delayed"]
    cols_to_drop = ["Is_Delayed", "Shipment_ID", "Transport_Mode", "Distance_km", "Traffic_Level", "Weather_Condition"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    new_encoders = {}
    for col in X.columns:
        if "Date" in col:
            X[col] = pd.to_datetime(X[col], format='mixed').map(lambda x: x.toordinal())
        elif X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            new_encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    new_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42)
    new_model.fit(X_train, y_train)

    joblib.dump(new_model, "model.pkl")
    joblib.dump(new_encoders, "encoders.pkl")
    joblib.dump(X.columns.tolist(), "feature_order.pkl")

    return new_model, accuracy_score(y_test, new_model.predict(X_test)), X_test, y_test

# ---------------- HEADER ----------------
st.title("🚚 Supply Chain Delay Prediction system")

st.markdown('<div class="card">', unsafe_allow_html=True)
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.subheader("Dashboard Overview")
with col_h2:
    st.info("📅 Last Update: Today")
with st.expander("🔔 Feature Updates"):
    st.write("- Feature Mismatch Solved: Traffic_Distance & Weather_Risk integrated.")
    st.write("- Label Encoding Fixed: safe_encode handles unseen values.")
    st.write("- Original UI styling and Insights page fully restored.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Dashboard", "Data Log", "Model Center", "Insights"])

# ================= DASHBOARD =================
if page == "Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 Shipment Input")

    col1, col2, col3 = st.columns(3)
    with col1:
        Order_Date = st.date_input("Order Date")
        Shipment_Priority = st.selectbox("Shipment Priority", ["Low", "Medium", "High"])
        Traffic_Level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    with col2:
        Dispatch_Date = st.date_input("Dispatch Date")
        Transport_Mode = st.selectbox("Transport Mode", ["Road", "Air", "Ship", "Rail"])
        Weather_Condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Fog", "Snow"])
        Public_Holiday = st.selectbox("Public Holiday", [0, 1])
    with col3:
        Estimated_Delivery_Date = st.date_input("Estimated Delivery Date")
        Distance_km = st.number_input("Distance (km)", 0, 5000, 500)
        Warehouse_Processing_Time_hours = st.number_input("Processing Time (hrs)", 0, 100, 10)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict"):
        if model is None or encoders is None or feature_order is None:
            st.error("Model assets missing. Please retrain in Model Center or verify .pkl files.")
        else:
            try:
                # 1. Feature Engineering (Match logic from model.py reference)
                traffic_map = {"Low": 1, "Medium": 2, "High": 3}
                weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Fog": 3, "Snow": 3}
                
                # 2. Build DataFrame with correct types and internal mappings
                input_df = pd.DataFrame([{
                    "Order_Date": Order_Date.strftime('%d-%m-%Y'),
                    "Dispatch_Date": Dispatch_Date.strftime('%d-%m-%Y'),
                    "Estimated_Delivery_Date": Estimated_Delivery_Date.strftime('%d-%m-%Y'),
                    "Shipment_Priority": Shipment_Priority,
                    "Warehouse_Processing_Time_hours": float(Warehouse_Processing_Time_hours),
                    "Public_Holiday": int(Public_Holiday),
                    "Traffic_Distance": Distance_km * traffic_map[Traffic_Level],
                    "Weather_Risk": weather_map[Weather_Condition]
                }])

                # 3. Transform dates and categorical values
                for col in input_df.columns:
                    if "Date" in col:
                        input_df[col] = pd.to_datetime(input_df[col], format='%d-%m-%Y').map(lambda x: x.toordinal())
                    elif col in encoders:
                        # Use the safe_encode helper to fix label errors
                        input_df[col] = safe_encode(encoders[col], input_df[col])

                # 4. Final Alignment - critical fix for feature alignment error
                input_df = input_df[feature_order]

                # 5. Predict
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                if prediction == 1:
                    st.error(f"### ⚠️ Prediction: Delayed ({probability:.1%})")
                else:
                    st.success(f"### ✅ Prediction: On time ({1-probability:.1%})")

                # 6. Log raw data for storage
                record = {
                    "Order_Date": Order_Date.strftime('%d-%m-%Y'),
                    "Dispatch_Date": Dispatch_Date.strftime('%d-%m-%Y'),
                    "Estimated_Delivery_Date": Estimated_Delivery_Date.strftime('%d-%m-%Y'),
                    "Shipment_Priority": Shipment_Priority, "Transport_Mode": Transport_Mode,
                    "Distance_km": Distance_km, "Traffic_Level": Traffic_Level,
                    "Weather_Condition": Weather_Condition,
                    "Warehouse_Processing_Time_hours": Warehouse_Processing_Time_hours,
                    "Public_Holiday": Public_Holiday, "Is_Delayed": int(prediction)
                }
                append_prediction(record)

            except Exception as e:
                st.error(f"Processing Error: {e}")

# ================= DATA LOG =================
elif page == "Data Log":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Stored Predictions")
    df = load_data()
    if df.empty:
        st.warning("No data available")
    else:
        st.dataframe(df, use_container_width=True)
        st.metric("Total Records", len(df))
    st.markdown('</div>', unsafe_allow_html=True)

# ================= MODEL CENTER =================
elif page == "Model Center":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Model Center")
    df = load_data()
    if df.empty:
        st.warning("Dataset missing")
    else:
        if st.button("🔁 Retrain Model"):
            with st.spinner("Synchronizing features..."):
                new_model, acc, X_test, y_test = retrain_model(df)
                if new_model:
                    st.success(f"Retrained Successfully! Accuracy: {acc:.2f}")
                    cm = confusion_matrix(y_test, new_model.predict(X_test))
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= INSIGHTS (Restored) =================
elif page == "Insights":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 insights")
    df = load_data()

    if df.empty:
        st.warning("No data available")
    else:
        st.subheader("Delay Distribution")
        if "Is_Delayed" in df.columns:
            st.bar_chart(df["Is_Delayed"].value_counts().rename(index={0: "On-Time", 1: "Delayed"}))
        else:
            st.error("Column 'Is_Delayed' not found in dataset.")

        if model and hasattr(model, "feature_importances_") and feature_order:
            st.subheader("Top Risk Factors")
            imp_df = pd.DataFrame({
                "Feature": feature_order,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.bar_chart(imp_df.set_index("Feature"))
    st.markdown('</div>', unsafe_allow_html=True)
