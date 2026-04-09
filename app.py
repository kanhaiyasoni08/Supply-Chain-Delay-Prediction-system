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
import pandas as pd
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"
feature_order = joblib.load("feature_order.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------- CUSTOM UI ----------------
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

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

model = load_model()

# ---------------- DATA FUNCTIONS ----------------

def append_prediction(record):
    if os.path.exists(DATA_FILE):
        # 1. Get the Exact Headers from the file
        existing_df_head = pd.read_csv(DATA_FILE, nrows=0)
        existing_cols = existing_df_head.columns.tolist()

        # 2. Generate Shipment_ID
        existing_df_full = pd.read_csv(DATA_FILE)
        if "Shipment_ID" in existing_df_full.columns and len(existing_df_full) > 0:
            last_id = str(existing_df_full["Shipment_ID"].iloc[-1])
            try:
                last_num = int(last_id.replace("SHP", ""))
                new_id = f"SHP{last_num + 1:07d}"
            except:
                new_id = "SHP0000001"
        else:
            new_id = "SHP0000001"

        # 3. Create the Row and FORCE KEY MATCHING
        full_record = {"Shipment_ID": new_id}
        full_record.update(record)
        
        # Convert record keys to match CSV header case-sensitivity
        # This prevents "Is_delayed" vs "Is_Delayed" issues
        normalized_record = {}
        for col in existing_cols:
            # Search for the key in a case-insensitive way
            match = next((k for k in full_record if k.lower() == col.lower()), None)
            if match:
                normalized_record[col] = full_record[match]
            else:
                normalized_record[col] = np.nan

        new_row_df = pd.DataFrame([normalized_record])

        # 4. Final Reindex and Save
        new_row_df = new_row_df.reindex(columns=existing_cols)
        new_row_df.to_csv(DATA_FILE, mode='a', header=False, index=False)

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            # on_bad_lines='warn' skips the error rows and prints a warning
            return pd.read_csv(DATA_FILE, on_bad_lines='warn')
        except Exception as e:
            st.error(f"Fatal error reading CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_row_count():
    if os.path.exists(DATA_FILE):
        return len(pd.read_csv(DATA_FILE))
    return 0

def get_input_columns(df):
    return [col for col in df.columns if col not in ["Is_Delayed", "Shipment_ID", "Prediction"]]

# ---------------- RETRAIN FUNCTION ----------------
def retrain_model(df):
    # 1. Remove rows where the target 'Is_Delayed' is missing (NaN)
    # This prevents the "Input y contains NaN" error
    df = df.dropna(subset=["Is_Delayed"])
    
    if df.empty:
        st.error("Not enough valid data to retrain!")
        return None, 0, None, None

    # 2. Separate Target (y) and Features (X)
    y = df["Is_Delayed"]
    
    # Drop non-feature columns
    cols_to_drop = ["Is_Delayed", "Shipment_ID", "Transport_Mode", "Prediction"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. Apply Preprocessing (Ordinal dates and Label Encoding)
    # -----------------------------
# Inside retrain_model(df):
# -----------------------------
    for col in X.columns:
        if "Date" in col:
            # Use format='mixed' to handle both YYYY-MM-DD and DD-MM-YYYY
            X[col] = pd.to_datetime(X[col], format='mixed').map(lambda x: x.toordinal())
        elif X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 4. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    new_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    new_model.fit(X_train, y_train)

    # 5. Evaluate and Save
    preds = new_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump(new_model, MODEL_FILE)

    return new_model, acc, X_test, y_test

# ---------------- HEADER ----------------
st.title("🚚 Supply Chain Delay Prediction system")

st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("Dashboard Overview")

with col2:
    st.info("📅 Last Update: Today")

with st.expander("🔔 Feature Updates"):
    st.write("- Dynamic input system added")
    st.write("- Uses all dataset features automatically")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Data Log",
    "Model Center",
    "Insights"
])

# ================= DASHBOARD =================
if page == "Dashboard":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📦 Shipment Input")

    col1, col2, col3 = st.columns(3)

    # -------- INPUT --------
    with col1:
        Order_Date = st.date_input("Order Date")
        Shipment_Priority = st.selectbox("Shipment Priority", ["Low", "Medium", "High"])
        Traffic_Level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])

    with col2:
        Dispatch_Date = st.date_input("Dispatch Date")
        Transport_Mode = st.selectbox("Transport Mode", ["Road", "Air", "Ship"])  # UI + storage only
        Weather_Condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Fog"])
        Public_Holiday = st.selectbox("Public Holiday", [0, 1])

    with col3:
        Estimated_Delivery_Date = st.date_input("Estimated Delivery Date")
        Distance_km = st.number_input("Distance (km)", 0, 5000, 500)
        Warehouse_Processing_Time_hours = st.number_input("Processing Time (hrs)", 0, 100, 10)


    st.markdown('</div>', unsafe_allow_html=True)

    # -------- DATE CONVERSION --------
    order_days = Order_Date.toordinal()
    dispatch_days = Dispatch_Date.toordinal()
    delivery_days = Estimated_Delivery_Date.toordinal()

    # -------- TRAFFIC LEVEL MAPPING --------
    traffic_map = {"Low": 1, "Medium": 2, "High": 3}
    traffic_num = traffic_map[Traffic_Level]

    # -------- FEATURE ENGINEERING --------
    Traffic_Distance = Distance_km * traffic_num
    weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Fog": 3}
    Weather_Risk = weather_map[Weather_Condition]

    # -------- MODEL INPUT (NO Transport_Mode) --------
    input_df = pd.DataFrame([{
        "Order_Date": order_days,
        "Dispatch_Date": dispatch_days,
        "Estimated_Delivery_Date": delivery_days,
        "Shipment_Priority": Shipment_Priority,
        "Traffic_Distance": Traffic_Distance,
        "Weather_Risk": Weather_Risk,
        "Warehouse_Processing_Time_hours": Warehouse_Processing_Time_hours,
        "Public_Holiday": Public_Holiday
    }])

    if st.button("🚀 Predict"):

        if model is None:
            st.error("Model not available!")
        else:
            # Load saved objects
            feature_order = joblib.load("feature_order.pkl")
            encoders = joblib.load("encoders.pkl")

            # Apply label encoding (same as training)
            for col, le in encoders.items():
                if col in input_df.columns and input_df[col].dtype == 'object':
                    input_df[col] = le.transform(input_df[col])

            # FIX COLUMN ORDER (VERY IMPORTANT)
            input_df = input_df[feature_order]

            # Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # Display result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎯 Prediction Result")
            
            if prediction == 1:
                st.markdown(
                    f'<p class="metric-red">❌ Delayed ({probability*100:.2f}%)</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p class="metric-green">✅ On-Time ({(1-probability)*100:.2f}%)</p>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # -------- STORE FULL DATA --------
            record = {
                "Order_Date": Order_Date,
                "Dispatch_Date": Dispatch_Date,
                "Estimated_Delivery_Date": Estimated_Delivery_Date,
                "Shipment_Priority": Shipment_Priority,
                "Transport_Mode": Transport_Mode,  # stored but NOT used in model
                "Distance_km": Distance_km,
                "Traffic_Level": Traffic_Level,
                "Weather_Condition": Weather_Condition,
                "Warehouse_Processing_Time_hours": Warehouse_Processing_Time_hours,
                "Public_Holiday": Public_Holiday,
                "Prediction": prediction
            }

            append_prediction(record)

# ================= DATA LOG =================
elif page == "Data Log":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Stored Predictions")

    df = load_data()

    if df.empty:
        st.warning("No data available")
    else:
        st.dataframe(df)
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

            model, acc, X_test, y_test = retrain_model(df)

            st.success(f"Accuracy: {acc:.2f}")

            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.text(classification_report(y_test, preds))

    st.markdown('</div>', unsafe_allow_html=True)

# ================= INSIGHTS =================
# ================= INSIGHTS =================
elif page == "Insights":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Insights")

    df = load_data()

    if df.empty:
        st.warning("No data available")
    else:
        # FIX 1: Change "Prediction" to "Is_Delayed"
        st.subheader("Delay Distribution")
        if "Is_Delayed" in df.columns:
            st.bar_chart(df["Is_Delayed"].value_counts())
        else:
            st.error("Column 'Is_Delayed' not found in dataset.")

        # FIX 2: Feature Importance
        if model and hasattr(model, "feature_importances_"):
            # Ensure we only get columns used for training
            # We drop non-feature columns like ID and the Target itself
            cols_to_ignore = ["Is_Delayed", "Shipment_ID", "Transport_Mode"]
            features = [col for col in df.columns if col not in cols_to_ignore]
            
            importance = model.feature_importances_

            # Ensure the length of features matches the length of importance
            if len(features) == len(importance):
                imp_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                st.subheader("Top Risk Factors")
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.info("Feature importance alignment pending model refresh.")

    st.markdown('</div>', unsafe_allow_html=True)