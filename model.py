import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data.csv")

# Drop unnecessary column
df.drop(['Shipment_ID'], axis=1, inplace=True)
df.drop(['Transport_Mode'], axis=1, inplace=True)

# -----------------------------
# created powerful features
# -----------------------------
traffic_map = {"Low": 1, "Medium": 2, "High": 3}
weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Fog": 3}

df["Traffic_Distance"] = df["Distance_km"] * df["Traffic_Level"].map(traffic_map)
df["Weather_Risk"] = df["Weather_Condition"].map(weather_map)
df.drop(["Distance_km", "Traffic_Level", "Weather_Condition"], axis=1, inplace=True)

# -----------------------------
# 2. Convert Categorical → Numeric
# -----------------------------
label_encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Save encoders for UI use
joblib.dump(label_encoders, "encoders.pkl")

# -----------------------------
# 3. Split Data
# -----------------------------
y = df["Is_Delayed"]
X = df.drop("Is_Delayed", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Feature Order Preservation for UI
# -----------------------------
joblib.dump(X.columns.tolist(), "feature_order.pkl")

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


importances = model.feature_importances_
plt.barh(X.columns, importances)
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")