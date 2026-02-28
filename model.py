# data handeling
import pandas as pd
import numpy as np

# data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# model
from sklearn.linear_model import LogisticRegression

# evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# save model    
import joblib


# loading data
data = pd.read_csv('shipment_data_sample2.csv')

# data preprocessing
data.dropna(inplace=True)  # remove missing values

# Convert Date Columns
data['Order_Date'] = pd.to_datetime(data['Order_Date'])
data['Dispatch_Date'] = pd.to_datetime(data['Dispatch_Date'])
data['Estimated_Delivery_Date'] = pd.to_datetime(data['Estimated_Delivery_Date'])

# Create new numeric features
data['Processing_Days'] = (data['Dispatch_Date'] - data['Order_Date']).dt.days
data['Estimated_Delivery_Days'] = (data['Estimated_Delivery_Date'] - data['Dispatch_Date']).dt.days

# Drop original date columns
data.drop(['Order_Date', 'Dispatch_Date',
           'Estimated_Delivery_Date'], axis=1, inplace=True)

# Drop identifier columns
data.drop(['Shipment_ID'], axis=1, inplace=True)

# Encode Categorical Variables
data = pd.get_dummies(data, drop_first=True)

# Split Features & Target
X = data.drop('Is_Delayed', axis=1)
y = data['Is_Delayed']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create & Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# 3. Calculate AUC-ROC Score
auc_score = roc_auc_score(y_test, y_pred_proba)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:\n", accuracy_score(y_test, y_pred))
print("AUC-ROC Score:\n", auc_score)

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='Is_Delayed', data=data)
plt.title('Distribution of Delayed vs On-Time Shipments')
plt.xlabel('Shipment Delay Status')
plt.ylabel('Count')
plt.show()

# Save Model
# joblib.dump(model, 'shipment_delay_model.pkl')