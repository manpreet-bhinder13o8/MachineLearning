from catboost import CatBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
import joblib

df = pd.read_excel('Telco_customer_churn.xlsx')

print("\nFirst few rows:")
print(df.head())

df.info()

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Separated numeric and categorical columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

if "CustomerID" in categorical_cols:
    categorical_cols.remove("CustomerID")


# filled numeric missing values with mean
for col in numeric_cols:
    if df[col].isnull().any():
       df[col] = df[col].fillna(df[col].mean())

# filled categorical missing values with mode
for col in categorical_cols:
    if df[col].isnull().any():
         df[col] = df[col].fillna(df[col].mode()[0])


print("\nMissing values after handling:\n", df.isnull().sum()[df.isnull().sum() > 0])

# df["Churn Reason"] = df["Churn Reason"].fillna(df["Churn Reason"].mode()[0])

# To remove duplicates
initial_len = len(df)
df = df.drop_duplicates()
print(f"\nRemoved {initial_len - len(df)} duplicate rows.")




# Basic feature engineering

def tenure_group(months):
    if months <= 12:
        return "New"
    elif months <= 36:
        return "Medium"
    else:
        return "Long"

df["TenureGroup"] = df["Tenure Months"].apply(tenure_group)

# b) Average monthly usage ((Tenure + 1) to avoid division by zero)
df["AvgUsage"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)

# c) Payment risk based on Monthly Charges
def payment_risk(charge):
    if charge < 30:
        return "Low"
    elif charge < 70:
        return "Medium"
    else:
        return "High"

df["PaymentRisk"] = df["Monthly Charges"].apply(payment_risk)

print("\nNew features created: TenureGroup, AvgUsage, PaymentRisk")


# EDA and generate insights

# style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

#Churn distribution-------

plt.figure()
sns.countplot(x="Churn Value", data=df, palette="Set2")
plt.title("Churn Distribution (0 = No, 1 = Yes)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()
churn_rate = df["Churn Value"].mean() * 100
print(f"\nInsight 1: Overall churn rate = {churn_rate:.2f}%")

# Correlation heatmap ---
# Made a copy for encoding (for safe side)
df_encoded = df.copy()
# Encoding object columns
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap (after encoding)")
plt.show()
# Top correlations with churn (because i could not understand what the map was showing) :))
churn_corr = df_encoded.corr()["Churn Value"].abs().sort_values(ascending=False)
print("\nInsight 2: Top 20 features correlated with churn:")
print(churn_corr.head(20))  


# Feature importance using Random Forest (on encoded data) ---
X = df_encoded.drop("Churn Value", axis=1)
y = df_encoded["Churn Value"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure()
importances.head(10).plot(kind="barh", color="teal")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()
print("\nInsight 3: Top 11 most important features for predicting churn:")
print(importances.head(11))


# High risk vs low risk segmentation
high_risk = df[df["Monthly Charges"] > 70]

low_risk = df[df["Monthly Charges"] <= 70]

print("High Risk Customers:", len(high_risk))
print("Low Risk Customers:", len(low_risk))


# LLM models 
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["Churn Value", "Churn Label", "Churn Score", "CLTV", "Churn Reason"], axis=1, errors='ignore')
y = df["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:,1]

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:,1]

cat = CatBoostClassifier(
    verbose=0,
    random_state=42
)

cat.fit(X_train, y_train)

cat_pred = cat.predict(X_test)
cat_prob = cat.predict_proba(X_test)[:,1]

def evaluate_model(name, y_true, pred, prob):

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", accuracy_score(y_true, pred))
    print("Precision:", precision_score(y_true, pred))
    print("Recall   :", recall_score(y_true, pred))
    print("F1 Score :", f1_score(y_true, pred))
    print("ROC AUC  :", roc_auc_score(y_true, prob))

evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred,
    lr_prob
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)

evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred,
    xgb_prob
)

evaluate_model(
    "CatBoost",
    y_test,
    cat_pred,
    cat_prob
)

results = pd.DataFrame({
    "Actual_Churn": y_test,
    "Churn_Risk_Percentage": cat_prob * 100
})

print("\nCustomer Churn Probability")
print(results.head(20))

df.info()
# df.to_excel('Telco_customer_churn_updated.xlsx', index=False)


# Save the best model (I'll use CatBoost, but you can change it)
joblib.dump(cat, "churn_model.pkl")          # your trained model

# Save the scaler (used to scale features)
joblib.dump(scaler, "scaler.pkl")

# Save the list of feature column names (the ones used for training)
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Save the entire processed dataset (after cleaning and feature engineering)
df.to_csv("processed_data.csv", index=False)