import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import r3

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")
st.title("🎓 Student Dropout Analysis Dashboard")

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
csv_path = r"C:\Users\nrosh\OneDrive\Documents\student dropout.csv"

try:
    df = pd.read_csv(csv_path)
    st.success("CSV Loaded Successfully!")
except:
    st.error(" Failed to load CSV. Check file path.")
    st.stop()

# Convert TRUE/FALSE → 1/0
df["Dropped_Out"] = df["Dropped_Out"].replace(
    {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
).astype(int)

# -------------------------------------------------------
# DATASET PREVIEW
# -------------------------------------------------------
st.subheader(" Dataset Preview")
st.dataframe(df)

# -------------------------------------------------------
# KEY STATISTICS
# -------------------------------------------------------
st.markdown("---")
st.header("📊 Key Statistics Overview")

total_students = len(df)
dropped = df["Dropped_Out"].sum()
retained = total_students - dropped
dropout_rate = (dropped / total_students) * 100

# Card Layout
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #f9d976, #f39f86);
    text-align: center;
    margin: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
.card h2 { font-size: 40px; margin: 0; font-weight: bold; }
.card p { font-size: 18px; margin: 0; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><h2>{total_students}</h2><p>Total Students</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><h2>{dropout_rate:.1f}%</h2><p>Dropout Rate</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><h2>{dropped}</h2><p>Dropped</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><h2>{retained}</h2><p>Retained</p></div>', unsafe_allow_html=True)

# -------------------------------------------------------
# VISUAL ANALYSIS
# -------------------------------------------------------
st.markdown("---")
st.header("📊 Visual Analysis")

# Basic graphs
col1, col2 = st.columns(2)
with col1:
    st.subheader("Gender-Based Dropout Rates")
    st.pyplot(r3.gender_dropout_graph(df))

with col2:
    st.subheader("Dropout Pie Chart")
    st.pyplot(r3.dropout_pie_chart(df))

col3, col4 = st.columns(2)
with col3:
    st.subheader("Age Demographics")
    st.pyplot(r3.age_distribution(df))

with col4:
    st.subheader("Risk Factors Analysis")
    st.pyplot(r3.risk_factor_donut())

# -------------------------------------------------------
# ADDITIONAL GRAPHS
# -------------------------------------------------------
st.markdown("## 📈 Additional Analysis")

colA, colB = st.columns(2)
with colA:
    st.subheader(" Failures vs Dropout")
    st.pyplot(r3.failures_vs_dropout(df))

with colB:
    st.subheader(" Parental Education vs Dropout")
    st.pyplot(r3.parental_education_vs_dropout(df))

st.subheader(" Family Support vs Final Grade")
st.pyplot(r3.family_support_vs_grades(df))

# -------------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------------
st.markdown("---")
st.header("🤖 Machine Learning: Model Comparison")

df_ml = df.copy()

# Remove leakage columns
df_ml.drop(columns=["Final_Grade", "Number_of_Absences"], inplace=True)

numeric_df = df_ml.select_dtypes(include=["int64", "float64"])
numeric_df["Dropped_Out"] = df_ml["Dropped_Out"]

X = numeric_df.drop("Dropped_Out", axis=1)
y = numeric_df["Dropped_Out"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scores = {}

# Logistic Regression
log = LogisticRegression(max_iter=2000)
log.fit(X_train, y_train)
log_prob = log.predict_proba(X_test)[:, 1]
scores["Logistic Regression"] = roc_auc_score(y_test, log_prob)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)
dt_prob = dt.predict_proba(X_test)[:, 1]
scores["Decision Tree"] = roc_auc_score(y_test, dt_prob)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=12)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
scores["Random Forest"] = roc_auc_score(y_test, rf_prob)

st.subheader("📈 AUC – ROC Scores")
for model, score in scores.items():
    st.write(f"**{model}: {score:.4f}**")

best_model = max(scores, key=scores.get)
st.success(f"🏆 Best Model: {best_model} (AUC = {scores[best_model]:.4f})")

# -------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------
st.subheader("📉 ROC Curve Comparison")

fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr_log, tpr_log, label="Logistic Regression")
ax.plot(fpr_dt, tpr_dt, label="Decision Tree")
ax.plot(fpr_rf, tpr_rf, label="Random Forest")
ax.plot([0, 1], [0, 1], "k--")
ax.set_title("ROC Curve Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)
