import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------------
# 1. Gender-based dropout graph
# -------------------------------------------------------
def gender_dropout_graph(df):
    gender_drop = df.groupby("Gender")["Dropped_Out"].mean() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(gender_drop.index, gender_drop.values, color=["#6A5ACD", "#8A2BE2"])
    ax.set_title("Dropout Rates by Gender", fontsize=14)
    ax.set_ylabel("Dropout Rate (%)")
    ax.set_xlabel("Gender")
    return fig

# -------------------------------------------------------
# 2. Dropout Pie Chart
# -------------------------------------------------------
def dropout_pie_chart(df):
    labels = ["Retained", "Dropped Out"]
    values = [
        (df["Dropped_Out"] == 0).sum(),
        (df["Dropped_Out"] == 1).sum()
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#E53935"],
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    ax.set_title("Overall Dropout Distribution")
    return fig

# -------------------------------------------------------
# 3. Age Distribution
# -------------------------------------------------------
def age_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Age"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#6495ED")

    ax.set_title("Student Age Distribution", fontsize=14)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Students")
    return fig

# -------------------------------------------------------
# 4. Risk Factor Donut Chart
# -------------------------------------------------------
def risk_factor_donut():
    labels = ["Academic Issues", "Financial Issues", "Family Issues", "Health Issues"]
    values = [40, 30, 20, 10]
    colors = ["#5A67D8", "#805AD5", "#ECC94B", "#F6AD55"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
           wedgeprops={"linewidth": 1, "edgecolor": "white"})
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.set_title("Key Contributing Factors to Dropout")
    return fig

# -------------------------------------------------------
# NEW GRAPH 1 — Failures vs Dropout
# -------------------------------------------------------
def failures_vs_dropout(df):
    failure_drop = df.groupby("Number_of_Failures")["Dropped_Out"].mean() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(failure_drop.index, failure_drop.values, color="#E67373")
    ax.set_title("Dropout Rate vs Number of Failures", fontsize=14)
    ax.set_xlabel("Number of Failures")
    ax.set_ylabel("Dropout Rate (%)")
    return fig

# -------------------------------------------------------
# NEW GRAPH 4 — Parental Education vs Dropout
# -------------------------------------------------------
def parental_education_vs_dropout(df):
    df_copy = df.copy()
    df_copy["Avg_Parent_Edu"] = (df_copy["Mother_Education"] + df_copy["Father_Education"]) / 2
    edu_drop = df_copy.groupby("Avg_Parent_Edu")["Dropped_Out"].mean() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(edu_drop.index, edu_drop.values, marker="o", color="#8A2BE2")
    ax.set_title("Parental Education Level vs Dropout Rate", fontsize=14)
    ax.set_xlabel("Avg Parent Education Level")
    ax.set_ylabel("Dropout Rate (%)")
    return fig

# -------------------------------------------------------
# NEW GRAPH 7 — Family Support vs Grades (Reduced Size)
# -------------------------------------------------------
def family_support_vs_grades(df):
    grade_avg = df.groupby("Family_Support")["Final_Grade"].mean()

    fig, ax = plt.subplots(figsize=(4, 3))  # ⬅ Smaller graph
    ax.bar(grade_avg.index, grade_avg.values, color="#4CAF50")
    ax.set_title("Family Support vs Final Grade", fontsize=12)
    ax.set_xlabel("Family Support (yes/no)", fontsize=10)
    ax.set_ylabel("Avg Final Grade", fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    return fig
