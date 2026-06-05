import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------
# Load CSS
# --------------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>",
                unsafe_allow_html=True)

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("data/Mall_Customers.csv")

# Load Model
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# --------------------------
# Header
# --------------------------
st.markdown(
    """
    <div class='main-header'>
    🛍️ Mall Customer Segmentation using K-Means Clustering
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================
# DATASET OVERVIEW
# ==========================

st.markdown("<div class='section'>📊 Dataset Overview</div>",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Average Age",
              round(df["Age"].mean(), 1))

with col3:
    st.metric("Average Income",
              round(df["Annual Income (k$)"].mean(), 1))

st.dataframe(df.head(10),
             use_container_width=True)

st.write("### Statistical Summary")
st.dataframe(df.describe(),
             use_container_width=True)

# ==========================
# VISUALIZATIONS
# ==========================

st.markdown("<div class='section'>📈 Visualizations</div>",
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        df["Age"],
        bins=15,
        kde=True,
        color="purple",
        ax=ax
    )

    ax.set_title("Age Distribution")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        df["Annual Income (k$)"],
        bins=15,
        kde=True,
        color="green",
        ax=ax
    )

    ax.set_title("Income Distribution")

    st.pyplot(fig)

# Spending Score

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="red",
    s=100,
    ax=ax
)

ax.set_title("Income vs Spending Score")

st.pyplot(fig)

# Correlation Heatmap

fig, ax = plt.subplots(figsize=(7,5))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)

# ==========================
# PREDICTION SECTION
# ==========================

st.markdown("<div class='section'>🎯 Predict Customer Segment</div>",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

with col2:
    income = st.number_input(
        "Annual Income (k$)",
        min_value=1,
        max_value=150,
        value=50
    )

with col3:
    spending = st.number_input(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

if st.button("Predict Segment"):

    sample = np.array([
        [age, income, spending]
    ])

    sample_scaled = scaler.transform(sample)

    cluster = model.predict(sample_scaled)[0]

    segment_names = {
        0: "💎 Premium Customers",
        1: "🛒 Regular Customers",
        2: "🔥 High Spenders",
        3: "💰 Wealthy Customers",
        4: "📉 Budget Customers"
    }

    st.success(
        f"Predicted Segment: {segment_names.get(cluster, f'Segment {cluster}')}"
    )

    st.balloons()

# Footer

st.markdown(
    """
    <div class='footer'>
    Developed using Streamlit, Scikit-Learn & K-Means Clustering
    </div>
    """,
    unsafe_allow_html=True
)