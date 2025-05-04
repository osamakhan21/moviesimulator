# CineScope – AI Box Office Predictor

import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

# -----------------------------
# ----- Utility Functions -----
# -----------------------------

MODEL_DIR = Path("models")

def load_models():
    try:
        revenue_model = joblib.load(MODEL_DIR / "gbm_revenue.pkl")
        rating_model = joblib.load(MODEL_DIR / "ordinal_rating.pkl")
        feature_columns = joblib.load(MODEL_DIR / "feature_cols.pkl")
        return revenue_model, rating_model, feature_columns
    except FileNotFoundError:
        return None, None, None

def engineer_features(selections: dict, feature_columns):
    if feature_columns is None:
        return pd.DataFrame({"dummy": [0]})
    X = pd.DataFrame([0]*len(feature_columns), index=feature_columns).T
    for actor in selections["cast"]:
        col = f"actor_{actor}"
        if col in X.columns:
            X[col] = 1
    for g in selections["genres"]:
        col = f"genre_{g}"
        if col in X.columns:
            X[col] = 1
    m = selections["month"]
    X["rel_month_sin"] = np.sin(2*np.pi*m/12)
    X["rel_month_cos"] = np.cos(2*np.pi*m/12)
    return X

def predict(selections):
    revenue_model, rating_model, feature_cols = load_models()
    X = engineer_features(selections, feature_cols)

    if revenue_model is None or rating_model is None:
        st.warning("Models not found – using random demo predictions.")
        gross_pred = np.random.uniform(10e6, 500e6)
        rating_probs = np.random.dirichlet(np.ones(10))
        return gross_pred, rating_probs

    gross_pred = revenue_model.predict(X)[0]
    rating_probs = rating_model.predict_proba(X)[0]
    return gross_pred, rating_probs

# -----------------------------
# --------- Streamlit ---------
# -----------------------------

st.set_page_config(page_title="CineScope – AI Movie Simulator", layout="wide")

theme = st.sidebar.radio("Choose mode", ["Light", "Dark"])

# Inject background styling
if theme == "Dark":
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #0a0a0a !important;
        color: #f5f5f5 !important;
        transition: all 0.4s ease-in-out;
    }
    .stApp {
        background-color: #0a0a0a;
        transition: all 0.4s ease-in-out;
    }
    .css-1d391kg, .css-1v3fvcr, .css-18ni7ap.e8zbici2 {
        background-color: #1a1a1a !important;
        transition: all 0.4s ease-in-out;
    }
    .stButton>button {
        background-color: #ffaa00;
        color: black;
        border-radius: 12px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    h1, h2, h3 {
        color: #ffd700;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #FFFFFFF !important;
        color: #222 !important;
        transition: all 0.4s ease-in-out;
    }
    .stApp {
        background-color: #FFFFFF;
        transition: all 0.4s ease-in-out;
    }
    .css-1d391kg, .css-1v3fvcr, .css-18ni7ap.e8zbici2 {
        background-color: #fff0e0 !important;
        transition: all 0.4s ease-in-out;
    }
    .stButton>button {
        background-color: #ff471a;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    h1, h2, h3 {
        color: #cc3300;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Logo (Main Page)
col_logo, col_title = st.columns([1, 6])
with col_logo:
    try:
        logo = Image.open("Blue_Flat_Illustrative_Human_Artificial_Intelligence_Technology_Logo-removebg-preview.png")
        st.image(logo, use_container_width=True)
    except:
        st.write("")
with col_title:
    st.title("CineScope – AI Movie Simulator")
    st.markdown("<h4>Pick a hypothetical movie recipe and see how it might perform!</h4>", unsafe_allow_html=True)

# Sample data for dropdowns
actors = ["Tom Hanks", "Scarlett Johansson", "Dwayne Johnson", "Zendaya", "Leonardo DiCaprio"]
genres = ["Action", "Comedy", "Drama", "Horror", "Sci‑Fi"]

with st.sidebar:
    st.header("Select Movie Recipe")
    sel_cast = st.multiselect("Celebirities Names", actors)
    sel_genres = st.multiselect("Genres", genres)
    sel_month = st.slider("Release Month", 1, 12, 7)
    if st.button("Simulate!"):
        selections = {"cast": sel_cast, "genres": sel_genres, "month": sel_month}
        gross, probs = predict(selections)
        st.session_state["latest"] = {"gross": gross, "probs": probs}

if "latest" in st.session_state:
    res = st.session_state["latest"]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Projected Worldwide Gross")
        st.metric("USD (Millions)", f"{res['gross'] / 1e6:,.1f}M")
    with col2:
        st.subheader("IMDb‑Style Rating Distribution")
        df = pd.DataFrame({"rating": range(1, 11), "prob": res["probs"]})
        chart = alt.Chart(df).mark_bar().encode(x="rating:O", y="prob:Q")
        st.altair_chart(chart, use_container_width=True)
