import streamlit as st
import pandas as pd
import requests
import os
from pathlib import Path
import json

# ===================================
# Config
# ===================================

"""
Trocar pela url do Railway (requests.post("https://.../predict", json=data)
"""

# Para converter csv em json e fazer a predict

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data = df.to_dict(orient="records")  # lista de dicts
    response = requests.post("https://lol-prediction-prodution.up.railway.app/predict", json=data)
    st.write(response.json())

# ===================================
# UI
# ===================================
st.title("LOL match prediction model 🎮")
st.markdown(
    "<div style='text-align: center;'><img src='Graficos_projeto/league-of-legends7103.jpg' width='150'></div>",
    unsafe_allow_html=True
)
"""
Para converter csv em json e fazer a predict
"""
st.header("Upload the match data as csv")
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data = df.to_dict(orient="records")  # lista de dicts
    if st.button("Show prediction🚀"):
        st.write("📅 Running prediction for the match")
        response = requests.post("http://localhost:8000/predict", json=data)
        results = response.json()["results"]
        for r in results:
            if r["predicted_winner"] == 1:
                st.write("**:blue[Blue team won!]**")
            else:
                st.write("**:red[Red team won!]**")

