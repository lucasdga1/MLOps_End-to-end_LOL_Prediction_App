import streamlit as st
import pandas as pd
import requests

# ===================================
# Config
# ===================================


# ===================================
# UI
# ===================================
st.title("LOL match prediction model 🎮")
st.markdown(
    "<div style='text-align: center;'><img src='.streamlit/league-of-legends7103.jpg' width='150'></div>",
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
        response = requests.post("https://mlops-end-to-end-lol-prediction-app.onrender.com/predict", json=data)
        if response.status_code != 200:
            st.error(f"Erro na API: {response.status_code}")
            st.write(response.text)
        else:
            response_json = response.json()
            results = response_json.get("results")
            if results is None:
                st.error("Resposta da API não contém 'results'.")
                st.write(response_json)
            else:
                for r in results:
                    if r["predicted_winner"] == 1:
                        st.write("**:blue[Blue team won!]**")
                    else:
                        st.write("**:red[Red team won!]**")

