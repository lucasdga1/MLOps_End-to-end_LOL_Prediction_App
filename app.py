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
st.image(
    'league-of-legends7103.jpg', width='150'
)

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
            st.subheader("Results:")
            results = response.json()["results"]
            for idx, r in enumerate(results, start=1):
                st.write(f"**Match {idx}:**")
                if r["predicted_winner"] == 1:
                    st.write("**:blue[Blue team predicted to win!]**")
                else:
                    st.write("**:red[Red team predicted to win!]**")

                if "actual_winner" in r:
                    if r["actual_winner"] == 1:
                        st.write("Actual: :blue[Blue team won]")
                    else:
                        st.write("Actual: :red[Red team won]")
                st.markdown("---")

