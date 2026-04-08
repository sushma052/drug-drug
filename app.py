import streamlit as st
import pandas as pd
import pickle

st.title("💊 Drug–Drug Interaction Predictor")

# Load saved model + encoder
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    return model, le

model, le = load_artifacts()

drug1 = st.text_input("Enter Drug 1 (e.g., DB00005)")
drug2 = st.text_input("Enter Drug 2 (e.g., DB00006)")

if st.button("Predict"):
    try:
        d1 = le.transform([drug1])[0]
        d2 = le.transform([drug2])[0]

        input_df = pd.DataFrame([[d1, d2]], columns=["Drug1", "Drug2"])
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error("⚠️ Interaction Exists")
        else:
            st.success("✅ No Interaction")
    except Exception as e:
        st.warning("Invalid Drug IDs or missing model files.")
