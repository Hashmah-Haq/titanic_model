import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to get a model prediction.")

model_path = Path(__file__).with_name("titanic_model.pkl")
if not model_path.exists():
    st.error("titanic_model.pkl not found. Run Create_Titanic_Model.ipynb and place the downloaded model beside app.py.")
    st.stop()
with open(model_path, "rb") as f:
    model = pickle.load(f)

pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["female", "male"])
sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])

if st.button("Predict survival"):
    passenger = pd.DataFrame([{"pclass": pclass, "sex": sex, "sibsp": sibsp, "embarked": embarked}])
    pred = int(model.predict(passenger)[0])
    prob = float(model.predict_proba(passenger)[0, 1])
    st.subheader("Model prediction: " + ("Survived" if pred == 1 else "Did not survive"))
    st.write(f"Estimated survival probability: {prob:.1%}")
    st.caption("Coursework demonstration only.")
