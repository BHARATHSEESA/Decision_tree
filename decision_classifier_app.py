import streamlit as st
import pickle
import numpy as np

# title
st.title("Wine Quality Prediction")

st.write("Decision Tree Classifier App")

try:
    # load model
    model = pickle.load(open("wine_model.pkl", "rb"))

    # load label encoder
    le = pickle.load(open("label_encoder.pkl", "rb"))

    st.success("Model Loaded Successfully")

    # user inputs
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        step=1.0
    )

    residual_sugar = st.number_input(
        "Residual Sugar",
        min_value=0.0,
        step=1.0
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        step=1.0
    )

    density = st.number_input(
        "Density",
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )

    # predict button
    if st.button("Predict"):

        features = np.array([
            [
                fixed_acidity,
                residual_sugar,
                alcohol,
                density
            ]
        ])

        prediction = model.predict(features)

        result = le.inverse_transform(prediction)

        st.success(f"Predicted Wine Quality: {result[0]}")

except Exception as e:
    st.error(f"Error: {e}")
