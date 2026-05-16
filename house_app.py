import streamlit as st
import pickle
import numpy as np

# title
st.title("House Price Prediction")

st.write("Decision Tree Regressor App")

try:

    # load model
    model = pickle.load(open("house_price_model.pkl", "rb"))

    st.success("Model Loaded Successfully")

    # inputs
    bedrooms = st.number_input("Bedrooms", min_value=1)

    bathrooms = st.number_input("Bathrooms", min_value=1)

    sqft_living = st.number_input("Sqft Living", min_value=500)

    sqft_lot = st.number_input("Sqft Lot", min_value=500)

    floors = st.number_input("Floors", min_value=1)

    waterfront = st.number_input("Waterfront", min_value=0, max_value=1)

    view = st.number_input("View", min_value=0)

    condition = st.number_input("Condition", min_value=1)

    sqft_above = st.number_input("Sqft Above", min_value=500)

    sqft_basement = st.number_input("Sqft Basement", min_value=0)

    yr_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025
    )

    # prediction
    if st.button("Predict Price"):

        features = np.array([
            [
                bedrooms,
                bathrooms,
                sqft_living,
                sqft_lot,
                floors,
                waterfront,
                view,
                condition,
                sqft_above,
                sqft_basement,
                yr_built
            ]
        ])

        prediction = model.predict(features)

        st.success(
            f"Predicted House Price: ${prediction[0]:,.2f}"
        )

except Exception as e:

    st.error(f"Error: {e}")