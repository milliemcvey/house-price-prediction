import numpy as np
import pandas as pd
import streamlit as st

from house_price_prediction.data import clean_house_data, load_raw_data
from house_price_prediction.model import (
    make_coefficients_table,
    predict_house_value,
    train_test_model,
)


st.set_page_config(
    page_title="House Price Prediction",
    layout="wide",
)


@st.cache_data
def load_cleaned_data():
    raw_data = load_raw_data()
    return clean_house_data(raw_data)


@st.cache_resource
def train_model(cleaned_data):
    return train_test_model(cleaned_data)


cleaned_data = load_cleaned_data()
model, metrics, feature_names = train_model(cleaned_data)
coefficients = make_coefficients_table(feature_names, model.coef_)

st.title("House Price Prediction")
st.caption("A beginner-friendly linear regression demo using the California housing dataset.")

st.sidebar.header("House details")

total_rooms = st.sidebar.slider(
    "Total rooms",
    min_value=100,
    max_value=10000,
    value=2500,
    step=100,
)

total_bedrooms = st.sidebar.slider(
    "Total bedrooms",
    min_value=20,
    max_value=3000,
    value=500,
    step=20,
)

housing_median_age = st.sidebar.slider(
    "House age",
    min_value=1,
    max_value=52,
    value=25,
)

households = st.sidebar.slider(
    "Households",
    min_value=20,
    max_value=3000,
    value=500,
    step=20,
)

median_income = st.sidebar.slider(
    "Median income",
    min_value=0.5,
    max_value=15.0,
    value=4.5,
    step=0.1,
)

location = st.sidebar.selectbox(
    "Location category",
    ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
)

input_values = {
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "housing_median_age": housing_median_age,
    "households": households,
    "median_income": median_income,
    "ocean_proximity_INLAND": location == "INLAND",
    "ocean_proximity_ISLAND": location == "ISLAND",
    "ocean_proximity_NEAR BAY": location == "NEAR BAY",
    "ocean_proximity_NEAR OCEAN": location == "NEAR OCEAN",
}

prediction = predict_house_value(model, feature_names, input_values)

metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
metric_column_1.metric("Predicted value", f"${prediction:,.0f}")
metric_column_2.metric("Model MAE", f"${metrics['mae']:,.0f}")
metric_column_3.metric("R-squared", f"{metrics['r2']:.3f}")

st.info(
    "This is a simple baseline model. Treat the prediction as a learning example, "
    "not as a real property valuation."
)

chart_column, table_column = st.columns([1.4, 1])

with chart_column:
    st.subheader("Price distribution")
    price_counts, price_bins = np.histogram(cleaned_data["median_house_value"], bins=30)
    price_distribution = pd.DataFrame(
        {
            "price": price_bins[:-1],
            "district_count": price_counts,
        }
    )
    st.bar_chart(price_distribution, x="price", y="district_count")

    st.subheader("Actual data: rooms vs price")
    scatter_data = cleaned_data.sample(1000, random_state=42)[
        ["total_rooms", "median_house_value"]
    ]
    st.scatter_chart(
        scatter_data,
        x="total_rooms",
        y="median_house_value",
    )

with table_column:
    st.subheader("Model coefficients")
    st.dataframe(
        coefficients,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Current input")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [
                    "Total rooms",
                    "Total bedrooms",
                    "House age",
                    "Households",
                    "Median income",
                    "Location",
                ],
                "Value": [
                    total_rooms,
                    total_bedrooms,
                    housing_median_age,
                    households,
                    median_income,
                    location,
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

st.subheader("How to read this")
st.write(
    "Use the controls in the sidebar to change the input values. The model updates "
    "the predicted value using the same linear regression approach from the training notebook."
)
