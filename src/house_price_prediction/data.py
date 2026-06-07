import pandas as pd

from house_price_prediction.config import DATA_URL


SELECTED_COLUMNS = [
    "total_rooms",
    "total_bedrooms",
    "housing_median_age",
    "households",
    "median_income",
    "ocean_proximity",
    "median_house_value",
]


def load_raw_data(data_url=DATA_URL):
    """Load the original housing dataset from a public CSV URL."""
    return pd.read_csv(data_url)


def clean_house_data(housing_data):
    """Return a cleaned dataset that is ready for linear regression."""
    house_data = housing_data[SELECTED_COLUMNS].copy()

    # Fill missing bedroom values with the median from the same column.
    median_bedrooms = house_data["total_bedrooms"].median()
    house_data["total_bedrooms"] = house_data["total_bedrooms"].fillna(median_bedrooms)

    house_data = house_data.drop_duplicates()

    # Convert the text location column into numeric 0/1 columns.
    house_data = pd.get_dummies(house_data, columns=["ocean_proximity"], drop_first=True)

    return house_data
