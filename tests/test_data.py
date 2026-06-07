import pandas as pd

from house_price_prediction.data import clean_house_data


def test_clean_house_data_fills_missing_values_and_encodes_category():
    raw_data = pd.DataFrame(
        {
            "total_rooms": [1000, 1500, 1500],
            "total_bedrooms": [200, None, None],
            "housing_median_age": [20, 10, 10],
            "households": [300, 450, 450],
            "median_income": [4.5, 6.0, 6.0],
            "ocean_proximity": ["INLAND", "NEAR OCEAN", "NEAR OCEAN"],
            "median_house_value": [120000, 200000, 200000],
        }
    )

    cleaned_data = clean_house_data(raw_data)

    assert cleaned_data.isna().sum().sum() == 0
    assert "ocean_proximity" not in cleaned_data.columns
    assert "ocean_proximity_NEAR OCEAN" in cleaned_data.columns
    assert len(cleaned_data) == 2
