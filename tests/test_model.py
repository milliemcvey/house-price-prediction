import pandas as pd

from house_price_prediction.model import split_features_and_target, train_test_model


def test_split_features_and_target_separates_price_column():
    house_data = pd.DataFrame(
        {
            "total_rooms": [1000, 1500],
            "median_income": [4.5, 6.0],
            "median_house_value": [120000, 200000],
        }
    )

    X, y = split_features_and_target(house_data)

    assert "median_house_value" not in X.columns
    assert y.tolist() == [120000, 200000]


def test_train_test_model_returns_metrics():
    house_data = pd.DataFrame(
        {
            "total_rooms": [800, 1000, 1200, 1400, 1600, 1800],
            "total_bedrooms": [150, 200, 220, 260, 300, 340],
            "housing_median_age": [35, 30, 25, 20, 15, 10],
            "households": [250, 300, 340, 380, 420, 460],
            "median_income": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "ocean_proximity_NEAR OCEAN": [False, False, False, True, True, True],
            "median_house_value": [100000, 120000, 150000, 180000, 210000, 240000],
        }
    )

    model, metrics, feature_names = train_test_model(house_data, test_size=0.33)

    assert model is not None
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert len(feature_names) == len(model.coef_)
