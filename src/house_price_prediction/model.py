import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from house_price_prediction.config import RANDOM_SEED


TARGET_COLUMN = "median_house_value"


def split_features_and_target(house_data):
    """Separate model input columns from the price column."""
    X = house_data.drop(columns=[TARGET_COLUMN])
    y = house_data[TARGET_COLUMN]
    return X, y


def train_linear_regression(X_train, y_train):
    """Train a simple baseline linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(y_test, predictions):
    """Calculate common regression metrics."""
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


def train_test_model(house_data, test_size=0.2, random_state=RANDOM_SEED):
    """Train and evaluate the baseline model from a cleaned dataset."""
    X, y = split_features_and_target(house_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = train_linear_regression(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_regression_model(y_test, predictions)

    return model, metrics, X.columns


def make_coefficients_table(feature_names, coefficients):
    """Create a readable table of model coefficients."""
    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    ).sort_values(by="Coefficient", ascending=False)


def predict_house_value(model, feature_names, input_values):
    """Predict one house value from a dictionary of feature values."""
    input_data = pd.DataFrame([input_values])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return model.predict(input_data)[0]
