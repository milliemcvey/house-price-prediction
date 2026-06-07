import pickle

import pandas as pd

from house_price_prediction.config import CLEANED_DATA_PATH, MODEL_PATH, MODELS_DIR
from house_price_prediction.model import make_coefficients_table, train_test_model


def main():
    house_data = pd.read_csv(CLEANED_DATA_PATH)
    model, metrics, feature_names = train_test_model(house_data)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    coefficients = make_coefficients_table(feature_names, model.coef_)

    print("Model metrics")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2:   {metrics['r2']:.3f}")
    print()
    print("Largest coefficients")
    print(coefficients.head())
    print()
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
