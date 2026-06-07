# House Price Prediction with Linear Regression

## Project overview

This beginner-friendly project builds a linear regression model to predict house prices from a small set of understandable housing features.

The project is split into three notebooks:

1. Explore the dataset.
2. Clean and prepare the data.
3. Train and evaluate a linear regression model.

The goal is not to create a perfect price prediction system. The goal is to practice the basic machine learning workflow using simple Python code and clear explanations.

## Learning objectives

- Load a public dataset with pandas.
- Explore rows, columns, missing values, and summary statistics.
- Create simple visualisations with matplotlib.
- Select useful features for a regression problem.
- Handle missing values and categorical data.
- Train a baseline linear regression model with scikit-learn.
- Evaluate predictions using MAE, RMSE, and R-squared.
- Interpret model coefficients in a beginner-friendly way.

## Dataset source

This project uses the California housing dataset CSV from the Hands-On Machine Learning repository:

https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv

The dataset can be loaded directly with pandas and does not require an API key.

The original dataset does not include exact square footage, bedroom count, or bathroom count. To keep the project beginner friendly, this project uses similar understandable features:

- `total_rooms` as a simple house-size related feature.
- `total_bedrooms` as a bedroom-related feature.
- `housing_median_age` as house age.
- `ocean_proximity` as a location category.
- `median_house_value` as the target price.

## Folder structure

```text
house-price-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_training.ipynb
└── .gitignore
```

After running the cleaning notebook, this file will also be created:

```text
data/cleaned_house_data.csv
```

## Setup instructions

From the project folder, install the required packages:

```bash
pip install -r requirements.txt
```

Then open Jupyter:

```bash
jupyter notebook
```

Open the notebooks in this order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_model_training.ipynb`

The third notebook depends on the cleaned CSV created by the second notebook.

## Model and metrics

The model is a baseline linear regression model. Linear regression tries to learn a straight-line relationship between the input features and the target price.

The model is evaluated with:

- Mean Absolute Error: the average size of the prediction error.
- Root Mean Squared Error: similar to MAE, but larger errors are penalized more.
- R-squared: shows how much of the variation in house prices is explained by the model.

These metrics should be interpreted after running the notebook. A simple linear model is useful as a first baseline, but it may not capture all patterns in real housing prices.

## Future improvements

- Add more useful features, such as rooms per household or bedrooms per room.
- Compare linear regression with another simple model after the baseline is complete.
- Improve the visualisations and write a short final report based on the results.
