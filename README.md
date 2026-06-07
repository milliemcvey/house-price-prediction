# House Price Prediction with Linear Regression

## Project overview

This beginner-friendly project builds a linear regression model to predict house prices from a small set of understandable housing features.

The main learning path is notebook based:

1. Explore the dataset.
2. Clean and prepare the data.
3. Train and evaluate a linear regression model.

The repository also includes a small `src/` package, command-line scripts, tests, and a GitHub Actions workflow. These additions keep the project beginner friendly while making the structure closer to a real-world Python machine learning repository.

## Learning objectives

- Load a public dataset with pandas.
- Explore rows, columns, missing values, and summary statistics.
- Create simple visualisations with matplotlib.
- Select useful features for a regression problem.
- Handle missing values and categorical data.
- Train a baseline linear regression model with scikit-learn.
- Evaluate predictions using MAE, RMSE, and R-squared.
- Organise a Python project with notebooks, reusable code, scripts, and tests.

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
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_training.ipynb
├── reports/
│   ├── README.md
│   └── figures/
├── scripts/
│   ├── prepare_data.py
│   └── train_model.py
├── src/
│   └── house_price_prediction/
│       ├── config.py
│       ├── data.py
│       └── model.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .editorconfig
├── .gitignore
├── LICENSE
├── PROJECT_CHECKLIST.md
├── pyproject.toml
├── README.md
└── requirements.txt
```

Generated files are not committed:

- `data/cleaned_house_data.csv`
- `models/linear_regression_model.pkl`
- files inside `reports/figures/`

## Setup instructions

From the project folder, create and activate a virtual environment if you want to keep dependencies separate:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` command installs the local `src/` package in editable mode. That means the scripts and tests can import the project helper functions.

## How to use the notebooks

Open Jupyter:

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_model_training.ipynb`

The third notebook depends on the cleaned CSV created by the second notebook.

## Optional script workflow

The notebooks are best for learning. The scripts are included to show how the same steps can be repeated from the command line.

Prepare the cleaned dataset:

```bash
python scripts/prepare_data.py
```

Train the baseline model:

```bash
python scripts/train_model.py
```

Run the tests:

```bash
pytest
```

## Model and metrics

The model is a baseline linear regression model. Linear regression tries to learn a straight-line relationship between the input features and the target price.

The model is evaluated with:

- Mean Absolute Error: the average size of the prediction error.
- Root Mean Squared Error: similar to MAE, but larger errors are penalized more.
- R-squared: shows how much of the variation in house prices is explained by the model.

These metrics should be interpreted after running the notebook. A simple linear model is useful as a first baseline, but it may not capture all patterns in real housing prices.

## What makes this portfolio ready

- Clear README and setup instructions.
- Notebook workflow for learning and explanation.
- Reusable `src/` code for data cleaning and modelling.
- Scripts for repeatable command-line runs.
- Basic tests for important cleaning and modelling functions.
- GitHub Actions workflow to run tests automatically.
- Generated data and model files excluded from git.

## Future improvements

- Add more useful features, such as rooms per household or bedrooms per room.
- Compare linear regression with another simple model after the baseline is complete.
- Add a short report in `reports/` that explains the final results in your own words.
