from house_price_prediction.config import CLEANED_DATA_PATH, DATA_DIR
from house_price_prediction.data import clean_house_data, load_raw_data


def main():
    raw_data = load_raw_data()
    cleaned_data = clean_house_data(raw_data)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Saved cleaned data to {CLEANED_DATA_PATH}")
    print(f"Rows: {cleaned_data.shape[0]}")
    print(f"Columns: {cleaned_data.shape[1]}")


if __name__ == "__main__":
    main()
