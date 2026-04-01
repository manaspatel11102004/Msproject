# India Climate Forecast Dashboard

This project analyzes recent Indian climate data and predicts future average temperature in a Streamlit app.

## Project idea

The app uses a larger recent dataset, trains a forecasting model, and shows trend charts, AQI visuals, rainfall patterns, and future temperature predictions.

## Kaggle dataset

- Dataset: `Indian Climate Dataset (2024-2025)`
- Kaggle link: <https://www.kaggle.com/datasets/ankushnarwade/indian-climate-dataset-20242025>
- File used: `Indian_Climate_Dataset_2024_2025.csv`
- Coverage: January 2024 to December 2025
- Size: 7,310+ rows
- Columns: 13

This dataset is recent, larger than the earlier one, and supports forecasting, EDA, and city-level climate analysis.

## Files

- `app.py` - Streamlit application
- `src/data_utils.py` - dataset schema, validation, and download helper
- `src/train_model.py` - forecasting pipeline, evaluation, and artifact saving
- `data/Indian_Climate_Dataset_2024_2025.csv` - Kaggle dataset file
- `artifacts/` - trained model, metrics, forecast base data, and predictions

## Local run

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Make sure the dataset file exists at:

```text
data/Indian_Climate_Dataset_2024_2025.csv
```

3. Train the model and generate artifacts:

```powershell
python -m src.train_model
```

4. Launch the app:

```powershell
streamlit run app.py
```

## Model details

- Algorithm: `RandomForestRegressor`
- Features:
  - city and state
  - month and day-of-year
  - recent temperature lags
  - rolling temperature trends
  - humidity, rainfall, AQI, pressure, and cloud cover lags
- Target:
  - next average temperature value

## Deploy on Streamlit Community Cloud

1. Upload this project to a GitHub repository.
2. Include `app.py`, `requirements.txt`, the `src` folder, and `data/Indian_Climate_Dataset_2024_2025.csv`.
3. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
4. Click `New app`, choose your repository, and set the main file path to `app.py`.
5. Deploy the app.
