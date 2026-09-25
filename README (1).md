# 🚗 Used Car Price Predictor

A Streamlit web app that predicts the price of a used car using a Linear
Regression model trained on used-car sales data (Brand, Body type, Mileage,
Engine volume, Engine type, Registration, and Year).

The model was trained in `Linear_Ridge___Lasso_Regression.ipynb`, comparing
Linear, Ridge, and Lasso Regression. Linear Regression performed best and was
saved as `linear_regression_model.pkl`, which this app loads to serve
predictions.

## Project structure

```
.
├── app.py                        # Streamlit app
├── requirements.txt              # Python dependencies
├── linear_regression_model.pkl   # Trained model (from the notebook)
└── README.md
```

## How it works

1. The user enters car details (brand, body type, mileage, engine volume,
   engine type, registration status, year) in the app's form.
2. The app label-encodes the categorical fields using the same mappings the
   model was trained with (sklearn's `LabelEncoder`, alphabetical order).
3. The model predicts `log(price)`, which the app converts back with
   `np.exp()` to show the estimated price in dollars.

## Running locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploying on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `linear_regression_model.pkl` to a
   GitHub repository (all three files must be present — the model file is
   required for the app to load).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select your repo/branch, and set the main file path to
   `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and starts the app.

## Retraining the model

If you want to retrain on fresh data, rerun the notebook end to end (it pulls
the dataset via `kagglehub`) and it will overwrite
`linear_regression_model.pkl` with the newly trained model. Copy that file
into this project folder before redeploying.

## Notes

- Rows with missing `Price` or `EngineV`, and cars with `EngineV > 10`
  (data-entry errors), were dropped during training — predictions for
  extreme or out-of-range inputs may be less reliable.
- The `Model` column was dropped before training, so the app does not ask
  for a specific car model, only brand and body type.

## License

Add your preferred license here (e.g., MIT).
