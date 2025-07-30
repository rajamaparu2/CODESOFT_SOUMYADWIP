import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_sales():
    print("--- Starting Sales Prediction (Python) ---")

    df = None
    filename = None
    file_found = False

    filenames = ['advertising.csv', 'Advertising.csv']

    for name in filenames:
        try:
            df = pd.read_csv(name)
            filename = name
            file_found = True
            print(f"Loaded dataset: '{filename}' | Shape: {df.shape}")
            break
        except:
            continue

    if not file_found:
        print("\nSearching for advertising-related CSV files...")
        try:
            for file in os.listdir('.'):
                if 'advertising' in file.lower() and file.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file)
                        filename = file
                        file_found = True
                        print(f"Auto-loaded file: '{filename}' | Shape: {df.shape}")
                        break
                    except:
                        continue
        except:
            print("Error: Cannot list directory contents.")
            return

    if not file_found:
        print("\nDataset not found. Upload 'advertising.csv' and try again.")
        return

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Null Value Check ---")
    print(df.isnull().sum())

    df.columns = df.columns.str.lower().str.replace('.', '_').str.replace(' ', '_')

    features = ['tv', 'radio', 'newspaper']
    target = 'sales'

    if not all(f in df.columns for f in features) or target not in df.columns:
        print(f"Missing required columns. Found columns: {df.columns.tolist()}")
        return

    before = df.shape[0]
    df = df.dropna(subset=features + [target])
    after = df.shape[0]
    if before != after:
        print(f"\nDropped {before - after} rows with missing values.")

    X = df[features]
    y = df[target]

    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print("\n--- X Head ---")
    print(X.head())
    print("\n--- y Head ---")
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain/Test Split -> Train: {X_train.shape}, Test: {X_test.shape}")

    model = LinearRegression()
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Metrics ---")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    print("\n--- Coefficients ---")
    for i, col in enumerate(features):
        print(f"{col}: {model.coef_[i]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    samples = pd.DataFrame([
        [200, 30, 15],
        [10, 40, 50],
        [150, 5, 5]
    ], columns=features)

    print("\nExample Budgets:")
    print(samples)

    preds = model.predict(samples)

    print("\n--- Predictions ---")
    for i, row in samples.iterrows():
        print(f"\nSample {i+1}:")
        print(f"TV: {row['tv']} | Radio: {row['radio']} | Newspaper: {row['newspaper']}")
        print(f"Predicted Sales: {preds[i]:.2f}")

    print("\n--- Test Set Predictions ---")
    result_df = pd.DataFrame({
        'Actual_Sales': y_test.reset_index(drop=True),
        'Predicted_Sales': y_pred,
        'TV': X_test['tv'].reset_index(drop=True),
        'Radio': X_test['radio'].reset_index(drop=True),
        'Newspaper': X_test['newspaper'].reset_index(drop=True)
    })

    print("\nFirst 10:")
    print(result_df.head(10).to_string(index=False))

    print("\nLast 10:")
    print(result_df.tail(10).to_string(index=False))

    print("\n--- Sales Prediction Complete ---")

if __name__ == "__main__":
    predict_sales()
