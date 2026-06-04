# house_price_predictor.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATASET_PATH = "housing_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def generate_synthetic_data(num_samples=1000, save_path=DATASET_PATH):
    """
    Generates a realistic synthetic housing dataset and saves it to a CSV.
    """
    print(f"Generating synthetic housing dataset with {num_samples} records...")
    np.random.seed(42)

    # Features
    square_feet = np.random.randint(800, 5000, size=num_samples)
    bedrooms = np.random.randint(1, 6, size=num_samples)
    bathrooms = np.random.randint(1, 5, size=num_samples)
    # Age of the house in years
    age = np.random.randint(0, 50, size=num_samples)
    # Location score from 1 (poor) to 5 (excellent)
    location_score = np.random.randint(1, 6, size=num_samples)

    # Base price calculation with a realistic formula
    # Base: $50,000 + $150/sqft + $25,000/bedroom + $18,000/bathroom - $1,000/year of age + $45,000/location_point
    base_price = (
        50000
        + (square_feet * 150)
        + (bedrooms * 25000)
        + (bathrooms * 18000)
        - (age * 1000)
        + (location_score * 45000)
    )

    # Add random noise (random market variance)
    noise = np.random.normal(0, 15000, size=num_samples)
    price = base_price + noise

    # Ensure no prices are negative (highly unlikely but good practice)
    price = np.clip(price, 30000, None)

    # Create DataFrame
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age': age,
        'LocationScore': location_score,
        'Price': price
    })

    df.to_csv(save_path, index=False)
    print(f"Dataset successfully created and saved to '{save_path}'!")
    return df


def explore_and_visualize_data(df):
    """
    Performs basic Exploratory Data Analysis (EDA) and saves plots.
    """
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print("Dataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # Correlation matrix
    corr = df.corr()
    print("\nCorrelation with Price:")
    print(corr['Price'].sort_values(ascending=False))

    # Save correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Housing Features")
    plt.tight_layout()
    plot_path = "correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Correlation heatmap plot saved as '{plot_path}'")


def train_and_evaluate_model():
    """
    Preprocesses data, trains Linear Regression and Random Forest models,
    evaluates them, and saves the best model.
    """
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(DATASET_PATH)

    explore_and_visualize_data(df)

    # Separate Features and Target
    X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age', 'LocationScore']]
    y = df['Price']

    # Split into Train and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (important for Linear Regression and other distance-based models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # 2. Random Forest Regressor Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    # Evaluation Helper
    def evaluate(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"\n[{model_name} Performance]")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"R-squared (R2) Score: {r2:.4f}")
        return r2

    r2_lr = evaluate(y_test, y_pred_lr, "Linear Regression")
    r2_rf = evaluate(y_test, y_pred_rf, "Random Forest Regressor")

    # Select and Save Best Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    if r2_rf > r2_lr:
        best_model = rf_model
        best_model_name = "Random Forest Regressor"
    else:
        best_model = lr_model
        best_model_name = "Linear Regression"

    print(f"\nSaving the best model ({best_model_name}) and scaler...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[SUCCESS] Model and Scaler saved successfully!")


def predict_price():
    """
    CLI interface to get input features from the user and predict the price of a house.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found! Running training first...")
        train_and_evaluate_model()

    # Load Model and Scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("\n==================================================")
    print("        HOUSE PRICE PREDICTION SYSTEM             ")
    print("==================================================")
    print("Please enter the details of the house below:")

    try:
        sqft = float(input("1. Square Footage (e.g. 1500): "))
        bedrooms = int(input("2. Number of Bedrooms (e.g. 3): "))
        bathrooms = int(input("3. Number of Bathrooms (e.g. 2): "))
        age = int(input("4. Age of the house in years (e.g. 5): "))
        location = int(input("5. Location Score (1: Poor, 3: Average, 5: Excellent): "))

        # Input validation
        if sqft <= 0 or bedrooms < 1 or bathrooms < 1 or age < 0 or not (1 <= location <= 5):
            print("[ERROR] Invalid input parameters. Please try again with realistic values.")
            return

        # Prepare user input for prediction
        input_data = pd.DataFrame([{
            'SquareFeet': sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age': age,
            'LocationScore': location
        }])

        # Scale input details (MUST use same scaler fit during training)
        input_scaled = scaler.transform(input_data)

        # Make Prediction
        predicted_price = model.predict(input_scaled)[0]

        print("\n--------------------------------------------------")
        print(f"[ESTIMATE] Estimated Market Price: ${predicted_price:,.2f}")
        print("--------------------------------------------------")

    except ValueError:
        print("[ERROR] Error: Please enter numeric values only.")


if __name__ == "__main__":
    while True:
        print("\n--- Housing Price Prediction Menu ---")
        print("1. Generate Data & Train Model")
        print("2. Predict House Price (Interactive)")
        print("3. Exit")
        choice = input("Enter choice (1-3): ")

        if choice == "1":
            train_and_evaluate_model()
        elif choice == "2":
            predict_price()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please select 1, 2, or 3.")
