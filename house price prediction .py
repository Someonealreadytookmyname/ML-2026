import pandas as pd
from sklearn.linear_model import LinearRegression

def get_dataset():
    """Creates and returns the initial house dataset."""
    data = {
        'Size_sqft': [850, 900, 1200, 1500, 1600, 1800, 2200, 2500],
        'Condition': [2, 4, 3, 5, 2, 4, 3, 5],
        'Price': [150000, 210000, 250000, 340000, 280000, 390000, 420000, 550000]
    }
    return pd.DataFrame(data)

def train_linear_regression(df):
    """Trains a Linear Regression model on the dataset."""
    X = df[['Size_sqft', 'Condition']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    return model

def get_user_input():
    """Gets validated house features from user input."""
    print("\n--- Predict House Price ---")
    try:
        size_sqft = float(input("Enter size in sqft (e.g., 1500): "))
        condition = float(input("Enter condition rating (1 to 5): "))
        if size_sqft <= 0 or not (1 <= condition <= 5):
            print("Error: Size must be positive and condition must be between 1 and 5.")
            return None, None
        return size_sqft, condition
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return None, None

def main():
    # Load dataset and train the model
    df = get_dataset()
    model = train_linear_regression(df)
    
    # Print model parameters
    print("Coefficients (Size_sqft, Condition):", model.coef_)
    print("Intercept:", model.intercept_)
    
    # Predict and display training results
    df['Predicted_Price'] = model.predict(df[['Size_sqft', 'Condition']])
    print("\nActual vs Predicted Prices:")
    print(df)
    
    # Prompt user for custom prediction
    size, cond = get_user_input()
    if size is not None and cond is not None:
        # Use pandas DataFrame to avoid warnings about feature names
        input_df = pd.DataFrame([[size, cond]], columns=['Size_sqft', 'Condition'])
        prediction = model.predict(input_df)[0]
        print(f"\nEstimated Price for {size:.0f} sqft with Condition {cond:.1f}: ${prediction:,.2f}")

if __name__ == "__main__":
    main()