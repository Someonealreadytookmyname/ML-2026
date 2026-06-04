import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page configuration for a premium look
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism cards, beautiful buttons, and gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply custom font */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle Styling */
    .sub-title {
        font-size: 1.2rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    
    /* Premium card container */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        margin-top: 1rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.05) 0%, rgba(255, 143, 107, 0.05) 100%);
        border-left: 5px solid #FF4B4B;
    }
    
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FF4B4B;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(255, 75, 75, 0.2);
    }
    
    .metric-label {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8C96A0;
        font-weight: 600;
    }
    
    /* Input section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Configurations for Advanced Model
DATASET_PATH = "housing_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# --- Model Loading & Training Helper Functions ---

@st.cache_resource
def load_advanced_model():
    """Loads pre-trained model and scaler, training them if they do not exist."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        # We need to train the model first if not present
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
        else:
            # Generate synthetic data
            np.random.seed(42)
            num_samples = 1000
            sqft = np.random.randint(800, 5000, size=num_samples)
            bedrooms = np.random.randint(1, 6, size=num_samples)
            bathrooms = np.random.randint(1, 5, size=num_samples)
            age = np.random.randint(0, 50, size=num_samples)
            location = np.random.randint(1, 6, size=num_samples)
            base_price = (50000 + sqft * 150 + bedrooms * 25000 + bathrooms * 18000 - age * 1000 + location * 45000)
            noise = np.random.normal(0, 15000, size=num_samples)
            price = np.clip(base_price + noise, 30000, None)
            df = pd.DataFrame({
                'SquareFeet': sqft,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Age': age,
                'LocationScore': location,
                'Price': price
            })
            df.to_csv(DATASET_PATH, index=False)
            
        # Train model
        X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age', 'LocationScore']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest by default
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def get_advanced_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None

# --- Simple Model Helper Functions ---

@st.cache_data
def get_simple_dataset():
    data = {
        'Size_sqft': [850, 900, 1200, 1500, 1600, 1800, 2200, 2500],
        'Condition': [2, 4, 3, 5, 2, 4, 3, 5],
        'Price': [150000, 210000, 250000, 340000, 280000, 390000, 420000, 550000]
    }
    return pd.DataFrame(data)

@st.cache_resource
def train_simple_model(df):
    X = df[['Size_sqft', 'Condition']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    return model


# --- Main UI Layout ---

st.markdown('<div class="main-title">🏡 Smart House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An interactive machine learning dashboard for estimating real estate value.</div>', unsafe_allow_html=True)

# Sidebar for Model Selection and Info
st.sidebar.markdown("### 🛠️ Model Configuration")
model_type = st.sidebar.radio(
    "Choose Prediction Model:",
    ["Advanced Model (5 Features)", "Simple Model (2 Features)"],
    help="Select the machine learning model to use for predictions."
)

if model_type == "Advanced Model (5 Features)":
    st.sidebar.info(
        "**Advanced Model Details:**\n\n"
        "- **Algorithm:** Random Forest Regressor / Linear Regression\n"
        "- **Dataset:** 1,000 generated records with typical housing features\n"
        "- **Features:** Sq Footage, Bedrooms, Bathrooms, Age, Location Quality"
    )
else:
    st.sidebar.info(
        "**Simple Model Details:**\n\n"
        "- **Algorithm:** Linear Regression (no scaling)\n"
        "- **Dataset:** Small 8-record house dataset\n"
        "- **Features:** Size (Sqft), Condition Rating (1 to 5)"
    )

# Setup Tabs
tab1, tab2 = st.tabs(["🔮 Estimate House Price", "📊 Model Insights & Analytics"])

# Load datasets and models
adv_df = get_advanced_dataset()
adv_model, adv_scaler = load_advanced_model()

simp_df = get_simple_dataset()
simp_model = train_simple_model(simp_df)

if model_type == "Advanced Model (5 Features)":
    # ----------------------------------------------------
    # TAB 1: Advanced Model Prediction
    # ----------------------------------------------------
    with tab1:
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            st.markdown('<div class="section-header">1. Enter Property Characteristics</div>', unsafe_allow_html=True)
            
            # Interactive input controls
            sqft = st.slider("Square Footage (sq ft)", min_value=800, max_value=5000, value=1500, step=50)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
            with sub_col2:
                bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
                
            sub_col3, sub_col4 = st.columns(2)
            with sub_col3:
                age = st.slider("Age of the House (years)", min_value=0, max_value=50, value=5, step=1)
            with sub_col4:
                location = st.select_slider(
                    "Location Quality Score",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: {1: "1 - Poor", 2: "2 - Below Avg", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}[x]
                )
                
        with col2:
            st.markdown('<div class="section-header">2. Estimation Output</div>', unsafe_allow_html=True)
            
            # Perform prediction
            input_df = pd.DataFrame([{
                'SquareFeet': sqft,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Age': age,
                'LocationScore': location
            }])
            
            input_scaled = adv_scaler.transform(input_df)
            predicted_price = adv_model.predict(input_scaled)[0]
            
            # Large styled metric box
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">Estimated Market Price</div>
                <div class="price-display">${predicted_price:,.2f}</div>
                <p style="color: #6C757D; font-size: 0.95rem;">
                    *This price estimate is generated using a Random Forest model trained on local market dynamics.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add dynamic highlights/warnings based on characteristics
            if age > 35:
                st.warning("⚠️ Note: Older houses might require renovation costs not fully captured by this estimate.")
            if location == 5:
                st.success("🌟 Premium Location: High demand in this zone increases value resilience.")

    # ----------------------------------------------------
    # TAB 2: Advanced Model Insights
    # ----------------------------------------------------
    with tab2:
        st.markdown("### 📊 Advanced Model Insights & Data Analytics")
        st.write("This tab provides insights into the dataset and correlations between house features and market prices.")
        
        # Display dataset preview
        if adv_df is not None:
            st.subheader("📋 Dataset Preview (First 5 records)")
            st.dataframe(adv_df.head(), use_container_width=True)
            
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.subheader("🔥 Feature Correlation Matrix")
                corr = adv_df.corr()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
                plt.title("Correlation Heatmap")
                st.pyplot(fig)
                
            with col_graph2:
                st.subheader("📐 Price Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(adv_df['Price'], kde=True, color="#FF4B4B", ax=ax)
                plt.title("Distribution of Housing Prices")
                plt.xlabel("Price ($)")
                st.pyplot(fig)
                
        else:
            st.warning("Advanced dataset housing_data.csv not found to generate insights.")

else:
    # ----------------------------------------------------
    # TAB 1: Simple Model Prediction
    # ----------------------------------------------------
    with tab1:
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            st.markdown('<div class="section-header">1. Enter Property Characteristics</div>', unsafe_allow_html=True)
            
            # Sliders matching the features of the simple model
            sqft = st.slider("Size in Square Feet (sqft)", min_value=500, max_value=3000, value=1500, step=50)
            condition = st.slider("Condition Rating (1: Poor, 5: Excellent)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
            
        with col2:
            st.markdown('<div class="section-header">2. Estimation Output</div>', unsafe_allow_html=True)
            
            # Prediction dataframe matching column names in training
            input_df = pd.DataFrame([[sqft, condition]], columns=['Size_sqft', 'Condition'])
            predicted_price = simp_model.predict(input_df)[0]
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">Estimated Market Price</div>
                <div class="price-display">${predicted_price:,.2f}</div>
                <p style="color: #6C757D; font-size: 0.95rem;">
                    *This price estimate is generated using a Linear Regression model trained dynamically on 8 sample data points.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: Simple Model Insights
    # ----------------------------------------------------
    with tab2:
        st.markdown("### 📊 Simple Model Details & Training Data")
        
        st.write("Below is the complete dataset used to train the Simple Model, along with the model's predictions on those records.")
        
        # Calculate training predictions
        simp_df_copy = simp_df.copy()
        simp_df_copy['Predicted_Price'] = simp_model.predict(simp_df[['Size_sqft', 'Condition']])
        simp_df_copy['Difference'] = simp_df_copy['Price'] - simp_df_copy['Predicted_Price']
        
        st.dataframe(simp_df_copy.style.format({
            'Price': '${:,.2f}',
            'Predicted_Price': '${:,.2f}',
            'Difference': '${:,.2f}'
        }), use_container_width=True)
        
        # Model Parameters
        st.subheader("⚙️ Model Coefficients (Linear Regression Parameters)")
        intercept = simp_model.intercept_
        coef_sqft = simp_model.coef_[0]
        coef_cond = simp_model.coef_[1]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Intercept (Base Price)", f"${intercept:,.2f}")
        col_m2.metric("Size Coefficient (per SqFt)", f"${coef_sqft:,.2f}")
        col_m3.metric("Condition Coefficient (per Unit)", f"${coef_cond:,.2f}")
        
        st.markdown(f"**Regression Formula:**  \n`Price = ${intercept:,.2f} + ({coef_sqft:,.2f} × Size) + ({coef_cond:,.2f} × Condition)`")
