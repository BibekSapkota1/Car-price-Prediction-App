

# import joblib
# import streamlit as st
# import numpy as np
# import pandas as pd
# import sklearn

# def load_model_compat(filename):
#     return joblib.load(filename, mmap_mode="r", fix_imports=True)

# # Load models
# models = {
#   #  "Decision Tree": joblib.load("decisiontree_model.pkl"),
#   #  "Random Forest": joblib.load("randomforest_model.pkl"),
#     "Linear Regression": joblib.load("linear_regression_model.pkl"),
#     "XGBoost": joblib.load("xgboost_model.pkl"),
#   #  "CatBoost": joblib.load("catboost_model.pkl"),
#     "LG Boost": joblib.load("lightgbm_model.pkl"),
#   #  "Aada Boost": joblib.load("adaboost_model.pkl"),
#     "Ridge Regression": joblib.load("ridge_model.pkl"),
#     "Lasso Regression": joblib.load("lasso_model.pkl"),
#     "Mlp Regression": joblib.load("mlp_model.pkl"),
# }

# print(f"✅ Models loaded successfully with scikit-learn {sklearn.__version__}")

# # Define ensemble weights
# model_weights = {
#     "XGBoost": 0.185013,
#     "LG Boost": 0.000181,
#     "CatBoost": 0.012551,
#     "Aada Boost": 0.153472,
#     "Decision Tree": 0.023876,
#     "Random Forest": 0.000225,
#     "Linear Regression": 0.162482,
#     "Lasso Regression": 0.096746,
#     "Ridge Regression": 0.200036,
#     "Mlp Regression": 0.000190
# }

# # Load encoders
# brand_encoder = joblib.load("brand_encoder.pkl")
# fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
# transmission_encoder = joblib.load("transmission_encoder.pkl")
# ext_col_encoder = joblib.load("ext_col_encoder.pkl")
# int_col_encoder = joblib.load("int_col_encoder.pkl")
# accident_encoder = joblib.load("accident_encoder.pkl")
# clean_title_encoder = joblib.load("clean_title_encoder.pkl")

# # Streamlit UI
# st.set_page_config(layout="wide")
# st.title("🚗 Car Price Prediction")
# st.markdown("Enter the car details below, and select a machine learning model to estimate the price.")

# # Sidebar for model selection (improved dropdown)
# with st.sidebar:
#     st.header("Select Prediction Model")
#     st.markdown("### Choose your model for prediction")
#     model_choice = st.selectbox(
#         "Prediction Model", 
#         ["Ensemble Model"] + list(models.keys()), 
#         index=0,
#         help="Select the machine learning model you want to use for price prediction.",
#         format_func=lambda x: f"✨ {x}"  # This adds a fun style to model names in the dropdown
#     )

# # Input fields
# col1, col2 = st.columns(2)

# with col1:
#     brand = st.selectbox("Brand", brand_encoder.classes_, help="Choose the car's brand")
#     mileage = st.number_input("Total Km Travelled (in km)", min_value=0, help="Enter the total mileage of the car")
#     fuel_type = st.selectbox("Fuel Type", fuel_type_encoder.classes_, help="Select fuel type")
#     transmission = st.selectbox("Transmission", transmission_encoder.classes_, help="Select the transmission type")
#     ext_col = st.selectbox("Exterior Color", ext_col_encoder.classes_, help="Select the exterior color")
#     int_col = st.selectbox("Interior Color", int_col_encoder.classes_, help="Select the interior color")

# with col2:
#     accident = st.selectbox("Accident History", accident_encoder.classes_, help="Select accident history")
#     clean_title = st.selectbox("Clean Title", clean_title_encoder.classes_, help="Select if the car has a clean title")
#     horsepower = st.number_input("Horsepower (HP)", min_value=0, help="Enter the horsepower of the car")
#     displacement = st.number_input("Displacement (in L)", min_value=0.0, help="Enter the engine displacement in liters")
#     cylinder_count = st.number_input("Cylinder Count", min_value=1, help="Enter the number of cylinders")
#     model_age = st.number_input("Model Age (in years)", min_value=0, help="Enter the model age of the car")

# # Function to safely transform categorical features
# def safe_transform(encoder, value):
#     try:
#         return encoder.transform([value])[0]
#     except ValueError:
#         return -1  # Assign -1 for unseen labels

# if st.button("🔍 Predict Price"):
#     # Encode categorical features
#     encoded_features = [
#         safe_transform(brand_encoder, brand), mileage,
#         safe_transform(fuel_type_encoder, fuel_type),
#         safe_transform(transmission_encoder, transmission),
#         safe_transform(ext_col_encoder, ext_col),
#         safe_transform(int_col_encoder, int_col),
#         safe_transform(accident_encoder, accident),
#         safe_transform(clean_title_encoder, clean_title),
#         horsepower, displacement, cylinder_count, model_age
#     ]
    
#     # Convert features into a DataFrame
#     feature_columns = ['brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age']
#     features_df = pd.DataFrame([encoded_features], columns=feature_columns)
    
#     # Make prediction with Ensemble Model
#     if model_choice == "Ensemble Model":
#         weighted_sum = 0
#         weight_total = sum(model_weights.values())

#         for name, model in models.items():
#             if name in model_weights:
#                 weighted_sum += model_weights[name] * model.predict(features_df)[0]

#         predicted_price = weighted_sum / weight_total  # Normalize

#     else:
#         predicted_price = models[model_choice].predict(features_df)[0]
    
#     # Display result
#     if predicted_price < 0:
#         st.error(f"💰 Predicted Car Price: ${predicted_price:,.2f}")
#         st.error("⚠️ The predicted price seems unrealistic. Please check the input values.")
#     else:
#         st.success(f"💰 Predicted Car Price: ${predicted_price:,.2f}")

# Vesrion mismatch

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import sklearn

# Safe model loading function
def load_model_compat(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"⚠️ Error loading {filename}: {e}")
        return None  # Return None if loading fails

# Load models safely
models = {
#    "Decision Tree": load_model_compat("decisiontree_model.pkl"),
 #   "Random Forest": load_model_compat("randomforest_model.pkl"),
    "Linear Regression": load_model_compat("linear_regression_model.pkl"),
    "XGBoost": load_model_compat("xgboost_model.pkl"),
 #   "CatBoost": load_model_compat("catboost_model.pkl"),
    "LG Boost": load_model_compat("lightgbm_model.pkl"),
 #   "Aada Boost": load_model_compat("adaboost_model.pkl"),
    "Ridge Regression": load_model_compat("ridge_model.pkl"),
    "Lasso Regression": load_model_compat("lasso_model.pkl"),
    "Mlp Regression": load_model_compat("mlp_model.pkl"),
}

# Remove models that failed to load
models = {name: model for name, model in models.items() if model is not None}

if not models:
    st.error("❌ No models could be loaded. Please check the pickle files or scikit-learn version.")

print(f"✅ Models loaded successfully with scikit-learn {sklearn.__version__}")

# Define ensemble weights
model_weights = {
    "XGBoost": 0.185013,
    "LG Boost": 0.000181,
    "CatBoost": 0.012551,
    "Aada Boost": 0.153472,
    "Decision Tree": 0.023876,
    "Random Forest": 0.000225,
    "Linear Regression": 0.162482,
    "Lasso Regression": 0.096746,
    "Ridge Regression": 0.200036,
    "Mlp Regression": 0.000190
}

# Load encoders safely
def safe_load_encoder(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"⚠️ Error loading encoder {filename}: {e}")
        return None

brand_encoder = safe_load_encoder("brand_encoder.pkl")
fuel_type_encoder = safe_load_encoder("fuel_type_encoder.pkl")
transmission_encoder = safe_load_encoder("transmission_encoder.pkl")
ext_col_encoder = safe_load_encoder("ext_col_encoder.pkl")
int_col_encoder = safe_load_encoder("int_col_encoder.pkl")
accident_encoder = safe_load_encoder("accident_encoder.pkl")
clean_title_encoder = safe_load_encoder("clean_title_encoder.pkl")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction")
st.markdown("Enter the car details below, and select a machine learning model to estimate the price.")

# Sidebar for model selection
with st.sidebar:
    st.header("Select Prediction Model")
    model_choice = st.selectbox(
        "Prediction Model", 
        ["Ensemble Model"] + list(models.keys()), 
        index=0,
        help="Select the ML model for price prediction."
    )

# Input fields
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_encoder.classes_ if brand_encoder else ["N/A"])
    mileage = st.number_input("Total Km Travelled (in km)", min_value=0)
    fuel_type = st.selectbox("Fuel Type", fuel_type_encoder.classes_ if fuel_type_encoder else ["N/A"])
    transmission = st.selectbox("Transmission", transmission_encoder.classes_ if transmission_encoder else ["N/A"])
    ext_col = st.selectbox("Exterior Color", ext_col_encoder.classes_ if ext_col_encoder else ["N/A"])
    int_col = st.selectbox("Interior Color", int_col_encoder.classes_ if int_col_encoder else ["N/A"])

with col2:
    accident = st.selectbox("Accident History", accident_encoder.classes_ if accident_encoder else ["N/A"])
    clean_title = st.selectbox("Clean Title", clean_title_encoder.classes_ if clean_title_encoder else ["N/A"])
    horsepower = st.number_input("Horsepower (HP)", min_value=0)
    displacement = st.number_input("Displacement (in L)", min_value=0.0)
    cylinder_count = st.number_input("Cylinder Count", min_value=1)
    model_age = st.number_input("Model Age (in years)", min_value=0)

# Function to safely transform categorical features
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except (ValueError, AttributeError):
        return -1  # Assign -1 for unseen labels or missing encoder

if st.button("🔍 Predict Price"):
    # Encode categorical features
    encoded_features = [
        safe_transform(brand_encoder, brand), mileage,
        safe_transform(fuel_type_encoder, fuel_type),
        safe_transform(transmission_encoder, transmission),
        safe_transform(ext_col_encoder, ext_col),
        safe_transform(int_col_encoder, int_col),
        safe_transform(accident_encoder, accident),
        safe_transform(clean_title_encoder, clean_title),
        horsepower, displacement, cylinder_count, model_age
    ]
    
    # Convert features into a DataFrame
    feature_columns = ['brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age']
    features_df = pd.DataFrame([encoded_features], columns=feature_columns)
    
    # Make prediction with Ensemble Model
    if model_choice == "Ensemble Model":
        weighted_sum = 0
        weight_total = sum(model_weights.values())

        for name, model in models.items():
            if name in model_weights:
                try:
                    weighted_sum += model_weights[name] * model.predict(features_df)[0]
                except Exception as e:
                    st.warning(f"⚠️ Error predicting with {name}: {e}")

        predicted_price = weighted_sum / weight_total if weight_total > 0 else 0

    else:
        try:
            predicted_price = models[model_choice].predict(features_df)[0]
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
            predicted_price = None
    
    # Display result
    if predicted_price is None:
        st.error("🚨 Prediction failed. Try another model.")
    elif predicted_price < 0:
        st.error(f"💰 Predicted Car Price: ${predicted_price:,.2f}")
        st.error("⚠️ The predicted price seems unrealistic. Please check the input values.")
    else:
        st.success(f"💰 Predicted Car Price: ${predicted_price:,.2f}")


