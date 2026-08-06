import streamlit as st
import joblib
import pandas as pd

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Malaysia Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("malaysia_house_price_data_2025.csv")

df = load_data()

# Get unique categorical values from the dataset
areas = sorted( 
    df["Area"].dropna().unique().tolist() 
)

states = sorted(
    df["State"].dropna().unique().tolist()
)

tenures = sorted(
    df["Tenure"].dropna().unique().tolist()
)

property_types = sorted(
    df["Type"].dropna().unique().tolist()
)

# ==============================
# Load Trained Models
# ==============================

@st.cache_resource
def load_models():
    return joblib.load("housing_models.joblib")

model_bundle = load_models()

preprocessor = model_bundle["preprocessor"]
models = model_bundle["models"]
metrics = model_bundle["metrics"]

# ==============================
# Main Page
# ==============================

st.title("🏠 Malaysia Housing Price Predictor")

st.write(
    "This application predicts the median property price "
    "of residential properties in Malaysia."
)

st.write(
    "Enter the property information below to get an estimated "
    "median property price."
)

# ==============================
# Model Selection
# ==============================

st.header("🤖 Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a model:",
    [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "CatBoost"
    ]
)

st.info(
    f"Selected model: **{selected_model}**"
)

# ==============================
# Property Information
# ==============================

st.header("🏡 Property Information")

st.write(
    "Please provide the following property details:"
)

# ============================== 
# State and Area Selection 
# ============================== 
# Initialize default selections 
if "selected_state" not in st.session_state: 
  st.session_state.selected_state = states[0] 
  
if "selected_area" not in st.session_state: 
  state_areas = sorted( 
      df.loc[ 
          df["State"] == st.session_state.selected_state, "Area" 
      ].dropna().unique().tolist() 
  ) 
  
  st.session_state.selected_area = state_areas[0] 
  
# Function called when State changes 
def update_area_from_state(): 
  selected_state = st.session_state.selected_state 
  
  available_areas = sorted( 
      df.loc[ 
          df["State"] == selected_state, "Area"
     ].dropna().unique().tolist() 
  ) 

  if available_areas: 
    st.session_state.selected_area = available_areas[0] 
    
# Function called when Area changes 
def update_state_from_area(): 
  selected_area = st.session_state.selected_area 
  matching_states = ( 
      df.loc[ 
          df["Area"] == selected_area, "State" 
      ] .dropna() .unique() .tolist() 
  ) 
  
  if matching_states: 
    st.session_state.selected_state = matching_states[0] 
    
# State dropdown 
state = st.selectbox( 
    "State", 
    states, 
    key="selected_state", 
    on_change=update_area_from_state 
)

# Get areas belonging to the selected State 
filtered_areas = sorted( df.loc[ df["State"] == state, "Area" ] .dropna() .unique() .tolist() ) 

# Make sure the current Area is valid for the selected State 
if st.session_state.selected_area not in filtered_areas: 
  st.session_state.selected_area = filtered_areas[0] 
  
# Area dropdown 
area = st.selectbox( "Area", filtered_areas, key="selected_area", on_change=update_state_from_area )

# Tenure
tenure = st.selectbox(
    "Tenure",
    tenures
)

# Property Type
property_type = st.selectbox(
    "Property Type",
    property_types
)

# Median PSF
median_psf = st.number_input(
    "Median Price Per Square Foot (RM)",
    min_value=0.0,
    value=300.0,
    step=1.0
)

# Transactions
transactions = st.number_input(
    "Number of Transactions",
    min_value=0,
    value=100,
    step=1
)

# ==============================
# Prediction
# ==============================

st.divider()

st.header("💰 Price Prediction")

if st.button("Predict Median Price"):

    # Create input data for the model
    input_data = pd.DataFrame({
        "Area": [area],
        "State": [state],
        "Tenure": [tenure],
        "Type": [property_type],
        "Median_PSF": [median_psf],
        "Transactions": [transactions]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(
        f"Estimated Median Property Price: RM {prediction:,.2f}"
    )
