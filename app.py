import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(
    page_title="AI Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
    }
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 8px solid #6366f1;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
    }
    .score-display {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>AI Student Score Predictor</h1>'
    '<p>Predicting final scores based on study habits and AI usage</p></div>',
    unsafe_allow_html=True
)

# 3. Data Loading
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists("ai_impact_student_performance_dataset.csv"):
        return pd.read_csv("ai_impact_student_performance_dataset.csv")
    return None

with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Please upload the dataset CSV file.")
    st.stop()

# 4. Model Training
@st.cache_resource
def train_model(df):
    if "final_score" not in df.columns:
        return None, None, None, None, None, None

    X = df.drop("final_score", axis=1)
    y = df["final_score"]

    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X = X.fillna(X.mean(numeric_only=True))

    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    test_data = {"y_test": y_test, "y_pred": y_pred}

    return model, feature_names, label_encoders, metrics, test_data, scaler

model, feature_names, label_encoders, metrics, test_data, scaler = train_model(df)

if model is None:
    st.error("Dataset must contain 'final_score' column.")
    st.stop()

def prepare_input(input_df):
    X = input_df.copy()
    for col in label_encoders:
        if col in X.columns:
            X[col] = label_encoders[col].transform(X[col].astype(str))
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[feature_names]
    return pd.DataFrame(scaler.transform(X), columns=feature_names)

# 5. Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analytics", "Features", "About"])

with tab1:
    age = st.number_input("Age", 15, 35, 20)
    study = st.number_input("Study Hours/Day", 0.0, 15.0, 3.5)

    input_data = pd.DataFrame({
        "age": [age],
        "study_hours_per_day": [study],
    })

    if st.button("Predict Final Score"):
        prepared = prepare_input(input_data)
        score = model.predict(prepared)[0]
        st.success(f"🎯 Predicted Final Score: **{score:.1f} / 100**")

with tab2:
    st.metric("R² Score", f"{metrics['r2']:.3f}")
    plot_df = pd.DataFrame({
        "Actual": test_data["y_test"],
        "Predicted": test_data["y_pred"]
    })

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Scores",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(df.head())

with tab4:
    st.markdown("""
    **Model:** Random Forest Regressor  
    **Target:** Final Score  
    **Purpose:** Academic performance prediction
    """)
