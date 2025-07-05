import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.metrics import classification_report

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('../data/ObesityDataSet_Cleaned.csv')
    return df

# --- Load Model ---
# @st.cache_resource
# def load_model():
#     with open('../model/logistic_model.pkl', 'rb') as f:
#         return pickle.load(f)

# df = load_data()
# model_data = load_model()
# model = model_data['model']
# scaler = model_data['scaler']
# encoders = model_data['encoders']

# --- Page 1A: Factor Distribution (categorical count) ---
def show_attribute_distribution():
    st.subheader("Attributes Distribution")

    df = load_data()

    numeric_cols = ['Age', 'Height', 'Weight','BMI']
    categorical_cols = ['Gender', 'MonitorCaloriesHabit', 'GeneticsOverweight', 'SnackHabit','LevelObesity']

    # --- Numeric Charts (Top) ---
    st.markdown("##### 🔢 Numeric Feature Distribution")
    num_cols = st.columns(len(numeric_cols))
    for i, col in enumerate(numeric_cols):
        with num_cols[i]:
            fig = px.histogram(df, x=col, nbins=20, marginal="box", color_discrete_sequence=["skyblue"])
            fig.update_layout(title=f"{col} Distribution", height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- Categorical Charts (Bottom) ---
    st.markdown("##### 🏷️ Categorical Feature Distribution")
    for i in range(0, len(categorical_cols), 3):
        cat_row = st.columns(3)
        for j in range(3):
            if i + j < len(categorical_cols):
                with cat_row[j]:
                    col = categorical_cols[i + j]
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count']
                    fig = px.bar(counts, x=col, y='count', color=col,
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_layout(title=f"{col} Count", height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

# --- Page 1B: Obesity Distribution ---
def show_obesity_distribution_by_attribute(df, attribute):
    st.markdown(f"##### 🔍 {attribute}")

    # Calculate percentage distribution
    counts = df.groupby([attribute, 'LevelObesity']).size().unstack().fillna(0)
    percentages = counts.div(counts.sum(axis=1), axis=0)
    st.dataframe(percentages.style.format("{:.2%}"))

    # Plotly stacked bar chart
    fig = px.histogram(df, x=attribute, color='LevelObesity',
                       barmode='group',  # Use 'stack' if you prefer stacked bars
                       title=f"{attribute} vs Obesity",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=350, xaxis_title=attribute, yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    # Highlight group with highest obesity distribution
    dominant_group = percentages.sum(axis=1).idxmax()
    st.info(f"✅ Group *{dominant_group}* in {attribute} shows highest obesity distribution.")

def show_obesity_distribution_analysis():
    st.subheader("Obesity Distribution by Attributes")

    df = load_data()

    categorical_cols = ['Gender', 'MonitorCaloriesHabit', 'GeneticsOverweight', 'SnackHabit', 'AgeGroup']

    # Just loop through all attributes one by one, no columns
    for attr in categorical_cols:
        show_obesity_distribution_by_attribute(df, attr)


# --- Page 2: Obesity Classification ---
# Load model, scaler, label encoder, and training column names
model = joblib.load("../model/model_saved_file/random_forest_model.pkl")
scaler = joblib.load("../model/model_saved_file/scaler.pkl")
label_encoder = joblib.load("../model/model_saved_file/label_encoder.pkl")
training_columns = joblib.load("../model/model_saved_file/feature_columns.pkl")

def obesity_prediction():
    st.subheader("Predict Obesity Level")

    # Initialize/reset session state
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
        st.session_state.bmi = None
        st.session_state.label = None

    # --- Input Form with a border ---
    with st.container():
        st.markdown("""
            <div style="border: 1px solid #ccc; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.markdown("#### 📝 Input Personal Details")

            age = st.slider("Age", 10, 100, 25)
            gender = st.selectbox("Gender", ['Male', 'Female'])
            height = st.number_input("Height (in meters)", 1.0, 2.5, 1.70)
            weight = st.number_input("Weight (in kg)", 30.0, 200.0, 70.0)
            monitor = st.selectbox("Do you monitor your calorie intake?", ['yes', 'no'])
            history = st.selectbox("Family history of overweight?", ['yes', 'no'])
            snack = st.selectbox("Snack habits", ['no', 'sometimes', 'frequently', 'always'])

            col1, col2 = st.columns([1, 1])
            with col1:
                predict = st.form_submit_button("🎯 Predict")
            with col2:
                clear = st.form_submit_button("🗑️ Clear")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Handle buttons ---
    if predict:
        bmi = weight / (height ** 2)
        st.session_state.bmi = bmi

        # Build input
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'MonitorCaloriesHabit': monitor,
            'GeneticsOverweight': history,
            'SnackHabit': snack
        }])
        input_encoded = pd.get_dummies(input_df)
        for col in training_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[training_columns]
        input_scaled = scaler.transform(input_encoded)

        # Predict
        pred = model.predict(input_scaled)[0]
        label = label_encoder.inverse_transform([pred])[0]

        st.session_state.label = label
        st.session_state.show_result = True

    elif clear:
        st.session_state.show_result = False
        st.session_state.bmi = None
        st.session_state.label = None

    # --- Show result if available ---
    if st.session_state.show_result:
        st.subheader("Prediction Result")
        st.markdown(f"📏 **Calculated BMI:** `{st.session_state.bmi:.2f}`")
        st.success(f"🎯 **Predicted Obesity Level:** `{st.session_state.label}`")

        suggestions = {
            "Normal_Weight": "Maintain your healthy lifestyle! 👍",
            "Overweight_Level_I": "Consider regular physical activity and better eating habits.",
            "Overweight_Level_II": "Consult a health professional for a personalized plan.",
            "Obesity_Type_I": "Strongly consider medical advice and regular exercise.",
            "Obesity_Type_II": "High risk. Please seek professional help.",
            "Obesity_Type_III": "Severe risk. A medical intervention is highly recommended.",
            "Insufficient_Weight": "Consider a balanced diet to reach healthy weight."
        }
        st.info("💡 Suggestion: " + suggestions.get(st.session_state.label, "Consult a health expert."))

        # --- Feature Importance Chart ---
        st.subheader("Feature Importance (Top 10)")
        importance_df = pd.DataFrame({
            'Feature': training_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)

        fig = px.bar(importance_df, x='Importance', y='Feature',
                     orientation='h', color='Importance',
                     color_continuous_scale='Blues',
                     title="Top 10 Feature Importances")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Model Performance ---
def show_model_performance():
    st.header("Random Forest Model Performance")

    # Load saved components
    model = joblib.load("../model/model_saved_file/random_forest_model.pkl")
    scaler = joblib.load("../model/model_saved_file/scaler.pkl")
    le = joblib.load("../model/model_saved_file/label_encoder.pkl")
    X_test, y_test, y_pred, acc, cm, report_text = joblib.load("../model/model_saved_file/evaluation_results.pkl")

    # Show Accuracy
    st.metric("Model Accuracy:", f"{acc:.2%}")

    # Convert classification report to dataframe
    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Show classification report as a table
    st.subheader("Classification Report")
    st.dataframe(report_df.round(2))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_labels = le.classes_
    cm_fig = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=cm_labels.tolist(),
        y=cm_labels.tolist(),
        colorscale='Blues',
        showscale=True,
        annotation_text=cm.astype(str).tolist()
    )
    cm_fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(cm_fig, use_container_width=True)

# --- Page Setting ---
st.set_page_config(page_title="Obesity Dashboard", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        "📊 Obesity Dashboard",  # title
        ["Obesity Analysis", "Obesity Classification", "Model Performance"],  # menu items
        icons=["bar-chart", "robot", "activity"],  # optional icons
        menu_icon="list",  # icon for the menu title
        default_index=0,
        orientation="vertical"
    )

# Inject custom CSS
st.markdown("""
    <style>
    .css-1d391kg, .css-1n76uvr {
        font-size: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Content ---
if selected == "Obesity Analysis":
    st.title("📊 Obesity Data Analysis Dashboard")
    tab1, tab2 = st.tabs(["Attributes Distribution", "Obesity Distribution by Attributes"])

    with tab1:
        show_attribute_distribution()

    with tab2:
        show_obesity_distribution_analysis()

elif selected == "Obesity Classification":
    st.title("🤖 Obesity Classification")
    obesity_prediction()

elif selected == "Model Performance":
    st.title("📈 Model Performance")
    show_model_performance()