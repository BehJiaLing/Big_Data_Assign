import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
from streamlit_option_menu import option_menu

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('../data/ObesityDataSet_Cleaned.csv')
    # Create AgeGroup column
    bins = [0, 20, 30, 40, 50, 100]
    labels = ['<20', '21–30', '31–40', '41–50', '50+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
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

    numeric_cols = ['Age', 'Height', 'Weight']
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
    st.info(f"✅ Group **`{dominant_group}`** in `{attribute}` shows highest obesity distribution.")

def show_obesity_distribution_analysis():
    st.subheader("Obesity Distribution by Factor")

    df = load_data()

    # Create AgeGroup column
    bins = [0, 20, 30, 40, 50, 100]
    labels = ['<20', '21–30', '31–40', '41–50', '50+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    categorical_cols = ['Gender', 'MonitorCaloriesHabit', 'GeneticsOverweight', 'SnackHabit', 'AgeGroup']

    # Just loop through all attributes one by one, no columns
    for attr in categorical_cols:
        show_obesity_distribution_by_attribute(df, attr)


# --- Page 2: Obesity Classification ---
def show_tab2_prediction():
    st.subheader("🤖 Predict Obesity Level")

    age = st.slider("Age", 10, 100, 25)
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    height = st.number_input("Height (in meters)", 1.0, 2.5, 1.70)
    weight = st.number_input("Weight (in kg)", 30.0, 200.0, 70.0)
    monitor = st.selectbox("Do you monitor your calorie intake?", encoders['MonitorCaloriesHabit'].classes_)
    history = st.selectbox("Family history of overweight?", encoders['GeneticsOverweight'].classes_)
    snack = st.selectbox("Snack habits", encoders['SnackHabit'].classes_)

    if st.button("Predict Obesity Level"):
        input_df = pd.DataFrame([[age, gender, height, weight, monitor, history, snack]],
                                columns=['Age', 'Gender', 'Height', 'Weight', 'MonitorCaloriesHabit', 'GeneticsOverweight', 'SnackHabit'])

        for col in ['Gender', 'MonitorCaloriesHabit', 'GeneticsOverweight', 'SnackHabit']:
            input_df[col] = encoders[col].transform(input_df[col])

        input_df[['Age', 'Height', 'Weight']] = scaler.transform(input_df[['Age', 'Height', 'Weight']])
        prediction = model.predict(input_df)[0]
        label = encoders['LevelObesity'].inverse_transform([prediction])[0]

        st.success(f"🎯 Predicted Obesity Level: **{label}**")

        suggestions = {
            "Normal_Weight": "Maintain your healthy lifestyle!",
            "Overweight_Level_I": "Consider regular physical activity and better eating habits.",
            "Overweight_Level_II": "Consult a health professional for a personalized plan.",
            "Obesity_Type_I": "Strongly consider medical advice and regular exercise.",
            "Obesity_Type_II": "High risk. Please seek professional help.",
            "Obesity_Type_III": "Severe risk. A medical intervention is highly recommended.",
            "Insufficient_Weight": "Consider a balanced diet to reach healthy weight."
        }
        st.info("💡 Suggestion: " + suggestions.get(label, "Consult a health expert."))

# --- Page 3: Model Performance ---
def show_tab3_performance():
    st.subheader("📈 Model Performance (Example)")
    st.markdown("""
    - **Accuracy**: 85%
    - **Precision**: ~82%
    - **Recall**: ~84%
    - **F1 Score**: ~83%
    """)
    st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png", caption="Sample Confusion Matrix (illustrative)")
    st.markdown("Model: **Logistic Regression** trained on cleaned structured dataset.")

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
    st.title("📊 Obesity Analysis Dashboard")
    tab1, tab2 = st.tabs(["Attributes Distribution", "Obesity Distribution by Factor"])

    with tab1:
        show_attribute_distribution()

    with tab2:
        show_obesity_distribution_analysis()

elif selected == "Obesity Classification":
    st.title("🤖 Obesity Classification")
    show_tab2_prediction()

elif selected == "Model Performance":
    st.title("📈 Model Performance")
    show_tab3_performance()

