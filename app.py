# ==============================================
# Crop Yield & Fertilizer Prediction - Streamlit UI + Report PDF
# ==============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import base64, io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ----------------------------
# Load dataset
# ----------------------------
file_path = r"C:\Users\swana\Downloads\modified_crop_yield_dataset_v2.xlsx"
df = pd.read_excel(file_path)

X = df[["Soil_type", "Temperature", "Rainfall"]]
y_crop = df["Crop_Type"]
y_yield = df["Yield"]
y_fertilizer = df["Fertilizer"]

categorical_cols = ["Soil_type"]
numerical_cols = ["Temperature", "Rainfall"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

crop_classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

yield_regressor = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

fertilizer_classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train models
X_train, X_test, y_train_crop, y_test_crop = train_test_split(X, y_crop, test_size=0.2, random_state=42)
crop_classifier.fit(X_train, y_train_crop)

X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)
yield_regressor.fit(X_train_y, y_train_y)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
fertilizer_classifier.fit(X_train_f, y_train_f)

# ----------------------------
# Background & Styling
# ----------------------------
def add_bg_and_styles(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        .title-text {{
            font-size: 34px;
            font-weight: 800;
            color: #FFFFFF;
            text-align: center;
        }}
        .subtitle-text {{
            font-size: 18px;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 25px;
        }}
        div[data-baseweb="input"] input {{
            background-color: #ffffff !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: #ffffff !important;
            border-radius: 8px !important;
        }}
        .stNumberInput label, .stSelectbox label {{
            color: #FFFFFF !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }}
        .result-card {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 25px;
            border-radius: 15px;
            margin-top: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        .result-card * {{
            color: #FFFFFF !important;
        }}
        .input-params-section {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 25px;
            border-radius: 15px;
            margin-top: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        .input-params-section * {{
            color: #FFFFFF !important;
        }}
        .report-section {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            color: #FFFFFF !important;
        }}
        .report-section * {{
            color: #FFFFFF !important;
        }}
        .report-header {{
            color: #FFFFFF !important;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .report-subheader {{
            color: #FFFFFF !important;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .stMetric {{
            color: #FFFFFF !important;
        }}
        .stCaption, .stCaption * {{
            color: #FFFFFF !important;
            font-weight: bold !important;
            text-align: center !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply styles
add_bg_and_styles(r"C:\Users\swana\OneDrive\Desktop\crp.jpg")

# ----------------------------
# Generate Graphs Function
# ----------------------------
def generate_graphs(predicted_crop):
    figs = []
    crop_data = df[df["Crop_Type"] == predicted_crop]

    if len(crop_data) > 0:
        # 1. Pie chart - Rainfall distribution for crop
        fig1, ax1 = plt.subplots(figsize=(6, 5), facecolor="white")
        rainfall_counts = crop_data["Rainfall"].value_counts().head(5)
        ax1.pie(rainfall_counts.values, labels=rainfall_counts.index,
                autopct="%1.1f%%", startangle=90)
        ax1.set_title(f"{predicted_crop}: Rainfall Distribution (Top 5 values)")
        figs.append(fig1)

        # 2. Bar graph - Soil type vs crop count
        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor="white")
        soil_counts = crop_data["Soil_type"].value_counts()
        soil_counts.plot(kind="bar", ax=ax2, color="skyblue")
        ax2.set_title(f"{predicted_crop}: Soil Type Distribution")
        ax2.set_xlabel("Soil Type")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        figs.append(fig2)

        # 3. Scatter plot - Fertilizer vs Yield
        fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor="white")
        ax3.scatter(crop_data["Fertilizer"], crop_data["Yield"], alpha=0.6, c="green")
        ax3.set_title(f"{predicted_crop}: Fertilizer vs Yield")
        ax3.set_xlabel("Fertilizer")
        ax3.set_ylabel("Yield")
        ax3.tick_params(axis='x', rotation=45)
        figs.append(fig3)
    else:
        for i in range(3):
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
            ax.text(0.5, 0.5, f'No data available for {predicted_crop}',
                    ha='center', va='center', fontsize=14)
            figs.append(fig)

    return figs

# ----------------------------
# Generate PDF Report Function
# ----------------------------
def generate_pdf_report(predicted_crop, predicted_yield, recommended_fertilizer,
                       temperature, rainfall, soil_type, img_buffers):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("🌾 Crop Prediction Report", styles["Title"]))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("This report is generated based on your inputs. "
                           "The system predicts the most suitable crop, expected yield, and recommended fertilizer."
                           " In addition to predicting the expected yield, it also recommends the appropriate fertilizer that can help improve productivity."
                           "The purpose of this report is to assist farmers and decision-makers in making data-driven choices for better crop management and sustainable farming practices.", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Machine learning models (Random Forest) analyze soil type, temperature, and rainfall "
                           "to assist farmers in decision-making."
                           " These models work by combining multiple decision trees to analyze the complex relationships between soil type, temperature, and rainfall. "
                           " Instead of relying on a single model, Random Forest aggregates the results of many models, which significantly improves both accuracy and stability. "
                           "By considering environmental and soil parameters, the model identifies patterns in agricultural data that would otherwise be difficult to detect manually. "
                           "For instance, it can capture subtle variations in soil fertility or rainfall distribution that directly influence crop productivity. "
                           "Such predictive capability empowers farmers to make proactive decisions about which crops are best suited for their land. "
                            "Furthermore, the model evaluates historical data to recommend fertilizers that align with both soil conditions and the chosen crop, "
                            "reducing the chances of overuse or misuse of chemical inputs. "
                            "This approach not only improves yield outcomes but also strengthens food security by aligning production with resource availability. "
                            "In essence, the application of machine learning in agriculture bridges the gap between traditional farming practices and modern data science innovations. "
                            "Through this predictive system, farmers are empowered with tools to maximize efficiency, reduce uncertainty, and contribute to long-term agricultural sustainability.", styles["Normal"]))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(f"🌱 Predicted Crop Type: <b>{predicted_crop}</b>", styles["Normal"]))
    story.append(Paragraph(f"📊 Predicted Yield: <b>{round(predicted_yield,2)} tons/hectare</b>", styles["Normal"]))
    story.append(Paragraph(f"💊 Recommended Fertilizer: <b>{recommended_fertilizer}</b>", styles["Normal"]))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Input Parameters:", styles["Heading2"]))
    story.append(Paragraph(f"🌡️ Temperature: {temperature}°C", styles["Normal"]))
    story.append(Paragraph(f"🌧️ Rainfall: {rainfall} mm", styles["Normal"]))
    story.append(Paragraph(f"🟤 Soil Type: {soil_type}", styles["Normal"]))
    story.append(Spacer(1, 0.3*inch))

    graph_titles = [f"{predicted_crop}: Rainfall Distribution",
                   f"{predicted_crop}: Soil Type Distribution",
                   f"{predicted_crop}: Fertilizer vs Yield"]
    for i, (buf, title) in enumerate(zip(img_buffers, graph_titles), start=1):
        story.append(Paragraph(f"Graph {i}: {title}", styles["Heading3"]))
        story.append(RLImage(buf, width=400, height=250))
        story.append(Spacer(1, 0.2*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Crop Prediction System", layout="centered")

st.markdown('<div class="title-text">🌾 Crop Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Predict Crop Type, Yield, and Fertilizer Recommendation</div>', unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    with col2:
        soil_type = st.selectbox("🟤 Soil Type", df["Soil_type"].unique())
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'pdf_buffer' not in st.session_state:
    st.session_state.pdf_buffer = None

if submitted:
    user_input = pd.DataFrame([{
        "Soil_type": soil_type,
        "Temperature": temperature,
        "Rainfall": rainfall
    }])

    predicted_crop = crop_classifier.predict(user_input)[0]
    predicted_yield = yield_regressor.predict(user_input)[0]
    recommended_fertilizer = fertilizer_classifier.predict(user_input)[0]

    # Display Input Parameters Section (with white font)
    st.markdown(
        f"""
        <div class="input-params-section">
            <h2>📋 Input Parameters</h2>
            <p>🌡️ <b>Temperature:</b> {temperature}°C</p>
            <p>🌧️ <b>Rainfall:</b> {rainfall} mm</p>
            <p>🟤 <b>Soil Type:</b> {soil_type}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display Prediction Results
    st.markdown(
        f"""
        <div class="result-card">
            <h2>✅ Prediction Results</h2>
            <p>🌱 <b>Predicted Crop Type:</b> {predicted_crop}</p>
            <p>📊 <b>Predicted Yield:</b> {round(predicted_yield, 2)} tons/hectare</p>
            <p>💊 <b>Recommended Fertilizer:</b> {recommended_fertilizer}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    figs = generate_graphs(predicted_crop)

    img_buffers = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_buffers.append(buf)

    st.session_state.pdf_buffer = generate_pdf_report(
        predicted_crop, predicted_yield, recommended_fertilizer,
        temperature, rainfall, soil_type, img_buffers
    )
    st.session_state.report_generated = True
    st.session_state.predicted_crop = predicted_crop
    st.session_state.predicted_yield = predicted_yield
    st.session_state.recommended_fertilizer = recommended_fertilizer
    st.session_state.temperature = temperature
    st.session_state.rainfall = rainfall
    st.session_state.soil_type = soil_type
    st.session_state.figs = figs

if st.session_state.report_generated:
    figs = st.session_state.figs
    st.markdown("---")
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown('<div class="report-header">📄 Crop Prediction Report</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(figs[0])
        st.markdown('<p class="stCaption">Rainfall Distribution</p>', unsafe_allow_html=True)
        st.pyplot(figs[1])
        st.markdown('<p class="stCaption">Soil Type Distribution</p>', unsafe_allow_html=True)
    with col2:
        st.pyplot(figs[2])
        st.markdown('<p class="stCaption">Fertilizer vs Yield</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Full Report (PDF)",
        data=st.session_state.pdf_buffer,
        file_name="crop_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )