# 🌾 Crop Yield & Fertilizer Prediction System

This is a Streamlit-based web application that predicts the most suitable crop type, expected yield, and recommended fertilizer based on user inputs like soil type, temperature, and rainfall. The app also generates a detailed PDF report with interactive visualizations to assist farmers and decision-makers in making data-driven choices for sustainable agriculture.

## Features

- **User-friendly Streamlit UI** with modern styling and background image support.
- **Machine Learning Models**:
  - Crop Type Prediction (RandomForestClassifier)
  - Yield Prediction (RandomForestRegressor)
  - Fertilizer Recommendation (RandomForestClassifier)
- **Customizable Input Parameters**: Soil type, temperature, rainfall.
- **Interactive Visualizations**: Pie chart, bar graph, and scatter plot based on prediction.
- **PDF Report Generation**: Downloadable report with all results and graphs.
- **Fully customizable for your own dataset.**

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/swanand67/crop_app.git
cd crop_app
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

**Required Packages:**
- streamlit
- pandas
- matplotlib
- scikit-learn
- reportlab
- numpy

### 3. Prepare Your Dataset

The app expects an Excel file (`modified_crop_yield_dataset_v2.xlsx`) with the following columns:

- `Soil_type`
- `Temperature`
- `Rainfall`
- `Crop_Type`
- `Yield`
- `Fertilizer`

Update the `file_path` variable in `app.py` to point to your dataset's location.

### 4. Add a Background Image

Update the path in `add_bg_and_styles` in `app.py` to use your own image file.

### 5. Run the Application

```bash
streamlit run app.py
```

## Usage

1. Enter the temperature (°C), rainfall (mm), and select the soil type.
2. Click **Predict**.
3. View prediction results, graphs, and download the full PDF report.

## Screenshots

![UI Screenshot](assets/ui_screenshot.png)

## Project Structure

```
crop_app/
│
├── app.py                # Main Streamlit application
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── assets/
│   └── ui_screenshot.png # Example screenshot
└── modified_crop_yield_dataset_v2.xlsx # Your input dataset (not included)
```

## Customization

- **Model Parameters**: Tune the random forest hyperparameters in `app.py`.
- **Input Columns**: Modify for additional parameters as needed.
- **Styling**: Update CSS in `add_bg_and_styles` for different themes.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Author

Developed by [swanand67](https://github.com/swanand67).

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [ReportLab](https://www.reportlab.com/)
