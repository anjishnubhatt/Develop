
# 📊 Sales Forecasting & EDA App

This is a Streamlit-based interactive application for sales data analysis and time series forecasting. It combines powerful EDA capabilities with forecasting models like **Prophet** and **SARIMA** to help users gain insights and predict future sales.

---

## 🔍 Features

- 📈 **EDA (Exploratory Data Analysis)**:
  - Time series plots
  - Monthly trends
  - Product movement classification (slow/medium/fast)
  - Sales variation buckets (low/medium/high)

- 📅 **Forecasting Tab**:
  - Facebook Prophet for flexible forecasting
  - SARIMA with auto grid search
  - In-sample evaluation using MAPE
  - 30-day future forecast visualizations

- 🧮 **SQL Query Tab**:
  - Run SQL queries on forecast results
  - Custom filter and slice output

---



### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

Navigate to your project directory and launch the app with:

```bash
streamlit run main.py
```

> Replace `main.py` with your actual Streamlit file name if different.

---

## 📁 Project Structure

```
📦 forecasting-app/
├── main.py
├── eda.py
├── prophet_forecast.py
├── sarima_forecast.py
├── requirements.txt
└── README.md
```
