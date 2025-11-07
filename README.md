
# 📘 Electric Vehicle (EV) Cost Prediction using Machine Learning

This project predicts the **Base MSRP (Manufacturer Suggested Retail Price)** of electric vehicles (EVs) using machine learning.  
The model leverages key features such as battery range, brand, model year, and efficiency metrics to estimate the cost of an EV accurately.

The workflow includes:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model building and evaluation  
- Saving a reusable prediction pipeline  

---

## ⚙️ Tech Stack
- **Language:** Python 3.x  
- **Platform:** Google Colab
- **Libraries:**  
  - pandas, numpy, matplotlib, seaborn  
  - scikit-learn  
  - joblib  

---

### 📁 Dataset Access
The dataset used in this project can be downloaded from:

[🔗EV Data ](https://www.kaggle.com/datasets/rithurajnambiar/electric-vehicle-data?utm_source=chatgpt.com)

---

## 📂 Dataset Description

| Column | Description |
|--------|--------------|
| Make | Vehicle brand (e.g., Tesla, Nissan, BMW) |
| Model | Model name of the EV |
| Model Year | Year of manufacturing |
| Electric Vehicle Type | BEV (Battery Electric Vehicle) or PHEV (Plug-in Hybrid) |
| Electric Range | Driving range on a full charge |
| State, City | Registration or sales region |
| Base MSRP | Price of the vehicle |
| Vehicle Location, Electric Utility | Removed as irrelevant columns |

---

## 🧹 Data Cleaning & Preprocessing Steps
1. **Dropped irrelevant columns:** Vehicle Location, Electric Utility  
2. **Feature Engineering:**  
   - EV_Age = 2025 - Model Year  
   - Price_per_Mile = Base MSRP / Electric Range  
3. **Outlier Handling:**  
   - Removed rows with Base MSRP <= 0  
   - Trimmed top 0.1% extreme prices  
4. **Cardinality Reduction:**  
   - Kept top-10 Makes and top-20 Cities  
5. **Encoding:**  
   - Frequency encoding for Model  
   - One-Hot Encoding for categorical features  
6. **Scaling:** Standardized numeric features  
7. **Train-Test Split:** 80% training, 20% testing  

---

## 📊 Exploratory Data Analysis (EDA)
Performed several visualizations:
- Histogram — Base MSRP  
- Histogram — Electric Range  
- Scatter Plot — Base MSRP vs Range  
- Boxplot — Price by EV Type  
- Bar chart — Top 10 Makes  
- Correlation Heatmap & Pairplot  

---

## 🤖 Modeling
**Algorithm:** RandomForestRegressor (n_estimators=200)

**Pipeline:**
```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])
```

**Features Used:**
['Make_top', 'Model_freq', 'Model Year', 'Electric Vehicle Type',
 'Electric Range', 'State', 'City_top', 'EV_Age', 'Price_per_Mile']

---

## 📈 Model Evaluation
| Metric | Score | Interpretation |
|--------|--------|----------------|
| MAE | 54.15 | Average error of $54 per prediction |
| RMSE | 961.54 | Small deviation considering EV prices range from $20k–$100k |
| R² | 0.9976 | Model explains 99.76% of variance in price |

---

## 🔍 Top 10 Feature Importances
| Rank | Feature | Importance | Meaning |
|------|----------|-------------|----------|
| 1 | Model Year | 0.324 | Newer cars cost more |
| 2 | Price_per_Mile | 0.306 | Efficiency indicator |
| 3 | EV_Age | 0.258 | Age-depreciation effect |
| 4 | Model_freq | 0.068 | Model popularity trend |
| 5 | Make_top_CADILLAC | 0.019 | Brand premium |
| 6 | Make_top_TESLA | 0.015 | Brand premium |
| 7 | Electric Range | 0.006 | Range increases cost |
| 8 | Make_top_VOLVO | 0.001 | Brand effect |
| 9 | City_top_OLYMPIA | 0.001 | Minor regional influence |
| 10 | EV Type (PHEV) | 0.0002 | Plug-in hybrids slightly cheaper |

---

## 💾 Model Saving
Both **model** and **preprocessor** were saved as:
```
ev_price_model_and_preprocessor.joblib
```

Usage example:
```python
import joblib
bundle = joblib.load("ev_price_model_and_preprocessor.joblib")
model = bundle['model']
preprocessor = bundle['preprocessor']

# Predict
X_new_prep = preprocessor.transform(X_new)
predicted_price = model.predict(X_new_prep)
```

---

## 🔮 Future Improvements
- Add battery capacity, charging speed, efficiency  
- Adding AI ChatBot
- Create Streamlit / Flask app  
- Apply SHAP explainability  

---
