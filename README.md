### **README for Gold Price Prediction Project**  

# **Gold Price Prediction Using Machine Learning**  
This project implements **machine learning models** to accurately predict **gold price trends** based on historical financial indicators. It compares multiple regression algorithms and evaluates their effectiveness in forecasting gold prices.

---

## **Project Overview**
Gold prices are influenced by various economic factors such as **stock market trends, currency exchange rates, inflation rates, and crude oil prices**. Traditional forecasting methods struggle to capture these dependencies, making **machine learning an effective solution** for gold price prediction.

The project employs **four key regression models**:
✅ **Linear Regression** – Establishes a basic trend relationship.  
✅ **Support Vector Regression (SVR)** – Captures complex non-linear dependencies.  
✅ **Random Forest Regressor** – Uses multiple decision trees for robust predictions.  
✅ **XGBoost Regressor** – A powerful boosting algorithm for optimized forecasting.  

📌 **Final Evaluation:** XGBoost achieved the highest accuracy (R² ≈ 98.6%), proving most effective for gold price prediction.

---

## **Dataset**
The dataset consists of structured financial indicators, including:
- **Gold price trends (Target Variable)**
- **Stock market indices (S&P 500, Dow Jones)**
- **USD exchange rate fluctuations**
- **Crude oil prices & inflation rates**
- **Interest rates & global financial events**

📌 **Data Source:** Kaggle, Yahoo Finance, and Federal Reserve Economic Data (FRED).

---

## **System Requirements**
### **Software & Libraries**
- **Python** (Programming Language)
- **Jupyter Notebook / Google Colab** (Development Environment)
- **Libraries Used:**
  - `pandas` – Data manipulation
  - `numpy` – Numerical computations
  - `matplotlib & seaborn` – Data visualization
  - `scikit-learn` – Machine learning models
  - `xgboost` – Advanced boosting algorithm

### **Hardware Requirements**
- **Minimum 8GB RAM** for efficient data processing
- **Intel Core i5/i7 or AMD Ryzen 5/7** for model training
- **GPU (Optional)** – Enhances performance for large datasets

---

## **Project Workflow**
### **1️⃣ Data Preprocessing**
✅ **Handling Missing Values** – Impute missing data using statistical methods.  
✅ **Outlier Detection & Removal** – Boxplot & IQR method applied.  
✅ **Feature Scaling** – Applied `StandardScaler()` for normalization.  
✅ **Correlation Analysis** – Identified impactful features using heatmaps.

### **2️⃣ Model Training & Evaluation**
✅ **Splitting Data:** 80% training, 20% testing (`train_test_split`).  
✅ **Cross-Validation:** 5-fold validation ensures model reliability.  
✅ **Hyperparameter Tuning:** Used `GridSearchCV` for optimization.  

```python
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for XGBoost
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(xgb.XGBRegressor(objective="reg:squarederror"), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_xgb = grid_search.best_estimator_
```

### **3️⃣ Prediction & Model Validation**
✅ **Comparison Plot:** **Actual vs. Predicted Gold Prices** to visualize accuracy.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Gold Prices", marker="o")
plt.plot(best_xgb.predict(X_test_scaled), label="Predicted Gold Prices", linestyle="dashed", marker="s")

plt.xlabel("Time")
plt.ylabel("Gold Price")
plt.title("Actual vs. Predicted Gold Prices")
plt.legend()
plt.show()
```

📌 **XGBoost delivered the most reliable predictions with minimal deviation.**

---

## **Results & Insights**
✅ **XGBoost model achieved an R² Score of ~98.6%, making it the best model for forecasting.**  
✅ **Actual vs. predicted price trends show minimal deviations, confirming strong predictive reliability.**  
✅ **Gold price movements align with macroeconomic trends, reinforcing model accuracy.**  

---

## **Future Enhancements**
🚀 **Integrating real-time financial APIs** for dynamic updates.  
🚀 **Exploring deep learning models (LSTM)** for advanced time-series forecasting.  
🚀 **Refining hyperparameters using Bayesian optimization** for improved accuracy.  

---

## **References**
📚 **Gold Price Prediction Using Machine Learning Techniques** – [Read Here](https://www.ijraset.com/research-paper/gold-price-prediction-using-machine-learning-techniques)  
📚 **Gold Price Forecasting Using Machine Learning** – [Read Here](https://link.springer.com/chapter/10.1007/978-3-031-82383-1_22)  
📚 **Kaggle Gold Price Dataset** – [Access Here](https://www.kaggle.com/)  
📚 **Yahoo Finance API** – [Live Financial Data](https://www.yahoofinance.com/)  

---

### **GitHub Repository**
🔗 **GitHub Link:** [https://github.com/UROOJR1/GOLD-PRICE-PREDICTION]  

