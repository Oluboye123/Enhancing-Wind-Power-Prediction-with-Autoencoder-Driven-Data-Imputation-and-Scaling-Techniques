# 🌬️ Wind Power Prediction Using Autoencoder-Driven Data Imputation & Machine Learning  
### *Enhancing Renewable Energy Forecasting with AI & Deep Learning*  

## 📌 Project Overview  
This project explores the use of **machine learning and deep learning models** to improve wind power prediction accuracy using **autoencoder-driven data imputation and scaling techniques**. Accurate wind energy forecasting is essential for **renewable energy management and grid stability**.  

By comparing **six regression models**, we investigate how different **feature engineering, data preprocessing, and dimensionality reduction techniques** impact model performance. A novel **autoencoder-based imputation method** was introduced to handle missing values and optimize prediction accuracy.  

---

## 🧐 Key Research Questions  
- How do different machine learning models perform in wind power prediction?  
- Can **autoencoders** improve the accuracy of traditional regression models?  
- What is the impact of **different data imputation and scaling techniques** on prediction errors?  
- How do **deep learning techniques (CNN, LSTM)** compare with traditional ML models?  

---

## 🔬 Methodology  

### 📌 Data Collection  
- **Dataset:** Kaggle’s Wind Power Dataset  
- **Size:** 50,000+ rows, 22 features (meteorological & turbine operational data)  

### 📌 Data Preprocessing  
- **Missing Values Handling:** Mean, Median, Mode Imputation  
- **Scaling Techniques:** Min-Max Scaling, Standardization, Robust Scaling  
- **Feature Selection:** ExtraTrees Regressor for importance ranking  
- **Outlier Removal:** Isolation Forest  

### 📌 Models Used  
- **Machine Learning Models**:  
  - Decision Tree (DT)  
  - Random Forest (RF)  
  - Extreme Gradient Boosting (XGBoost)  
  - Support Vector Regression (SVR)  
- **Deep Learning Models**:  
  - Convolutional Neural Network (CNN)  
  - Long Short-Term Memory (LSTM)  

### 📌 Dimensionality Reduction & Feature Extraction  
- **General Autoencoder (GAE)**  
- **Denoising Autoencoder (DAE)**  
- **Variational Autoencoder (VAE)**  

### 📌 Model Optimization  
- **Hyperparameter Tuning**: GridSearchCV (5-fold cross-validation)  
- **Pipeline Implementation** for automation  
- **Feature Engineering**: One-Hot Encoding, Label Encoding  

---

## 📊 Results & Key Findings  

### 📈 Model Performance (Baseline)  
| Model | Accuracy (%) | MAE | RMSE |
|--------|-------------|------|------|
| **XGBoost** | **92%** | 0.5151 | 0.7777 |
| **Random Forest** | **90%** | 0.5201 | 0.8292 |
| **CNN** | **83%** | 0.7555 | 1.0819 |
| **Decision Tree** | **84%** | 0.6284 | 1.080 |
| **LSTM** | **81%** | 0.8173 | 1.1774 |
| **SVR** | **53%** | 1.8062 | 2.3217 |

✅ **XGBoost had the highest accuracy (92%)**.  
✅ **Random Forest (90%) and CNN (83%) also performed well**.  
✅ **SVR had the worst performance (53%)** but improved with VAE imputation.  

---

### 🏆 Impact of Autoencoder-Based Data Imputation  
| Model + Autoencoder | Accuracy (%) |
|---------------------|-------------|
| **XGBoost + GAE** | **87%** |
| **CNN + GAE** | **81%** |
| **SVR + VAE** | **70%** |
| **LSTM + VAE** | **79%** |
| **Random Forest + VAE** | **86%** |

🚀 **SVR improved from 53% to 70% with VAE**.  
🚀 **XGBoost with General Autoencoder achieved 87% accuracy**.  

---

### ⚖️ Effect of Scaling & Imputation on Model Performance  
| Model | Best Imputation Method | Best Scaling Technique |
|--------|----------------------|----------------------|
| **XGBoost** | Mean Imputation | Min-Max Scaling |
| **Random Forest** | Mean Imputation | Standard Scaling |
| **CNN** | Mean Imputation | Standard Scaling |
| **Decision Tree** | Median Imputation | Standard Scaling |
| **SVR** | Median Imputation | Standard Scaling |
| **LSTM** | Mean Imputation | Min-Max Scaling |

---

## 🔎 Key Insights & Ethical Considerations  
✔️ **Autoencoders improved prediction accuracy, especially for weaker models.**  
✔️ **Deep learning models (CNN, LSTM) require more computational power but capture complex patterns better.**  
✔️ **Bias in data selection can impact energy forecasting accuracy.**  
✔️ **Explainability and transparency of ML models are crucial for real-world applications.**  

---

## 📌 Implications for Renewable Energy  
- **More accurate predictions can improve wind energy grid management.**  
- **Minimizing prediction errors reduces financial losses.**  
- **AI-driven models help optimize renewable energy targets.**  

---

## 🔮 Future Work & Improvements  
🔹 Implement **Deep Ensemble Learning** to combine multiple models for higher accuracy.  
🔹 Apply **Reinforcement Learning (RL)** for adaptive energy forecasting.  
🔹 Develop **AI-powered dashboards** for real-time visualization.  
🔹 Expand dataset with **global wind energy data** for broader applicability.  

## ✍️ Self-Reflection  
### 📌 My Role & Contributions  
✔️ Developed **machine learning & deep learning models** for wind power forecasting.  
✔️ Designed **data preprocessing pipelines** using feature engineering and scaling techniques.  
✔️ Implemented **autoencoder-driven imputation** to enhance model accuracy.  
✔️ Evaluated **model performance using key regression metrics**.  
✔️ Presented recommendations for improving renewable energy forecasting with AI.  

### 📌 Key Learnings from the Project  
✅ **Strengthened expertise in ML/DL models & feature engineering.**  
✅ **Learned hyperparameter tuning techniques for optimal model performance.**  
✅ **Developed structured approaches for model evaluation & comparative analysis.**  
✅ **Improved understanding of AI applications in energy forecasting.**  
