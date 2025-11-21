#  Racing for the 'Score' Across Linear and Non-Linear Models

> A comprehensive data science project predicting student academic performance using machine learning techniques 

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-green)
![License](https://img.shields.io/badge/License-MIT-blue)

--- 
### Documentation link :[ https://data-pageup.github.io/LR/](https://data-pageup.github.io/Score-Across-Linear-and-Non-Linear-Models/)
---

##  Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results & Performance](#results--performance)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

##  Overview

This project develops a predictive model for student academic performance by analyzing 80,000 synthetically generated student records. By leveraging both linear and non-linear machine learning algorithms, we identify key factors influencing exam scores and provide actionable insights for educational institutions.

The study demonstrates that **previous academic performance** is the strongest predictor of exam scores (correlation: 0.93), alongside behavioral and environmental factors like motivation level, study hours, and stress levels.

---

##  Key Features

- **Comprehensive EDA**: In-depth exploratory analysis with 47 visualizations
- **Advanced Feature Engineering**: Statistical significance testing for numerical and categorical variables
- **Multiple Algorithms**: Implementation of 5 different regression models
- **Assumption Validation**: Rigorous checking of linear regression assumptions
- **Model Comparison**: Detailed performance metrics across all models
- **Production-Ready**: Clean code structure with proper preprocessing pipeline

---

##  Dataset

**Source**: Kaggle (Synthetically Generated)  
**Size**: 80,000 student records  
**Features**: 31 attributes across 5 categories
 
### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Academic & Study** | 9 | study_hours_per_day, attendance_percentage, previous_gpa |
| **Psychological & Health** | 7 | mental_health_rating, stress_level, exam_anxiety_score |
| **Lifestyle & Demographics** | 7 | age, gender, part_time_job, learning_style |
| **Family & Support** | 3 | parental_education_level, family_income_range |
| **Technology & Screen** | 4 | social_media_hours, netflix_hours, internet_quality |

---

## 📁 Project Structure

```
project/
├── data/
│   └── student_performance_dataset.csv (80,000 × 31 columns)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_building.ipynb
│   └── 05_assumption_testing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   ├── visualizations/
│   └── model_performance.csv
└── README.md
```

---

## 🔬 Methodology

### 1. **Data Cleaning** 
- Removed irrelevant identifier columns (student_id)
- Verified data completeness: 0 missing values
- Dataset quality: 100% ready for analysis

### 2. **Exploratory Data Analysis** 
- Univariate analysis on all 30 features
- Correlation heatmap: identified 0.93 correlation with previous_gpa
- Distribution analysis: balanced dataset across all major categories
- Outlier detection: minimal outliers except in study_hours_per_day

### 3. **Feature Selection** 
**Numerical Variables** (Pearson Correlation Test):
- 9 significant features identified (p < 0.05)
- Top predictors: previous_gpa, motivation_level, study_hours_per_day

**Categorical Variables** (Chi-Square Test):
- 4 significant features identified (p < 0.05)
- Most impactful: access_to_tutoring (χ² = 420.48), dropout_risk (χ² = 332.68)

### 4. **Data Preprocessing**
- Label encoding for binary categorical variables
- One-hot encoding for multi-class features
- Standard scaling: normalized numerical features (mean=0, std=1)
- **Final dataset**: 80,000 × 21 features

### 5. **Model Building & Evaluation**
- Train-test split: 80-20 ratio with random_state=42
- 5 models trained and compared
- Performance metrics: R² Score, RMSE, MAE

---

##  Models Implemented

### Linear Models

| Model | R² Score | RMSE | MAE | Key Advantage |
|-------|----------|------|-----|---------------|
| **Multiple Linear Regression** | 0.8705 | 0.1305 | 0.2757 | Interpretability |
| **Ridge Regression** | 0.8705 | 0.1305 | 0.2757 | Multicollinearity handling |
| **Lasso Regression** | 0.8703 | 0.1307 | 0.2769 | Feature selection |

### Non-Linear Models

| Model | R² Score | RMSE | MAE | Key Advantage |
|-------|----------|------|-----|---------------|
| **Random Forest** | 0.8669 | 0.1341 | 0.2862 | Robustness |
| **Support Vector Regressor (RBF)** | 0.8663 | 0.1347 | 0.2760 | Non-linear mapping |

---

##  Results & Performance

### Model Ranking
```
🥇 Linear Regression & Ridge Regression:    R² = 0.8705
🥈 Lasso Regression:                        R² = 0.8703
🥉 Random Forest:                            R² = 0.8669
4️⃣  SVR (RBF):                              R² = 0.8663
```

### Key Insights
- **Linear models outperform** non-linear models by ~0.4% due to predominantly linear relationships
- **Previous GPA** explains 87% of exam score variance (r = 0.93)
- **Motivation level** shows moderate positive correlation (r = 0.25)
- **Exam anxiety** negatively impacts scores (r = -0.24)

### Assumption Testing Results
✅ **Linearity**: Confirmed through residual plots  
✅ **Autocorrelation**: Durbin-Watson ≈ 2.0 (no autocorrelation)  
⚠️ **Homoscedasticity**: Minor deviations detected  
⚠️ **Normality**: Slight non-normality in residuals (Shapiro-Wilk p < 0.001)  
⚠️ **Multicollinearity**: Moderate (VIF: motivation_level=6.55)

---

### Dependencies

```
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
statsmodels==0.13.5
missingno==0.5.1
```

---

##  Usage

### Quick Start

```python
# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load and preprocess data
import pandas as pd
df = pd.read_csv('student_performance_dataset.csv')

# Prepare features and target
X = df.drop('exam_score', axis=1)
y = df['exam_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
```

### Running Full Pipeline

```bash
# Execute all analyses
jupyter notebook notebooks/01_data_cleaning.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_feature_selection.ipynb
jupyter notebook notebooks/04_model_building.ipynb
jupyter notebook notebooks/05_assumption_testing.ipynb
```

---

## 🔍 Key Findings

### 📌 Top Predictors of Exam Score

1. **Previous GPA** (r = 0.93) - Strongest indicator of current performance
2. **Motivation Level** (r = 0.25) - Moderate positive impact
3. **Study Hours/Day** (r = 0.24) - Consistent study habits matter
4. **Screen Time** (r = 0.17) - Time management indicator

### ⚠️ Negative Factors

1. **Exam Anxiety** (r = -0.24) - Highest negative impact
2. **Stress Level** (r = -0.12) - Inverse relationship with performance
3. **Study Environment** - Dorm settings negatively correlated

### 💡 Actionable Insights

- Educational institutions should focus on supporting **high-anxiety students**
- **Mentoring programs** targeting low-motivation students can improve outcomes
- Optimal study hour range: **2-5.5 hours/day** (where most high performers study)
- Access to tutoring shows positive association with dropout prevention
- Mental health support directly correlates with academic success

---

## 🔮 Future Enhancements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Deploy model as REST API
- [ ] Create interactive dashboard for predictions
- [ ] Time-series analysis for semester progression
- [ ] Feature importance visualization with SHAP values
- [ ] Cross-validation for robust performance metrics
- [ ] Hyperparameter tuning (GridSearchCV, RandomSearchCV)
- [ ] Integration with institutional data systems
- [ ] Real-time prediction pipeline
- [ ] Student intervention recommendation system

---

## 👥 Contributors

Developed by a passionate data science team focused on educational analytics.



---

## 📚 References

- Kaggle: Student Performance Dataset
- Scikit-Learn Documentation: https://scikit-learn.org
- Statsmodels Documentation: https://www.statsmodels.org
- Statistical Testing: SciPy Documentation

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Support

For questions, issues, or suggestions, please contact the project team or open an issue in the repository.

**Happy Predicting! 🚀**

--- 

<div align="center">

**Made with ❤️ by the Data Science Team**

*"Racing for the Score - Predicting Academic Excellence Through Data"*

</div>
