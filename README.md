# Customer Churn Prediction using Machine Learning


## Project Overview

This project develops a machine learning model to predict customer churn (attrition) for a telecommunications company. The model helps identify high-risk customers, enabling proactive retention strategies to reduce revenue loss.

**Key Features:**
- Random Forest classifier with 76.5% accuracy
- Handles class imbalance (26.5% churn rate)
- Identifies key churn drivers (tenure, contract type, service usage)
- Production-ready model with prediction function

## Business Problem

Customer churn significantly impacts revenue in competitive markets. This solution:
- Predicts which customers are likely to churn
- Provides probabilities for risk assessment
- Highlights important churn factors for targeted interventions

## Dataset

The dataset contains 7,043 customer records with 21 features including:
- **Demographics**: Gender, senior citizen status, partner/dependents
- **Account details**: Tenure, contract type, payment method
- **Service usage**: Phone/internet service, online security
- **Charges**: Monthly and total amounts
- **Target**: Churn (Yes/No)

**Preprocessing Steps:**
- Handled missing values in TotalCharges
- Encoded categorical variables (one-hot & binary)
- Final feature set: 30 numeric columns

## Methodology

### Model Development
- **Algorithm**: Random Forest (handles mixed features, non-linear relationships)
- **Hyperparameter Tuning**: Grid search with 5-fold CV
- **Class Weight**: Balanced to address class imbalance

**Best Parameters:**
```python
{
    'class_weight': 'balanced',
    'max_depth': 10,
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 100
}

```

### Performance Metrics
| Metric       | No Churn | Churn | Weighted Avg |
|--------------|----------|-------|--------------|
| Precision    | 0.90     | 0.54  | 0.80         |
| Recall       | 0.77     | 0.76  | 0.77         |
| F1-score     | 0.83     | 0.63  | 0.78         |
| Accuracy     | 0.7651   |       |              |

## Key Findings

**Top Predictive Features:**
1. Tenure (18.3%)
2. TotalCharges (14.2%)
3. Two-year contract (9.3%)
4. MonthlyCharges (9.3%)
5. Fiber optic internet service (7.6%)

**Business Insights:**
- Longer contracts reduce churn risk
- Fiber optic customers more likely to churn
- Electronic check payments correlate with higher churn

## Repository Structure

```
customer-churn-prediction/
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for analysis
│   └── customer_churn_prediction_using_ML.ipynb  
├── models/                  # Saved models
├── gradio-app/              
│   └── app.py               
└── README.md 
```


## Future Improvements

1. Address class imbalance with SMOTE
2. Experiment with XGBoost and Neural Networks
3. Incorporate temporal behavior patterns
4. Combine with customer lifetime value model

## Tools Used

- Python 3.8+
- pandas (data manipulation)
- scikit-learn (modeling)
- joblib (model persistence)
- Jupyter Notebook (development)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
