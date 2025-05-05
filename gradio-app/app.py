import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and required files
model = joblib.load('customer_churn_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Title and description
title = "Customer Churn Prediction"
description = """
### Problem Statement:
By predicting customer churn, the company can proactively design retention strategies to
keep these customers, thereby improving customer satisfaction and reducing financial loss.
"""

# Sample test case
sample_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, 
                 PhoneService, MultipleLines, InternetService, OnlineSecurity,
                 OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                 StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                 MonthlyCharges, TotalCharges):
    
    # Create input dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': int(SeniorCitizen),
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    try:
        # Convert to DataFrame
        test_df = pd.DataFrame([input_data])

        # Preprocess the test data
        test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce').fillna(0)

        # Convert binary categorical variables
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            test_df[col] = test_df[col].map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})

        # One-hot encode categorical variables
        test_df = pd.get_dummies(test_df)

        # Ensure all columns from training are present
        for col in model_columns:
            if col not in test_df.columns:
                test_df[col] = 0

        # Reorder columns to match training
        test_df = test_df[model_columns]

        # Make prediction
        prediction = model.predict(test_df)
        probability = model.predict_proba(test_df)[:, 1]

        return {
            "Prediction": "Yes" if prediction[0] == 1 else "No",
            "Churn Probability": f"{probability[0]:.2%}",
            "Recommendation": "Offer retention package" if prediction[0] == 1 else "Customer is stable"
        }
        
    except Exception as e:
        return {"Error": str(e)}

with gr.Blocks(title=title) as demo:
    # Header with title and right-aligned buttons
    with gr.Row():
        gr.Markdown(f"# {title}")
        # with gr.Column(min_width=200):
        #     with gr.Row():
        #         dataset_btn = gr.Button("Dataset", link="https://example.com/dataset")
        #         github_btn = gr.Button("GitHub", link="https://github.com/example/repo")
    
    gr.Markdown(description)
    
    # Input sections
    with gr.Row():
        with gr.Column():
            # Customer demographics
            gr.Markdown("### Customer Demographics")
            gender = gr.Radio(["Female", "Male"], value=sample_data['gender'], label="Gender")
            SeniorCitizen = gr.Checkbox(value=sample_data['SeniorCitizen'], label="Senior Citizen")
            Partner = gr.Radio(["Yes", "No"], value=sample_data['Partner'], label="Partner")
            Dependents = gr.Radio(["Yes", "No"], value=sample_data['Dependents'], label="Dependents")
            tenure = gr.Number(value=sample_data['tenure'], label="Tenure (months)")
            
        with gr.Column():
            # Services
            gr.Markdown("### Services")
            PhoneService = gr.Radio(["Yes", "No"], value=sample_data['PhoneService'], label="Phone Service")
            MultipleLines = gr.Dropdown(
                ["No phone service", "No", "Yes"], 
                value=sample_data['MultipleLines'], 
                label="Multiple Lines"
            )
            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"], 
                value=sample_data['InternetService'], 
                label="Internet Service"
            )
            OnlineSecurity = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['OnlineSecurity'], 
                label="Online Security"
            )
            OnlineBackup = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['OnlineBackup'], 
                label="Online Backup"
            )
            
        with gr.Column():
            # Additional services
            gr.Markdown("### Additional Services")
            DeviceProtection = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['DeviceProtection'], 
                label="Device Protection"
            )
            TechSupport = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['TechSupport'], 
                label="Tech Support"
            )
            StreamingTV = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['StreamingTV'], 
                label="Streaming TV"
            )
            StreamingMovies = gr.Dropdown(
                ["No", "Yes", "No internet service"], 
                value=sample_data['StreamingMovies'], 
                label="Streaming Movies"
            )
            
    with gr.Row():
        with gr.Column():
            # Contract and payment
            gr.Markdown("### Contract & Payment")
            Contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"], 
                value=sample_data['Contract'], 
                label="Contract"
            )
            PaperlessBilling = gr.Radio(
                ["Yes", "No"], 
                value=sample_data['PaperlessBilling'], 
                label="Paperless Billing"
            )
            PaymentMethod = gr.Dropdown(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], 
                value=sample_data['PaymentMethod'], 
                label="Payment Method"
            )
            
        with gr.Column():
            # Charges
            gr.Markdown("### Charges")
            MonthlyCharges = gr.Number(
                value=sample_data['MonthlyCharges'], 
                label="Monthly Charges ($)"
            )
            TotalCharges = gr.Number(
                value=sample_data['TotalCharges'], 
                label="Total Charges ($)"
            )
    
    # Submit button
    submit_btn = gr.Button("Predict Churn", variant="primary")
    
    # Output
    output = gr.JSON(label="Prediction Results")
    
    # Footer
    gr.Markdown("Thank you for using our prediction service! 😊")
    
    # Connect the button to the function
    submit_btn.click(
        fn=predict_churn,
        inputs=[gender, SeniorCitizen, Partner, Dependents, tenure, 
               PhoneService, MultipleLines, InternetService, OnlineSecurity,
               OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
               StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
               MonthlyCharges, TotalCharges],
        outputs=output
    )

demo.launch(share=True)