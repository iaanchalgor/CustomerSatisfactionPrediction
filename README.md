Customer Satisfaction Prediction
Overview

This project aims to predict Customer Satisfaction Ratings based on support ticket information using machine learning techniques. The goal is to help companies improve customer experience by analyzing ticket data and predicting satisfaction levels.

Dataset Information

Rows: 8469

Columns: 16

Key Columns:

Ticket ID, Ticket Description, Resolution, Ticket Priority, Ticket Channel, Product Purchased, First Response Time, Time to Resolution, Customer Satisfaction Rating

Missing values were handled using median imputation for numerical features and cosine similarity for text-based Resolution data.

Project Workflow

Data Preprocessing

Handling missing values (median imputation, cosine similarity for Resolution)

Date-time conversion for response and resolution times

Encoding categorical variables using One-Hot Encoding

Exploratory Data Analysis (EDA)

Distribution of resolution time and response time

Correlation analysis for numerical features

WordCloud for Ticket Descriptions

Feature Engineering

TF-IDF for text features

Derived features: Resolution_Delay, Response_Speed, Resolution_Speed, Description_Length, Purchase_Weekday, etc.

Model Building
Models Trained:

Random Forest Classifier

Gradient Boosting

XGBoost (Optional)

Hyperparameter Tuning

GridSearchCV used for tuning parameters like n_estimators, max_depth, min_samples_split, and min_samples_leaf.

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Model Results Summary
Model	Accuracy	Key Findings
Random Forest (Base)	0.54	Initial model with default parameters
Gradient Boosting	0.33	Tuned model, moderate performance
Random Forest (Tuned)	0.65	Best performing model with GridSearchCV

Confusion Matrix and classification reports were used for detailed analysis.

Tools & Libraries Used

Python 3.x

pandas, numpy, matplotlib, seaborn (Data Analysis & Visualization)

scikit-learn (ML Models, Preprocessing, Hyperparameter Tuning)

xgboost (Optional Model)

wordcloud (For text visualization)

Future Improvements

Use advanced NLP models (BERT, RoBERTa) for ticket description analysis

Ensemble methods for better accuracy

Deploy model using Flask or FastAPI for real-time predictions
