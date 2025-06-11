# Classical ML - Dataset Selection Guide

This guide provides curated datasets suitable for your Classical ML milestone project, organized by domain and learning objective.

## Recommended Datasets by Domain

### Finance & Economics
1. **Credit Card Fraud Detection**
   - Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
   - Skills: Imbalanced classification, anomaly detection, feature engineering
   - Algorithms: Random Forest, XGBoost, Isolation Forest, SMOTE

2. **Stock Price Prediction**
   - Source: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
   - Skills: Time series analysis, feature creation, multi-step forecasting
   - Algorithms: ARIMA, Prophet, LSTM, ensemble methods

3. **Loan Default Prediction**
   - Source: https://www.kaggle.com/wordsforthewise/lending-club
   - Skills: Classification, risk modeling, feature importance
   - Algorithms: Logistic Regression, Random Forest, Gradient Boosting

### Healthcare & Medicine
1. **Diabetes Prediction**
   - Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database
   - Skills: Binary classification, feature importance, model calibration
   - Algorithms: Logistic Regression, SVM, Random Forest

2. **Heart Disease Prediction**
   - Source: https://www.kaggle.com/ronitf/heart-disease-uci
   - Skills: Classification, feature selection, model interpretation
   - Algorithms: Decision Trees, KNN, SVM, Ensemble methods

3. **Medical Cost Prediction**
   - Source: https://www.kaggle.com/mirichoi0218/insurance
   - Skills: Regression, feature engineering, outlier handling
   - Algorithms: Linear Regression, Random Forest, Gradient Boosting

### Retail & E-commerce
1. **Customer Segmentation**
   - Source: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
   - Skills: Clustering, feature engineering, dimensionality reduction
   - Algorithms: K-means, DBSCAN, Gaussian Mixture Models

2. **Sales Forecasting**
   - Source: https://www.kaggle.com/c/rossmann-store-sales
   - Skills: Time series forecasting, feature engineering, hierarchical models
   - Algorithms: ARIMA, Prophet, XGBoost, LightGBM

3. **Product Recommendation**
   - Source: https://www.kaggle.com/retailrocket/ecommerce-dataset
   - Skills: Recommendation systems, collaborative filtering
   - Algorithms: Matrix factorization, Nearest neighbors, Association rules

### Environmental & Climate
1. **Air Quality Prediction**
   - Source: https://www.kaggle.com/amankumar1/air-quality-prediction-in-india-using-aiml
   - Skills: Regression, time series, environmental modeling
   - Algorithms: Random Forest, XGBoost, ARIMA

2. **Temperature Forecasting**
   - Source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
   - Skills: Time series forecasting, seasonal decomposition
   - Algorithms: SARIMA, Prophet, Deep learning

3. **Rainfall Prediction**
   - Source: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
   - Skills: Classification, meteorological feature engineering
   - Algorithms: Random Forest, Gradient Boosting, Neural Networks

### Transportation & Logistics
1. **Taxi Trip Duration Prediction**
   - Source: https://www.kaggle.com/c/nyc-taxi-trip-duration
   - Skills: Regression, geospatial features, datetime features
   - Algorithms: Linear Regression, Random Forest, Gradient Boosting

2. **Flight Delay Prediction**
   - Source: https://www.kaggle.com/usdot/flight-delays
   - Skills: Classification, feature engineering, time series
   - Algorithms: Random Forest, XGBoost, Logistic Regression

3. **Traffic Volume Prediction**
   - Source: https://www.kaggle.com/allarasanidm/metro-traffic-volume
   - Skills: Time series forecasting, multivariate analysis
   - Algorithms: ARIMA, Prophet, LSTM, XGBoost

## Datasets by Machine Learning Technique

### For Regression Analysis
1. **House Price Prediction**
   - Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
   - Features: 79 explanatory variables describing aspects of residential homes
   - Techniques to demonstrate: Regularization, feature selection, ensemble methods

2. **Bike Sharing Demand**
   - Source: https://www.kaggle.com/c/bike-sharing-demand
   - Features: Temporal features, weather conditions
   - Techniques to demonstrate: Time series, feature engineering, regression

### For Classification
1. **Titanic Survival Prediction**
   - Source: https://www.kaggle.com/c/titanic
   - Features: Passenger information (age, class, fare, etc.)
   - Techniques to demonstrate: Classification, feature engineering, missing data handling

2. **Wine Quality Classification**
   - Source: https://archive.ics.uci.edu/ml/datasets/wine+quality
   - Features: Physicochemical properties of wine
   - Techniques to demonstrate: Multiclass classification, feature importance

### For Clustering
1. **Mall Customer Segmentation**
   - Source: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
   - Features: Customer demographics and spending patterns
   - Techniques to demonstrate: K-means, hierarchical clustering, DBSCAN

2. **Countries Clustering**
   - Source: https://www.kaggle.com/rohan0301/unsupervised-learning-on-country-data
   - Features: Socioeconomic and health factors
   - Techniques to demonstrate: Clustering, dimensionality reduction

### For Time Series Analysis
1. **Energy Consumption Forecasting**
   - Source: https://www.kaggle.com/robikscube/hourly-energy-consumption
   - Features: Hourly power consumption with seasonal patterns
   - Techniques to demonstrate: Time series decomposition, ARIMA, Prophet

2. **Stock Market Data**
   - Source: https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231
   - Features: Daily open, high, low, close prices
   - Techniques to demonstrate: Technical indicators, auto regression, volatility modeling

### For AutoML Demonstration
1. **Amazon Employee Access Challenge**
   - Source: https://www.kaggle.com/c/amazon-employee-access-challenge
   - Features: Hierarchical categorical features
   - Techniques to demonstrate: AutoML, feature encoding, hyperparameter optimization

2. **Mercedes-Benz Greener Manufacturing**
   - Source: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
   - Features: Anonymized manufacturing features
   - Techniques to demonstrate: AutoML, dimensionality reduction, hyperparameter tuning

## Selection Criteria Checklist

When selecting your dataset, ensure it:

- [ ] Contains at least 1,000 samples (more is better)
- [ ] Has a mix of numerical and categorical features
- [ ] Requires multiple preprocessing steps
- [ ] Allows demonstration of several algorithms
- [ ] Presents a clear business problem to solve
- [ ] Has clean enough data to make progress but messy enough to show cleaning skills
- [ ] Has complexity suitable for a multi-week project
- [ ] Does not require excessive computational resources

## Getting Started Guide

1. **Explore multiple datasets** before committing to one
2. **Perform basic EDA** to understand data structure and challenges
3. **Identify baseline models** appropriate for your problem
4. **Split your dataset** properly (train, validation, test)
5. **Start with simple models** before moving to complex ones
6. **Document your approach** from the beginning
7. **Map your dataset characteristics** to milestone requirements
