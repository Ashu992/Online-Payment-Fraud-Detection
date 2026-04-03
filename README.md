# Online Payment Fraud Detection
**Developed by: Ashutosh Tripathi**

## 📌 Executive Summary
This project focuses on identifying fraudulent transactions within a massive synthetic financial dataset of over 6.3 million records. By leveraging advanced data preprocessing and machine learning, I developed a system capable of distinguishing legitimate transfers from sophisticated fraud with high precision.

The final solution employs a Random Forest Classifier with a tuned classification threshold to minimize False Positives, ensuring a seamless user experience while maintaining robust security.

## 📊 Dataset Overview
The dataset contains historical transaction data with the following key attributes:
* **Type**: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT.
* **Amount**: The transaction value in local currency.
* **Balance Discrepancies**: Features tracking the balance before and after for both Origin and Destination accounts.
* **isFraud**: The target variable identifying fraudulent activity.
dataset link : https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset

## 🛠️ Key Technical Implementations

### 1. High-Signal Feature Engineering
Standard transaction data was augmented with engineered features to capture "account draining" behaviors:
* **Balance Differences**: Calculated actual change vs. expected change for origin and destination accounts to find discrepancies.
* **Zero-Balance Indicators**: Binary flags for accounts starting or ending with zero balances, a common trait in shell-account fraud.

### 2. Handling Extreme Class Imbalance
With fraud representing only 0.13% of the data, I implemented **Ratio-based Undersampling** (1:5 ratio) to provide the model with sufficient exposure to fraud signals while maintaining a large enough sample for generalization.

### 3. Threshold Optimization
Rather than using the default 0.5 probability threshold, the classification boundary was shifted to **0.95**. This strategic move prioritizes **Precision**, ensuring that legitimate customers are not inconvenienced by false flags while still capturing nearly 80% of actual fraud.

## 📈 Model Performance
The Random Forest approach proved superior in capturing non-linear fraud patterns compared to Logistic Regression.

| Metric | Random Forest (Tuned) | Logistic Regression |
| :--- | :--- | :--- |
| ROC-AUC | 0.999 | 0.991 |
| Precision (Fraud) | 0.90 | 0.02 |
| Recall (Fraud) | 0.79 | 0.96 |
| Overall Accuracy | 99.9% | 93.0% |

## 🚀 Deployment
The project includes a serialized version of the model and preprocessing pipeline:
* `fraud_model.pkl`: The trained Random Forest model.
* `features.pkl`: List of features required for prediction.

## 💡 Conclusion
The analysis confirms that fraud is most prevalent in TRANSFER and CASH_OUT transactions. By deploying this model, financial institutions can automate the monitoring of millions of transactions, flagging 79% of fraud automatically while maintaining a 90% accuracy rate on those flags.