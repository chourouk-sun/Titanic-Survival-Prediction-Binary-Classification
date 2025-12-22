# -Titanic-Survival-Prediction-Binary-Classification-


# Titanic – Machine Learning from Disaster

**Project Type:** Machine Learning | Binary Classification | Supervised Learning
**Author:**  Chourouk  
**Level:** Junior Data Scientist  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib  


## Project Overview

This project aims to predict the survival of passengers on the Titanic using machine learning. The dataset contains passenger information such as age, gender, class, and fare, and the goal is to determine whether a passenger survived the disaster.

**Skills Demonstrated:**

- Data cleaning and preprocessing  
- Feature engineering  
- Binary classification with Logistic Regression  
- Model evaluation (accuracy, confusion matrix, classification report)  
- End-to-end ML workflow from raw data to predictions  
- Visualization of model predictions  

## Tools & Libraries
- **Programming Language:** Python 3.x  
- **Libraries:**
  - Pandas & NumPy – data manipulation
  - Matplotlib & Seaborn – data visualization
  - Scikit-learn – machine learning models and evaluation
- **Environment:** Jupyter Notebook 

## Project Motivation
The sinking of the RMS Titanic is one of the most well-known disasters in history.  
This project aims to apply **machine learning techniques** to predict whether a passenger survived the Titanic disaster based on available passenger data.
This project demonstrates the **complete end-to-end workflow of a machine learning classification problem**, from raw data exploration to model evaluation and prediction generation.

## Problem Statement
Given passenger information such as:
- Passenger class
- Sex
- Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket fare
- Port of embarkation
**Goal:**  
Predict whether a passenger **survived (1)** or **did not survive (0)**.


## Dataset Information
The dataset is provided by Kaggle: **Titanic – Machine Learning from Disaster**
- **Training data:** `train.csv`  
- **Test data:** `test.csv`  
**Kaggle link:** [Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

### Dataset Size
- Train set: 891 rows × 12 columns
- Test set: 418 rows × 11 columns


## Feature Description
| Feature | Description |
|------|-----------|
| PassengerId | Unique passenger identifier |
| Survived | Target variable (0 = No, 1 = Yes) |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C, Q, S) |


## Project Structure
Titanic-ML-Project/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── Titanic_ML.ipynb # Full machine learning workflow
├── submission.csv # Predictions for Kaggle submission
├── README.md # Project documentation
└── requirements.txt # Python dependencies








## Machine Learning Workflow

### 1 Data Loading
- Import training and test datasets
- Verify data integrity and structure

### 2️ Exploratory Data Analysis (EDA)
- Analyze missing values
- Examine distributions of numerical features
- Study relationships between features and survival
- Visualize survival by gender and passenger class

### 3️ Data Cleaning
- Fill missing values:
  - Age → median
  - Embarked → most frequent value
  - Fare (test set) → median
- Drop irrelevant or high-cardinality features:
  - Name
  - Ticket
  - Cabin
  - PassengerId (temporarily)

### 4️ Feature Engineering
- Convert categorical variables to numerical using one-hot encoding
- Prepare features for model training
- Separate input features (X) and target variable (y)

### 5️ Train-Validation Split
- Split training data into:
  - 80% training
  - 20% validation
- Ensure reproducibility using a fixed random seed

### 6️ Feature Scaling
- Standardize numerical features using `StandardScaler`
- Improve convergence and model performance

### 7️ Model Selection & Training
- **Model Used:** Logistic Regression
- Justification:
  - Simple
  - Interpretable
  - Suitable for binary classification
- Train model on training set

### 8️ Model Evaluation
- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score
- Visualization using heatmaps

### 9️ Prediction on Test Set
- Train final model on full training data
- Generate survival predictions for unseen test data

### 10 Submission File Creation
- Create `submission.csv` with:
  - PassengerId
  - Predicted survival value
- File is compatible with Kaggle submission format
  

## Results & Observations
- The model achieves solid baseline accuracy for a simple classifier
- Predictions show:
  - Higher survival probability for females
  - Higher survival rate for 1st class passengers
- Results align with historical expectations of the Titanic disaster



## Visualizations Included
- Survival distribution
- Survival by gender
- Survival by passenger class
- Survival by age group
- Confusion matrix heatmap

All visualizations are generated inside the Jupyter Notebook.


## Future Improvements
- Use advanced models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Hyperparameter tuning with GridSearchCV
- Create new features:
  - Family size
  - Title extraction from names
- Cross-validation for better generalization
- Build an interactive dashboard (Streamlit)


## What This Project Demonstrates
✅ End-to-end ML pipeline  
✅ Data preprocessing & feature engineering  
✅ Classification modeling  
✅ Model evaluation and interpretation  
✅ Professional project documentation  
✅ Kaggle-ready submission workflow  


## Author
**[Your Name]**  
Junior Data Scientist  
📍 Interested in data-driven problem solving and machine learning projects  

🔗 GitHub: [your-github-profile]  
🔗 LinkedIn: [your-linkedin-profile]

---

## 📚 References

- Kaggle Titanic Competition  
  https://www.kaggle.com/competitions/titanic
- Scikit-learn Documentation  
  https://scikit-learn.org/stable/
- Pandas Documentation  
  https://pandas.pydata.org/docs/





























