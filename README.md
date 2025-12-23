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
**[Chourouk]**  
Junior Data Scientist  
Interested in data-driven problem solving and machine learning projects  
🔗 GitHub: [https://github.com/chourouk-sun]  
🔗 LinkedIn: [https://www.linkedin.com/in/chourouk-djilanii-bb5931378/]

## References
- Kaggle Titanic Competition  
  https://www.kaggle.com/competitions/titanic
- Scikit-learn Documentation  
  https://scikit-learn.org/stable/
- Pandas Documentation  
  https://pandas.pydata.org/docs/

##  Confusion Matrix Analysis 
![[Confusion Matrix ]( https://github.com/chourouk-sun/Titanic-Survival-Prediction-Binary-Classification/blob/02a62e13eea3a7536d33ae354b6e350e1a5c3f40/images/image1.png)

### What each value means: 
The confusion matrix shows the performance of the Logistic Regression classifier on the validation set.
* **True Negatives (90)**
  → Passengers who **did not survive** and were **correctly predicted** as not surviving.

* **False Positives (15)**
  → Passengers who **did not survive**, but the model **incorrectly predicted** they survived.

* **False Negatives (19)**
  → Passengers who **did survive**, but the model **incorrectly predicted** they did not survive.

* **True Positives (55)**
  → Passengers who **survived** and were **correctly predicted** as survivors.

## Model Interpretation
* The model performs **well at identifying non-survivors**, with a high number of correct predictions (90).
* It also correctly identifies a significant number of survivors (55).
* There are **more false negatives (19) than false positives (15)**, meaning the model is slightly more conservative and sometimes predicts death when the passenger actually survived.
* Overall performance is **balanced and reliable**, supporting the achieved **81% validation accuracy**.

## Predicted Survival by Age
![Survival by Age ](https://github.com/chourouk-sun/Titanic-Survival-Prediction-Binary-Classification/blob/37dacead425a9754aac0012cea69b1b1e14d6b37/images/image2.png)

This histogram displays the **model’s predicted survival outcomes by age** for passengers in the test dataset.
* **X-axis:** Passenger age
* **Y-axis:** Number of passengers (count)
* **Colors (hue):**
  * **0 (Blue):** Predicted *not survived*
  * **1 (Orange):** Predicted *survived*
* The bars are **stacked**, allowing comparison of survival predictions across age groups.

## Interpretation of the Results
- **Younger passengers (children and young adults)** show a **higher proportion of predicted survivors**, especially in the age range **0–15** and **20–30**.
- **Middle-aged passengers (30–50)** are more frequently predicted as **non-survivors**, indicating lower survival probability.
- **Older passengers (50+)** have the **lowest predicted survival rate**, with very few survivors predicted.
- The distribution reflects realistic Titanic survival patterns, where **age played an important role** in survival chances.

- The model predicts higher survival rates for younger passengers, particularly children and young adults, while older passengers show significantly lower survival probabilities. Middle-aged passengers are more frequently predicted as non-survivors.
These patterns are consistent with historical accounts of the Titanic disaster, indicating that age was an important factor influencing survival. The visualization helps confirm that the model captures meaningful relationships between passenger age and survival likelihood.

### Why This Visualization Is Important
* Demonstrates how **age influences model predictions**
* Helps validate that the model learned **meaningful patterns**
* Provides intuitive, real-world interpretation of ML results
* Supports historical knowledge of the Titanic disaster
  

## Predicted Survival Distribution (Test Set)
![Predicted Survival Count ](https://github.com/chourouk-sun/Titanic-Survival-Prediction-Binary-Classification/blob/600ee27fbf77e4f10f18fd74d8e47e58cea57b6b/images/image3.png)

This visualization shows the **distribution of survival predictions** generated by the trained machine learning model on the **unseen test dataset**.

* The x-axis represents the predicted survival outcome:
  * **0 – Did Not Survive**
  * **1 – Survived**
* The y-axis represents the **number of passengers** predicted in each category.

**Interpretation:**
* The model predicts that **more passengers did not survive than survived**, which is consistent with the historical reality of the Titanic disaster.
* This imbalance indicates that the model has learned meaningful patterns from the training data rather than predicting survival randomly.
* The predicted distribution aligns with known trends from the dataset, where overall survival rates were lower than non-survival rates.

**Why this visualization matters:**
* It provides a **high-level sanity check** of model behavior on unseen data.
* It helps verify that predictions are realistic and not biased toward a single class.
* It improves model interpretability by showing how the classifier generalizes beyond the training set.
This plot is useful for understanding the **overall decision tendency of the model** before submitting predictions to Kaggle or deploying the model.
























