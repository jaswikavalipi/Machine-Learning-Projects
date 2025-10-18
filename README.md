🛳️ Titanic Survival Prediction — Machine Learning Project
📖 Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning.
It is based on the famous Kaggle Titanic dataset, which contains demographic and travel information about passengers, such as age, gender, ticket class, and fare price.

The goal is to build, train, and evaluate a classification model that can accurately predict survival based on passenger data.

⚙️ How It Works

Data Loading
The dataset (titanic.csv) is loaded using pandas. It contains columns like PassengerId, Pclass, Sex, Age, Fare, Embarked, etc.

Data Cleaning

Unnecessary columns (PassengerId, Name, Ticket, Cabin) are dropped.

Missing values in Age are filled with the median age.

Missing values in Embarked are filled with the most frequent value (mode).

Data Encoding

The categorical columns Sex and Embarked are converted into numerical form using LabelEncoder.

Example: Sex → male = 1, female = 0

Embarked → C = 0, Q = 1, S = 2

Feature Selection & Splitting

Features (X) are all independent columns.

The target (y) is the Survived column.

The dataset is split into training and test sets using an 80/20 ratio.

Model Training

A Random Forest Classifier is trained on the training set.

This ensemble model combines multiple decision trees to improve accuracy and reduce overfitting.

Prediction & Evaluation

The trained model predicts survival on the test set.

Performance is evaluated using metrics such as:

Accuracy

Precision

Recall

F1-score

A classification report is printed for detailed performance analysis.

Feature Importance Visualization

The Random Forest model provides feature_importances_, showing which features contribute most to survival prediction.

A bar chart displays the Top 10 important features (e.g., Sex, Fare, Age, Pclass).

🧠 Key Insights

Sex is the most influential factor — females had a much higher survival rate.

Passenger Class (Pclass) and Fare are also strong predictors (wealthier passengers were more likely to survive).

Younger passengers tended to have higher survival chances.

🧰 Tools & Libraries Used

Python 3

Pandas – Data manipulation and analysis

NumPy – Numerical operations

Scikit-learn – Machine learning algorithms and preprocessing

Matplotlib / Seaborn – Data visualization

🚀 How to Run

Open Google Colab or Jupyter Notebook.

Upload your Titanic dataset (titanic.csv) to the environment.

Copy and paste the Python script from this project.

Run all cells in order.

View the accuracy, classification report, and feature importance plot.

📈 Possible Improvements

Perform hyperparameter tuning using GridSearchCV for better accuracy.

Try Gradient Boosting or XGBoost models for improved performance.

Add more feature engineering, like:

Extracting passenger titles (Mr., Mrs., etc.) from names

Creating a FamilySize feature

Handling outliers in Fare and Age

Use cross-validation for more robust model evaluation.

✨ Project Outcome

By the end of this project, you’ll have:

A trained Random Forest model capable of predicting Titanic survival.

A strong understanding of end-to-end machine learning workflow — from data cleaning to model interpretation.

Visual insights into what factors most affected passenger survival.
