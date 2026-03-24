# 🎬 Movie Genre Classification & Data Analysis

This project demonstrates basic machine learning and data analysis techniques using Python, Pandas, and Scikit-learn.

---

## 📌 Project Overview

The project includes:

- Movie genre classification using a Decision Tree model
- Data analysis on video game sales dataset
- Basic data preprocessing and evaluation techniques

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 📊 Dataset

- `movies.csv` → Used for genre classification  
- `vgsales.csv` → Used for exploratory data analysis  

---

## 🚀 Machine Learning Model

### Decision Tree Classifier

Steps performed:

1. Load dataset using Pandas  
2. Separate features (X) and target (y)  
3. Split dataset into training and testing sets  
4. Train model using `DecisionTreeClassifier`  
5. Predict test data  
6. Evaluate performance using accuracy score  

---

## 📈 Example Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

movies_data = pd.read_csv('movies.csv')

X = movies_data.drop(columns=['genre'])
y = movies_data['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train.values, y_train.values)

prediction = model.predict(X_test.values)
score = accuracy_score(y_test, prediction)

print("Accuracy:", score)
