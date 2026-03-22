# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, separate features and target, scale the features using MinMaxScaler, and encode the target labels using LabelEncoder.

2.Split the dataset into training and testing sets using train_test_split() with stratified sampling.

3.Train a Logistic Regression model with L2 regularization (multinomial) on the training data and make predictions on the test data.

4.Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix, then visualize the confusion matrix using a heatmap.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: SHRIHARI M
RegisterNumber:  212225230265
*/

Program to implement SVM
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
svm = SVC()
param_grid = {
'C': [0.1, 10, 100],
'kernel': ['linear', 'rbf'],
'gamma': ['scale','auto']
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: SHRIHARI M")
print("Register Number:25013276")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="622" height="570" alt="Screenshot 2026-03-22 165023" src="https://github.com/user-attachments/assets/acff8101-f9a3-4c75-9af3-2e1f376fe233" />
<img width="608" height="711" alt="Screenshot 2026-03-22 165035" src="https://github.com/user-attachments/assets/ff61e2fd-fe92-4dc8-be54-4489ac92a1b4" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
