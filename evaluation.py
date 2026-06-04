# Evaluation Metrics

import pandas as pd
from data_helpers import ensure_file

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Ensure placement dataset exists
ensure_file("placement_dataset.csv")

df = pd.read_csv("../data/placement_dataset.csv")

X = df[["CGPA","Python","Communication"]]
y = df["Placement"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,prediction))
print("Precision:",precision_score(y_test,prediction))
print("Recall   :",recall_score(y_test,prediction))
print("F1 Score :",f1_score(y_test,prediction))

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)