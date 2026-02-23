from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


data = load_iris()
X = data.data
y= data.target

base_model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
bagging_model = BaggingClassifier(
    estimator=base_model,
    n_estimators=50,
    random_state=42)

bagging_model.fit(X_train,y_train)

y_pred = bagging_model.predict(X_test)

print(f"Accuracy = {accuracy_score(y_test, y_pred)}")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
