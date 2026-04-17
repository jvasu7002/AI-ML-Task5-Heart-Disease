# Decision Tree & Random Forest - Heart Disease Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("C:/Users/vasu jain/Downloads/archive (2)/heart.csv")   # change file name if needed

# Step 2: Features & Target
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Decision Tree
# -----------------------------
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# -----------------------------
# Feature Importance
# -----------------------------
importances = rf.feature_importances_

plt.barh(X.columns, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# -----------------------------
# Decision Tree Visualization
# -----------------------------
plt.figure(figsize=(12,8))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.title("Decision Tree")
plt.show()
