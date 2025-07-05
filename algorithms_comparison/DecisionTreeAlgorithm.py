import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load cleaned dataset
df = pd.read_csv("./data/ObesityDataSet_Cleaned.csv")
df.columns = df.columns.str.strip()

# 2. Define features and target
X = df.drop(['LevelObesity'], axis=1)
y = df['LevelObesity']

# 3. Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 4. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Feature scaling (optional for decision trees, but good for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Decision Tree Classifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Predict
y_pred = model.predict(X_test_scaled)

# 9. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 10. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Decision Tree - Confusion Matrix')
plt.tight_layout()
plt.show()

# 11. Plot Decision Tree Structure
plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=X.columns,
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()
