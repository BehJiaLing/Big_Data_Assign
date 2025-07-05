import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load cleaned dataset
df = pd.read_csv("./data/ObesityDataSet_Cleaned.csv")

# 2. Define features and target
X = df.drop(['LevelObesity'], axis=1)
y = df['LevelObesity']

# 3. One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 4. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Scale features (optional for tree-based models but okay for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Predict
y_pred = model.predict(X_test_scaled)

# 9. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))