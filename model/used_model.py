import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("./data/ObesityDataSet_Cleaned.csv")

# Features and target
X = df.drop(['LevelObesity'], axis=1)
y = df['LevelObesity']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Save everything
joblib.dump(model, "./model/model_saved_file/logistic_model.pkl")
joblib.dump(scaler, "./model/model_saved_file/scaler.pkl")
joblib.dump(le, "./model/model_saved_file/label_encoder.pkl")
joblib.dump((X_test_scaled, y_test, y_pred, acc, cm, report), "./model/model_saved_file/evaluation_results.pkl")

print("✅ Model, scaler, and evaluation saved.")
