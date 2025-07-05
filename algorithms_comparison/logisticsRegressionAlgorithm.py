import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load cleaned dataset
df = pd.read_csv("./data/ObesityDataSet_Cleaned.csv")

# 2. Define features (X) and target (y)
X = df.drop(['LevelObesity'], axis=1)
y = df['LevelObesity']

# 3. Encode categorical features (Gender, MonitorCaloriesHabit, etc.)
X = pd.get_dummies(X, drop_first=True)

# 4. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)  # you can use le.inverse_transform() later to decode

# 5. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2%}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
