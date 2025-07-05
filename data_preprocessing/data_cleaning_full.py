import pandas as pd

# 1. Load dataset
df = pd.read_csv('./data/ObesityDataSet_Ori.csv')

# 2. Shape of dataset
print("Shape of dataset:", df.shape)

# 3. Show first few rows
print("First 5 rows:")
print(df.head())

# 4. Info (column types, non-null counts)
print("\nDataset Info:")
print(df.info())

# 5. Column reduction – keep only relevant columns
cols_to_keep = ['Age', 'Gender', 'Height', 'Weight', 'SCC', 'family_history_with_overweight', 'CAEC', 'NObeyesdad']
df = df[cols_to_keep]

# Rename columns to cleaner names
df = df.rename(columns={
    'SCC': 'MonitorCaloriesHabit',
    'family_history_with_overweight': 'GeneticsOverweight',
    'CAEC': 'SnackHabit',
    'NObeyesdad': 'LevelObesity'
})

print("\nColumns after reduction and renaming:", df.columns.tolist())

# 6. Remove duplicates
duplicate_rows = df[df.duplicated()]
print(f"\nDuplicate rows: {len(duplicate_rows)}")
df = df.drop_duplicates()

# 7. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 8. Handle inconsistencies (strip spaces, normalize text)
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['SnackHabit'] = df['SnackHabit'].str.strip()
df['MonitorCaloriesHabit'] = df['MonitorCaloriesHabit'].astype(str).str.strip()
df['GeneticsOverweight'] = df['GeneticsOverweight'].astype(str).str.strip()
df['LevelObesity'] = df['LevelObesity'].str.strip()

# 9. Feature Engineering – Add BMI column
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 10. Feature Engineering – Create AgeGroup column
bins = [0, 20, 30, 40, 50, 100]
labels = ['<20', '21–30', '31–40', '41–50', '50+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Format numeric columns
df['Age'] = df['Age'].astype(int)
df['Height'] = df['Height'].round(2)
df['Weight'] = df['Weight'].round(2)
df['BMI'] = df['BMI'].round(2)

# Final preview
print("\nFinal cleaned dataset preview:")
print(df.head())

# Save as cleaned file
df.to_csv("./data/ObesityDataSet_Cleaned.csv", index=False)
print("\n✅ Cleaned data saved to '../data/ObesityDataSet_Cleaned.csv'")