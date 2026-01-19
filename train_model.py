import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("DATA/simplified_learning_dataset.csv")

# Encode Interest_Area (string → numeric)
encoder = LabelEncoder()
df['Interest_Area'] = encoder.fit_transform(df['Interest_Area'])

# Features and target
X = df[['CGPA', 'Completed_Course_Count', 'Interest_Area']]
y = df['Recommended_Course']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(criterion="gini", max_depth=4)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(encoder, "interest_encoder.pkl")

print("✅ Model trained and saved successfully!")
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))