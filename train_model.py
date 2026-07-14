import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/placement.csv")

# Keep important columns
df = df[
    [
        "cgpa",
        "internships_count",
        "projects_count",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "placement_status",
    ]
]

# Convert target to numbers
df["placement_status"] = df["placement_status"].map({
    "Not Placed": 0,
    "Placed": 1
})

# Features and target
X = df.drop("placement_status", axis=1)
y = df["placement_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "models/placement_model.pkl")

print("Model saved successfully!")