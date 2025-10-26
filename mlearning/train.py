import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Get input/output file names from Snakemake
input_file, output_file = sys.argv[1], sys.argv[2]

# Load data
df = pd.read_csv(input_file)
X = df[["feature1", "feature2"]]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)

# Save metrics
with open(output_file, "w") as f:
    f.write("Classification Metrics:\n")
    f.write(report)

print("✅ Metrics saved to", output_file)

