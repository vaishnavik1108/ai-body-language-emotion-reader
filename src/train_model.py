import os
import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

# ── Ensure output folder exists before saving anything ──
os.makedirs('model', exist_ok=True)

# ── Step 1: Load Dataset ──
print("📊 Loading coordinate dataset...")
csv_path = os.path.join('data', 'engagement_coords.csv')

if not os.path.exists(csv_path):
    print(f"❌ ERROR: Dataset not found at '{csv_path}'.")
    print("   Please run 'data_collection.py' first to generate the dataset.")
    sys.exit()

df = pd.read_csv(csv_path)
print(f"   Total rows loaded: {len(df)}")

if len(df) == 0:
    print("❌ ERROR: Dataset is empty. Please collect data first.")
    sys.exit()

# ── Step 2: Data Cleaning ──
# Fill missing values with 0 instead of dropping rows
# This preserves 100% of recorded frames
df.fillna(0, inplace=True)

# ── Step 3: Separate Features and Labels ──
X = df.drop('class', axis=1)
y = df['class']

print(f"   Classes found: {list(y.unique())}")
print(f"   Class distribution:\n{y.value_counts().to_string()}")

# ── Step 4: Train / Test Split (80/20) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# ── Step 5: Train Random Forest Classifier ──
print("\n🤖 Training Random Forest Classifier (100 trees)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Step 6: Evaluate Model ──
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("\n--- 📈 EVALUATION REPORT ---")
print(f"Model Accuracy Score : {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(report)

# ── Step 7: Save Model ──
model_path = os.path.join('model', 'engagement_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"💾 Model saved → {model_path}")

# ── Step 8: Save Text Report ──
report_path = os.path.join('model', 'model_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.6f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"📄 Report saved → {report_path}")

# ── Step 9: Save Confusion Matrix ──
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Engagement Classifier — Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()

cm_path = os.path.join('model', 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()   # close cleanly — no hanging windows
print(f"📊 Confusion matrix saved → {cm_path}")

print("\n✅ All done! You can now run:  streamlit run app.py")