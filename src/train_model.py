import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# === Load preprocessed data ===
df = pd.read_csv("data/processed_diabetes.csv")  # assumes output from preprocessing.py

# === Separate features and target ===
X = df.drop(columns=["readmitted_binary"])
y = df["readmitted_binary"]

# === Train/val split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Optional: class weights (tune based on your distribution) ===
sample_weights = y_train.map({0: 0.618, 1: 2.987, 2: 0.954})

# === Initialize tuned model ===
model = XGBClassifier(
    n_estimators=260,
    learning_rate=0.0203,
    max_depth=10,
    min_child_weight=3,
    gamma=0.33,
    subsample=0.5899,
    colsample_bytree=0.5294,
    reg_alpha=0.3815,
    reg_lambda=1.5034,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1
)

# === Fit the model ===
model.fit(X_train, y_train, sample_weight=sample_weights)

# === Evaluate ===
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

# === Save model ===
joblib.dump(model, "models/xgb_readmission_model.pkl")