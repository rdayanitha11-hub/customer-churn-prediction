import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

DATASET_PATH = "data/archive/Telco-Customer-Churn.csv"
MODEL_PATH = "models/model.pkl"

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df = df.dropna(subset=['tenure', 'MonthlyCharges', 'TotalCharges'])

    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    df = pd.get_dummies(df, drop_first=True)
    return df

def main():
    df = load_data(DATASET_PATH)
    df = preprocess_data(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    file_size = os.path.getsize(MODEL_PATH)
    print(f"✅ Model successfully saved to: {MODEL_PATH}")
    print(f"✅ File size: {file_size / 1024:.2f} KB")

if __name__ == "__main__":
    main()