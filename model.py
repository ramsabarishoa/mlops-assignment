import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib

def train_model():
    """Trains a RandomForestClassifier on the Iris dataset and saves it."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Added split for better practice
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "iris_model.joblib")
    print("Model trained and saved as iris_model.joblib")

if __name__ == "__main__":
    train_model()