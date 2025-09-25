from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Train model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    # Save model to file
    joblib.dump(clf, "iris_model.joblib")
    print("Model trained and saved as iris_model.joblib")

if __name__ == "__main__":
    main()
