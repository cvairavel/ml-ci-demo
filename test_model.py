import joblib
from sklearn.datasets import load_iris

def test_model_prediction():
    model = joblib.load("iris_model.joblib")
    X, y = load_iris(return_X_y=True)
    preds = model.predict(X[:5])
    assert len(preds) == 5
