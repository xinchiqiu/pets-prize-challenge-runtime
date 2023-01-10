import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


class SwiftModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst