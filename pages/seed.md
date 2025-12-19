# Seed Model

Your submission should be a Python file named `submission.py` that contains a `get_model()` function:

```python
def get_model():
    """Return a model for the FLIP AAV fitness prediction challenge."""
    return YourModel()
```

## Example Baseline

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

class SimpleBaseline(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = Ridge()
    
    def fit(self, X, y):
        features = X['sequence'].apply(len).values.reshape(-1, 1)
        self.model.fit(features, y)
        return self
    
    def predict(self, X):
        features = X['sequence'].apply(len).values.reshape(-1, 1)
        return self.model.predict(features)

def get_model():
    return SimpleBaseline()
```

Better approaches: K-mer features, ESM-2 embeddings, CNNs, Transformers
