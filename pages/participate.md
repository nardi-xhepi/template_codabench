# How to Participate

## Challenge Overview

The **FLIP AAV Capsid Optimization Challenge** tests your ability to predict protein fitness from amino acid sequences.

## The Task

- **Input:** Amino acid sequence of AAV capsid protein (~735 characters)
- **Output:** Fitness score (float)
- **Metric:** Spearman's rank correlation

## Submission Format

Submit a Python file `submission.py` containing a `get_model()` function:

```python
def get_model():
    return YourModel()
```

Your model must implement:
- `fit(X, y)`: Train on sequences and fitness values
- `predict(X)`: Return predictions
