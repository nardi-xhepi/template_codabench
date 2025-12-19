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

## The Twist: Epistasis

You'll train on variants with 1-2 mutations but test on variants with 3+ mutations. Mutations are not additive: they interact!

## Evaluation Metric: Spearman's Rank Correlation (ρ)

In protein engineering, **ranking matters more than absolute values**.

| Score | Meaning |
|-------|---------|
| ρ = 1.0 | Perfect ranking |
| ρ = 0.5 | Moderate correlation |
| ρ = 0.0 | Random ranking |

Your final score is the Spearman correlation on the test set. Higher ρ = higher leaderboard position.
