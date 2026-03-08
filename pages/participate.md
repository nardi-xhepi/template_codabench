# How to Participate

## The Challenge: 

Welcome to the **AAV Capsid Optimization Challenge** — where machine learning meets the future of gene therapy.

### The Real-World Problem

Every year, thousands of patients with genetic diseases wait for treatments that could save or transform their lives. **Adeno-Associated Viruses (AAV)** are the workhorses of modern gene therapy, delivering therapeutic genes into patients' cells to treat conditions like spinal muscular atrophy, hemophilia, and inherited blindness.

But here's the challenge: **engineering better AAV capsids is expensive and time-consuming**. Each experimental cycle costs thousands of dollars and takes weeks. Researchers need to test tens of thousands of variants to find the rare ones that work better — improving tissue targeting, evading immune responses, or packaging larger therapeutic genes.

**What if we could predict which variants will work *before* stepping into the lab?**

This is where you come in.

---

## Your Mission

You are tasked with **predicting the fitness of AAV capsid variants** from their amino acid sequences. Your predictions will help researchers prioritize which variants to test experimentally, dramatically accelerating the design of next-generation gene therapies.

### The Task

- **Input:** Amino acid sequence of an AAV capsid protein variant (~735 residues)
- **Output:** Fitness score (float) — how well the capsid packages viral DNA
- **Primary metric:** **Spearman's rank correlation (ρ)** — because in protein engineering, **ranking matters more than exact values**
- **Bonus metric:** **Precision@100** — the fraction of the true top-100 variants that appear in your model's predicted top 100. This reflects a real experimental constraint: if you could only test 100 variants in the lab, how many would actually be the best?

Your model will be evaluated on its ability to **rank variants correctly**: which ones are the top performers, and which ones fail?

---

## The Scientific Challenge: Epistasis

Here's where it gets interesting.

In the lab, we can easily create and test variants with **1 or 2 mutations**. But the real therapeutic breakthroughs often come from variants with **3, 4, or even 10 mutations** — and these are astronomically expensive to test exhaustively.

The problem? **Mutations don't simply add up.** They **interact**.

- Two mutations that each improve fitness by +1 might together give +3 (synergistic) or -2 (antagonistic).
- A mutation that's harmful alone might be beneficial when combined with others.

This phenomenon is called **epistasis**, and it's the Achilles' heel of naive machine learning models.

### Your Challenge

- **Train** on variants with **1–2 mutations** (31,807 sequences)
- **Test** on variants with **3+ mutations** (50,776 sequences)

A simple linear model that assumes mutations are independent will **fail spectacularly**. You'll need to capture the **combinatorial interactions** between mutations — the hidden patterns that make protein engineering so hard.

---

## Evaluation:

We evaluate your model using **Spearman's rank correlation coefficient (ρ)**.

In real-world protein engineering, researchers don't care about predicting exact fitness values — they care about **which variants to test first**. If your model correctly identifies the top 100 variants out of 50,000, it's a success.

| Spearman ρ | Interpretation |
|------------|----------------|
| **ρ = 1.0** | Perfect ranking — you found all the best variants |
| **ρ ≥ 0.5** | Strong correlation — your model is useful |
| **ρ = 0.0** | Random guessing — no better than chance |
| **ρ < 0.0** | Anti-correlated — something went very wrong |

Your **leaderboard score** is the Spearman correlation on the public test set (50% of test data).
Your **final ranking** is determined by the private test set (revealed at the end).

### Bonus Metric: Precision@100

We also compute **Precision@100** — the fraction of true top-100 variants that your model correctly identifies. This metric reflects the **real experimental constraint**: if you could only test 100 variants in the lab, how many of them would actually be the best?

---

## Submission Format

Submit a Python file named `submission.py` containing a `get_model()` function that returns a scikit-learn compatible model.

### Requirements

Your model must implement:
- **`fit(X, y)`**: Train on a DataFrame `X` with a `sequence` column and target `y` (fitness scores)
- **`predict(X)`**: Return predicted fitness scores for new sequences

### Example Submission

```python
def get_model():
    return YourModel()
```

Your model must implement:
- `fit(X, y)`: Train on sequences and fitness values
- `predict(X)`: Return predictions

Zip your file and upload:
```bash
zip submission.zip submission.py
```

---

## Getting Started

1. **Download the starting kit** — includes baseline code, data exploration, and local testing pipeline
2. **Understand the baseline** — Ridge regression with k-mer features (Spearman ρ ≈ 0.42)
3. **Improve the model** — try better features, non-linear models, or protein language models
4. **Submit and iterate** — test on the leaderboard, learn from the data, and refine your approach

---

**Are you ready to accelerate the future of gene therapy?**

Let's get started.
