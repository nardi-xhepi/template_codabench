### Creators of the challenge
Creators: Alizée Mesnard, Cécile Luc, Hamza Boukhriss, Laurian Truong, Nardi Xhepi, Nicolas Bieszczad

# AAV Capsid Optimization Challenge
## From Research Benchmark to Educational Experience

**We transformed a cutting-edge protein engineering benchmark into a hands-on learning platform for real-world machine learning in biology.**

---

## Real challenges have real impact !

This competition bridges the gap between **academic ML** and **real-world protein engineering**. Instead of toy datasets, you'll work with actual experimental data from AAV gene therapy research, the same data scientists use to design treatments for genetic diseases.

In concrete terms, your goal is to build a machine learning regression model that predicts how well a modified virus functions. You will be provided with datasets where the input is a sequence of amino acids (a long text string representing the protein) and the target is its experimental "fitness" score (a numerical float). The catch is that you will train your model on slightly modified proteins (those with only 1 or 2 mutations), but you must predict the fitness of heavily mutated ones with 3 or more mutations!

What makes this challenge unique?

1. **We target the real bottleneck in protein engineering**: Predicting high-order epistatic interactions (3+ mutations) from low-order data (1-2 mutations)
2. **We industrialized the evaluation pipeline**: Automated scoring with both correlation metrics (Spearman) and experimental relevance (Precision@100)
3. **We designed it for learning** !

---

## The Scientific Context

This challenge uses the **AAV dataset** from the [FLIP benchmark](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/2b44928ae11fb9384c4cf38708677c48-Paper-round2.pdf) (Fitness Landscape Inference for Proteins) — BUT with a pedagogical twist, with additionnal features.

### The Data

Generated through **deep mutational scanning (DMS)** experiments measuring the packaging fitness of ~280,000 variants of the AAV2 capsid protein VP1 (~735 amino acids each).

In this challenge, the data is provided as standard CSV files:
- **Features (`train_features.csv`, `test_features.csv`, `private_test_features.csv`)**: A `sequence` column containing the full amino acid chain (a string of ~735 characters) for each variant.
- **Labels (`train_labels.csv`)**: A `target` column containing the fitness score (a log-enrichment ratio). A higher score indicates a highly functional virus that successfully assembled and packaged its DNA, while a lower score means the mutations broke the viral structure.

### The Application : How is this challenge useful? -> Gene Therapy

**Adeno-Associated Viruses (AAV)** are the workhorse of modern gene therapy, delivering therapeutic genes to treat:
- Spinal muscular atrophy (Zolgensma — $2.1M treatment)
- Hemophilia (multiple FDA-approved therapies)
- Inherited blindness (Luxturna)
- And dozens more in clinical trials

Engineering better AAV capsids is crucial for:
- **Improved tissue targeting** (getting genes to the right cells)
- **Immune evasion** (avoiding patient antibodies)
- **Packaging efficiency** (fitting larger therapeutic genes)

### The Challenge: Epistasis

The core difficulty is **epistasis** — when mutations interact non-additively.

Because of this non-linear landscape, the challenge is set up as an extrapolation problem:
- You will train on variants with **1–2 mutations** (which are cheap and easy to test experimentally).
- You must predict the fitness of variants with **3+ mutations** (which are exponentially expensive to explore in a lab).
- Simple linear models usually fail catastrophically here. You must capture **combinatorial interactions**.

This mirrors the **real experimental bottleneck** in protein engineering labs worldwide.

---

## The Task

- **Input**: Amino acid sequence of an AAV capsid variant (~735 residues)
- **Output**: Fitness score (float) — how well the capsid packages viral DNA
- **Primary metric**: **Spearman's rank correlation (ρ)** — because experimentalists care about ranking, not exact values
- **Bonus metric**: **Precision@100** — can you identify the true top-100 variants? This reflects the real constraint: labs can only test ~100 candidates

### Dataset Split

| Split | Samples | Mutations | Purpose |
|-------|---------|-----------|---------|
| **Train** | 31,807 | 1–2 | Model training |
| **Public test** | 25,388 | 3+ | Leaderboard score |
| **Private test** | 25,388 | 3+ | Final ranking |

The distribution shift (1-2 mutations → 3+ mutations) is **the core challenge** and reflects real-world constraints in protein engineering.

---

## Our Pedagogical Innovation

### We transformed FLIP into a learning experience

The original FLIP benchmark is a research tool used in top-tier ML/biology papers. We adapted it for education by:

1. **Scaffolded Baselines (See `solution/submission.py`)**
   Machine learning algorithms cannot read raw strings; sequences must be translated into numerical representations. We provide a starting codebase with progressive difficulty:
   - **Level 1 (The active baseline)**: A simple Ridge regression model. It translates the string into numbers by computing the amino acid composition (e.g., frequency of Alanine) and K-mer frequencies (counting 2-letter and 3-letter substrings like "AC" or "ACD"). This yields a ρ ≈ 0.42.
   - **Level 2 (Your turn to explore)**: We added 4 biophysical features, pedagogically chosen to teach domain knowledge integration. The method to compute those features is in the `submission.py` example file! Using them will teach you how to integrate biological domain knowledge:
     * *Net Charge*: Influences the capsid's interactions with cellular receptors.
     * *Average Hydrophobicity*: Drives the core folding of the protein.
     * *Hydrophobic Moment (Amphipathicity)*: Indicates if secondary structures have distinct hydrophobic/hydrophilic faces.
     * *Instability Index*: A proxy for how rapidly the protein might degrade.
   - **Level 3 (Advanced)**: Implement Protein Language Models (ESM-2, ProtBERT), gradient boosting, or neural networks to capture deep epistatic interactions.

2. **Dual Evaluation Metrics**
   - **Spearman ρ**: Standard ML metric, easy to interpret
   - **Precision@100**: Real-world relevance

3. **Industrialized Pipeline**
   - Fully automated ingestion → training → prediction → scoring
   - Docker-based reproducibility
   - Local testing tools before submission
   - Clear error messages and debugging support

4. **Documentation**
   - Jupyter notebook with full EDA, visualization, and baseline explanation
   - Biological context for every design choice
   - Ideas for improvement with increasing difficulty

---

## 🚀 What Participants Will Learn

- **Feature engineering for biological sequences**: k-mers, composition, physicochemical properties
- **Handling distribution shift**: Training and test data are fundamentally different (epistasis)
- **Choosing metrics wisely**: When correlation != practical utility
- **Protein language models**: How to use pre-trained embeddings (ESM-2, ProtBERT) for biology
- **Ensemble methods**: Combining diverse models for robustness
- **Real-world constraints**: Computational biology isn't just about accuracy, it's about guiding experiments

## Structure of the bundle

- `competition.yaml`: configuration file for the Codabench competition, specifying phases, tasks, and evaluation metrics.
- `ingestion_program/`: contains the ingestion program that will be run on participant's submissions. It is responsible for loading the submission code, training the model on the AAV fitness data, and generating predictions on the test sequences. It contains:
    * `metadata.yaml`: a file describing how to run the ingestion program for Codabench.
    * `ingestion.py`: loads the submission code and produces predictions evaluated by the `scoring_program`. The `submission.py` must define a `get_model()` function returning a scikit-learn compatible model. This model is fitted on training sequences calling `fit`, and `predict` is used to generate fitness score predictions on the test data.
- `scoring_program/`: contains the scoring program that evaluates the predictions. It computes **Spearman's rank correlation** between predicted and true fitness scores. It contains:
    * `metadata.yaml`: a file describing how to run the scoring program for Codabench.
    * `scoring.py`: loads the predictions from the ingestion program and produces a JSON file containing the Spearman correlation on both public and private test sets, as well as runtime.
- `solution/`: contains a baseline submission that participants can use as a reference. It implements a `SequenceFitnessRegressor` using amino acid composition and k-mer features with Ridge regression. Participants must submit a `submission.py` file with a `get_model()` function returning a scikit-learn compatible model.
- `*_phase/`: contains the data for a given phase, including input sequences and reference fitness labels. Running `tools/setup_data.py` will generate the data for the development phase.
- `pages/`: contains markdown files rendered as web pages in the Codabench competition.
- `requirements.txt`: contains the required Python dependencies to run the challenge.

## Submission format

Participants must submit a `submission.py` file containing a `get_model()` function:

```python
from sklearn.base import BaseEstimator, RegressorMixin

class MyModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # X is a DataFrame with a 'sequence' column (amino acid strings)
        # y contains the fitness scores
        ...
        return self

    def predict(self, X):
        # Return predicted fitness scores
        ...
        return predictions

def get_model():
    return MyModel()
```

Submit by zipping the file:
```bash
zip submission.zip submission.py
```

## Extra scripts in the `tools/` folder

- `tools/setup_data.py`: script to load and preprocess the AAV fitness data for the competition phases.
- `tools/create_bundle.py`: script to create the Codabench bundle archive from the repository structure.
- `tools/Dockerfile`: Dockerfile to build the Docker image used to run the ingestion and scoring programs.
- `tools/run_docker.py`: convenience script to build and test the Docker image locally. See [below](#setting-up-and-testing-the-docker-image) for more details.

## Instructions to create the Codabench bundle

Make sure that the `setup_data.py` script has been run to generate the data for the competition.

Then, run the `create_bundle.py` script to create the Codabench bundle archive:

```bash
python create_bundle.py
```

You can then upload the generated `bundle.zip` file to Codabench on this [page](https://www.codabench.org/competitions/upload/).

## Instructions to test the bundle locally

To test the ingestion program, run:

```bash
python ingestion_program/ingestion.py --data-dir dev_phase/input_data/ --output-dir ingestion_res/ --submission-dir solution/
```

To test the scoring program, run:

```bash
python scoring_program/scoring.py --reference-dir dev_phase/reference_data/ --output-dir scoring_res --prediction-dir ingestion_res/
```

### Setting up and testing the Docker image

For convenience, a Python script `tools/run_docker.py` is provided to build the Docker image and run the ingestion and scoring programs inside the container. This script requires the `docker` Python package:

```bash
pip install docker
python tools/run_docker.py
```

You can also perform these steps manually. First, build the Docker image from the `Dockerfile`:

```bash
docker build -t docker-image tools
```

Then run the ingestion and scoring programs:

```bash
docker run --rm -u root \
    -v "./ingestion_program":"/app/ingestion_program" \
    -v "./dev_phase/input_data":/app/input_data \
    -v "./ingestion_res":/app/output \
    -v "./solution":/app/ingested_program \
    --name ingestion docker-image \
        python /app/ingestion_program/ingestion.py

docker run --rm -u root \
    -v "./scoring_program":"/app/scoring_program" \
    -v "./dev_phase/reference_data":/app/input/ref \
    -v "./ingestion_res":/app/input/res \
    -v "./scoring_res":/app/output \
    --name scoring docker-image \
        python /app/scoring_program/scoring.py
```

### CI for the bundle

This repo defines a CI for the bundle, which builds a Docker image from `tools/Dockerfile` and runs `tools/setup_data.py` followed by the ingestion and scoring programs.
