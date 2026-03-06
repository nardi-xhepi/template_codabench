### Creators of the challenge
Creators: Alizée Mesnard, Cécile Luc, Hamza Boukhriss, Laurian Truong, Nardi Xhepi, Nicolas Bieszczad

# FLIP AAV Capsid Optimization Challenge

A Codabench competition on **protein fitness prediction from amino acid sequences**.

Given a mutated AAV2 capsid protein sequence (~735 amino acids), the goal is to **predict its fitness score** — how well the capsid packages viral DNA. The challenge is evaluated using **Spearman's rank correlation (ρ)**.

The key difficulty is **epistasis**: the model is trained on variants with 1–2 mutations but must generalize to variants with 3 or more mutations, where mutation effects are non-additive.

## Background

The data comes from the [FLIP](https://github.com/J-SNACKKB/FLIP) (Fitness Landscape Inference for Proteins) benchmark, specifically the **AAV** dataset. It was generated through a deep mutational scanning (DMS) experiment measuring the packaging fitness of ~280,000 variants of the AAV2 capsid protein VP1 ([Bryant et al., 2021](https://www.nature.com/articles/s41587-021-00922-7)).

AAV vectors are the leading platform for **gene therapy**. Engineering better capsids is crucial for improving tissue targeting, evading immune responses, and increasing packaging efficiency for larger therapeutic genes.

## The Task

- **Input**: Amino acid sequence of an AAV capsid variant (~735 characters)
- **Output**: Fitness score (float) — higher = better packaging ability
- **Metric**: Spearman's rank correlation coefficient (ρ)

## Dataset

| Split | Samples | Mutations | Purpose |
|-------|---------|-----------|---------|
| Train | 31,807 | 1–2 | Model training |
| Public test | 25,388 | 3+ | Leaderboard score |
| Private test | 25,388 | 3+ | Final ranking |

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
