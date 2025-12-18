import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

EVAL_SETS = ["test", "private_test"]


def compute_spearman(predictions, targets):
    """Compute Spearman's rank correlation coefficient."""
    y_pred = predictions.values.flatten()
    y_true = targets.values.flatten()
    
    mask = ~(pd.isna(y_pred) | pd.isna(y_true))
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    if len(y_pred) == 0:
        return 0.0
    
    corr, _ = spearmanr(y_pred, y_true)
    return 0.0 if pd.isna(corr) else corr


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')
        predictions = pd.read_csv(prediction_dir / f'{eval_set}_predictions.csv')
        targets = pd.read_csv(reference_dir / f'{eval_set}_labels.csv')
        scores[eval_set] = float(compute_spearman(predictions, targets))

    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scoring program for codabench")
    parser.add_argument("--reference-dir", type=str, default="/app/input/ref")
    parser.add_argument("--prediction-dir", type=str, default="/app/input/res")
    parser.add_argument("--output-dir", type=str, default="/app/output")
    args = parser.parse_args()
    main(Path(args.reference_dir), Path(args.prediction_dir), Path(args.output_dir))
