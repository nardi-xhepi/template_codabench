from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

class SequenceFitnessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.kmer_vocab = None
    
    def _build_vocab(self, sequences, k=3):
        all_kmers = set()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                all_kmers.add(seq[i:i+k])
        return sorted(all_kmers)
    
    def _seq_to_features(self, seq):
        # AA composition
        counts = Counter(seq)
        aa_freq = [counts.get(aa, 0) / len(seq) for aa in AMINO_ACIDS]
        # K-mer counts
        kmer_counts = Counter([seq[i:i+3] for i in range(len(seq)-2)])
        kmer_freq = [kmer_counts.get(k, 0) / (len(seq)-2) for k in self.kmer_vocab]
        return np.array(aa_freq + kmer_freq + [len(seq)/735.0])
    
    def fit(self, X, y):
        sequences = X['sequence'].values
        self.kmer_vocab = self._build_vocab(sequences)
        X_feat = np.array([self._seq_to_features(s) for s in sequences])
        X_scaled = self.scaler.fit_transform(X_feat)
        # BUG: y might be DataFrame, sklearn expects array!
        # Handle DataFrame y
        import pandas as pd
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        sequences = X['sequence'].values
        X_feat = np.array([self._seq_to_features(s) for s in sequences])
        return self.model.predict(self.scaler.transform(X_feat))

def get_model():
    return SequenceFitnessRegressor()
