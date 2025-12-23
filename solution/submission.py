from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter

# Standard amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class SequenceFitnessRegressor(BaseEstimator, RegressorMixin):
    """
    Fitness prediction model using multiple sequence features:
    - Amino acid composition (frequency of each amino acid)
    - K-mer frequencies (2-mers and 3-mers)
    - Sequence length
    
    Uses Ridge regression for robust high-dimensional fitting.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.kmer2_vocab = None
        self.kmer3_vocab = None
    
    def _get_aa_composition(self, sequence):
        """Get amino acid composition (frequency of each AA)."""
        counts = Counter(sequence)
        total = len(sequence)
        return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS])
    
    def _get_kmer_counts(self, sequence, k, vocab):
        """Get k-mer frequency vector."""
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        counts = Counter(kmers)
        total = len(kmers) if kmers else 1
        return np.array([counts.get(kmer, 0) / total for kmer in vocab])
    
    def _build_kmer_vocab(self, sequences, k):
        """Build vocabulary of k-mers from sequences."""
        all_kmers = set()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                all_kmers.add(seq[i:i+k])
        return sorted(all_kmers)
    
    def _sequence_to_features(self, sequence):
        """Convert sequence to feature vector."""
        features = []
        
        # Amino acid composition (20 features)
        features.extend(self._get_aa_composition(sequence))
        
        # 2-mer frequencies
        features.extend(self._get_kmer_counts(sequence, 2, self.kmer2_vocab))
        
        # 3-mer frequencies
        features.extend(self._get_kmer_counts(sequence, 3, self.kmer3_vocab))
        
        # Sequence length (normalized)
        features.append(len(sequence) / 735.0)  # Normalize by typical AAV length
        
        return np.array(features)
    
    def fit(self, X, y):
        """Fit the model on sequences and fitness values.
        
        Args:
            X: DataFrame with 'sequence' column
            y: Target values (DataFrame or array-like)
        """
        sequences = X['sequence'].values
        
        # Handle y as DataFrame or array
        import pandas as pd
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        elif hasattr(y, 'values'):
            y = y.values.ravel()
        
        # Build k-mer vocabularies
        print("  Building k-mer vocabularies...")
        self.kmer2_vocab = self._build_kmer_vocab(sequences, 2)
        self.kmer3_vocab = self._build_kmer_vocab(sequences, 3)
        
        print(f"  2-mer vocab size: {len(self.kmer2_vocab)}")
        print(f"  3-mer vocab size: {len(self.kmer3_vocab)}")
        
        # Convert all sequences to features
        print("  Extracting features...")
        X_features = np.array([self._sequence_to_features(seq) for seq in sequences])
        
        print(f"  Total features: {X_features.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit Ridge regression
        print("  Training Ridge regression...")
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """Predict fitness for new sequences."""
        sequences = X['sequence'].values
        X_features = np.array([self._sequence_to_features(seq) for seq in sequences])
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)


def get_model():
    """Return the model for the FLIP AAV fitness prediction challenge."""
    return SequenceFitnessRegressor(alpha=10.0)
