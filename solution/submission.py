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

    def _get_bio_features(self, sequence):
        """Extract biologically-relevant physicochemical features."""
        CHARGED = {'K': 1, 'R': 1, 'D': -1, 'E': -1}
        KYTE_DOOLITTLE = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        net_charge = sum(CHARGED.get(aa, 0) for aa in sequence)
        normalized_charge = net_charge / len(sequence)

        avg_hydrophobicity = np.mean([KYTE_DOOLITTLE.get(aa, 0) for aa in sequence])

        window_size = 11
        angle = 100
        moments = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            sin_sum = sum(KYTE_DOOLITTLE.get(aa, 0) * np.sin(np.radians(angle * j))
                          for j, aa in enumerate(window))
            cos_sum = sum(KYTE_DOOLITTLE.get(aa, 0) * np.cos(np.radians(angle * j))
                          for j, aa in enumerate(window))
            moment = np.sqrt(sin_sum**2 + cos_sum**2) / window_size
            moments.append(moment)
        hydrophobic_moment = max(moments) if moments else 0.0

        instability_proxy = 0
        for i in range(len(sequence) - 1):
            aa1, aa2 = sequence[i], sequence[i+1]
            if (KYTE_DOOLITTLE.get(aa1, 0) > 2.0 and aa2 in CHARGED) or \
               (aa1 in CHARGED and KYTE_DOOLITTLE.get(aa2, 0) > 2.0):
                instability_proxy += 1
        instability_index = (instability_proxy / (len(sequence) - 1)) * 100

        return [normalized_charge, avg_hydrophobicity, hydrophobic_moment, instability_index]

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
