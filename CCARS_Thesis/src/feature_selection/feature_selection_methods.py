"""
Additional Feature Selection Methods: MRMR, BOSS, FISHER

Integrates multiple feature selection algorithms to compare with CCARS:
- MRMR: Minimum Redundancy Maximum Relevance
- BOSS: Bootstrapping-based feature selection  
- FISHER: Fisher Score ranking
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')


class MRMRSelector:
    """
    Minimum Redundancy Maximum Relevance (MRMR) Feature Selection
    
    Selects features that maximize relevance with target while
    minimizing redundancy with already selected features.
    """
    
    def __init__(self, n_features=30):
        """
        Args:
            n_features: Number of features to select
        """
        self.n_features = n_features
        self.selected_features_ = None
        self.scores_ = None
    
    def fit(self, X, y):
        """
        Select features using MRMR
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
        """
        n_samples, n_total_features = X.shape
        
        # Calculate relevance (mutual information with target)
        relevance = mutual_info_classif(X, y, random_state=42)
        
        # Initialize
        selected = []
        remaining = list(range(n_total_features))
        
        # Select first feature (highest relevance)
        first_idx = np.argmax(relevance)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        print(f"MRMR selecting {self.n_features} features...")
        
        # Iteratively select features
        for k in range(1, min(self.n_features, n_total_features)):
            if k % 10 == 0:
                print(f"  Selected {k}/{self.n_features} features...")
            
            max_score = -np.inf
            best_feature = None
            
            for i in remaining:
                # Relevance of feature i
                rel = relevance[i]
                
                # Redundancy with already selected features
                if len(selected) > 0:
                    redundancy = 0
                    for j in selected:
                        # Correlation as redundancy measure
                        redundancy += np.abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
                    redundancy /= len(selected)
                else:
                    redundancy = 0
                
                # MRMR score
                score = rel - redundancy
                
                if score > max_score:
                    max_score = score
                    best_feature = i
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
        
        self.selected_features_ = np.array(selected)
        print(f"✓ MRMR selected {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X):
        """Transform X to selected features"""
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class BOSSSelector:
    """
    Bootstrapping-based Orthogonal Signal Selection (BOSS)
    
    Uses bootstrap sampling to identify stable/important features
    """
    
    def __init__(self, n_features=30, n_bootstrap=100, random_state=42):
        """
        Args:
            n_features: Number of features to select
            n_bootstrap: Number of bootstrap iterations
            random_state: Random seed
        """
        self.n_features = n_features
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.selected_features_ = None
        self.importance_scores_ = None
    
    def fit(self, X, y):
        """
        Select features using BOSS
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        n_samples, n_total_features = X.shape
        np.random.seed(self.random_state)
        
        # Track feature selection frequency
        feature_counts = np.zeros(n_total_features)
        
        print(f"BOSS with {self.n_bootstrap} bootstrap samples...")
        
        for b in range(self.n_bootstrap):
            if (b + 1) % 20 == 0:
                print(f"  Bootstrap iteration {b+1}/{self.n_bootstrap}...")
            
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Compute feature importance (F-statistic)
            f_stats, _ = f_classif(X_boot, y_boot)
            
            # Select top features in this bootstrap
            top_features = np.argsort(f_stats)[-self.n_features:]
            feature_counts[top_features] += 1
        
        # Select features by frequency
        self.importance_scores_ = feature_counts / self.n_bootstrap
        self.selected_features_ = np.argsort(feature_counts)[-self.n_features:]
        
        print(f"✓ BOSS selected {len(self.selected_features_)} features")
        print(f"  Average selection frequency: {self.importance_scores_[self.selected_features_].mean():.2%}")
        
        return self
    
    def transform(self, X):
        """Transform X to selected features"""
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class FisherScoreSelector:
    """
    Fisher Score Feature Selection
    
    Ranks features by Fisher criterion:
    (between-class variance) / (within-class variance)
    """
    
    def __init__(self, n_features=30):
        """
        Args:
            n_features: Number of features to select
        """
        self.n_features = n_features
        self.selected_features_ = None
        self.scores_ = None
    
    def _compute_fisher_score(self, X, y):
        """
        Compute Fisher score for each feature
        
        Fisher Score = (Σ n_i (μ_i - μ)²) / (Σ n_i σ_i²)
        where:
        - n_i: samples in class i
        - μ_i: mean of class i
        - μ: overall mean
        - σ_i²: variance of class i
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        scores = np.zeros(n_features)
        
        # Overall statistics
        overall_mean = np.mean(X, axis=0)
        
        for f in range(n_features):
            feature_data = X[:, f]
            
            # Between-class variance
            between_var = 0
            # Within-class variance
            within_var = 0
            
            for c in classes:
                class_mask = (y == c)
                class_data = feature_data[class_mask]
                n_c = len(class_data)
                
                if n_c > 1:
                    class_mean = np.mean(class_data)
                    class_var = np.var(class_data)
                    
                    between_var += n_c * (class_mean - overall_mean[f]) ** 2
                    within_var += n_c * class_var
            
            # Fisher score
            if within_var > 0:
                scores[f] = between_var / within_var
            else:
                scores[f] = 0
        
        return scores
    
    def fit(self, X, y):
        """
        Select features using Fisher Score
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        print(f"Computing Fisher scores for {X.shape[1]} features...")
        
        # Compute Fisher scores
        self.scores_ = self._compute_fisher_score(X, y)
        
        # Select top features
        self.selected_features_ = np.argsort(self.scores_)[-self.n_features:]
        
        print(f"✓ Fisher Score selected {len(self.selected_features_)} features")
        print(f"  Score range: {self.scores_[self.selected_features_].min():.2f} - "
              f"{self.scores_[self.selected_features_].max():.2f}")
        
        return self
    
    def transform(self, X):
        """Transform X to selected features"""
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class UnifiedFeatureSelector:
    """
    Unified interface for all feature selection methods
    """
    
    def __init__(self, method='mrmr', n_features=30, **kwargs):
        """
        Args:
            method: 'mrmr', 'boss', 'fisher', or 'ccars'
            n_features: Number of features to select
            **kwargs: Additional arguments for specific methods
        """
        self.method = method.lower()
        self.n_features = n_features
        
        if self.method == 'mrmr':
            self.selector = MRMRSelector(n_features=n_features)
        elif self.method == 'boss':
            n_bootstrap = kwargs.get('n_bootstrap', 100)
            random_state = kwargs.get('random_state', 42)
            self.selector = BOSSSelector(
                n_features=n_features,
                n_bootstrap=n_bootstrap,
                random_state=random_state
            )
        elif self.method == 'fisher':
            self.selector = FisherScoreSelector(n_features=n_features)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: mrmr, boss, fisher")
    
    def fit(self, X, y):
        """Fit the selector"""
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        """Transform to selected features"""
        return self.selector.transform(X)
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        return self.selector.fit_transform(X, y)
    
    @property
    def selected_features_(self):
        """Get selected feature indices"""
        return self.selector.selected_features_


# Test the implementations
if __name__ == '__main__':
    print("Testing Feature Selection Methods\n")
    print("=" * 70)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=200,
        n_informative=30,
        n_classes=16,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n")
    
    # Test each method
    methods = ['mrmr', 'boss', 'fisher']
    n_features = 30
    
    for method in methods:
        print(f"\n{method.upper()}")
        print("-" * 70)
        
        selector = UnifiedFeatureSelector(method=method, n_features=n_features)
        X_selected = selector.fit_transform(X, y)
        
        print(f"Selected features shape: {X_selected.shape}")
        print(f"Selected indices (first 10): {selector.selected_features_[:10]}")
    
    print("\n" + "=" * 70)
    print("✓ All methods tested successfully!")
