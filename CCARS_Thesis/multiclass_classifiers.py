"""
Multi-Classifier Evaluation Module for CCARS

Provides multiple classifiers (SVM, Random Forest, etc.) to validate
CCARS wavelength selection across different algorithms.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from multiclass_plsda import MultiClassPLSDA


class MultiClassifierFramework:
    """
    Framework to evaluate multiple classifiers with CCARS-selected wavelengths
    
    For each classifier, trains and evaluates on:
    - Selected wavelengths (from CCARS)
    - Full spectrum (baseline)
    """
    
    def __init__(self, n_components=3, random_state=42):
        """
        Initialize multi-classifier framework
        
        Args:
            n_components: Number of PLS components for PLS-DA
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.n_components = n_components
        
        # Define classifiers
        self.classifiers = {
            'PLS-DA': MultiClassPLSDA(n_components=n_components),
            'SVM-Linear': SVC(kernel='linear', random_state=random_state),
            'SVM-RBF': SVC(kernel='rbf', C=10, gamma='scale', random_state=random_state),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                random_state=random_state,
                n_jobs=-1
            ),
            'k-NN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        }
        
        self.results = {}
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                          classifier_name, wavelength_type):
        """
        Train and evaluate a single classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            classifier_name: Name of classifier to use
            wavelength_type: Description (e.g., '10_selected', '204_full')
        
        Returns:
            Dictionary with results
        """
        classifier = self.classifiers[classifier_name]
        
        # Train
        start_time = time.time()
        classifier.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        predict_time = time.time() - start_time
        
        # Compute metrics
        results = {
            'classifier': classifier_name,
            'wavelength_type': wavelength_type,
            'n_features': X_train.shape[1],
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_time': train_time,
            'predict_time': predict_time,
            'predictions': y_pred,
            'model': classifier  # ADD: Include model for permutation testing
        }
        
        return results
    
    def evaluate_all_classifiers(self, X_train, y_train, X_test, y_test,
                                 wavelength_type='unknown'):
        """
        Evaluate all classifiers on given data
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            wavelength_type: Description of wavelength selection
        
        Returns:
            Dictionary mapping classifier names to results
        """
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Evaluating All Classifiers - {wavelength_type}")
        print(f"{'='*70}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        for clf_name in self.classifiers.keys():
            print(f"\n  Training {clf_name}...", end=' ')
            
            try:
                result = self.train_and_evaluate(
                    X_train, y_train, X_test, y_test,
                    clf_name, wavelength_type
                )
                results[clf_name] = result
                
                print(f"✓ Accuracy: {result['accuracy']:.4f} "
                      f"(Train: {result['train_time']:.2f}s)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                results[clf_name] = None
        
        return results
    
    def compare_selected_vs_full(self, X_train_sel, X_train_full,
                                 y_train, X_test_sel, X_test_full, y_test,
                                 n_selected):
        """
        Compare all classifiers on selected vs full wavelengths
        
        Args:
            X_train_sel: Training data with selected wavelengths
            X_train_full: Training data with all wavelengths
            y_train: Training labels
            X_test_sel: Test data with selected wavelengths
            X_test_full: Test data with all wavelengths
            y_test: Test labels
            n_selected: Number of selected wavelengths
        
        Returns:
            Dictionary with comprehensive comparison
        """
        # Evaluate on selected wavelengths
        results_selected = self.evaluate_all_classifiers(
            X_train_sel, y_train, X_test_sel, y_test,
            wavelength_type=f'{n_selected}_selected'
        )
        
        # Evaluate on full spectrum
        results_full = self.evaluate_all_classifiers(
            X_train_full, y_train, X_test_full, y_test,
            wavelength_type=f'{X_train_full.shape[1]}_full'
        )
        
        # Combine results
        comparison = {
            'selected': results_selected,
            'full': results_full,
            'summary': self._create_summary(results_selected, results_full, n_selected)
        }
        
        return comparison
    
    def _create_summary(self, results_sel, results_full, n_selected):
        """Create summary comparison table"""
        summary = []
        
        for clf_name in self.classifiers.keys():
            if results_sel.get(clf_name) and results_full.get(clf_name):
                sel = results_sel[clf_name]
                full = results_full[clf_name]
                
                summary.append({
                    'classifier': clf_name,
                    'accuracy_selected': sel['accuracy'],
                    'accuracy_full': full['accuracy'],
                    'accuracy_diff': sel['accuracy'] - full['accuracy'],
                    'accuracy_retention': sel['accuracy'] / full['accuracy'] if full['accuracy'] > 0 else 0,
                    'f1_selected': sel['f1_weighted'],
                    'f1_full': full['f1_weighted'],
                    'train_time_selected': sel['train_time'],
                    'train_time_full': full['train_time'],
                    'speedup': full['train_time'] / sel['train_time'] if sel['train_time'] > 0 else 0,
                    'n_wavelengths_selected': n_selected,
                    'n_wavelengths_full': full['n_features'],
                    'reduction_percent': (1 - n_selected / full['n_features']) * 100
                })
        
        return summary
    
    def print_comparison_table(self, summary):
        """Print formatted comparison table"""
        print("\n" + "="*100)
        print("COMPREHENSIVE CLASSIFIER COMPARISON")
        print("="*100)
        
        print(f"\n{'Classifier':<20} {'Selected':<12} {'Full':<12} {'Diff':<10} {'Retention':<12}")
        print("-"*100)
        
        for row in summary:
            retention_pct = row['accuracy_retention'] * 100
            print(f"{row['classifier']:<20} "
                  f"{row['accuracy_selected']:<12.2%} "
                  f"{row['accuracy_full']:<12.2%} "
                  f"{row['accuracy_diff']:<10.2%} "
                  f"{retention_pct:<12.1f}%")
        
        print("\n" + "="*100)


if __name__ == '__main__':
    """Test multi-classifier framework"""
    print("Testing Multi-Classifier Framework\n")
    
    # Create dummy data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=204, n_informative=50,
        n_classes=16, n_clusters_per_class=1, random_state=42
    )
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select top 10 features (dummy selection)
    selected_indices = list(range(10))
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # Test framework
    framework = MultiClassifierFramework(n_components=3)
    
    comparison = framework.compare_selected_vs_full(
        X_train_sel, X_train, y_train,
        X_test_sel, X_test, y_test,
        n_selected=10
    )
    
    framework.print_comparison_table(comparison['summary'])
    
    print("\n✓ Multi-Classifier Framework Test Complete!")
