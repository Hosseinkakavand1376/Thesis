# HSI Feature Selection - Complete Analysis Pipeline

## Quick Start

**Run complete analysis (both datasets, all methods):**
```bash
python scripts/RUN_ALL_MASTER.py
```

**Quick test:**
```bash
python scripts/RUN_ALL_MASTER.py --quick-test --datasets salinas
```

**Single dataset:**
```bash
python scripts/RUN_ALL_MASTER.py --datasets salinas
```

---

## File Organization

### Main Executable (Master Pipeline)
- **`scripts/RUN_ALL_MASTER.py`** - Main orchestrator, runs everything

### Core CCARS Scripts  
- `scripts/main_hsi_cars_comprehensive.py` - CCARS with all classifiers
- `scripts/main_hsi_cars.py` - Original CCARS implementation
- `scripts/compare_all_methods.py` - Compare MRMR/BOSS/FISHER

### Core Modules (Required by scripts)
- `multiclass_cars.py` - CCARS algorithm implementation
- `multiclass_plsda.py` - PLS-DA classifier
- `feature_selection_methods.py` - MRMR, BOSS, FISHER implementations
- `multiclass_classifiers.py` - Multi-classifier framework

### Data Loading & Preprocessing
- `hsi_data_loader.py` - Load hyperspectral datasets
- `hsi_preprocessing.py` - Log10 + SNV preprocessing
- `hsi_config.py` - Dataset configurations  
- `hsi_evaluation.py` - Model evaluation metrics

### Visualization
- `src/visualization/create_publication_plots.py` - Publication plots
- `src/visualization/create_advanced_visualizations.py` - Advanced plots
- `src/visualization/generate_comparison_report.py` - Comparison reports

### Utilities
- `utils/wavelength_loader.py` - Load wavelengths from CSV
- `utils/cleanup_results.py` - Clean result directories

### Data Files
- `data/wavelengths_salinas_corrected_204.csv` - Salinas wavelengths
- `data/indianpines_wavelengths_200.csv` - Indian Pines wavelengths

### Documentation
- `docs/FEATURE_SELECTION_README.md` - Feature selection methods
- `docs/MASTER_PIPELINE_README.md` - Pipeline documentation
- `README.md` - This file

---

## Output Directories (Created after running)

```
HSI_CARS_comprehensive/         # CCARS results
Feature_Selection_Comparison/   # MRMR/BOSS/FISHER comparison
Publication_Plots/              # Publication-quality plots  
Advanced_Visualizations/        # Advanced analysis plots
```

---

## Methods

- **CCARS** - Competitive Adaptive Reweighted Sampling (500 MC runs)
- **MRMR** - Minimum Redundancy Maximum Relevance
- **BOSS** - Bootstrap-based selection (100 iterations)
- **FISHER** - Fisher Score ranking

## Classifiers

PLS-DA, SVM (Linear, RBF), Random Forest, k-NN

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy

---

## Citation

Based on Nicola Dilillo's CCARS methodology:
"Enhancing lettuce classification: Optimizing spectral wavelength selection via CCARS and PLS-DA"  
Smart Agricultural Technology, Volume 11, 2025
