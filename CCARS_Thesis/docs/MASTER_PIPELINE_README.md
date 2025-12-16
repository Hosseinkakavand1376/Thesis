# MASTER PIPELINE - Complete Analysis Guide

## 🚀 **ONE-COMMAND EXECUTION**

Run the complete analysis on both datasets with all methods from scratch:

```bash
python RUN_ALL_MASTER.py
```

That's it! This single command will:
1. ✅ Run CCARS (500 runs) on Salinas and Indian Pines
2. ✅ Run MRMR, BOSS, FISHER on both datasets  
3. ✅ Test with ALL classifiers (PLS-DA, SVM, RF, k-NN)
4. ✅ Generate ALL publication plots
5. ✅ Create advanced visualizations
6. ✅ Save comprehensive results

---

## 📂 **What Gets Created**

After running, you'll have:

```
CCARS_Thesis/
├── HSI_CARS_comprehensive/
│   ├── salinas/
│   │   ├── comprehensive_results.csv       # All CCARS results
│   │   ├── selected_wavelengths.csv         # Selected wavelengths
│   │   ├── confusion_matrix_selected.png
│   │   └── wavelength_frequency.png
│   └── indian_pines/
│       └── (same structure)
│
├── Feature_Selection_Comparison/
│   ├── salinas/
│   │   ├── all_methods_comparison.csv      # MRMR/BOSS/FISHER results
│   │   └── comparison_report.txt
│   └── indian_pines/
│       └── (same structure)
│
├── Publication_Plots/
│   ├── salinas/
│   │   ├── salinas_mean_spectra.png        # Mean spectra per class
│   │   ├── salinas_wavelength_overlay.png  # Selected wavelengths
│   │   └── salinas_summary_table.csv       # Complete results table
│   └── indian_pines/
│       └── (same structure)
│
├── Advanced_Visualizations/
│   ├── salinas/
│   │   ├── salinas_confusion_matrices_grid.png
│   │   ├── salinas_feature_importance.png
│   │   ├── salinas_method_comparison_boxplot.png
│   │   ├── salinas_accuracy_vs_features_detailed.png
│   │   └── salinas_performance_summary_table.png
│   └── indian_pines/
│       └── (same structure)
│
└── pipeline_summary.json                   # Execution summary
```

---

## ⏱️ **Estimated Runtime**

**Full Analysis (both datasets):**
- CCARS (500 runs): ~15-20 minutes per dataset
- MRMR/BOSS/FISHER: ~2-3 minutes per dataset
- Plotting: ~1 minute per dataset
- **Total: ~40-50 minutes**

**Quick Test Mode:**
- CCARS (100 runs): ~3-4 minutes per dataset
- Other methods: ~1 minute per dataset  
- **Total: ~10 minutes**

---

## 🎛️ **Options & Customization**

### **Run Only Specific Datasets**

```bash
# Only Salinas
python RUN_ALL_MASTER.py --datasets salinas

# Only Indian Pines
python RUN_ALL_MASTER.py --datasets indian_pines
```

### **Quick Test (Reduced Parameters)**

Perfect for testing before full run:

```bash
python RUN_ALL_MASTER.py --quick-test
```

Changes:
- CCARS: 100 runs instead of 500
- Feature counts: [20, 30] instead of [10, 20, 30, 50]
- Classifiers: Only SVM-RBF and Random Forest

### **Skip CCARS (If Already Run)**

If you already have CCARS results and just want to re-run other methods:

```bash
python RUN_ALL_MASTER.py --skip-ccars
```

### **Combine Options**

```bash
# Quick test on Salinas only
python RUN_ALL_MASTER.py --datasets salinas --quick-test

# Skip CCARS, only Salinas
python RUN_ALL_MASTER.py --datasets salinas --skip-ccars
```

---

## 📊 **Results Summary**

After completion, check `pipeline_summary.json` for execution times:

```json
{
  "salinas": {
    "ccars": {"success": true, "time": 1234.5},
    "other_methods": {"success": true, "time": 156.7},
    "pub_plots": {"success": true, "time": 45.2},
    "adv_plots": {"success": true, "time": 32.1}
  },
  "indian_pines": {
    "ccars": {"success": true, "time": 987.6},
    ...
  }
}
```

---

## 🎓 **Best Results to Report**

### **Salinas:**
- **Best:** MRMR (50 features) + Random Forest = **92.50%**
- **CCARS:** 30 features + Random Forest = **88.93%** (96% retention)
- **Fastest:** FISHER (any features) = 0.1s selection time

### **Indian Pines:**
- **Best:** Random Forest (full) = **88.20%**
- **CCARS:** 30 features + Random Forest = **77.17%** (87% retention)
- **Note:** k-NN achieved 101% retention (better than full!)

---

## 📋 **Publication-Ready Outputs**

### **Tables**
- `Publication_Plots/salinas/salinas_summary_table.csv`
- Complete results for all methods × features × classifiers

### **Key Plots**
1. **Mean Spectra per Class** - Shows spectral signatures
2. **Confusion Matrices** - Classification performance  
3. **Wavelength Overlay** - Shows selected wavelengths
4. **Feature Importance** - CCARS selection frequency
5. **Method Comparison** - Box plots comparing all methods
6. **Performance Summary** - Ranked table with metrics

---

## 🐛 **Troubleshooting**

### **If Pipeline Fails:**

1. **Check individual scripts work:**
   ```bash
   # Test CCARS
   python main_hsi_cars_comprehensive.py --dataset salinas --wavelengths 20 --cars_runs 50
   
   # Test other methods
   python compare_all_methods.py --dataset salinas --features 20 --methods MRMR
   
   # Test plotting
   python create_publication_plots.py --dataset salinas
   ```

2. **Check data files exist:**
   - `Salinas_corrected.mat`
   - `Salinas_gt.mat`
   - `wavelengths_salinas_corrected_204.csv`
   - Same for Indian Pines

3. **Check dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn scipy
   ```

---

## 📝 **Manual Step-by-Step**

If you prefer to run steps individually:

```bash
# Step 1: CCARS (both datasets)
python main_hsi_cars_comprehensive.py --dataset salinas --wavelengths 10 20 30 50 --cars_runs 500
python main_hsi_cars_comprehensive.py --dataset indian_pines --wavelengths 10 20 30 50 --cars_runs 500

# Step 2: Other methods (both datasets)
python compare_all_methods.py --dataset salinas --features 10 20 30 50 --methods MRMR BOSS FISHER --classifiers SVM-RBF "Random Forest"
python compare_all_methods.py --dataset indian_pines --features 10 20 30 50 --methods MRMR BOSS FISHER --classifiers SVM-RBF "Random Forest"

# Step 3: Publication plots
python create_publication_plots.py --dataset salinas
python create_publication_plots.py --dataset indian_pines

# Step 4: Advanced visualizations
python create_advanced_visualizations.py --dataset salinas
python create_advanced_visualizations.py --dataset indian_pines
```

---

## ✅ **Verification**

After completion, verify you have:

- [ ] CCARS results in `HSI_CARS_comprehensive/`
- [ ] Method comparison in `Feature_Selection_Comparison/`
- [ ] Publication plots in `Publication_Plots/`
- [ ] Advanced plots in `Advanced_Visualizations/`
- [ ] Summary tables (.csv files)
- [ ] All PNG plot files

---

## 🎉 **You're Done!**

- All results saved
- All plots generated
- Ready for thesis/publication

**Total output files: ~30-40 files** (results, plots, tables)

**Next step:** Write your thesis! 📝
