# Alternative Dataset Download Script
#
# The automatic download from ehu.eus can be unreliable.
# This script provides alternative methods to get the Salinas dataset.

"""
OPTION 1: Manual Download (Recommended)
========================================

1. Visit one of these mirrors:
   
   Primary Source:
   http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
   
   Alternative Sources:
   https://rslab.ut.ac.ir/dataDownload
   https://www.google.com/drive/your-link-here (if user has Google Drive link)

2. Download these two files:
   - Salinas_corrected.mat (~26 MB)
   - Salinas_gt.mat (smaller file)

3. Place them in:
   CCARS_Thesis/dataset/

4. Run the pipeline again:
   python main_hsi_cars.py --dataset salinas --runs 10 --iterations 20


OPTION 2: Use wget/curl (if available)
=======================================

# For Salinas:
wget http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat -P dataset/
wget http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat -P dataset/

# Or with curl:
curl -o dataset/Salinas_corrected.mat http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat
curl -o dataset/Salinas_gt.mat http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat


OPTION 3: Test with Indian Pines (Smaller Dataset)
===================================================

Indian Pines is smaller and may download successfully:

python main_hsi_cars.py --dataset indian_pines --runs 10 --iterations 20


OPTION 4: Use Synthetic Data for Testing
=========================================

Run the individual module tests which use synthetic data:

python multiclass_plsda.py    # Test multi-class PLS-DA
python multiclass_cars.py     # Test CARS algorithm
python hsi_evaluation.py      # Test evaluation functions

These will verify the implementation works correctly even without real HSI data.
"""

# Quick test with minimal data
if __name__ == '__main__':
    print(__doc__)
