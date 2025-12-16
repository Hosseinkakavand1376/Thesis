"""
MASTER PIPELINE - Complete Feature Selection and Classification Analysis

This script runs the ENTIRE analysis pipeline from scratch:
1. Feature Selection: CCARS (500 runs), MRMR, BOSS, FISHER
2. Classification: PLS-DA, SVM-Linear, SVM-RBF, Random Forest, k-NN
3. Datasets: Salinas and Indian Pines
4. Visualizations: All publication-quality plots

Usage:
    python RUN_ALL_MASTER.py

Options:
    --datasets: salinas indian_pines (default: both)
    --skip-ccars: Skip CCARS if already run
    --quick-test: Run with reduced parameters for testing
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import json


class MasterPipeline:
    """Orchestrates the complete analysis pipeline"""
    
    def __init__(self, datasets=None, skip_ccars=False, quick_test=False):
        if datasets is None:
            datasets = ['salinas', 'indian_pines']
        self.datasets = datasets
        self.skip_ccars = skip_ccars
        self.quick_test = quick_test
        self.results_summary = {}
        
        # Configuration
        if quick_test:
            self.feature_counts = [20, 30]
            self.cars_runs = 100
            self.classifiers = ['SVM-RBF', 'Random Forest']
            self.methods = ['MRMR', 'BOSS', 'FISHER']
        else:
            self.feature_counts = [10, 20, 30, 50]
            self.cars_runs = 500
            self.classifiers = ['PLS-DA', 'SVM-Linear', 'SVM-RBF', 'Random Forest', 'k-NN']
            self.methods = ['MRMR', 'BOSS', 'FISHER']
    
    def print_header(self, text):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80 + "\n")
    
    def run_command(self, cmd, description):
        """Run a command and track results"""
        self.print_header(description)
        print(f"Command: {' '.join(cmd)}\n")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            elapsed = time.time() - start_time
            print(f"\n✅ {description} - Completed in {elapsed:.1f}s")
            return True, elapsed
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} - Failed after {elapsed:.1f}s")
            print(f"Error: {e}")
            return False, elapsed
    
    def step1_run_ccars(self, dataset):
        """Step 1: Run CCARS comprehensive analysis (500 runs, all classifiers)"""
        if self.skip_ccars:
            print(f"⏭️  Skipping CCARS for {dataset} (already run)")
            return True, 0
        
        cmd = ['python', 'main_hsi_cars_comprehensive.py', '--dataset', dataset, '--wavelengths']
        cmd.extend([str(w) for w in self.feature_counts])
        cmd.extend(['--cars_runs', str(self.cars_runs)])
        
        return self.run_command(cmd, f"STEP 1: CCARS Analysis - {dataset.upper()}")
    
    def step2_run_other_methods(self, dataset):
        """Step 2: Run MRMR, BOSS, FISHER with all classifiers"""
        cmd = ['python', 'compare_all_methods.py', '--dataset', dataset, '--features']
        cmd.extend([str(w) for w in self.feature_counts])
        cmd.append('--methods')
        cmd.extend(self.methods)
        cmd.append('--classifiers')
        cmd.extend(self.classifiers)
        
        return self.run_command(cmd, f"STEP 2: MRMR/BOSS/FISHER - {dataset.upper()}")
    
    def step3_create_publication_plots(self, dataset):
        """Step 3: Generate all publication plots"""
        cmd = ['python', 'create_publication_plots.py', '--dataset', dataset]
        
        return self.run_command(cmd, f"STEP 3: Publication Plots - {dataset.upper()}")
    
    def step4_create_advanced_plots(self, dataset):
        """Step 4: Generate advanced visualizations"""
        cmd = ['python', 'create_advanced_visualizations.py', '--dataset', dataset]
        
        return self.run_command(cmd, f"STEP 4: Advanced Visualizations - {dataset.upper()}")
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        self.print_header("MASTER PIPELINE - COMPLETE ANALYSIS")
        
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Feature counts: {self.feature_counts}")
        print(f"CCARS runs: {self.cars_runs}")
        print(f"Methods: CCARS, {', '.join(self.methods)}")
        print(f"Classifiers: {', '.join(self.classifiers)}")
        print(f"Quick test mode: {self.quick_test}")
        
        total_start = time.time()
        
        for dataset in self.datasets:
            dataset_start = time.time()
            self.results_summary[dataset] = {}
            
            # Step 1: CCARS
            success, elapsed = self.step1_run_ccars(dataset)
            self.results_summary[dataset]['ccars'] = {'success': success, 'time': elapsed}
            
            if not success and not self.skip_ccars:
                print(f"\n⚠️  CCARS failed for {dataset}, continuing with other methods...")
            
            # Step 2: Other methods
            success, elapsed = self.step2_run_other_methods(dataset)
            self.results_summary[dataset]['other_methods'] = {'success': success, 'time': elapsed}
            
            if not success:
                print(f"\n⚠️  Other methods failed for {dataset}, skipping plots...")
                continue
            
            # Step 3: Publication plots
            success, elapsed = self.step3_create_publication_plots(dataset)
            self.results_summary[dataset]['pub_plots'] = {'success': success, 'time': elapsed}
            
            # Step 4: Advanced plots
            success, elapsed = self.step4_create_advanced_plots(dataset)
            self.results_summary[dataset]['adv_plots'] = {'success': success, 'time': elapsed}
            
            dataset_elapsed = time.time() - dataset_start
            print(f"\n✅ {dataset.upper()} complete in {dataset_elapsed/60:.1f} minutes")
        
        total_elapsed = time.time() - total_start
        
        # Final summary
        self.print_final_summary(total_elapsed)
    
    def print_final_summary(self, total_time):
        """Print final execution summary"""
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        print(f"Total execution time: {total_time/60:.1f} minutes\n")
        
        for dataset, results in self.results_summary.items():
            print(f"\n{dataset.upper()}:")
            print("-" * 40)
            for step, data in results.items():
                status = "✅" if data['success'] else "❌"
                print(f"  {status} {step:15s}: {data['time']:.1f}s")
        
        # Save summary
        summary_path = Path('pipeline_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
        
        print(f"\n📄 Summary saved to: {summary_path}")
        
        # Print output locations
        self.print_header("OUTPUT LOCATIONS")
        
        for dataset in self.datasets:
            print(f"\n{dataset.upper()}:")
            print(f"  CCARS Results:        HSI_CARS_comprehensive/{dataset}/")
            print(f"  Method Comparison:    Feature_Selection_Comparison/{dataset}/")
            print(f"  Publication Plots:    Publication_Plots/{dataset}/")
            print(f"  Advanced Plots:       Advanced_Visualizations/{dataset}/")
        
        print("\n" + "="*80)
        print("🎉 MASTER PIPELINE COMPLETE!")
        print("="*80)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Run complete feature selection analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis on both datasets
  python RUN_ALL_MASTER.py
  
  # Run only Salinas
  python RUN_ALL_MASTER.py --datasets salinas
  
  # Quick test (reduced parameters)
  python RUN_ALL_MASTER.py --quick-test
  
  # Skip CCARS if already run
  python RUN_ALL_MASTER.py --skip-ccars
        """
    )
    
    parser.add_argument('--datasets', nargs='+', 
                       choices=['salinas', 'indian_pines'],
                       default=['salinas', 'indian_pines'],
                       help='Datasets to process')
    
    parser.add_argument('--skip-ccars', action='store_true',
                       help='Skip CCARS execution (use existing results)')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced parameters for testing')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = MasterPipeline(
        datasets=args.datasets,
        skip_ccars=args.skip_ccars,
        quick_test=args.quick_test
    )
    
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()
