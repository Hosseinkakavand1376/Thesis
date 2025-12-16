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
import zipfile
from datetime import datetime


class MasterPipeline:
    """Orchestrates the complete analysis pipeline"""
    
    def __init__(self, datasets=None, skip_ccars=False, quick_test=False, pls_components=None, use_validation=False, use_optimization=False):
        if datasets is None:
            datasets = ['salinas', 'indian_pines']
        self.datasets = datasets
        self.skip_ccars = skip_ccars
        self.quick_test = quick_test
        self.use_validation = use_validation
        self.use_optimization = use_optimization
        self.results_summary = {}
        
        # PLS components to test
        if pls_components is None:
            self.pls_components = [2, 3, 4]
        else:
            self.pls_components = pls_components
        
        # Configuration
        if quick_test:
            self.feature_counts = [20, 30]
            self.cars_runs = 100
            self.classifiers = ['SVM-RBF', 'Random Forest']
            self.methods = ['CCARS', 'MRMR', 'BOSS', 'FISHER']
            # Still test all components in quick mode
        else:
            self.feature_counts = [10, 20, 30, 50]
            self.cars_runs = 500
            self.classifiers = ['PLS-DA', 'SVM-Linear', 'SVM-RBF', 'Random Forest', 'k-NN']
            self.methods = ['CCARS', 'MRMR', 'BOSS', 'FISHER']
    
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
    
    
    def step1_run_ccars(self, dataset, n_components=3):
        """Step 1: Run CCARS comprehensive analysis (500 runs, all classifiers)"""
        if self.skip_ccars:
            print(f"⏭️  Skipping CCARS for {dataset} (already run)")
            return True, 0
        
        cmd = ['python', 'main_hsi_cars_comprehensive.py', '--dataset', dataset]
        # Use wavelength range for optimization (will be auto-determined)
        cmd.extend(['--cars_runs', str(self.cars_runs)])
        cmd.extend(['--components', str(n_components)])
        
        # Add optimization flag if enabled
        if self.use_optimization:
            cmd.append('--optimize_wavelengths')
        
        # Add validation flag if enabled
        if self.use_validation:
            cmd.append('--validation')
        
        return self.run_command(cmd, f"STEP 1: CCARS Analysis - {dataset.upper()} (PLS {n_components})")
    
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
        self.print_header("MASTER PIPELINE - COMPLETE ANALYSIS WITH MULTI-COMPONENT TESTING")
        
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"PLS Components to test: {self.pls_components}")
        print(f"CCARS runs: {self.cars_runs}")
        print(f"Methods: CCARS, {', '.join(self.methods)}")
        print(f"Classifiers: {', '.join(self.classifiers)}")
        print(f"Quick test mode: {self.quick_test}")
        print(f"Wavelength optimization: {'ENHANCED (range [5-50])' if self.use_optimization else 'NICOLA (fixed [10,20,30,50])'}")
        print(f"Hold-out validation: {'ON' if self.use_validation else 'OFF'}")
        
        total_start = time.time()
        
        # Loop over datasets and components
        for dataset in self.datasets:
            dataset_start = time.time()
            self.results_summary[dataset] = {}
            
            for n_components in self.pls_components:
                component_key = f'components_{n_components}'
                self.results_summary[dataset][component_key] = {}
                
                self.print_header(f"{dataset.upper()} - PLS COMPONENTS: {n_components}")
                
                # Step 1: CCARS with this component count
                success, elapsed = self.step1_run_ccars(dataset, n_components=n_components)
                self.results_summary[dataset][component_key]['ccars'] = {
                    'success': success, 'time': elapsed
                }
                
                if not success and not self.skip_ccars:
                    print(f"\n⚠️  CCARS failed for {dataset} with {n_components} components, continuing...")
            
            # After all components tested, run comparison and plotting steps
            print(f"\n{'='*80}")
            print(f"Running comparison and visualization steps for {dataset.upper()}")
            print(f"{'='*80}\n")
            
            # Step 2: Compare with other feature selection methods
            # Note: This uses default component (3) for comparison
            success, elapsed = self.step2_run_other_methods(dataset)
            self.results_summary[dataset]['comparison'] = {
                'success': success, 'time': elapsed
            }
            
            # Step 3: Create publication-quality plots
            success, elapsed = self.step3_create_publication_plots(dataset)
            self.results_summary[dataset]['publication_plots'] = {
                'success': success, 'time': elapsed
            }
            
            # Step 4: Create advanced visualizations
            success, elapsed = self.step4_create_advanced_plots(dataset)
            self.results_summary[dataset]['advanced_plots'] = {
                'success': success, 'time': elapsed
            }
            
            dataset_elapsed = time.time() - dataset_start
            print(f"\n✅ {dataset.upper()} complete in {dataset_elapsed/60:.1f} minutes")
        
        total_elapsed = time.time() - total_start
        
        # Final summary
        self.print_final_summary(total_elapsed)
        
        # Archive all outputs
        self.create_output_archive()
    
    def print_final_summary(self, total_time):
        """Print final execution summary"""
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        print(f"Total execution time: {total_time/60:.1f} minutes\n")
        
        for dataset, component_results in self.results_summary.items():
            print(f"\n{dataset.upper()}:")
            print("-" * 40)
            for component_key, steps in component_results.items():
                # Check if this is a flat structure (e.g., comparison, plots) or nested (components_N)
                if isinstance(steps, dict) and 'success' in steps and 'time' in steps:
                    # Flat structure: directly at dataset level
                    status = "✅" if steps['success'] else "❌"
                    print(f"  {status} {component_key:15s}: {steps['time']:.1f}s")
                else:
                    # Nested structure: component level with sub-steps
                    print(f"  {component_key}:")
                    for step_name, data in steps.items():
                        status = "✅" if data['success'] else "❌"
                        print(f"    {status} {step_name:15s}: {data['time']:.1f}s")
        
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
    
    def create_output_archive(self):
        """Create a zip archive of all output directories"""
        print("\n" + "="*80)
        print("📦 CREATING OUTPUT ARCHIVE")
        print("="*80)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"CCARS_Results_{timestamp}.zip"
        
        # Define output directories to archive
        output_dirs = [
            'HSI_CARS_comprehensive',
            'Feature_Selection_Comparison',
            'Publication_Plots',
            'Advanced_Visualizations'
        ]
        
        # Check which directories exist
        existing_dirs = [d for d in output_dirs if Path(d).exists()]
        
        if not existing_dirs:
            print("\n⚠️  No output directories found to archive")
            return
        
        print(f"\nArchiving {len(existing_dirs)} output directories...")
        
        try:
            with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for output_dir in existing_dirs:
                    dir_path = Path(output_dir)
                    print(f"  Adding {output_dir}/...")
                    
                    # Walk through directory and add all files
                    for file_path in dir_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to('.')
                            zipf.write(file_path, arcname)
                
                # Also add pipeline summary
                if Path('pipeline_summary.json').exists():
                    zipf.write('pipeline_summary.json', 'pipeline_summary.json')
                    print(f"  Adding pipeline_summary.json")
            
            # Get file size
            archive_size = Path(archive_name).stat().st_size / (1024 * 1024)  # MB
            
            print(f"\n✅ Archive created successfully!")
            print(f"   File: {archive_name}")
            print(f"   Size: {archive_size:.2f} MB")
            print(f"   Contains: {len(existing_dirs)} directories + summary")
            print("\n📥 Ready for download!")
            
        except Exception as e:
            print(f"\n❌ Error creating archive: {e}")
        
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
  
  # Enable wavelength optimization (auto-detect best/min/knee)
  python RUN_ALL_MASTER.py --optimize
  
  # Enable hold-out validation
  python RUN_ALL_MASTER.py --validation
  
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
    
    parser.add_argument('--optimize', action='store_true',
                       help='Enable wavelength optimization mode (tests range [5-50] and auto-selects best/min/knee)')
    
    parser.add_argument('--validation', action='store_true',
                       help='Enable hold-out validation (final independent evaluation)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = MasterPipeline(
        datasets=args.datasets,
        skip_ccars=args.skip_ccars,
        quick_test=args.quick_test,
        use_validation=args.validation,
        use_optimization=args.optimize
    )
    
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()
