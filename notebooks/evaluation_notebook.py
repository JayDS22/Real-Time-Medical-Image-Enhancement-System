"""
Model Evaluation and Results Analysis
This script demonstrates how to evaluate the enhancement quality.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.metrics import ImageQualityMetrics
from src.visualization import MedicalImageVisualizer

sns.set_style('whitegrid')


def load_results():
    """Load original and enhanced images"""
    print("="*60)
    print("LOADING RESULTS")
    print("="*60)
    
    original_path = 'data/processed/sample.nii.gz'
    enhanced_path = 'results/enhanced_sample.nii.gz'
    
    if Path(original_path).exists() and Path(enhanced_path).exists():
        import nibabel as nib
        original = nib.load(original_path).get_fdata()
        enhanced = nib.load(enhanced_path).get_fdata()
        
        print(f"\nOriginal shape: {original.shape}")
        print(f"Enhanced shape: {enhanced.shape}")
        
        return original, enhanced, True
    else:
        print("\nFiles not found. Using synthetic data for demonstration.")
        print("To use real data, run:")
        print("  1. python src/preprocessing.py --input_dir data/raw --output_dir data/processed")
        print("  2. python src/inference.py --input data/processed/sample.nii.gz --output results/enhanced.nii.gz")
        
        # Create synthetic data
        original = np.random.randn(128, 128, 128) * 0.3 + 0.5
        noise = np.random.randn(*original.shape) * 0.15
        noisy = original + noise
        enhanced = original + noise * 0.3  # Simulated enhancement
        
        return noisy, enhanced, False


def compute_quality_metrics(original, enhanced):
    """Compute all quality metrics"""
    print("\n" + "="*60)
    print("COMPUTING QUALITY METRICS")
    print("="*60)
    
    metrics = ImageQualityMetrics()
    
    # Compute metrics
    results = {}
    results['snr_original'] = metrics.compute_snr(original)
    results['snr_enhanced'] = metrics.compute_snr(enhanced)
    results['ssim'] = metrics.compute_ssim(original, enhanced)
    results['psnr'] = metrics.compute_psnr(original, enhanced)
    results['mse'] = metrics.compute_mse(original, enhanced)
    results['mae'] = metrics.compute_mae(original, enhanced)
    results['contrast_original'] = metrics.compute_contrast(original)
    results['contrast_enhanced'] = metrics.compute_contrast(enhanced)
    results['sharpness_original'] = metrics.compute_sharpness(original)
    results['sharpness_enhanced'] = metrics.compute_sharpness(enhanced)
    
    # Compute improvements
    results['snr_improvement'] = ((results['snr_enhanced'] - results['snr_original']) / 
                                   results['snr_original'] * 100)
    results['contrast_improvement'] = ((results['contrast_enhanced'] - results['contrast_original']) / 
                                        results['contrast_original'] * 100)
    results['sharpness_improvement'] = ((results['sharpness_enhanced'] - results['sharpness_original']) / 
                                         results['sharpness_original'] * 100)
    
    # Print results
    print("\n" + "="*50)
    print("QUALITY METRICS")
    print("="*50)
    print(f"\nSNR:")
    print(f"  Original: {results['snr_original']:.2f} dB")
    print(f"  Enhanced: {results['snr_enhanced']:.2f} dB")
    print(f"  Improvement: {results['snr_improvement']:.1f}%")
    
    print(f"\nSSIM: {results['ssim']:.4f}")
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"MSE: {results['mse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    
    print(f"\nContrast:")
    print(f"  Original: {results['contrast_original']:.4f}")
    print(f"  Enhanced: {results['contrast_enhanced']:.4f}")
    print(f"  Improvement: {results['contrast_improvement']:.1f}%")
    
    print(f"\nSharpness:")
    print(f"  Original: {results['sharpness_original']:.4f}")
    print(f"  Enhanced: {results['sharpness_enhanced']:.4f}")
    print(f"  Improvement: {results['sharpness_improvement']:.1f}%")
    print("="*50)
    
    return results


def visualize_comparison(original, enhanced):
    """Visualize comparison of original and enhanced"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = MedicalImageVisualizer()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Slice comparison
    print("\nCreating slice comparison...")
    visualizer.plot_slice_comparison(
        original,
        enhanced,
        slice_idx=original.shape[2]//2,
        axis=2,
        save_path='results/slice_comparison.png'
    )
    
    # Multi-slice comparison
    print("Creating multi-slice comparison...")
    visualizer.plot_multi_slice_comparison(
        original,
        enhanced,
        num_slices=5,
        axis=2,
        save_path='results/multi_slice_comparison.png'
    )


def visualize_metrics(results):
    """Visualize metrics comparison"""
    print("\n" + "="*60)
    print("VISUALIZING METRICS")
    print("="*60)
    
    visualizer = MedicalImageVisualizer()
    visualizer.plot_metrics_comparison(
        results,
        save_path='results/metrics_comparison.png'
    )


def statistical_analysis(original, enhanced):
    """Perform statistical analysis"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    difference = enhanced - original
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of differences
    axes[0].hist(difference.ravel(), bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].set_title('Intensity Difference Distribution', fontweight='bold')
    axes[0].set_xlabel('Difference')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [original.ravel(), enhanced.ravel()]
    bp = axes[1].boxplot(data_to_plot, labels=['Original', 'Enhanced'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    axes[1].set_title('Intensity Distribution Comparison', fontweight='bold')
    axes[1].set_ylabel('Intensity')
    axes[1].grid(True, alpha=0.3)
    
    # Scatter plot
    sample_indices = np.random.choice(original.size, 10000, replace=False)
    axes[2].scatter(
        original.ravel()[sample_indices],
        enhanced.ravel()[sample_indices],
        alpha=0.3,
        s=1,
        c='steelblue'
    )
    axes[2].plot([original.min(), original.max()],
                 [original.min(), original.max()],
                 'r--', label='y=x', linewidth=2)
    axes[2].set_title('Original vs Enhanced Intensity', fontweight='bold')
    axes[2].set_xlabel('Original')
    axes[2].set_ylabel('Enhanced')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/statistical_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved statistical analysis to: results/statistical_analysis.png")
    plt.show()
    
    # Print statistics
    print(f"\nDifference Statistics:")
    print(f"  Mean: {difference.mean():.6f}")
    print(f"  Std: {difference.std():.6f}")
    print(f"  Min: {difference.min():.6f}")
    print(f"  Max: {difference.max():.6f}")


def save_results(results):
    """Save results to JSON"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_path = Path('results/evaluation_metrics.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("MODEL EVALUATION AND RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    original, enhanced, using_real_data = load_results()
    
    # Compute metrics
    results = compute_quality_metrics(original, enhanced)
    
    # Visualize comparison
    visualize_comparison(original, enhanced)
    
    # Visualize metrics
    visualize_metrics(results)
    
    # Statistical analysis
    statistical_analysis(original, enhanced)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    if not using_real_data:
        print("\nNote: This demonstration used synthetic data.")
        print("For real evaluation, process your medical images first.")
    
    print("\nAll visualizations saved to: results/")


if __name__ == '__main__':
    main()