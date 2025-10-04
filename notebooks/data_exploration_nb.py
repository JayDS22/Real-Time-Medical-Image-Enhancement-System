"""
Medical Image Data Exploration
This script explores medical imaging datasets and visualizes the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import seaborn as sns
import pandas as pd

sns.set_style('whitegrid')


def load_and_explore_sample():
    """Load and explore a sample medical image"""
    print("="*60)
    print("LOADING MEDICAL IMAGES")
    print("="*60)
    
    data_dir = Path('data/processed')
    sample_files = list(data_dir.glob('*.nii.gz'))
    
    if not sample_files:
        print("No data files found. Please run preprocessing first.")
        print("Run: python src/preprocessing.py --input_dir data/raw --output_dir data/processed")
        return None, None
    
    img_path = sample_files[0]
    img = nib.load(img_path)
    data = img.get_fdata()
    
    print(f"\nLoaded: {img_path.name}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Spacing: {img.header.get_zooms()}")
    
    return data, sample_files


def visualize_slices(data):
    """Visualize central slices in three orthogonal views"""
    print("\n" + "="*60)
    print("VISUALIZING ORTHOGONAL SLICES")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial
    axes[0].imshow(data[data.shape[0]//2, :, :], cmap='gray')
    axes[0].set_title('Axial View', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Coronal
    axes[1].imshow(data[:, data.shape[1]//2, :], cmap='gray')
    axes[1].set_title('Coronal View', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Sagittal
    axes[2].imshow(data[:, :, data.shape[2]//2], cmap='gray')
    axes[2].set_title('Sagittal View', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Orthogonal Views of Medical Image', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/orthogonal_views.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: results/orthogonal_views.png")
    plt.show()


def plot_intensity_distribution(data):
    """Plot intensity distribution histogram"""
    print("\n" + "="*60)
    print("INTENSITY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data.ravel(), bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    plt.xlabel('Intensity Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Intensity Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = data.mean()
    std_val = data.std()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'±1 STD')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/intensity_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: results/intensity_distribution.png")
    plt.show()
    
    print(f"\nIntensity Statistics:")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std Dev: {std_val:.4f}")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")
    print(f"  Median: {np.median(data):.4f}")


def analyze_dataset_statistics(sample_files):
    """Collect and analyze statistics from multiple files"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    stats = []
    
    max_files = min(10, len(sample_files))
    print(f"\nAnalyzing first {max_files} files...")
    
    for file_path in sample_files[:max_files]:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        stats.append({
            'filename': file_path.name,
            'shape': str(data.shape),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        })
    
    df = pd.DataFrame(stats)
    
    print("\nDataset Overview:")
    print(df.to_string(index=False))
    
    print("\nAggregate Statistics:")
    print(f"  Average Mean: {df['mean'].mean():.4f}")
    print(f"  Average Std: {df['std'].mean():.4f}")
    print(f"  Global Min: {df['min'].min():.4f}")
    print(f"  Global Max: {df['max'].max():.4f}")
    
    return df


def main():
    """Main exploration function"""
    print("\n" + "="*70)
    print("MEDICAL IMAGE DATA EXPLORATION")
    print("="*70)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Load sample
    data, sample_files = load_and_explore_sample()
    
    if data is None:
        return
    
    # Visualize slices
    visualize_slices(data)
    
    # Plot intensity distribution
    plot_intensity_distribution(data)
    
    # Analyze dataset
    if sample_files:
        df = analyze_dataset_statistics(sample_files)
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()