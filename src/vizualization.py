"""
Visualization tools for medical image enhancement results
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import seaborn as sns


class MedicalImageVisualizer:
    """Visualization tools for 3D medical images"""
    
    def __init__(self, figsize=(15, 10), cmap='gray'):
        self.figsize = figsize
        self.cmap = cmap
        sns.set_style("whitegrid")
    
    def plot_slice_comparison(
        self,
        original,
        enhanced,
        slice_idx=None,
        axis=2,
        save_path=None
    ):
        """
        Plot comparison of original and enhanced slices
        
        Args:
            original: Original 3D volume
            enhanced: Enhanced 3D volume
            slice_idx: Index of slice to display (if None, uses middle slice)
            axis: Axis along which to slice (0, 1, or 2)
            save_path: Path to save figure
        """
        if slice_idx is None:
            slice_idx = original.shape[axis] // 2
        
        # Extract slices
        if axis == 0:
            orig_slice = original[slice_idx, :, :]
            enh_slice = enhanced[slice_idx, :, :]
        elif axis == 1:
            orig_slice = original[:, slice_idx, :]
            enh_slice = enhanced[:, slice_idx, :]
        else:
            orig_slice = original[:, :, slice_idx]
            enh_slice = enhanced[:, :, slice_idx]
        
        # Compute difference
        diff_slice = enh_slice - orig_slice
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        # Original
        im1 = axes[0, 0].imshow(orig_slice, cmap=self.cmap)
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Enhanced
        im2 = axes[0, 1].imshow(enh_slice, cmap=self.cmap)
        axes[0, 1].set_title('Enhanced', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Difference
        im3 = axes[0, 2].imshow(diff_slice, cmap='RdBu_r')
        axes[0, 2].set_title('Difference', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # Histograms
        axes[1, 0].hist(orig_slice.ravel(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(enh_slice.ravel(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Enhanced Histogram')
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].hist(diff_slice.ravel(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 2].set_title('Difference Histogram')
        axes[1, 2].set_xlabel('Intensity Difference')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.suptitle(f'Slice {slice_idx} along axis {axis}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_multi_slice_comparison(
        self,
        original,
        enhanced,
        num_slices=5,
        axis=2,
        save_path=None
    ):
        """Plot multiple slice comparisons"""
        # Select evenly spaced slices
        total_slices = original.shape[axis]
        slice_indices = np.linspace(
            total_slices // 4,
            3 * total_slices // 4,
            num_slices,
            dtype=int
        )
        
        fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 3, 6))
        
        for i, slice_idx in enumerate(slice_indices):
            # Extract slices
            if axis == 0:
                orig_slice = original[slice_idx, :, :]
                enh_slice = enhanced[slice_idx, :, :]
            elif axis == 1:
                orig_slice = original[:, slice_idx, :]
                enh_slice = enhanced[:, slice_idx, :]
            else:
                orig_slice = original[:, :, slice_idx]
                enh_slice = enhanced[:, :, slice_idx]
            
            # Plot original
            axes[0, i].imshow(orig_slice, cmap=self.cmap)
            axes[0, i].set_title(f'Original\nSlice {slice_idx}')
            axes[0, i].axis('off')
            
            # Plot enhanced
            axes[1, i].imshow(enh_slice, cmap=self.cmap)
            axes[1, i].set_title(f'Enhanced\nSlice {slice_idx}')
            axes[1, i].axis('off')
        
        plt.suptitle('Multi-Slice Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_3d_volume(self, volume, threshold=None, save_path=None):
        """
        Plot 3D rendering of volume (orthogonal views)
        
        Args:
            volume: 3D numpy array
            threshold: Threshold for display (if None, uses median)
            save_path: Path to save figure
        """
        if threshold is None:
            threshold = np.median(volume)
        
        # Get middle slices
        mid_z = volume.shape[0] // 2
        mid_y = volume.shape[1] // 2
        mid_x = volume.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial view (z-axis)
        axes[0].imshow(volume[mid_z, :, :], cmap=self.cmap)
        axes[0].set_title('Axial View (z-plane)', fontweight='bold')
        axes[0].axis('off')
        
        # Coronal view (y-axis)
        axes[1].imshow(volume[:, mid_y, :], cmap=self.cmap)
        axes[1].set_title('Coronal View (y-plane)', fontweight='bold')
        axes[1].axis('off')
        
        # Sagittal view (x-axis)
        axes[2].imshow(volume[:, :, mid_x], cmap=self.cmap)
        axes[2].set_title('Sagittal View (x-plane)', fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle('3D Volume - Orthogonal Views', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of metrics
        
        Args:
            metrics_dict: Dictionary containing metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # SNR comparison
        snr_data = {
            'Original': metrics_dict.get('snr_original', 0),
            'Enhanced': metrics_dict.get('snr_enhanced', 0)
        }
        axes[0, 0].bar(snr_data.keys(), snr_data.values(), color=['#3498db', '#2ecc71'])
        axes[0, 0].set_title('Signal-to-Noise Ratio (SNR)', fontweight='bold')
        axes[0, 0].set_ylabel('SNR (dB)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # SSIM
        ssim_value = metrics_dict.get('ssim', 0)
        axes[0, 1].bar(['SSIM'], [ssim_value], color='#9b59b6')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('Structural Similarity Index (SSIM)', fontweight='bold')
        axes[0, 1].set_ylabel('SSIM Score')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].text(0, ssim_value + 0.05, f'{ssim_value:.4f}', 
                       ha='center', fontweight='bold')
        
        # Contrast comparison
        contrast_data = {
            'Original': metrics_dict.get('contrast_original', 0),
            'Enhanced': metrics_dict.get('contrast_enhanced', 0)
        }
        axes[1, 0].bar(contrast_data.keys(), contrast_data.values(), 
                      color=['#e74c3c', '#f39c12'])
        axes[1, 0].set_title('Image Contrast (RMS)', fontweight='bold')
        axes[1, 0].set_ylabel('Contrast')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Improvements
        improvements = {
            'SNR': metrics_dict.get('snr_improvement', 0),
            'Contrast': metrics_dict.get('contrast_improvement', 0),
            'Sharpness': metrics_dict.get('sharpness_improvement', 0)
        }
        axes[1, 1].bar(improvements.keys(), improvements.values(), 
                      color=['#1abc9c', '#e67e22', '#3498db'])
        axes[1, 1].set_title('Quality Improvements (%)', fontweight='bold')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Image Quality Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def visualize_enhancement_results(
    original_path,
    enhanced_path,
    output_dir=None
):
    """
    Complete visualization pipeline for enhancement results
    
    Args:
        original_path: Path to original image
        enhanced_path: Path to enhanced image
        output_dir: Directory to save visualizations
    """
    print("Loading images...")
    
    # Load images
    original = nib.load(original_path).get_fdata()
    enhanced = nib.load(enhanced_path).get_fdata()
    
    # Create visualizer
    visualizer = MedicalImageVisualizer()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot slice comparison
    print("Generating slice comparison...")
    save_path = output_dir / 'slice_comparison.png' if output_dir else None
    visualizer.plot_slice_comparison(original, enhanced, save_path=save_path)
    
    # Plot multi-slice comparison
    print("Generating multi-slice comparison...")
    save_path = output_dir / 'multi_slice_comparison.png' if output_dir else None
    visualizer.plot_multi_slice_comparison(original, enhanced, save_path=save_path)
    
    # Plot 3D views
    print("Generating 3D orthogonal views...")
    save_path = output_dir / '3d_original.png' if output_dir else None
    visualizer.plot_3d_volume(original, save_path=save_path)
    
    save_path = output_dir / '3d_enhanced.png' if output_dir else None
    visualizer.plot_3d_volume(enhanced, save_path=save_path)
    
    print("Visualization complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize enhancement results')
    parser.add_argument('--original', type=str, required=True, help='Original image path')
    parser.add_argument('--enhanced', type=str, required=True, help='Enhanced image path')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_enhancement_results(args.original, args.enhanced, args.output_dir)
