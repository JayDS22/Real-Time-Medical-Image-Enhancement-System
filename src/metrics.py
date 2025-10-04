"""
Evaluation Metrics for Medical Image Enhancement
Computes SNR, SSIM, PSNR, and other quality metrics
"""

import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
import argparse
from pathlib import Path
import json


class ImageQualityMetrics:
    """Compute image quality metrics"""
    
    @staticmethod
    def compute_snr(image, noise_region=None):
        """
        Compute Signal-to-Noise Ratio
        SNR = 20 * log10(mean_signal / std_noise)
        """
        if noise_region is not None:
            noise_std = np.std(noise_region)
        else:
            # Estimate noise from high-frequency components
            noise_std = ImageQualityMetrics.estimate_noise(image)
        
        signal_mean = np.mean(image)
        
        if noise_std == 0:
            return float('inf')
        
        snr = 20 * np.log10(np.abs(signal_mean) / noise_std)
        
        return snr
    
    @staticmethod
    def estimate_noise(image):
        """Estimate noise standard deviation using Median Absolute Deviation"""
        # Apply high-pass filter
        filtered = image - ndimage.gaussian_filter(image, sigma=1.0)
        
        # Compute MAD
        mad = np.median(np.abs(filtered - np.median(filtered)))
        
        # Convert to standard deviation
        sigma = 1.4826 * mad
        
        return sigma
    
    @staticmethod
    def compute_ssim(image1, image2, data_range=None):
        """
        Compute Structural Similarity Index (SSIM)
        """
        if data_range is None:
            data_range = max(image1.max() - image1.min(), image2.max() - image2.min())
        
        # Compute SSIM for 3D volumes
        ssim_value = ssim(
            image1,
            image2,
            data_range=data_range,
            win_size=7  # Use smaller window for 3D
        )
        
        return ssim_value
    
    @staticmethod
    def compute_psnr(image1, image2, data_range=None):
        """
        Compute Peak Signal-to-Noise Ratio (PSNR)
        """
        if data_range is None:
            data_range = max(image1.max() - image1.min(), image2.max() - image2.min())
        
        psnr_value = psnr(image1, image2, data_range=data_range)
        
        return psnr_value
    
    @staticmethod
    def compute_mse(image1, image2):
        """Compute Mean Squared Error"""
        mse = np.mean((image1 - image2) ** 2)
        return mse
    
    @staticmethod
    def compute_mae(image1, image2):
        """Compute Mean Absolute Error"""
        mae = np.mean(np.abs(image1 - image2))
        return mae
    
    @staticmethod
    def compute_contrast(image):
        """Compute image contrast (RMS contrast)"""
        return np.std(image)
    
    @staticmethod
    def compute_sharpness(image):
        """
        Compute image sharpness using Laplacian variance
        Higher values indicate sharper images
        """
        laplacian = ndimage.laplace(image)
        sharpness = np.var(laplacian)
        return sharpness
    
    @staticmethod
    def compute_entropy(image, bins=256):
        """Compute image entropy"""
        hist, _ = np.histogram(image.ravel(), bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    @staticmethod
    def compute_cnr(image, signal_roi, background_roi):
        """
        Compute Contrast-to-Noise Ratio
        CNR = |mean_signal - mean_background| / noise_std
        """
        signal_mean = np.mean(signal_roi)
        background_mean = np.mean(background_roi)
        noise_std = ImageQualityMetrics.estimate_noise(image)
        
        if noise_std == 0:
            return float('inf')
        
        cnr = np.abs(signal_mean - background_mean) / noise_std
        
        return cnr


def evaluate_enhancement(original_path, enhanced_path, output_path=None):
    """
    Evaluate enhancement quality by comparing original and enhanced images
    """
    print("Loading images...")
    
    # Load images
    original_img = nib.load(original_path)
    enhanced_img = nib.load(enhanced_path)
    
    original_data = original_img.get_fdata()
    enhanced_data = enhanced_img.get_fdata()
    
    # Ensure same shape
    if original_data.shape != enhanced_data.shape:
        print(f"Warning: Shape mismatch - Original: {original_data.shape}, Enhanced: {enhanced_data.shape}")
        return None
    
    print("Computing metrics...")
    
    # Initialize metrics calculator
    metrics_calc = ImageQualityMetrics()
    
    # Compute metrics
    results = {}
    
    # SNR
    results['snr_original'] = metrics_calc.compute_snr(original_data)
    results['snr_enhanced'] = metrics_calc.compute_snr(enhanced_data)
    results['snr_improvement'] = (
        (results['snr_enhanced'] - results['snr_original']) / results['snr_original'] * 100
    )
    
    # SSIM (structural similarity between original and enhanced)
    data_range = max(original_data.max() - original_data.min(), 
                     enhanced_data.max() - enhanced_data.min())
    results['ssim'] = metrics_calc.compute_ssim(original_data, enhanced_data, data_range)
    
    # PSNR
    results['psnr'] = metrics_calc.compute_psnr(original_data, enhanced_data, data_range)
    
    # MSE and MAE
    results['mse'] = metrics_calc.compute_mse(original_data, enhanced_data)
    results['mae'] = metrics_calc.compute_mae(original_data, enhanced_data)
    
    # Contrast
    results['contrast_original'] = metrics_calc.compute_contrast(original_data)
    results['contrast_enhanced'] = metrics_calc.compute_contrast(enhanced_data)
    results['contrast_improvement'] = (
        (results['contrast_enhanced'] - results['contrast_original']) / 
        results['contrast_original'] * 100
    )
    
    # Sharpness
    results['sharpness_original'] = metrics_calc.compute_sharpness(original_data)
    results['sharpness_enhanced'] = metrics_calc.compute_sharpness(enhanced_data)
    results['sharpness_improvement'] = (
        (results['sharpness_enhanced'] - results['sharpness_original']) / 
        results['sharpness_original'] * 100
    )
    
    # Entropy
    results['entropy_original'] = metrics_calc.compute_entropy(original_data)
    results['entropy_enhanced'] = metrics_calc.compute_entropy(enhanced_data)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nSignal-to-Noise Ratio (SNR):")
    print(f"  Original:    {results['snr_original']:.2f} dB")
    print(f"  Enhanced:    {results['snr_enhanced']:.2f} dB")
    print(f"  Improvement: {results['snr_improvement']:.2f}%")
    
    print(f"\nStructural Similarity (SSIM):")
    print(f"  {results['ssim']:.4f}")
    
    print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
    print(f"  {results['psnr']:.2f} dB")
    
    print(f"\nMean Squared Error (MSE):")
    print(f"  {results['mse']:.6f}")
    
    print(f"\nMean Absolute Error (MAE):")
    print(f"  {results['mae']:.6f}")
    
    print(f"\nContrast (RMS):")
    print(f"  Original:    {results['contrast_original']:.4f}")
    print(f"  Enhanced:    {results['contrast_enhanced']:.4f}")
    print(f"  Improvement: {results['contrast_improvement']:.2f}%")
    
    print(f"\nSharpness (Laplacian Variance):")
    print(f"  Original:    {results['sharpness_original']:.4f}")
    print(f"  Enhanced:    {results['sharpness_enhanced']:.4f}")
    print(f"  Improvement: {results['sharpness_improvement']:.2f}%")
    
    print(f"\nEntropy:")
    print(f"  Original: {results['entropy_original']:.4f}")
    print(f"  Enhanced: {results['entropy_enhanced']:.4f}")
    
    print("="*60)
    
    # Save results to JSON
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def batch_evaluate(original_dir, enhanced_dir, output_path=None):
    """Evaluate multiple image pairs"""
    original_dir = Path(original_dir)
    enhanced_dir = Path(enhanced_dir)
    
    # Find all image pairs
    original_files = list(original_dir.rglob('*.nii*'))
    
    all_results = []
    
    for original_file in original_files:
        # Find corresponding enhanced file
        relative_path = original_file.relative_to(original_dir)
        enhanced_file = enhanced_dir / relative_path
        
        if not enhanced_file.exists():
            print(f"Warning: Enhanced file not found for {original_file}")
            continue
        
        print(f"\nEvaluating {original_file.name}...")
        
        try:
            results = evaluate_enhancement(original_file, enhanced_file)
            if results:
                results['filename'] = str(relative_path)
                all_results.append(results)
        except Exception as e:
            print(f"Error processing {original_file}: {e}")
    
    # Compute average metrics
    if all_results:
        avg_results = {}
        for key in all_results[0].keys():
            if key != 'filename' and isinstance(all_results[0][key], (int, float)):
                values = [r[key] for r in all_results if key in r]
                avg_results[f'avg_{key}'] = np.mean(values)
                avg_results[f'std_{key}'] = np.std(values)
        
        print("\n" + "="*60)
        print("AVERAGE METRICS ACROSS ALL IMAGES")
        print("="*60)
        print(f"\nSNR Improvement: {avg_results['avg_snr_improvement']:.2f}% ± {avg_results['std_snr_improvement']:.2f}%")
        print(f"SSIM: {avg_results['avg_ssim']:.4f} ± {avg_results['std_ssim']:.4f}")
        print(f"PSNR: {avg_results['avg_psnr']:.2f} ± {avg_results['std_psnr']:.2f} dB")
        print("="*60)
        
        # Save all results
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump({
                    'individual_results': all_results,
                    'average_metrics': avg_results
                }, f, indent=2)
            print(f"\nResults saved to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate medical image enhancement')
    parser.add_argument('--original', type=str, required=True, 
                       help='Original image path or directory')
    parser.add_argument('--enhanced', type=str, required=True,
                       help='Enhanced image path or directory')
    parser.add_argument('--output', type=str, help='Output JSON path for results')
    parser.add_argument('--batch', action='store_true', 
                       help='Batch mode for directories')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_evaluate(args.original, args.enhanced, args.output)
    else:
        evaluate_enhancement(args.original, args.enhanced, args.output)


if __name__ == '__main__':
    main()
