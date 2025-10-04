"""
Integration tests for complete pipeline
"""

import pytest
import torch
import numpy as np
import tempfile
import nibabel as nib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import MedicalImagePreprocessor
from src.metrics import ImageQualityMetrics
from models.unet3d import UNet3D
from models.diffusion import GaussianDiffusion


class TestPreprocessingPipeline:
    """Test preprocessing pipeline"""
    
    @pytest.fixture
    def preprocessor(self):
        return MedicalImagePreprocessor(
            target_spacing=(1.0, 1.0, 1.0),
            target_size=(64, 64, 64),
            modality='CT'
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample 3D medical image"""
        return np.random.randn(80, 80, 80).astype(np.float32)
    
    def test_intensity_normalization(self, preprocessor, sample_image):
        """Test intensity normalization"""
        normalized = preprocessor.normalize_intensity(sample_image)
        
        # Should be in [-1, 1] range
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
    
    def test_denormalization(self, preprocessor, sample_image):
        """Test intensity denormalization"""
        normalized = preprocessor.normalize_intensity(sample_image)
        denormalized = preprocessor.denormalize_intensity(normalized)
        
        # Should be close to original range
        assert denormalized.min() >= sample_image.min() - 10
        assert denormalized.max() <= sample_image.max() + 10
    
    def test_resize_image(self, preprocessor, sample_image):
        """Test image resizing"""
        resized = preprocessor.resize_image(sample_image)
        
        assert resized.shape == preprocessor.target_size
    
    def test_patch_extraction_reconstruction(self, preprocessor, sample_image):
        """Test patch extraction and reconstruction"""
        # Extract patches
        patches, positions = preprocessor.extract_patches(
            sample_image,
            patch_size=(32, 32, 32),
            stride=(16, 16, 16)
        )
        
        assert len(patches) > 0
        assert len(patches) == len(positions)
        
        # Reconstruct
        reconstructed = preprocessor.reconstruct_from_patches(
            patches,
            positions,
            sample_image.shape,
            patch_size=(32, 32, 32)
        )
        
        assert reconstructed.shape == sample_image.shape
        
        # Reconstruction error should be small
        mse = np.mean((sample_image - reconstructed) ** 2)
        assert mse < 0.01
    
    def test_save_load_image(self, preprocessor, sample_image):
        """Test saving and loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.nii.gz'
            
            # Save
            preprocessor.save_image(sample_image, path)
            assert path.exists()
            
            # Load
            loaded, spacing = preprocessor.load_image(path)
            assert loaded.shape == sample_image.shape
            assert np.allclose(loaded, sample_image, rtol=1e-5)


class TestMetricsPipeline:
    """Test metrics computation pipeline"""
    
    @pytest.fixture
    def metrics(self):
        return ImageQualityMetrics()
    
    @pytest.fixture
    def clean_image(self):
        """Create clean synthetic image"""
        img = np.random.randn(64, 64, 64).astype(np.float32)
        return img * 0.3 + 0.5  # Moderate intensity
    
    @pytest.fixture
    def noisy_image(self, clean_image):
        """Create noisy version"""
        noise = np.random.randn(*clean_image.shape) * 0.1
        return clean_image + noise
    
    def test_snr_computation(self, metrics, clean_image, noisy_image):
        """Test SNR computation"""
        snr_clean = metrics.compute_snr(clean_image)
        snr_noisy = metrics.compute_snr(noisy_image)
        
        assert snr_clean > 0
        assert snr_noisy > 0
        # Clean should have higher SNR
        assert snr_clean > snr_noisy
    
    def test_ssim_computation(self, metrics, clean_image, noisy_image):
        """Test SSIM computation"""
        ssim = metrics.compute_ssim(clean_image, noisy_image)
        
        assert 0 <= ssim <= 1
        # Should be fairly similar
        assert ssim > 0.5
    
    def test_psnr_computation(self, metrics, clean_image, noisy_image):
        """Test PSNR computation"""
        psnr_val = metrics.compute_psnr(clean_image, noisy_image)
        
        assert psnr_val > 0
    
    def test_contrast_computation(self, metrics, clean_image):
        """Test contrast computation"""
        contrast = metrics.compute_contrast(clean_image)
        
        assert contrast > 0
    
    def test_sharpness_computation(self, metrics, clean_image):
        """Test sharpness computation"""
        sharpness = metrics.compute_sharpness(clean_image)
        
        assert sharpness >= 0
    
    def test_identical_images(self, metrics):
        """Test metrics on identical images"""
        img = np.random.randn(32, 32, 32).astype(np.float32)
        
        ssim = metrics.compute_ssim(img, img)
        mse = metrics.compute_mse(img, img)
        
        # SSIM should be 1 for identical images
        assert ssim > 0.99
        # MSE should be 0 for identical images
        assert mse < 1e-6


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def small_model(self, device):
        """Create small model for testing"""
        return UNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1
        ).to(device)
    
    @pytest.fixture
    def diffusion(self, small_model, device):
        """Create diffusion model"""
        return GaussianDiffusion(
            model=small_model,
            timesteps=50,
            device=device
        )
    
    def test_training_step(self, diffusion, device):
        """Test single training step"""
        # Create batch
        batch = torch.randn(2, 1, 32, 32, 32).to(device)
        t = torch.randint(0, 50, (2,)).to(device)
        
        # Compute loss
        loss = diffusion.training_losses(batch, t)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss).any()
        
        # Test backward pass
        loss.backward()
    
    def test_inference_pipeline(self, diffusion, device):
        """Test inference pipeline"""
        # Create noisy input
        noisy = torch.randn(1, 1, 32, 32, 32).to(device)
        
        # Denoise
        with torch.no_grad():
            denoised = diffusion.ddim_sample(
                shape=noisy.shape,
                ddim_timesteps=5
            )
        
        assert denoised.shape == noisy.shape
        assert not torch.isnan(denoised).any()
    
    def test_full_pipeline_with_metrics(self, diffusion, device):
        """Test full pipeline including metrics"""
        # Create synthetic data
        clean = torch.randn(1, 1, 32, 32, 32).to(device) * 0.3
        
        # Add noise
        t = torch.tensor([25]).to(device)
        noisy = diffusion.q_sample(clean, t)
        
        # Denoise
        with torch.no_grad():
            enhanced = diffusion.ddim_sample(
                shape=noisy.shape,
                ddim_timesteps=5
            )
        
        # Convert to numpy for metrics
        clean_np = clean.cpu().numpy()[0, 0]
        noisy_np = noisy.cpu().numpy()[0, 0]
        enhanced_np = enhanced.cpu().numpy()[0, 0]
        
        # Compute metrics
        metrics = ImageQualityMetrics()
        
        ssim_noisy = metrics.compute_ssim(clean_np, noisy_np)
        ssim_enhanced = metrics.compute_ssim(clean_np, enhanced_np)
        
        # Both should be reasonable
        assert 0 <= ssim_noisy <= 1
        assert 0 <= ssim_enhanced <= 1


class TestDataAugmentation:
    """Test data augmentation"""
    
    def test_random_flip(self):
        """Test random flipping"""
        img = np.random.randn(32, 32, 32).astype(np.float32)
        
        # Flip along each axis
        for axis in range(3):
            flipped = np.flip(img, axis=axis)
            assert flipped.shape == img.shape
            assert not np.array_equal(flipped, img)
    
    def test_random_rotation(self):
        """Test random rotation"""
        img = np.random.randn(32, 32, 32).astype(np.float32)
        
        # 90-degree rotation in xy plane
        rotated = np.rot90(img, k=1, axes=(1, 2))
        assert rotated.shape == img.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
