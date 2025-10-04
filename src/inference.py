"""
Inference Pipeline for Medical Image Enhancement
Fast enhancement using DDIM sampling
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet3d import UNet3D
from models.diffusion import GaussianDiffusion
from preprocessing import MedicalImagePreprocessor


class MedicalImageEnhancer:
    """Production-ready inference pipeline for medical image enhancement"""
    
    def __init__(
        self,
        model_path,
        device='cuda',
        patch_size=(64, 64, 64),
        patch_overlap=(16, 16, 16),
        ddim_steps=50
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            patch_size: Size of patches for processing
            patch_overlap: Overlap between patches
            ddim_steps: Number of DDIM sampling steps
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.ddim_steps = ddim_steps
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Create diffusion
        self.diffusion = GaussianDiffusion(
            model=self.model,
            timesteps=1000,
            device=self.device
        )
        
        # Create preprocessor
        self.preprocessor = MedicalImagePreprocessor()
        
        print("Model loaded successfully")
    
    def _load_model(self, model_path):
        """Load trained model"""
        # Create model architecture
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(2, 4)
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return model
    
    def extract_patches(self, volume):
        """Extract overlapping patches from volume"""
        patches = []
        positions = []
        
        stride = tuple(ps - po for ps, po in zip(self.patch_size, self.patch_overlap))
        
        for z in range(0, volume.shape[0] - self.patch_size[0] + 1, stride[0]):
            for y in range(0, volume.shape[1] - self.patch_size[1] + 1, stride[1]):
                for x in range(0, volume.shape[2] - self.patch_size[2] + 1, stride[2]):
                    patch = volume[
                        z:z + self.patch_size[0],
                        y:y + self.patch_size[1],
                        x:x + self.patch_size[2]
                    ]
                    
                    if patch.shape == self.patch_size:
                        patches.append(patch)
                        positions.append((z, y, x))
        
        return patches, positions
    
    def reconstruct_from_patches(self, patches, positions, volume_shape):
        """Reconstruct volume from patches with Gaussian weighting"""
        volume = np.zeros(volume_shape, dtype=np.float32)
        weights = np.zeros(volume_shape, dtype=np.float32)
        
        # Create Gaussian weight
        gaussian_weight = self._create_gaussian_weight()
        
        for patch, (z, y, x) in zip(patches, positions):
            volume[
                z:z + self.patch_size[0],
                y:y + self.patch_size[1],
                x:x + self.patch_size[2]
            ] += patch * gaussian_weight
            
            weights[
                z:z + self.patch_size[0],
                y:y + self.patch_size[1],
                x:x + self.patch_size[2]
            ] += gaussian_weight
        
        # Normalize
        volume = volume / (weights + 1e-8)
        
        return volume
    
    def _create_gaussian_weight(self):
        """Create Gaussian weight for patch blending"""
        weight = np.ones(self.patch_size, dtype=np.float32)
        
        for i in range(3):
            center = self.patch_size[i] / 2
            sigma = self.patch_size[i] / 6
            profile = np.exp(-0.5 * ((np.arange(self.patch_size[i]) - center) / sigma) ** 2)
            
            shape = [1, 1, 1]
            shape[i] = self.patch_size[i]
            weight = weight * profile.reshape(shape)
        
        return weight
    
    @torch.no_grad()
    def enhance_patch(self, patch):
        """Enhance single patch using diffusion model"""
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Add noise (simulate low quality)
        t = torch.tensor([250], device=self.device).long()  # Moderate noise
        noisy_patch = self.diffusion.q_sample(patch_tensor, t)
        
        # Denoise using DDIM
        enhanced = self.diffusion.ddim_sample(
            shape=noisy_patch.shape,
            ddim_timesteps=self.ddim_steps,
            eta=0.0
        )
        
        # Convert back to numpy
        enhanced_np = enhanced.cpu().numpy()[0, 0]
        
        return enhanced_np
    
    def enhance_volume(self, volume, show_progress=True):
        """Enhance entire volume"""
        start_time = time.time()
        
        # Extract patches
        patches, positions = self.extract_patches(volume)
        
        if len(patches) == 0:
            print("Warning: No valid patches extracted. Returning original volume.")
            return volume
        
        print(f"Extracted {len(patches)} patches")
        
        # Enhance each patch
        enhanced_patches = []
        
        iterator = tqdm(patches, desc="Enhancing patches") if show_progress else patches
        
        for patch in iterator:
            enhanced_patch = self.enhance_patch(patch)
            enhanced_patches.append(enhanced_patch)
        
        # Reconstruct volume
        enhanced_volume = self.reconstruct_from_patches(
            enhanced_patches,
            positions,
            volume.shape
        )
        
        elapsed_time = time.time() - start_time
        print(f"Enhancement completed in {elapsed_time:.2f}s")
        
        return enhanced_volume
    
    def enhance_file(self, input_path, output_path, modality='CT'):
        """Enhance medical image file"""
        print(f"Processing {input_path}")
        
        # Update preprocessor modality
        self.preprocessor.modality = modality
        
        # Load and preprocess
        print("Loading and preprocessing...")
        data, spacing = self.preprocessor.load_image(input_path)
        data = self.preprocessor.resample_image(data, spacing)
        data_normalized = self.preprocessor.normalize_intensity(data)
        
        # Enhance
        print("Enhancing...")
        enhanced_normalized = self.enhance_volume(data_normalized)
        
        # Denormalize
        enhanced = self.preprocessor.denormalize_intensity(enhanced_normalized)
        
        # Save
        print(f"Saving to {output_path}")
        self.preprocessor.save_image(enhanced, output_path, self.preprocessor.target_spacing)
        
        print("Done!")
        
        return enhanced


def main():
    parser = argparse.ArgumentParser(description='Enhance medical images')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--modality', type=str, default='CT', choices=['CT', 'MRI'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument('--patch_overlap', type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps')
    
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = MedicalImageEnhancer(
        model_path=args.model,
        device=args.device,
        patch_size=tuple(args.patch_size),
        patch_overlap=tuple(args.patch_overlap),
        ddim_steps=args.ddim_steps
    )
    
    # Enhance image
    enhancer.enhance_file(args.input, args.output, args.modality)


if __name__ == '__main__':
    main()
