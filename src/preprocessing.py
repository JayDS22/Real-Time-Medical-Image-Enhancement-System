"""
Preprocessing Pipeline for Medical Images
Handles CT and MRI data in NIfTI format
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import torch


class MedicalImagePreprocessor:
    """Preprocessor for medical imaging data"""
    
    def __init__(
        self,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(128, 128, 128),
        modality='CT',
        intensity_range=None
    ):
        """
        Args:
            target_spacing: Target voxel spacing in mm
            target_size: Target image dimensions
            modality: 'CT' or 'MRI'
            intensity_range: (min, max) for intensity normalization
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.modality = modality
        
        # Default intensity ranges
        if intensity_range is None:
            if modality == 'CT':
                # HU units for CT
                self.intensity_range = (-1000, 1000)
            else:
                # Percentile-based for MRI
                self.intensity_range = (0.5, 99.5)  # percentiles
        else:
            self.intensity_range = intensity_range
    
    def load_image(self, path):
        """Load medical image from file"""
        path = str(path)
        
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            # Load NIfTI
            img = nib.load(path)
            data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
            
        else:
            # Try loading with SimpleITK (DICOM, etc.)
            img = sitk.ReadImage(path)
            data = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()[::-1]  # ITK uses (x,y,z), numpy uses (z,y,x)
        
        return data, spacing
    
    def save_image(self, data, path, spacing=None):
        """Save medical image to NIfTI format"""
        if spacing is None:
            spacing = self.target_spacing
            
        # Create NIfTI image
        img = nib.Nifti1Image(data, affine=np.eye(4))
        
        # Set spacing in header
        img.header.set_zooms(spacing)
        
        # Save
        nib.save(img, str(path))
    
    def normalize_intensity(self, data):
        """Normalize intensity values"""
        if self.modality == 'CT':
            # Clip HU values
            data = np.clip(data, self.intensity_range[0], self.intensity_range[1])
            # Normalize to [0, 1]
            data = (data - self.intensity_range[0]) / (
                self.intensity_range[1] - self.intensity_range[0]
            )
        else:
            # MRI: percentile-based normalization
            lower = np.percentile(data, self.intensity_range[0])
            upper = np.percentile(data, self.intensity_range[1])
            data = np.clip(data, lower, upper)
            data = (data - lower) / (upper - lower + 1e-8)
        
        # Scale to [-1, 1] for diffusion model
        data = data * 2.0 - 1.0
        
        return data
    
    def denormalize_intensity(self, data):
        """Convert normalized values back to original range"""
        # From [-1, 1] to [0, 1]
        data = (data + 1.0) / 2.0
        
        if self.modality == 'CT':
            # Back to HU units
            data = data * (
                self.intensity_range[1] - self.intensity_range[0]
            ) + self.intensity_range[0]
        
        return data
    
    def resample_image(self, data, old_spacing):
        """Resample image to target spacing"""
        # Convert to SimpleITK
        img = sitk.GetImageFromArray(data)
        img.SetSpacing(old_spacing[::-1])  # numpy (z,y,x) to ITK (x,y,z)
        
        # Calculate new size
        old_size = img.GetSize()
        new_size = [
            int(round(old_size[i] * old_spacing[::-1][i] / self.target_spacing[::-1][i]))
            for i in range(3)
        ]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing[::-1])
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(data.min())
        resampler.SetInterpolator(sitk.sitkBSpline)
        
        resampled = resampler.Execute(img)
        
        # Back to numpy
        data = sitk.GetArrayFromImage(resampled)
        
        return data
    
    def resize_image(self, data):
        """Resize or pad image to target size"""
        current_size = data.shape
        
        # Calculate padding/cropping
        pad_or_crop = []
        for i in range(3):
            if current_size[i] < self.target_size[i]:
                # Pad
                pad_before = (self.target_size[i] - current_size[i]) // 2
                pad_after = self.target_size[i] - current_size[i] - pad_before
                pad_or_crop.append((pad_before, pad_after))
            elif current_size[i] > self.target_size[i]:
                # Crop
                crop_before = (current_size[i] - self.target_size[i]) // 2
                crop_after = crop_before + self.target_size[i]
                pad_or_crop.append((crop_before, crop_after))
            else:
                pad_or_crop.append((0, 0))
        
        # Apply padding or cropping
        result = data.copy()
        
        # Crop first
        for i in range(3):
            if current_size[i] > self.target_size[i]:
                indices = [slice(None)] * 3
                indices[i] = slice(pad_or_crop[i][0], pad_or_crop[i][1])
                result = result[tuple(indices)]
        
        # Then pad
        pad_width = [(max(0, p[0]), max(0, p[1])) for p in pad_or_crop]
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            result = np.pad(result, pad_width, mode='constant', constant_values=result.min())
        
        return result
    
    def extract_patches(self, data, patch_size=(64, 64, 64), stride=(32, 32, 32)):
        """Extract overlapping 3D patches"""
        patches = []
        positions = []
        
        for z in range(0, data.shape[0] - patch_size[0] + 1, stride[0]):
            for y in range(0, data.shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, data.shape[2] - patch_size[2] + 1, stride[2]):
                    patch = data[
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]
                    patches.append(patch)
                    positions.append((z, y, x))
        
        return np.array(patches), positions
    
    def reconstruct_from_patches(
        self,
        patches,
        positions,
        output_shape,
        patch_size=(64, 64, 64)
    ):
        """Reconstruct volume from overlapping patches with Gaussian weighting"""
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)
        
        # Create Gaussian weight map
        gaussian_weight = self.create_gaussian_weight(patch_size)
        
        for patch, (z, y, x) in zip(patches, positions):
            output[
                z:z+patch_size[0],
                y:y+patch_size[1],
                x:x+patch_size[2]
            ] += patch * gaussian_weight
            
            weights[
                z:z+patch_size[0],
                y:y+patch_size[1],
                x:x+patch_size[2]
            ] += gaussian_weight
        
        # Normalize by weights
        output = output / (weights + 1e-8)
        
        return output
    
    def create_gaussian_weight(self, size):
        """Create Gaussian weight map for patch blending"""
        weight = np.ones(size, dtype=np.float32)
        
        for i in range(3):
            profile = np.exp(-0.5 * ((np.arange(size[i]) - size[i]//2) / (size[i]/6))**2)
            shape = [1, 1, 1]
            shape[i] = size[i]
            weight = weight * profile.reshape(shape)
        
        return weight
    
    def preprocess(self, input_path, output_path=None):
        """Full preprocessing pipeline"""
        # Load image
        data, spacing = self.load_image(input_path)
        
        # Resample to target spacing
        data = self.resample_image(data, spacing)
        
        # Normalize intensity
        data = self.normalize_intensity(data)
        
        # Resize to target size
        data = self.resize_image(data)
        
        # Save if output path provided
        if output_path is not None:
            self.save_image(data, output_path)
        
        return data


def preprocess_dataset(
    input_dir,
    output_dir,
    modality='CT',
    target_spacing=(1.0, 1.0, 1.0),
    target_size=(128, 128, 128)
):
    """Preprocess entire dataset"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all medical image files
    extensions = ['.nii', '.nii.gz']
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f'*{ext}'))
    
    print(f"Found {len(files)} files")
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor(
        target_spacing=target_spacing,
        target_size=target_size,
        modality=modality
    )
    
    # Process each file
    processed_files = []
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Create output path
            relative_path = file_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preprocess
            preprocessor.preprocess(file_path, output_path)
            
            processed_files.append({
                'input': str(file_path),
                'output': str(output_path)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save metadata
    metadata = {
        'modality': modality,
        'target_spacing': target_spacing,
        'target_size': target_size,
        'num_files': len(processed_files),
        'files': processed_files
    }
    
    with open(output_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processed {len(processed_files)} files successfully")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess medical images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--modality', type=str, default='CT', choices=['CT', 'MRI'])
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument('--target_size', type=int, nargs=3, default=[128, 128, 128])
    
    args = parser.parse_args()
    
    preprocess_dataset(
        args.input_dir,
        args.output_dir,
        args.modality,
        tuple(args.target_spacing),
        tuple(args.target_size)
    )


if __name__ == '__main__':
    main()
