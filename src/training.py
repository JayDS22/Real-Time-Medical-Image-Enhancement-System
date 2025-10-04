"""
Training Script for Medical Image Enhancement
Using DDPM with 3D U-Net
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
import yaml
import json
from tqdm import tqdm
import nibabel as nib
from tensorboard import SummaryWriter
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.unet3d import UNet3D
from models.diffusion import GaussianDiffusion, DiffusionTrainer


class MedicalImageDataset(Dataset):
    """Dataset for medical images"""
    
    def __init__(self, data_dir, file_list=None, patch_size=(64, 64, 64), augment=True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # Get file list
        if file_list is not None:
            with open(file_list, 'r') as f:
                metadata = json.load(f)
                self.files = [f['output'] for f in metadata['files']]
        else:
            self.files = list(self.data_dir.rglob('*.nii.gz'))
            self.files.extend(list(self.data_dir.rglob('*.nii')))
        
        print(f"Found {len(self.files)} files in dataset")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.files[idx]
        img = nib.load(img_path)
        data = img.get_fdata().astype(np.float32)
        
        # Extract random patch
        patch = self.extract_random_patch(data)
        
        # Augmentation
        if self.augment:
            patch = self.augment_patch(patch)
        
        # Convert to tensor
        patch = torch.from_numpy(patch).unsqueeze(0)  # Add channel dimension
        
        return patch
    
    def extract_random_patch(self, data):
        """Extract random 3D patch"""
        # Get valid start positions
        max_start = [data.shape[i] - self.patch_size[i] for i in range(3)]
        
        if any(m < 0 for m in max_start):
            # Image smaller than patch, pad it
            pad_width = [
                (0, max(0, self.patch_size[i] - data.shape[i]))
                for i in range(3)
            ]
            data = np.pad(data, pad_width, mode='constant')
            max_start = [0, 0, 0]
        
        # Random start position
        start = [np.random.randint(0, m + 1) if m > 0 else 0 for m in max_start]
        
        # Extract patch
        patch = data[
            start[0]:start[0] + self.patch_size[0],
            start[1]:start[1] + self.patch_size[1],
            start[2]:start[2] + self.patch_size[2]
        ]
        
        return patch
    
    def augment_patch(self, patch):
        """Apply data augmentation"""
        # Random flip along each axis
        for axis in range(3):
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=axis).copy()
        
        # Random 90-degree rotations in xy plane
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
        
        # Random intensity shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            patch = np.clip(patch + shift, -1.0, 1.0)
        
        # Random intensity scale
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            patch = np.clip(patch * scale, -1.0, 1.0)
        
        return patch


def create_data_loaders(config):
    """Create training and validation data loaders"""
    
    # Training dataset
    train_dataset = MedicalImageDataset(
        data_dir=config['data']['train_dir'],
        file_list=config['data'].get('train_list'),
        patch_size=tuple(config['data']['patch_size']),
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    if 'val_dir' in config['data']:
        val_dataset = MedicalImageDataset(
            data_dir=config['data']['val_dir'],
            file_list=config['data'].get('val_list'),
            patch_size=tuple(config['data']['patch_size']),
            augment=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


def train_epoch(trainer, train_loader, epoch, writer, device):
    """Train for one epoch"""
    losses = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        loss = trainer.train_step(batch)
        losses.append(loss)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss:.4f}', 'avg_loss': f'{np.mean(losses):.4f}'})
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Train/Loss', loss, global_step)
    
    return np.mean(losses)


def validate_epoch(trainer, val_loader, epoch, writer):
    """Validate for one epoch"""
    losses = []
    
    pbar = tqdm(val_loader, desc=f'Validation {epoch}')
    for batch in pbar:
        loss = trainer.validate_step(batch)
        losses.append(loss)
        pbar.set_postfix({'val_loss': f'{loss:.4f}'})
    
    avg_loss = np.mean(losses)
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def train(config):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Create model
    print("Creating model...")
    model = UNet3D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        channel_mults=tuple(config['model']['channel_mults']),
        num_res_blocks=config['model']['num_res_blocks'],
        time_emb_dim=config['model']['time_emb_dim'],
        dropout=config['model']['dropout'],
        attention_resolutions=tuple(config['model']['attention_resolutions'])
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        beta_schedule=config['diffusion']['beta_schedule'],
        device=device
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=device,
        grad_clip=config['training']['grad_clip']
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['training'].get('resume_from'):
        print(f"Resuming from checkpoint: {config['training']['resume_from']}")
        start_epoch, _ = load_checkpoint(
            model, optimizer, scheduler,
            config['training']['resume_from'],
            device
        )
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_epoch(trainer, train_loader, epoch, writer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate_epoch(trainer, val_loader, epoch, writer)
            print(f"Val Loss: {val_loss:.4f}")
        else:
            val_loss = train_loss
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Generate sample
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("Generating sample...")
            generate_sample(diffusion, device, output_dir, epoch)
    
    writer.close()
    print("Training completed!")


def generate_sample(diffusion, device, output_dir, epoch):
    """Generate and save sample images"""
    diffusion.model.eval()
    
    with torch.no_grad():
        # Generate sample
        shape = (1, 1, 64, 64, 64)  # Small patch for visualization
        sample = diffusion.ddim_sample(shape, ddim_timesteps=50)
        
        # Save as numpy
        sample_np = sample.cpu().numpy()[0, 0]
        np.save(output_dir / f'sample_epoch_{epoch + 1}.npy', sample_np)
    
    diffusion.model.train()


def main():
    parser = argparse.ArgumentParser(description='Train medical image enhancement model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
