"""
Model Training Tutorial
This script demonstrates how to train the medical image enhancement model.
"""

import sys
sys.path.append('.')

import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import AdamW

from models.unet3d import UNet3D
from models.diffusion import GaussianDiffusion, DiffusionTrainer


def load_configuration():
    """Load training configuration"""
    print("="*60)
    print("LOADING CONFIGURATION")
    print("="*60)
    
    config_path = 'configs/training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))
    
    return config


def create_model(config, device):
    """Create and initialize model"""
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    print(f"Using device: {device}")
    
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
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created successfully")
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1e6:.1f} MB (float32)")
    
    return model


def create_diffusion_model(config, model, device):
    """Create diffusion model"""
    print("\n" + "="*60)
    print("CREATING DIFFUSION MODEL")
    print("="*60)
    
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        beta_schedule=config['diffusion']['beta_schedule'],
        device=device
    )
    
    print(f"Diffusion model created")
    print(f"Timesteps: {config['diffusion']['timesteps']}")
    print(f"Beta schedule: {config['diffusion']['beta_schedule']}")
    
    return diffusion


def setup_training(config, diffusion, device):
    """Setup optimizer and trainer"""
    print("\n" + "="*60)
    print("SETTING UP TRAINING")
    print("="*60)
    
    optimizer = AdamW(
        diffusion.model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=device,
        grad_clip=config['training']['grad_clip']
    )
    
    print("Trainer initialized")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']}")
    print(f"Gradient clipping: {config['training']['grad_clip']}")
    
    return trainer, optimizer


def training_demo(trainer, device):
    """Demonstrate training step"""
    print("\n" + "="*60)
    print("TRAINING DEMONSTRATION")
    print("="*60)
    
    # Create dummy batch
    batch_size = 2
    patch_size = 64
    batch = torch.randn(batch_size, 1, patch_size, patch_size, patch_size).to(device)
    
    print(f"\nBatch shape: {batch.shape}")
    print("Running training step...")
    
    # Training step
    loss = trainer.train_step(batch)
    
    print(f"Training loss: {loss:.6f}")
    print("✓ Training step completed successfully")


def generate_sample(diffusion, device):
    """Generate and visualize sample"""
    print("\n" + "="*60)
    print("GENERATING SAMPLE")
    print("="*60)
    
    print("Generating sample using DDIM (10 steps)...")
    
    with torch.no_grad():
        sample = diffusion.ddim_sample(
            shape=(1, 1, 64, 64, 64),
            ddim_timesteps=10
        )
    
    print(f"Generated sample shape: {sample.shape}")
    
    # Visualize central slice
    sample_np = sample[0, 0, 32, :, :].cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(sample_np, cmap='gray')
    plt.title('Generated Sample (Central Slice)', fontsize=14, fontweight='bold')
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/generated_sample.png', dpi=300, bbox_inches='tight')
    print("Saved sample to: results/generated_sample.png")
    plt.show()


def main():
    """Main training demonstration"""
    print("\n" + "="*70)
    print("MODEL TRAINING TUTORIAL")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    config = load_configuration()
    
    # Create model
    model = create_model(config, device)
    
    # Create diffusion
    diffusion = create_diffusion_model(config, model, device)
    
    # Setup training
    trainer, optimizer = setup_training(config, diffusion, device)
    
    # Run training demo
    training_demo(trainer, device)
    
    # Generate sample
    generate_sample(diffusion, device)
    
    print("\n" + "="*70)
    print("TUTORIAL COMPLETE")
    print("="*70)
    print("\nFor full training, run:")
    print("  python src/training.py --config configs/training_config.yaml")
    print("\nOr use the Makefile:")
    print("  make train")


if __name__ == '__main__':
    main()