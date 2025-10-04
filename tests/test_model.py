"""
Unit tests for model components
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.unet3d import UNet3D, ResBlock3D, Attention3D
from models.diffusion import GaussianDiffusion


class TestUNet3D:
    """Test 3D U-Net architecture"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        return UNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            channel_mults=(1, 2, 4),
            num_res_blocks=1
        ).to(device)
    
    def test_model_creation(self, model):
        """Test model can be created"""
        assert model is not None
        assert isinstance(model, UNet3D)
    
    def test_forward_pass(self, model, device):
        """Test forward pass with different input sizes"""
        batch_size = 2
        
        # Test with 64x64x64
        x = torch.randn(batch_size, 1, 64, 64, 64).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        
        output = model(x, t)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_timesteps(self, model, device):
        """Test with different timestep values"""
        x = torch.randn(1, 1, 32, 32, 32).to(device)
        
        for t_val in [0, 250, 500, 750, 999]:
            t = torch.tensor([t_val]).to(device)
            output = model(x, t)
            assert output.shape == x.shape
    
    def test_gradient_flow(self, model, device):
        """Test that gradients flow properly"""
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True).to(device)
        t = torch.tensor([500]).to(device)
        
        output = model(x, t)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestResBlock3D:
    """Test ResBlock3D component"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_resblock_same_channels(self, device):
        """Test ResBlock with same input/output channels"""
        block = ResBlock3D(64, 64, time_emb_dim=128).to(device)
        
        x = torch.randn(1, 64, 16, 16, 16).to(device)
        t_emb = torch.randn(1, 128).to(device)
        
        output = block(x, t_emb)
        assert output.shape == x.shape
    
    def test_resblock_different_channels(self, device):
        """Test ResBlock with different input/output channels"""
        block = ResBlock3D(64, 128, time_emb_dim=128).to(device)
        
        x = torch.randn(1, 64, 16, 16, 16).to(device)
        t_emb = torch.randn(1, 128).to(device)
        
        output = block(x, t_emb)
        assert output.shape == (1, 128, 16, 16, 16)


class TestAttention3D:
    """Test 3D Attention mechanism"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_attention_forward(self, device):
        """Test attention forward pass"""
        attn = Attention3D(channels=64, num_heads=4).to(device)
        
        x = torch.randn(1, 64, 8, 8, 8).to(device)
        output = attn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestGaussianDiffusion:
    """Test diffusion process"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        return UNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1
        ).to(device)
    
    @pytest.fixture
    def diffusion(self, model, device):
        return GaussianDiffusion(
            model=model,
            timesteps=100,
            device=device
        )
    
    def test_diffusion_creation(self, diffusion):
        """Test diffusion model creation"""
        assert diffusion is not None
        assert diffusion.timesteps == 100
    
    def test_forward_diffusion(self, diffusion, device):
        """Test forward diffusion q(x_t | x_0)"""
        x0 = torch.randn(1, 1, 16, 16, 16).to(device)
        t = torch.tensor([50]).to(device)
        
        xt = diffusion.q_sample(x0, t)
        
        assert xt.shape == x0.shape
        assert not torch.isnan(xt).any()
    
    def test_training_loss(self, diffusion, device):
        """Test loss computation"""
        x0 = torch.randn(1, 1, 16, 16, 16).to(device)
        t = torch.tensor([50]).to(device)
        
        loss = diffusion.training_losses(x0, t)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss).any()
    
    def test_ddim_sampling(self, diffusion, device):
        """Test DDIM sampling"""
        shape = (1, 1, 16, 16, 16)
        
        with torch.no_grad():
            sample = diffusion.ddim_sample(shape, ddim_timesteps=5)
        
        assert sample.shape == shape
        assert not torch.isnan(sample).any()
    
    def test_noise_schedule(self, diffusion):
        """Test beta schedule properties"""
        # Betas should be between 0 and 1
        assert (diffusion.betas >= 0).all()
        assert (diffusion.betas <= 1).all()
        
        # Alphas should be between 0 and 1
        assert (diffusion.alphas >= 0).all()
        assert (diffusion.alphas <= 1).all()
        
        # Alphas cumprod should be decreasing
        assert (diffusion.alphas_cumprod[1:] <= diffusion.alphas_cumprod[:-1]).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
