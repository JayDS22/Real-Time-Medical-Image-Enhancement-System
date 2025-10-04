# Quick Start Guide

This guide will help you get started with the Medical Image Enhancement System in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 16GB RAM minimum (32GB recommended)

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-image-enhancement.git
cd medical-image-enhancement

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Step 2: Test Installation (2 minutes)

Run the demo script to verify everything works:

```bash
python demo.py --test all
```

You should see output like:
```
✓ Model architecture test passed!
✓ Diffusion process test passed!
✓ Preprocessing test passed!
✓ Metrics test passed!
✓ Complete pipeline test passed!
ALL TESTS PASSED SUCCESSFULLY! ✓
```

## Step 3: Download Sample Data (3 minutes)

### Option A: Use Provided Medical Images

If you have the sample images from the project:

```bash
# Place your medical images in the data directory
mkdir -p data/raw
# Copy your .nii or .nii.gz files to data/raw/
```

### Option B: Download Public Dataset

```bash
# Download from Medical Segmentation Decathlon or TCIA
# Example: Brain MRI dataset
wget http://medicaldecathlon.com/...
unzip dataset.zip -d data/raw/
```

## Step 4: Preprocess Data (2 minutes)

```bash
python src/preprocessing.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --modality CT \
    --target_spacing 1.0 1.0 1.0 \
    --target_size 128 128 128
```

## Step 5: Quick Training Demo (Optional, 5 minutes)

For a quick test with small model:

```bash
# Create a minimal config for testing
cat > configs/quick_test.yaml << EOF
data:
  train_dir: "data/processed"
  patch_size: [64, 64, 64]

model:
  in_channels: 1
  out_channels: 1
  base_channels: 32
  channel_mults: [1, 2, 4]
  num_res_blocks: 1
  time_emb_dim: 128
  dropout: 0.1
  attention_resolutions: [2]

diffusion:
  timesteps: 100
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "linear"

training:
  epochs: 5
  batch_size: 2
  learning_rate: 0.0001
  output_dir: "models/quick_test"
  save_interval: 5
  num_workers: 2
EOF

# Train for just 5 epochs to test
python src/training.py --config configs/quick_test.yaml
```

## Step 6: Run Inference (1 minute)

If you have a trained model (or use the quick test model):

```bash
python src/inference.py \
    --input data/processed/sample.nii.gz \
    --output results/enhanced_sample.nii.gz \
    --model models/quick_test/best_model.pth \
    --ddim_steps 50
```

## Step 7: Evaluate Results (1 minute)

```bash
python src/metrics.py \
    --original data/processed/sample.nii.gz \
    --enhanced results/enhanced_sample.nii.gz \
    --output results/metrics.json
```

## Step 8: Visualize (1 minute)

```bash
python src/visualization.py \
    --original data/processed/sample.nii.gz \
    --enhanced results/enhanced_sample.nii.gz \
    --output_dir results/visualizations
```

---

## Using Command-Line Tools

After installation, you can use convenient command-line tools:

```bash
# Preprocess data
medical-preprocess --input_dir data/raw --output_dir data/processed

# Train model
medical-train --config configs/training_config.yaml

# Enhance images
medical-enhance --input scan.nii.gz --output enhanced.nii.gz --model best_model.pth

# Evaluate results
medical-evaluate --original scan.nii.gz --enhanced enhanced.nii.gz
```

---

## Common Issues and Solutions

### CUDA Out of Memory
```bash
# Reduce batch size in config
# Or use smaller patch size
# Or use CPU mode: --device cpu
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Data Format Issues
```bash
# Ensure your data is in NIfTI format (.nii or .nii.gz)
# Convert DICOM to NIfTI using dcm2niix or SimpleITK
```

---

## Next Steps

1. **For Training**: Edit `configs/training_config.yaml` to match your dataset
2. **For Production**: Use the full model configuration for best results
3. **For Research**: Check out the Jupyter notebooks in `notebooks/`

## Need Help?

- Check the full README.md for detailed documentation
- Review the example notebooks
- Open an issue on GitHub

---

**Ready to enhance medical images? Start with `python demo.py --test all`!**
