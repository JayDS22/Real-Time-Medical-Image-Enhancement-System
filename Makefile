# Makefile for Medical Image Enhancement System

.PHONY: help install test demo clean preprocess train inference evaluate visualize

help:
	@echo "Medical Image Enhancement System - Makefile Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install        Install all dependencies"
	@echo "  make test          Run all tests"
	@echo "  make demo          Run demo with synthetic data"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make preprocess    Preprocess medical images"
	@echo "  make train         Train the enhancement model"
	@echo "  make inference     Run inference on test images"
	@echo "  make evaluate      Evaluate enhancement quality"
	@echo "  make visualize     Create visualizations"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean         Clean generated files"
	@echo "  make dirs          Create necessary directories"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Installation complete!"

dirs:
	@echo "Creating directory structure..."
	mkdir -p data/raw data/processed
	mkdir -p models/pretrained
	mkdir -p results
	mkdir -p logs
	touch data/raw/.gitkeep
	touch data/processed/.gitkeep
	touch models/pretrained/.gitkeep
	touch results/.gitkeep
	@echo "✓ Directories created!"

test:
	@echo "Running tests..."
	python demo.py --test all

demo:
	@echo "Running demo..."
	python demo.py --test all

preprocess:
	@echo "Preprocessing medical images..."
	python src/preprocessing.py \
		--input_dir data/raw \
		--output_dir data/processed \
		--modality CT \
		--target_spacing 1.0 1.0 1.0 \
		--target_size 128 128 128

train:
	@echo "Training model..."
	python src/training.py --config configs/training_config.yaml

inference:
	@echo "Running inference..."
	@echo "Usage: make inference INPUT=path/to/input.nii.gz OUTPUT=path/to/output.nii.gz MODEL=path/to/model.pth"
	python src/inference.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL) \
		--ddim_steps 50

evaluate:
	@echo "Evaluating results..."
	@echo "Usage: make evaluate ORIGINAL=path/to/original.nii.gz ENHANCED=path/to/enhanced.nii.gz"
	python src/metrics.py \
		--original $(ORIGINAL) \
		--enhanced $(ENHANCED) \
		--output results/metrics.json

visualize:
	@echo "Creating visualizations..."
	@echo "Usage: make visualize ORIGINAL=path/to/original.nii.gz ENHANCED=path/to/enhanced.nii.gz"
	python src/visualization.py \
		--original $(ORIGINAL) \
		--enhanced $(ENHANCED) \
		--output_dir results/visualizations

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ *.egg-info
	@echo "✓ Cleanup complete!"

clean-all: clean
	@echo "Removing all generated data and models..."
	rm -rf data/processed/*
	rm -rf models/pretrained/*.pth
	rm -rf results/*
	rm -rf logs/*
	@echo "✓ Deep cleanup complete!"
