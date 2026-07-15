# distributed-llm-trainer

A lightweight research codebase for training, benchmarking, and generation with configurable language models.

## Overview

- `main.py` is the entry point.
- Uses Hydra and OmegaConf for configuration.
- Supports three modes: `train`, `benchmark`, and `generate`.
- Model implementations and training logic live under `src/lmr/`.

## Key directories

- `src/lmr/config/` - configuration loading and schema files
- `src/lmr/models/` - model implementations, including transformers and FST variants
- `src/lmr/data/` - dataset loading and preprocessing utilities
- `src/lmr/training/` - training loop and trainer utilities
- `src/lmr/benchmark/` - benchmarking wrappers and logic
- `src/lmr/generation/` - text generation utilities
- `config/` - Hydra configuration files for datasets, models, training, and generation
- `scripts/` - example shell scripts for training, benchmarking, and dataset tasks

## Getting started

1. Create the Python environment and install dependencies.
2. Edit `config/config.yaml` or use a config override on the command line.
3. Run with the desired mode:

```bash
python main.py mode=train
python main.py mode=benchmark
python main.py mode=generate
```

## Configuration

This repo uses Hydra and `config/config.yaml` as the main configuration entry point.

- `mode` selects the workflow: `train`, `benchmark`, or `generate`.
- `model` loads a model config from `config/model/`.
- `size` loads a size config from `config/size/`.
- `training` loads training settings from `config/training/`.
- `dataset` loads dataset settings from `config/dataset/`.
- `generation` and `benchmark` load settings from `config/generation/` and `config/benchmark/`.
- `tokenizer_base` selects the tokenizer and the tokenized dataset directory.
- `checkpoint_name` chooses the checkpoint folder under `checkpoints/`.

### How to override settings

Use Hydra overrides to change config values from the command line.

```bash
python main.py mode=train model=transformer size=medium training.batch_size=16 training.lr=0.0001 tokenizer_base=gpt2
python main.py mode=benchmark model=fst size=1_3b benchmark.batch_size=8 checkpoint_name=myrun
python main.py mode=generate model=transformer generation.max_new_tokens=128 checkpoint_name=myrun
```

### Configuration groups

- `config/model/` contains model architecture choices such as `transformer`, `fst`, `fst_clean`, and `fst_nor`.
- `config/size/` contains model size presets like `small`, `medium`, `7b`, `large`, etc.
- `config/training/` contains training defaults including batch size, learning rate, optimizer, and schedule settings.
- `config/dataset/` contains dataset loading settings, splits, and tokenization options.
- `config/generation/` defines generation settings such as sequence length and sampling options.
- `config/benchmark/` defines benchmark settings and evaluation behavior.

## Notes

- Training uses a dataset split loader and checkpointing under `checkpoints/`.
- Benchmark output is written to `output/` by default.
- Tokenizer choice affects the dataset directory and model setup.
