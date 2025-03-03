# MARC-AI: MARC Record Matching with Bibliographic Metadata

A machine learning system for matching and deduplicating MARC (Machine-Readable Cataloging) bibliographic records based solely on their metadata content, without relying on identifiers.

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/cdlib/marc-match-ai)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/cdlib/marc-match-ai-demo)

## Overview

Traditional matching of MARC records has relied heavily on identifiers like OCLC numbers, ISBNs, or LCCNs. However, this approach struggles with records having incorrect identifiers or lacking them altogether. MARC-AI matches records based solely on bibliographic metadata (title, author, publisher, etc.), enabling successful matches even when identifiers are missing or inaccurate.

## Key Features

- **Bibliographic Metadata Matching**: Performs matching based solely on bibliographic data
- **Text Field Flexibility**: Handles minor variations in metadata fields through normalization and various similarity metrics
- **Adjustable Matching Threshold**: Customizable threshold to balance precision and recall
- **Hard Negative Mining**: Improves model training by focusing on difficult examples
- **Scalable Processing**: Multiprocessing support for handling large datasets

## System Architecture

MARC-AI consists of several interconnected components:

1. **Processing Pipeline**:
   - Loads and normalizes MARC records
   - Generates similarity features between record pairs
   - Applies various text normalization techniques (lowercase, punctuation removal, diacritics removal)

2. **Machine Learning Model**:
   - Multi-layer perceptron trained on labeled record pairs
   - Predicts match probability based on field similarities
   - Built with PyTorch Lightning for reproducibility and scalability

3. **CLI Interface**:
   - `process`: Generate similarity features from MARC record pairs
   - `train`: Train a new model with configurable hyperparameters
   - `predict`: Apply trained model to predict matches
   - `pipeline`: End-to-end workflow from raw records to predictions

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/cdlib/marc-ai.git

# Or clone and install
git clone https://github.com/cdlib/marc-ai.git
cd marc-ai
pip install .
```

## Usage

### Processing Data

Generate similarity features from MARC records:

```bash
marc-ai process -i MARC_FILES... -o OUTPUT_CSV -p PAIR_INDICES [-C CHUNKSIZE] [-P PROCESSES]
```

### Training a Model

Train with hyperparameters from `config.yaml`:

```bash
marc-ai train -n RUN_NAME
```

### Making Predictions

Apply a trained model to processed data:

```bash
marc-ai predict -i INPUT_CSV -o OUTPUT_CSV [-m MODEL_PATH]
```

### End-to-End Pipeline

Process MARC records and predict matches in one step:

```bash
marc-ai pipeline -i MARC_FILES... -p PAIR_INDICES -o OUTPUT_CSV [-m MODEL_PATH] [-C CHUNKSIZE] [-P PROCESSES] [-t THRESHOLD]
```

## Configuration

The `config.yaml` file controls model architecture and training:

```yaml
model:
  # Input features
  features:
    - title_tokenset
    - title_agg
    - author
    - publisher
    - pub_date
    - pub_place
    - pagination
  
  # Training parameters
  batch_size: 512
  weight_decay: 0.0
  max_epochs: -1
  patience: 20
  lr: 0.006
  optimizer: Adam
  
  # Hard negative mining
  use_hard_negative_mining: true
  hard_negative_weight: 3.0
  mining_threshold: 0.7
  mining_frequency: 2
```

## Performance and Scalability

The system is optimized for processing large datasets, but since record matching grows quadratically with the number of records, we recommend using blocking strategies to reduce comparison pairs. Token blocking on title fields (using only the bottom 70% of words by occurrence) has proven effective at reducing comparisons while maintaining high recall.

## Dataset

The training dataset originates from HathiTrust contributors, with records anonymized by removing identifiers and custom fields. This dataset is designed for evaluating record pairing methods based on content alone. The data is licensed under CC0 with caveats detailed in `datasets/ht/LICENSE.md`.

## Analysis

Jupyter notebooks in the `analysis/` directory provide insights into dataset characteristics and model performance.

## Results

The `results/` folder contains model outcomes and baseline comparisons using string matching methods.

## Contributing

We welcome contributions including datasets, new matching methods, or performance improvements. If you'd like to contribute, please contact us.