# BERT Text Simplification

This repository demonstrates how to implement text simplification using small, domain-specific models while maintaining complete control over sensitive data. Instead of relying on external large language models, this approach shows how to fine-tune BART base (139M parameters) for specialized text simplification tasks.

## Overview

The project consists of several key components:
- Baseline model evaluation (`baseline.py`)
- Custom dataset handling (`classes/simplification_dataset.py`)
- Fine-tuning implementation (`finetuning.py`)
- Custom SARI score calculation (`sari.py`)

## Prerequisites

**Environment Requirements**
* Python (see requirements.txt for version)
* CUDA-compatible GPU recommended for efficient fine-tuning
* 8GB GPU memory minimum (for reasonable batch sizes)

**Required Knowledge**
* Basic understanding of Python programming
* Familiarity with machine learning concepts
* Basic understanding of text processing and NLP
* Experience with PyTorch or similar deep learning frameworks (recommended)

**Data Requirements**
* Domain-specific parallel corpus for fine-tuning
  - Source text (complex)
  - Target text (simplified)
* Validation dataset for model evaluation

## Installation

1. Clone this repository
```bash
git clone https://github.com/OfficialVreesie/bert-text-simplification.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Baseline Evaluation
baseline.py implements the initial evaluation using the base BART model. It:

* Loads the pre-trained BART base model
* Processes input text using beam search and sampling
* Evaluates using BERTScore and SARI metrics

To run the baseline evaluation:
```bash
python baseline.py
```

The script expects:
* Input data in CSV format with 'complex' and 'simple' columns
* Data file at 'data/medical/validation.csv'

Output metrics include:

* BERTScore (semantic similarity)
* SARI Score (overall simplification quality) with components: Add, Keep and Delete

### Model Parameters
The baseline uses the following generation parameters:

* max_length: 128 tokens
* num_beams: 4
* temperature: 0.7
* top_p: 0.9
* sampling enabled

## Project Structure

* `baseline.py`: Implements and evaluates the baseline BART model
* `classes/simplification_dataset.py`: Custom dataset class for handling parallel text data
* `finetuning.py`: Contains the fine-tuning implementation
* `sari.py`: Implements SARI score calculation for evaluation