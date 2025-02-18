# LLM Fine-tuning Project

This project involves working with small open-source LLMs (like GPT-2, GPT-Neo, DistilGPT-2) for various NLP tasks including classification, language modeling, and summarization.

## Project Structure

```
.
├── data/               # Dataset storage
├── models/            # Model checkpoints and configs
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model training code
│   └── evaluation/   # Evaluation scripts
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Tasks

1. Baseline Evaluation
   - Model setup and inference
   - Dataset preparation
   - Performance measurement

2. Fine-tuning and Improved Evaluation
   - Parameter-efficient fine-tuning (LoRA)
   - Results comparison
   - Performance analysis

## Datasets Options

- Classification: IMDB (50K movie reviews) or AG News (127.6K news articles)
- Language Modeling: WikiText-2 (under 1M tokens)
- Summarization: SAMSum or CNN/DailyMail subset

## Evaluation Metrics

- Classification: Accuracy, F1-score
- Language Modeling: Perplexity
- Summarization: ROUGE scores
