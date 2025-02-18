"""
Configuration settings for models and datasets.
"""

# Model configurations
MODEL_CONFIGS = {
    'gpt2-small': {
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 5e-5,
    },
    'gpt-neo-125m': {
        'max_length': 512,
        'batch_size': 4,
        'learning_rate': 3e-5,
    },
    'distilgpt2': {
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 5e-5,
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'imdb': {
        'task': 'classification',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 16,
    },
    'ag_news': {
        'task': 'classification',
        'num_labels': 4,
        'max_length': 256,
        'batch_size': 32,
    },
    'wikitext': {
        'task': 'language-modeling',
        'max_length': 512,
        'batch_size': 8,
    },
    'samsum': {
        'task': 'summarization',
        'max_length': 512,
        'target_max_length': 128,
        'batch_size': 8,
    }
}
