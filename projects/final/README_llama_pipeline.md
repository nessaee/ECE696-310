# Llama 7B and Llama Guard Pipeline

This project implements a flexible and modular pipeline for interacting with Llama 7B and Llama Guard models. The pipeline allows for content moderation of both inputs and outputs, with customizable processing steps. It supports both online and offline usage, allowing you to download models once and use them without an internet connection.

## Features

- Modular architecture with abstract interfaces for language models and content moderation
- Implementation for Llama 7B as the primary language model
- Implementation for Llama Guard as the content moderation model
- Pre-moderation of input prompts to filter harmful content
- Post-moderation of model responses to ensure safe outputs
- Customizable processing pipeline with the ability to add custom processing steps
- Command-line interface for easy usage
- **Offline support**: Download models once and use them without an internet connection
- **Local model loading**: Load models from local directories or the Hugging Face cache

## Project Structure

- `llm_interface.py`: Abstract base classes for language model and content moderation interfaces
- `llama_models.py`: Implementation of Llama 7B and Llama Guard model interfaces with offline support
- `llama_pipeline.py`: Pipeline for orchestrating interactions between models
- `run_llama_pipeline.py`: Command-line interface for running the pipeline
- `download_llama_models.py`: Script for downloading models for offline use

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Hugging Face account with access to Llama models

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install torch transformers
```

3. Set up your Hugging Face token:

```bash
export HF_TOKEN=your_hugging_face_token
```

## Usage

### Basic Usage

```bash
python run_llama_pipeline.py --prompt "Tell me about artificial intelligence"
```

### Using a File of Prompts

```bash
python run_llama_pipeline.py --prompt-file prompts.txt --output-file results.json
```

### Customizing Model Parameters

```bash
python run_llama_pipeline.py --prompt "Tell me about AI" \
    --llm-model "meta-llama/Llama-2-7b-hf" \
    --moderation-model "meta-llama/LlamaGuard-7b" \
    --temperature 0.8 \
    --max-length 1024
```

### Using the Custom Pipeline

```bash
python run_llama_pipeline.py --prompt "Tell me about AI" --custom-pipeline
```

### Disabling Moderation

```bash
python run_llama_pipeline.py --prompt "Tell me about AI" --no-pre-moderation --no-post-moderation
```

## Offline Usage

### Downloading Models for Offline Use

Before using the models offline, you need to download them first:

```bash
python download_llama_models.py --token YOUR_HUGGING_FACE_TOKEN
```

You can specify a custom download directory:

```bash
python download_llama_models.py --token YOUR_HUGGING_FACE_TOKEN --download-dir ~/llama_models
```

### Running in Offline Mode

Once the models are downloaded, you can run the pipeline in offline mode:

```bash
python run_llama_pipeline.py --offline --prompt "Tell me about AI"
```

If you specified a custom download directory, include it in the command:

```bash
python run_llama_pipeline.py --offline --download-dir ~/llama_models --prompt "Tell me about AI"
```

### Using Local Model Files

If you have the model files in a specific directory, you can point directly to them:

```bash
python run_llama_pipeline.py --offline \
    --llm-local-path /path/to/llama-7b \
    --moderation-local-path /path/to/llama-guard \
    --prompt "Tell me about AI"
```

## Extending the Pipeline

### Adding Custom Processing Steps

You can extend the pipeline by adding custom processing steps. Create a function that takes and returns a result dictionary, then add it to the pipeline:

```python
from llama_pipeline import CustomPipeline

def my_custom_step(result):
    # Process the result
    result["metadata"]["custom_field"] = "custom_value"
    return result

pipeline = CustomPipeline()
pipeline.add_processing_step(my_custom_step)
```

### Implementing Custom Models

You can implement custom models by extending the base interfaces:

```python
from llm_interface import LLMInterface

class MyCustomModel(LLMInterface):
    def __init__(self, model_name, **kwargs):
        # Initialize your model
        pass
    
    def generate(self, prompt, **kwargs):
        # Generate a response
        pass
    
    def batch_generate(self, prompts, **kwargs):
        # Generate responses for multiple prompts
        pass
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
