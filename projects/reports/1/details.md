# Model Training and Evaluation Technical Documentation

## Model Architectures and Implementation Details

### GPT-2 Small Architecture
- Base Model: `gpt2` from HuggingFace
- Architecture Type: Decoder-only transformer
- Core Components:
  - Multi-head self-attention layers
  - Position-wise feed-forward networks
  - Layer normalization
  - Residual connections
- Model Parameters:
  - Layers: 12 transformer blocks
  - Hidden Size: 768 dimensions
  - Attention Heads: 12 (64 dimensions per head)
  - Feed-forward Size: 3072 (4x hidden size)
  - Total Parameters: 124M
  - Vocabulary Size: 50,257 tokens
- Implementation Details:
  - Activation Function: GELU
  - Layer Normalization: Pre-norm configuration
  - Position Embeddings: Learned, absolute
  - Maximum Sequence Length: 1024 tokens
  - Dropout Rate: 0.1

### DistilGPT2 Architecture and Distillation
- Base Model: `distilgpt2` from HuggingFace
- Architecture Overview:
  - Distilled version of GPT-2 Small
  - Preserves core transformer architecture
  - Reduced number of layers while maintaining width

#### Model Parameters
- Architectural Dimensions:
  - Layers: 6 (reduced from 12)
  - Hidden Size: 768 (same as GPT-2)
  - Attention Heads: 12 (same as GPT-2)
  - Feed-forward Size: 3072 (same as GPT-2)
  - Total Parameters: 82M (34% reduction)
  - Vocabulary Size: 50,257 (same as GPT-2)

#### Architecture Differences from GPT-2
1. Layer Reduction:
   - Uses 6 transformer blocks instead of 12
   - Maintains layer architecture within blocks
   - Preserves attention head count and dimensionality

2. Knowledge Transfer:
   - Initialization: Student layers initialized from selected teacher layers
   - Layer Mapping: Every other layer from teacher mapped to student
   - Weight Averaging: Careful initialization to preserve knowledge

#### Benefits of Distillation
1. Computational Efficiency:
   - 40% faster inference time
   - 34% reduction in parameter count
   - Lower memory footprint

2. Knowledge Preservation:
   - Maintains 95% of GPT-2's performance
   - Better generalization on some tasks
   - More robust to input variations

3. Training Advantages:
   - Faster fine-tuning
   - Lower memory requirements
   - Enables larger batch sizes

#### Implementation Details
1. Tokenization:
   - Uses identical GPT-2 BPE tokenizer
   - Vocabulary size: 50,257 tokens
   - Special tokens handling preserved

2. Attention Mechanism:
   - Multi-head attention with 12 heads
   - Head size: 64 dimensions
   - Causal masking for autoregressive prediction

3. Position Embeddings:
   - Learned absolute positional embeddings
   - Maximum sequence length: 1024
   - Shared with GPT-2 initialization

## Model Initialization

### Classification Tasks (IMDB, AG News)
1. Base Model Loading:
   ```python
   model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
       num_labels=dataset_config['num_labels']
   )
   ```
   - Automatically adds a classification head on top of the base model
   - Classification head: Linear layer mapping hidden size (768) to num_labels
   - Uses default dropout rate of 0.1

### Language Modeling Tasks (WikiText)
1. Base Model Loading:
   ```python
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```
   - Uses original model architecture without modifications
   - Preserves pre-trained weights for the language modeling head

## Baseline Evaluation

### Process Flow
1. Model Initialization:
   - Load pre-trained model without any fine-tuning
   - Configure for task-specific output (classification or language modeling)

2. Dataset Loading:
   - Load evaluation dataset
   - Apply model-specific tokenization
   - Create DataLoader with batch size from config

3. Evaluation:
   - Run model in evaluation mode (`model.eval()`)
   - Compute task-specific metrics:
     - Classification: Accuracy, F1 Score
     - Language Modeling: Perplexity, Loss

4. Metrics Storage:
   - Save metrics to: `results/model_name/timestamp/baseline/metrics/`
   - Save analysis to: `results/model_name/timestamp/baseline/analysis/`

## Fine-tuning Approach and Process

### Fine-tuning Strategy

#### 1. Task Adaptation
- **Classification Tasks**:
  - Add classification head: Linear(hidden_size, num_classes)
  - Initialize with truncated normal distribution
  - Apply task-specific dropout (0.1)
  - Use cross-entropy loss for training

- **Language Modeling Tasks**:
  - Preserve original LM head
  - Share embeddings with output layer
  - Use causal language modeling loss
  - Apply token-wise cross-entropy

#### 2. Layer Freezing Strategy
- **First Phase (Optional)**:
  - Freeze transformer layers
  - Train only task-specific heads
  - Duration: 1 epoch

- **Second Phase**:
  - Unfreeze all layers
  - Apply differential learning rates:
    - Lower layers: 0.1x base learning rate
    - Middle layers: 0.5x base learning rate
    - Top layers: 1x base learning rate

#### 3. Optimization Configuration
1. Training Parameters:
   ```python
   {
       'num_epochs': 3,
       'warmup_ratio': 0.1,
       'weight_decay': 0.01,
       'max_grad_norm': 1.0,
       'learning_rates': {
           'classification': 2e-5,
           'language_modeling': 5e-5
       },
       'batch_sizes': {
           'gpt2': {
               'classification': 8,
               'language_modeling': 4
           },
           'distilgpt2': {
               'classification': 16,
               'language_modeling': 8
           }
       }
   }
   ```

2. Learning Rate Schedule:
   - Linear warmup for first 10% of steps
   - Linear decay for remaining steps
   - Minimum learning rate: 1e-7

3. Gradient Handling:
   - Gradient clipping at 1.0
   - Gradient accumulation for large batches
   - FP16 mixed precision training

#### 4. Regularization Techniques
1. Weight Decay:
   - Applied to all non-bias parameters
   - Rate: 0.01
   - Excluded from layer norm parameters

2. Dropout Scheme:
   - Attention dropout: 0.1
   - Hidden state dropout: 0.1
   - Classification head dropout: 0.1

3. Input Regularization:
   - Random sequence cropping
   - Dynamic sequence padding
   - Label smoothing (classification only): 0.1

2. Directory Structure:
   ```
   results/model_name/timestamp/
     ├── train/
     │   ├── metrics/
     │   ├── analysis/
     │   └── weights/
     └── test/
         ├── metrics/
         └── analysis/
   ```

### Training Process and Monitoring

#### 1. Model Preparation and Setup
```python
def prepare_model(model, task_config):
    # Add task-specific heads
    model = add_task_head(model, task_config)
    
    # Configure mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Move to GPU and parallelize if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    return model, scaler
```

#### 2. Optimization Setup
```python
def configure_optimizer(model, config):
    # Parameter groups with different learning rates
    param_groups = [
        {
            'params': model.transformer.h[:4].parameters(),
            'lr': config.learning_rate * 0.1
        },
        {
            'params': model.transformer.h[4:8].parameters(),
            'lr': config.learning_rate * 0.5
        },
        {
            'params': model.transformer.h[8:].parameters(),
            'lr': config.learning_rate
        }
    ]
    
    optimizer = AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        correct_bias=False
    )
    
    return optimizer
```

#### 3. Training Loop Implementation
```python
def train_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    
    with tqdm(dataloader) as pbar:
        for step, batch in enumerate(pbar):
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer and scheduler steps
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (step + 1)})
```

#### 4. Evaluation and Metrics Tracking
```python
def evaluate(model, eval_dataloader, metric_tracker):
    model.eval()
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            
            # Task-specific metric computation
            batch_metrics = compute_metrics(outputs, batch)
            for k, v in batch_metrics.items():
                metrics[k].append(v)
    
    # Average and log metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    metric_tracker.update(avg_metrics)
    
    return avg_metrics
```

#### 5. Checkpointing Strategy
```python
def save_checkpoint(model, optimizer, scheduler, metrics, config):
    checkpoint = {
        'epoch': config.current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    # Save based on metric improvement
    if metrics['eval_loss'] < config.best_loss:
        save_path = f'checkpoint_best.pt'
    else:
        save_path = f'checkpoint_epoch_{config.current_epoch}.pt'
    
    torch.save(checkpoint, config.weights_dir / save_path)
```

#### 6. Training Monitoring
1. Real-time Metrics:
   - Loss curves (training and validation)
   - Learning rate schedule visualization
   - GPU memory usage
   - Training speed (samples/second)

2. Validation Metrics:
   - Task-specific metrics (accuracy, F1, perplexity)
   - Confusion matrices for classification
   - Attention pattern visualization
   - Layer-wise gradient norms

3. System Monitoring:
   - CPU/GPU utilization
   - Memory consumption
   - Disk I/O for checkpoints
   - Training time per epoch

### Checkpointing
- Frequency: Every `save_steps` steps
- Saved Components:
  ```python
  {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'metrics': metrics,
      'config': config
  }
  ```

## Model-Specific Configurations

### GPT-2 Small
```python
{
    'name': 'gpt2-small',
    'pretrained_model': 'gpt2',
    'max_length': 1024,
    'batch_size': {
        'classification': 8,
        'language-modeling': 4
    },
    'learning_rate': {
        'classification': 2e-5,
        'language-modeling': 5e-5
    }
}
```

### DistilGPT2
```python
{
    'name': 'distilgpt2',
    'pretrained_model': 'distilgpt2',
    'max_length': 1024,
    'batch_size': {
        'classification': 16,  # Larger due to fewer parameters
        'language-modeling': 8
    },
    'learning_rate': {
        'classification': 2e-5,
        'language-modeling': 5e-5
    }
}
```

## Dataset Processing

### Classification (IMDB)
1. Text Processing:
   - Truncation: Max length 1024 tokens
   - Padding: Dynamic padding to batch max length
   - Special Tokens: 
     - `[CLS]` at start
     - `[SEP]` at end

2. Label Processing:
   - Binary classification (0: negative, 1: positive)
   - Label smoothing: None

### Language Modeling (WikiText)
1. Text Processing:
   - Truncation: Max length 1024 tokens
   - Stride: 512 tokens for overlapping contexts
   - Special Tokens: None (uses model's default tokens)

2. Target Processing:
   - Shifted input sequence for next token prediction
   - Masks padding tokens in loss computation
