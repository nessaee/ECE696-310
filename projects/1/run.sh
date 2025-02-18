# python src/data/download.py --models gpt2-small distilgpt2 --datasets imdb --task classification
# python src/main.py --mode evaluate --model gpt2-small --dataset imdb
# python src/main.py --mode train --model gpt2-small --dataset imdb --num_epochs 3 --eval_steps 1 --save_steps 1 --use_wandb


# 1. Download data and model
./scripts/download_data.sh --datasets imdb --models gpt2-small --task classification

# 2. Run baseline evaluation
./scripts/evaluate_baseline.sh --model gpt2-small --dataset imdb

# 3. Fine-tune model
./scripts/finetune_model.sh --model gpt2-small --dataset imdb --epochs 3 --use-wandb