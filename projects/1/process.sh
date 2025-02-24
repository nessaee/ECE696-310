#!/bin/bash
python3 scripts/consolidate_runs.py results/gpt2-small
python3 scripts/consolidate_runs.py results/distilgpt2


python3 scripts/performance_analysis.py results/gpt2-small
python3 scripts/performance_analysis.py results/distilgpt2
