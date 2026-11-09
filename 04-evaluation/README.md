
## Introduction
TODO: inference throughout (tokens/s), custom evaluation, scaling laws, ...

### Prototype
**Simple Eval:** generative (GSM8K) and categorical (MMLU) evaluation, backend by vLLM
```bash
# Generative Evaluation
python eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct

# Categorical Evaluation
python eval.py --task mmlu --model Qwen/Qwen2.5-0.5B-Instruct
```

## Custom Evaluation
TODO

## Local Eval Suit
A local evaluation suite for measuring model quality across several standard benchmarks using Inspect AI:
```bash
# Install dependencies
pip install inspect-ai inspect-evals vllm

# Run a full local evaluation suite with Inspect AI and vLLM
bash run_eval_suit.sh

# Interactive viewer
inspect view
```

## Scaling Laws
TODO: [paper-reproduction](scaling_laws/paper-reproductions.md)
1. Model size and data should be scaled roughly equally.
2. 20 tokens per parameter from empirical experience.

## Reference & Acknowledgement:

- UKGovernmentBEIS/inspect_ai: [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)