## Introduction
TODO, scaling laws, inference throughout (tokens/s)

### Prototype
**Simple Eval:** generative (GSM8K) and categorical (MMLU) evaluation, backend by vLLM
```bash
# Generative Evaluation
python eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct
# Categorical Evaluation
python eval.py --task mmlu --model Qwen/Qwen2.5-0.5B-Instruct
```

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

## Reference & Acknowledgement:

- Inspect AI: [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)