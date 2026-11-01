# All You Need for LLM Training

### This repo aims to cover what you need to know about LLM pretraining, post-training, and evaluation, plus a tracker of frontier models.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

TODO

### Prototype
**Model Architecture:** A mini GPT-2 (inspired by nanochat).
- Positions: RoPE
- Norm: RMSNorm
- Attention: MHA or GQA
- MLP: ReLU²
- Embed / head: untied
- Optimizer: Muon on matrices + AdamW on embeddings and lm_head

**Tokenizer:** Regex-pretokenized BPE, byte pairs merged by frequency up to vocab_size.
```bash
python tok_train.py --vocab-size 1024
```

**Document Packing:** Best-fit into a row of exactly seq_len, place the largest document that still fits, and crop the shortest only if nothing fits.

**Training:** Next-token pretrain on documents, --depth as size dial. Muon on matrices, AdamW on the rest.
```bash
python train.py --depth 2 --steps 400 --batch-size 8 --seq-len 128
```

**SFT:** Post-training with loss only on assistant tokens.
```bash
python train_sft.py --steps 100 --seq-len 64
```

**Reinforcement Learning:** Post-training with a minimal GRPO update on a verified reward.
```bash
python rl.py
```

**Simple Eval:** generative (GSM8K) and categorical (MMLU) evaluation, backend by vLLM
```bash
python eval.py --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --max-problems 50
python eval.py --task mmlu --model Qwen/Qwen2.5-0.5B-Instruct --max-problems 100
```

## Pretraining

### Baseline
WIP: Baseline model + all results in [RESULTS.md](01-pretraining/src/baseline/RESULTS.md)

- Model Architecture: RoPE + pre-norm + RMSNorm + SwiGLU
- Data: cleaning, mixture ratio
- Training: AdamW + scheduler (cosine/WSD)
- Hyperparameter: learning rate, batch size

## Midtraining
TODO:
- Annealing, long-context, domain/reasoning warmup

## Post-training
TODO:
- SFT, preference optimization, RL, distillation

## Evaluation
Run a full local evaluation suite with Inspect AI and vLLM:
```bash
bash run_eval_suit.sh
```

TODO:
- Scaling laws: [paper-reproduction](04-evaluation/scaling_laws/paper-reproductions.md)
- benchmarks, inference throughout

## Optimization
TODO:
- inference: kv cache, training efficiency...

## Frontier Tracker
Tech reports / blogs / papers

Families: ...

## Others
Tokenizers, distributed training...

## Reference & Acknowledgement:

- epfml/llm-baselines: [GitHub](https://github.com/epfml/llm-baselines)
- karpathy/nanochat: [GitHub](https://github.com/karpathy/nanochat)
- labmlai/annotated_deep_learning_paper_implementations: [GitHub](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- allenai/OLMo-core: [GitHub](https://github.com/allenai/OLMo-core/)