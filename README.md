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

## Pretraining
Model Architecture + Data + Scaling

## Midtraining
Annealing, long-context, domain/reasoning warmup.

## Post-training
SFT, preference optimization, RL, distillation.

## Evaluation
Harnesses, capability vs alignment

## Frontier Tracker
Papers / blogs / tech reports / news

Families: ...

## Others
Tokenizers, distributed training?

## Reference & Acknowledgement:

- epfml/llm-baselines: [GitHub](https://github.com/epfml/llm-baselines)
- karpathy/nanochat: [GitHub](https://github.com/karpathy/nanochat)
- labmlai/annotated_deep_learning_paper_implementations: [GitHub](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- allenai/OLMo-core: [GitHub](https://github.com/allenai/OLMo-core/)