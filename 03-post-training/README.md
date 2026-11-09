
## Introduction
TODO: turn base model to instruct model through: SFT, RL, DPO, Distillation, ...

TODO: Roadmaps:
1. with Hugging Face stack: TRL + PEFT + Unsloth
2. without TRL: from scratch
3. Megatron-style?? parallelism and efficiency

## SFT
TODO  
**Scripts:**
```bash
# Train SFT variants
python trl_train_sft.py

# Evaluate base + instruct + all SFT variants on leaderboard tasks
bash eval_sft.sh
```

## Preference
TODO

## RL
TODO: GRPO on MATH-lvl-1,2,3,4  

**Scripts:**
```bash
# GRPO
python trl_train_grpo.py

# GRPO (custom) 
python train_grpo.py
```

```bash
# Eval on MATH-lvl-5
...
```

## Distillation
TODO

## Results
We only include all the results on six tasks of LLM learderboard v2 here. Additional results on other benchmarks and custom evaluations 
are available in the [Evaluation Section](../04-evaluation/README.md).  

| Stage      | Model                  | IFEval | MuSR  | MATH Lvl-5 | BBH   | MMLU-Pro | GPQA  | Avg   |
|------------|------------------------|--------|-------|------------|-------|----------|-------|-------|
| Base       | Qwen/Qwen3.5-0.8B-Base | 25.88  | 38.10 | 7.33       | 40.72 | 24.84    | 29.19 | 27.68 |
| SFT        |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| Preference |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| RL         |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| Instruct   | Qwen/Qwen3.5-0.8B      | 53.97  | 42.72 | 12.99      | 33.76 | 16.27    | 26.01 | 30.95 |

## Benchmarks
TODO
1. Coding:
   - TODO
2. Agentic:
   - IFEval (Instruction Following Evaluation)
3. Reasoning & Knowledge & Multimodal:
   - MuSR (Multistep Soft Reasoning), MATH Level-5 (Mathematics Aptitude Test of Heuristics, Level 5 subset), BBH (Big 
   Bench Hard) 
   - MMLU-Pro (Massive Multitask Language Understanding - Pro version), GPQA (Google-Proof Q&A Benchmark)
   - TODO
4. Cybersecurity
   - TODO

## Others
TODO: potential problems/failure: KL collapse, reward hacking, eval contamination, verbosity bias, ...

## Reference & Acknowledgement:

- huggingface/trl: [GitHub](https://github.com/huggingface/trl)
- allenai/open-instruct: [GitHub](https://github.com/allenai/open-instruct)
- EleutherAI/lm-evaluation-harness: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)