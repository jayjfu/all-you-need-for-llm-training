
## Introduction
TODO: turn base model to instruct model through: SFT, RL, DPO, Distillation, ...

TODO: Roadmaps:
1. with Hugging Face stack: TRL + PEFT + Unsloth
2. without TRL: from scratch
3. Megatron-style?? parallelism and efficiency

## SFT
**Scripts:**
```bash
# Train SFT variants
python sft_train.py

# Evaluate base + instruct + all SFT variants on leaderboard tasks
bash sft_eval.sh
```

## Preference
TODO

## RL
TODO

## Distillation
TODO

## Results
We only include all the results on six tasks of LLM learderboard v2 here. Additional results on other benchmarks and custom evaluations 
are available in the [Evaluation Section](../04-evaluation/README.md).  

TODO: table here

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