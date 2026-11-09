
## Results
We compare the base model, the official instruct model and our SFT variants on six leaderboard tasks: IFEval, MuSR, MATH 
Level-5, BBH, MMLU-Pro, and GPQA. 

| Stage    | Model                  | IFEval | MuSR  | MATH Lvl-5 | BBH   | MMLU-Pro | GPQA  | Avg   |
|----------|------------------------|--------|-------|------------|-------|----------|-------|-------|
| Base     | Qwen/Qwen3.5-0.8B-Base | 25.88  | 38.10 | 7.33       | 40.72 | 24.84    | 29.19 | 27.68 |
| SFT      |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| SFT      |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| SFT      |                        | 0      | 0     | 0          | 0     | 0        | 0     | 0     |
| Instruct | Qwen/Qwen3.5-0.8B      | 53.97  | 42.72 | 12.99      | 33.76 | 16.27    | 26.01 | 30.95 |
