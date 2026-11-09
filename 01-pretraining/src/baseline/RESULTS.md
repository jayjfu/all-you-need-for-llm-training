
## Results

TODO:
- Data: cleaning, mixture ratio
- Training stability
- learning rate, batch size
- attention kernels

Others:
- gated linear units: SwiGLU + ...
- GQA ratio, RoPE theta/base, optional QK-norm, depth-vs-width balance, MoE at large scale??

Notes:
- Optimizer: AdamW (0.9, 0.95, 1e-8), Muon > ...
- Gradient clip = 1.0, dropout = 0
- Scheduler: cosine/linear decay, warmup-stable-decay (WSD)
- Pre-norm > post-norm, RMSNorm > LayerNorm
- RoPE > absolute positional embeddings
- SwiGLU > ReLU/GELU FFNs
- MHA > GQA/MQA
