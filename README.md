# VerifierRL

A controlled study of whether reinforcement learning (GRPO with verifiable rewards) improves a small language model's math reasoning beyond supervised fine-tuning. The model is Qwen2.5-1.5B-Instruct. Training uses GSM8K and evaluation uses GSM8K plus SVAMP. The evals are built to tell real reasoning gains apart from format learning and reward hacking.

**Status: not started.** This repo holds the design only. There is no code yet and no results. Every number in the docs is a target or a placeholder.

**Cost: $0.** It uses free-tier GPUs (Google Colab or Kaggle T4), open-weight models, open datasets, and open-source libraries.

## Study design

```
Qwen2.5-1.5B-Instruct ──► Base          (no training)
        │
        └─ LoRA SFT on GSM8K train ──► SFT
                 │
                 └─ GRPO (correctness + format rewards) ──► SFT+GRPO

All three checkpoints ──► one evaluator ──► GSM8K test (1,319) + SVAMP (held out)
                                            strict and lenient scoring, greedy decoding
```

## Documentation

- [docs/PROJECT.md](docs/PROJECT.md): what the project is, how it is built, how to run it for free, and the design decisions still open.

## License

[MIT](LICENSE). The model (Qwen2.5-1.5B-Instruct, Apache-2.0) and datasets (GSM8K, SVAMP, MIT) keep their own licenses.
