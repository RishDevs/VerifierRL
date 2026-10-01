# VerifierRL: Project Documentation

> **Status (2026-10-01): design stage.** No code has been written and no experiments have run. This document describes the planned system. Sections marked **Planned** describe things that don't exist yet. All result values are **TBD**. Section 8 lists the open decisions.

## Contents

1. [What the project is](#1-what-the-project-is)
2. [Research questions and claims under test](#2-research-questions-and-claims-under-test)
3. [Fixed choices](#3-fixed-choices)
4. [How it is built](#4-how-it-is-built)
5. [Evaluation protocol](#5-evaluation-protocol)
6. [Reward verifier and audit](#6-reward-verifier-and-audit)
7. [Setup and running for free](#7-setup-and-running-for-free)
8. [Open decisions](#8-open-decisions)
9. [Zero-cost constraints and their tradeoffs](#9-zero-cost-constraints-and-their-tradeoffs)
10. [Results](#10-results)

---

## 1. What the project is

VerifierRL is a controlled experiment. It trains one small model in three stages and compares the stages fairly:

| Checkpoint | How it is produced |
|---|---|
| **Base** | Qwen2.5-1.5B-Instruct with no changes |
| **SFT** | Base + a LoRA adapter trained with supervised fine-tuning on GSM8K training problems written in a fixed answer format |
| **SFT+GRPO** | SFT, then further trained with GRPO (Group Relative Policy Optimization). Rewards come from a program that checks answers, not from a learned reward model |

A single evaluator scores all three checkpoints with the same prompt, the same decoding settings, and the same answer-extraction code. As a result, any difference in scores comes from training and not from how the scores were measured.

**Main question:** Does RL with verifiable rewards improve reasoning beyond what SFT gives? Or does it mostly teach the model to produce the output format?

## 2. Research questions and claims under test

Each claim the project hopes to support, with the measurement that will test it. The bracketed values are targets, not results.

| # | Claim | Measured by | Target |
|---|---|---|---|
| C1 | GRPO beats SFT on GSM8K | Strict accuracy on GSM8K test (1,319 problems, greedy), SFT+GRPO minus SFT, with a paired significance test (§5.4) | [≥ 2 pp] |
| C2 | GRPO matches or beats Base | Strict accuracy on GSM8K test, SFT+GRPO vs Base | [SFT+GRPO ≥ Base] |
| C3 | Format compliance improves | % of outputs where strict extraction succeeds (§5.2) | [~60% → ≥ 90%] |
| C4 | Gains are reasoning, not format | Lenient accuracy, which ignores format, rises along with strict accuracy | [lenient also improves] |
| C5 | No overfitting to GSM8K | SVAMP accuracy (never trained on) for SFT+GRPO vs Base | [within 1 pp or better] |
| C6 | The verifier was hardened | Exploits found and patched in a manual audit of [200] rollouts (§6.3) | [N] exploits, each with a regression test |

The 73.2% Base GSM8K figure is a placeholder. The real Base number is whatever this project's evaluator measures (§5). That number depends on the prompt, the shot count, and the extraction rule, so it can differ from published figures.

## 3. Fixed choices

These are already decided and aren't open for discussion in this document.

| Component | Choice | License / cost |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face) | Apache-2.0, free, not gated |
| Training data | GSM8K, `openai/gsm8k`, config `main`: 7,473 train / 1,319 test | MIT, free |
| Held-out eval | SVAMP (1,000 problems; the HF mirror `ChilleD/SVAMP` splits it 700 train / 300 test) | MIT, free |
| Training library | TRL (`SFTTrainer`, `GRPOTrainer`) | Apache-2.0 |
| Efficiency layer | Unsloth (memory-efficient LoRA, optional vLLM generation) | Apache-2.0 |
| Fine-tuning method | LoRA adapters; base weights frozen | n/a |
| Compute | Free NVIDIA T4 (16 GB) | $0 |
| Decoding for eval | Greedy | n/a |
| Scoring | Strict (tag-parsed) and lenient (last number), every checkpoint | n/a |

**Hardware constraint that affects the design:** the T4 does not support bfloat16. All training and inference run in float16, or in 4-bit with float16 compute.

## 4. How it is built

### 4.1 Pipeline

```
 GSM8K train (7,473)
   ├─ dev split (held out from training, used for tuning and audit) ──────────────┐
   ├─ SFT split ──► format solutions into tags ──► LoRA SFT ──► SFT adapter       │
   └─ GRPO split ─────────────────────────────────► GRPO (from SFT adapter)       │
                                                       │  rewards: correctness    │
                                                       │           + format       │
                                                       ▼                          │
                                                 SFT+GRPO adapter                 │
                                                                                  │
 Base, SFT, SFT+GRPO ──► evaluator ──► GSM8K test (1,319), SVAMP, dev ◄───────────┘
                            │
                            └─► per-example JSONL + summary metrics CSV
```

The GSM8K test set is used only for final reported numbers. It is never used for tuning, checkpoint selection, or the verifier audit. Those use the dev split.

### 4.2 Output format

Every training and evaluation prompt uses the same system prompt. It asks the model to answer like this:

```
<reasoning>
step-by-step working
</reasoning>
<answer>
42
</answer>
```

An explicit system prompt replaces Qwen's default one ("You are Qwen, created by Alibaba Cloud…"). Every checkpoint, including Base, sees this instruction, so Base's format compliance is measured under the same request as the others.

### 4.3 Stage 1: SFT (Planned)

- **Input:** GSM8K training problems, with reference solutions rewritten into the tag format. GSM8K solutions contain calculator annotations such as `<<48/2=24>>`, which are removed. The final answer after `####` goes into `<answer>`.
- **Method:** LoRA on the attention and MLP projection layers. Loss is computed on the assistant response only, not on the prompt.
- **Output:** a LoRA adapter (tens of MB), saved off the runtime (§7.5).
- **Open choices:** the SFT data source (§8, D5) and the train-split strategy (§8, D4).

### 4.4 Stage 2: GRPO (Planned)

GRPO samples a **group** of *G* completions for each prompt and scores each one with the reward functions. Each completion's advantage is its reward minus the group mean, divided by the group's standard deviation. The policy is then pushed toward completions that scored above their group's average. No value network or learned reward model is needed. This is why GRPO fits on a single T4.

- **Starting point:** the SFT adapter.
- **Rewards** (details in §6):
  - *Correctness:* 1.0 if the strictly extracted answer equals the gold answer numerically, else 0.0.
  - *Format:* a small bonus for exactly one well-formed `<reasoning>` block and one well-formed `<answer>` block. It is kept small relative to correctness, so the model can't earn most of the reward from format alone.
- **Logged during training:** mean reward per component, reward standard deviation within groups, completion length, truncation rate, and KL to the reference model (if enabled). Every *k* steps, sampled rollouts are written to JSONL for the audit.
- **Monitoring for reward hacking:** strict accuracy on the dev split is checked regularly. If training reward rises while dev accuracy stays flat or falls, the verifier is being gamed.
- **Open choices:** group size, completion length, KL penalty, generation backend, precision (§8).

### 4.5 Planned repository layout

```
VerifierRL/
├── README.md
├── docs/PROJECT.md            this document
├── requirements.txt           pinned versions (pinned once a working set is confirmed on T4)
├── requirements-dev.txt       CPU-only: pytest + parsing deps for local tests
├── configs/
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/verifierrl/
│   ├── data.py                load GSM8K/SVAMP, build dev split, format SFT targets
│   ├── prompts.py             the one system prompt + chat-template wrapper
│   ├── extract.py             strict + lenient extraction, number normalization
│   ├── rewards.py             correctness + format rewards (built on extract.py)
│   ├── train_sft.py
│   ├── train_grpo.py
│   ├── evaluate.py            scores any checkpoint on any dataset
│   ├── stats.py               paired significance tests
│   └── audit.py               samples rollouts into a labeling sheet
├── notebooks/
│   ├── colab_run.ipynb        thin launcher: install, clone, call scripts
│   └── kaggle_run.ipynb
├── tests/
│   ├── test_extract.py
│   └── test_rewards.py        one regression test per exploit found in the audit
└── results/
    ├── eval/                  per-example JSONL per (checkpoint, dataset)
    ├── metrics.csv            one row per (checkpoint, dataset)
    └── audit/                 labeled rollouts + exploit log
```

Design rule: `rewards.py` and `evaluate.py` both import from `extract.py`. The reward used in training and the strict metric used in evaluation therefore apply exactly the same parsing rule. Because of this, a verifier exploit would also inflate strict accuracy. The lenient metric (§5.2) and the manual audit (§6.3) are the independent checks against that.

## 5. Evaluation protocol

### 5.1 Fixed settings (identical for every checkpoint)

- Same system prompt and chat template (§4.2), zero-shot (unless D11 changes this for all checkpoints).
- Greedy decoding, with the same `max_new_tokens` for every checkpoint. An output that hits the limit counts as truncated and is scored as is.
- Same precision and inference engine for every checkpoint (§8, D3 and D8). LoRA checkpoints run as the same base weights plus their adapter. This prevents a precision difference between checkpoints from posing as a training effect.

### 5.2 Two scoring modes

| Mode | Extraction rule | What it isolates |
|---|---|---|
| **Strict** | Exactly one `<answer>…</answer>` block that contains exactly one number. Otherwise the output fails to parse and is scored wrong | Correct *and* follows the format |
| **Lenient** | The last number anywhere in the output (this also handles `\boxed{}` and plain-text answers) | Correctness regardless of format |

Numbers are normalized before comparison: commas, currency symbols, and trailing periods are removed, and the value is compared as a number (so `27.0` equals `27`).

**Interpretation:**
- *Format compliance* is the share of outputs where strict extraction succeeds.
- If strict accuracy rises but lenient accuracy doesn't, the model learned the format, not reasoning.
- If lenient accuracy rises too, the model got better at the math itself.

### 5.3 Metrics recorded per (checkpoint, dataset)

Strict accuracy, lenient accuracy, format compliance, mean output length (tokens), truncation rate, and wall-clock eval time. Per-example outputs are kept in JSONL so any number can be traced back to individual outputs.

### 5.4 Statistical honesty

- With 1,319 problems at about 73% accuracy, one accuracy number has a standard error of about **1.2 pp**. On SVAMP (1,000 problems) it is about **1.4 pp**.
- The models are compared on the same problems, so the comparisons are **paired**. A paired test (McNemar's test, or a paired bootstrap over problems; see D12) has more statistical power than comparing two independent accuracies. Results are reported with a p-value or a confidence interval.
- A 2 pp gap (C1) may or may not be significant. The test decides, and the result gets reported either way.
- "Within 1 pp" on SVAMP (C5) is smaller than SVAMP's own noise. It shows the score didn't measurably drop. It doesn't show the scores are equal.
- If compute allows, GRPO is repeated with 2 or more seeds and the spread across seeds is reported. If not, the document states that the result comes from a single seed.

## 6. Reward verifier and audit

### 6.1 Correctness reward

1. Run strict extraction (§5.2). If it fails, the reward is 0.
2. Normalize the extracted number and the gold number.
3. The reward is 1.0 if they are equal within a small tolerance, else 0.0.

### 6.2 Format reward

A small bonus is given only when the output has exactly one `<reasoning>` block followed by exactly one `<answer>` block, with nothing meaningful after `</answer>`. The bonus's size relative to correctness is part of D6.

### 6.3 Audit procedure (before final runs)

1. Run a short pilot GRPO run on the training data and log rollouts with their rewards.
2. Sample **[200]** rollouts, oversampling high-reward and borderline cases.
3. Label each one by hand: *was the reward right?* If not, write down the exploit pattern.
4. For each exploit: patch `extract.py` or `rewards.py`, then add a failing case to `tests/test_rewards.py` that now passes.
5. Record everything in `results/audit/` (labels, the exploit list, and the commit that fixed each one).
6. Freeze the verifier, then run the final SFT+GRPO training.

**Exploit patterns to check for specifically** (hypotheses until the audit confirms or rules them out):
- Several candidate numbers in one `<answer>` block (`<answer>12 or 14</answer>`)
- Several `<answer>` blocks
- An answer that is right only because extraction grabbed the wrong number (e.g., a number from the restated question)
- Units or words in the answer tag that break or fool parsing
- Empty or degenerate `<reasoning>` that still collects the format reward
- Padding or repeating text until the output hits the length limit

The final count of exploits **[N]** comes from this audit and can't be filled in until the audit runs.

## 7. Setup and running for free

> The commands below are the **planned interface**. They don't work until the code in §4.5 exists.

### 7.1 Requirements

- A Google account (for Colab) or a Kaggle account (phone verification is required to enable GPUs).
- A Hugging Face account. This is optional: downloading the model and datasets needs no token, and an account is only needed to push adapters to the Hub (D10).
- No local GPU is needed. Unit tests for `extract.py` and `rewards.py` run on CPU on any laptop.

### 7.2 Google Colab (free tier)

1. Open `notebooks/colab_run.ipynb` in Colab.
2. Go to Runtime → Change runtime type → **T4 GPU**.
3. Run the setup cell (clone repo, `pip install -r requirements.txt`).
4. Run the stage cells:
   ```bash
   python -m verifierrl.evaluate   --config configs/eval.yaml --checkpoint base
   python -m verifierrl.train_sft  --config configs/sft.yaml
   python -m verifierrl.evaluate   --config configs/eval.yaml --checkpoint sft
   python -m verifierrl.train_grpo --config configs/grpo.yaml
   python -m verifierrl.evaluate   --config configs/eval.yaml --checkpoint sft_grpo
   python -m verifierrl.stats      --metrics results/metrics.csv
   ```
5. Save adapters and results off the runtime before it disconnects (§7.5).

### 7.3 Kaggle Notebooks (free tier)

1. Create a notebook, enable **GPU T4** under Settings → Accelerator, and turn **Internet on**.
2. Same steps as Colab. Files written to `/kaggle/working` are kept as notebook output when you use "Save Version".

### 7.4 Running locally (CPU)

```bash
pip install -r requirements-dev.txt
pytest tests/
```

This covers the verifier and extraction logic, which is the part most worth testing and doesn't need a GPU.

### 7.5 Surviving session limits

Free sessions disconnect: they have time limits and idle timeouts, and Colab doesn't guarantee a GPU will be available. To handle this:
- Training saves checkpoints every *n* steps and can resume from the latest one.
- Checkpoints and results are copied to persistent free storage (D10) during the run, not only at the end.
- Each stage is a separate script run, so a disconnect loses at most one stage's progress since its last checkpoint.

Free-tier quotas and session caps change over time. Check the current limits on each platform before planning a long run.

## 8. Open decisions

Every option listed is free. You make the call. Where a direction is suggested, it is labeled as a suggestion.

| ID | Decision | Options | Tradeoffs |
|---|---|---|---|
| **D1** | Compute platform | (a) Colab free T4 · (b) Kaggle T4 · (c) both, with Kaggle for long runs | Colab: quick to start, but GPU availability and session length vary. Kaggle: a fixed weekly GPU quota and more predictable sessions, but needs phone verification and has a weekly cap. Kaggle also offers 2×T4, but using both GPUs adds complexity that a 1.5B LoRA run doesn't need. |
| **D2** | GRPO generation backend | (a) Unsloth + vLLM (`fast_inference=True`) · (b) standard Hugging Face `generate` inside TRL | Generating rollouts is the main cost in GRPO. (a) is much faster per step, but vLLM and the model share the 16 GB, and version compatibility between Unsloth, TRL, and vLLM can break. (b) is simpler and more stable, but gives fewer GRPO steps in the same free GPU hours. |
| **D3** | Training precision | (a) 4-bit QLoRA · (b) 16-bit LoRA (float16) | (a) leaves more memory for larger groups and longer completions, with a small quality cost from quantization. (b) has no quantization loss, and a 1.5B model in float16 (~3 GB) fits on a T4. Either way, all checkpoints must be evaluated at the same precision (§5.1). |
| **D4** | How GSM8K train is split between SFT and GRPO | (a) both stages use the same training problems · (b) separate halves | (a) gives each stage more data, but GRPO then trains on problems whose solutions SFT already memorized, which makes "RL beyond SFT" harder to interpret. (b) keeps the stages clean, but each gets about half the data. In both cases a dev split is carved out first. |
| **D5** | SFT target source | (a) GSM8K reference solutions, reformatted · (b) rejection sampling: generate from Base, keep correct outputs in the right format | (a) needs no extra compute, but the human-written style doesn't match how the model naturally writes. (b) produces data in the model's own style (which usually makes SFT smoother) and is still free, but costs GPU time and only covers problems Base can already solve. |
| **D6** | Reward shape | (a) binary correctness + small format bonus · (b) add partial-credit or length terms | (a) is simplest and hardest to game. (b) gives the model more training signal, but every extra term is another thing it can exploit. |
| **D7** | KL penalty (β) to the reference model | (a) β = 0 · (b) small β > 0 | (a) uses less memory and compute, since no reference forward pass is needed, but the model can drift further from where it started. (b) keeps the policy close to SFT, which can protect SVAMP performance (C5), but costs memory and an extra hyperparameter. Check TRL's default for the pinned version. |
| **D8** | Eval inference engine | (a) vLLM · (b) Hugging Face `generate` | (a) evaluates 1,319 problems much faster. (b) has fewer dependencies. Greedy outputs can differ slightly between engines, so one engine must be used for every checkpoint. |
| **D9** | Experiment tracking | (a) Weights & Biases free personal tier · (b) TensorBoard · (c) CSV/JSONL only | (a) gives hosted dashboards and comparisons, but it's an external account and its free-tier terms could change. (b) is fully local and free, but logs must be saved off the runtime. (c) has zero dependencies but no live plots. |
| **D10** | Artifact storage | (a) Hugging Face Hub (free model repos) · (b) Google Drive (15 GB free) · (c) Kaggle notebook outputs and Datasets | (a) is versioned and shareable, and adapters load directly from the Hub, but needs a write token stored as a notebook secret. (b) mounts easily in Colab but is harder to use from Kaggle. (c) is easy on Kaggle but awkward to reach from Colab. |
| **D11** | Eval prompting | (a) zero-shot with a format instruction · (b) few-shot examples | (a) is cheaper, and it directly measures how well each checkpoint follows the format. (b) raises Base's scores, which shrinks the measured format gap and lengthens prompts. Either choice must be the same for all checkpoints. |
| **D12** | Significance test | (a) McNemar's test · (b) paired bootstrap confidence interval | (a) is standard for paired right/wrong outcomes and gives a p-value. (b) gives a confidence interval on the accuracy difference, which is easier to explain. Both are free; reporting both is cheap. |
| **D13** | GRPO budget | group size *G* (e.g., 4–8), max completion length (e.g., 256–512 tokens), number of steps, seeds | A larger *G* and longer completions give a better learning signal but use more memory and more time per step. Completions that are too short get cut off before the answer, which zeroes the reward. A cheap pilot run is how to set these. |

## 9. Zero-cost constraints and their tradeoffs

| Normally you'd use | Free alternative here | What you give up |
|---|---|---|
| A100/H100 cloud GPUs | Free T4 (Colab / Kaggle) | No bfloat16, 16 GB memory, smaller GRPO groups and shorter rollouts, fewer steps, interrupted sessions |
| Full fine-tuning | LoRA adapters | Fewer trainable parameters, which may cap how much the model can improve. The upside is that it fits in memory and saves are tiny |
| Learned reward model / human preference data | Programmatic verifier | Only works where answers can be checked automatically (math with numeric answers), and the verifier itself can be gamed (§6) |
| Large eval suites, many seeds | GSM8K test + SVAMP, 1 to a few seeds | Wider confidence intervals, so small effects may not reach significance (§5.4) |
| Paid experiment tracking / storage | TensorBoard or W&B free tier, HF Hub / Drive | More manual saving, and depends on free-tier terms |

## 10. Results

**None yet.** Results will be filled in from `results/metrics.csv` once the runs are finished.

| Checkpoint | GSM8K strict | GSM8K lenient | Format compliance | SVAMP strict | SVAMP lenient |
|---|---|---|---|---|---|
| Base | TBD | TBD | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD | TBD | TBD |
| SFT+GRPO | TBD | TBD | TBD | TBD | TBD |

| Comparison | Δ GSM8K strict | p-value / 95% CI |
|---|---|---|
| SFT+GRPO − SFT | TBD | TBD |
| SFT+GRPO − Base | TBD | TBD |

Verifier audit: [200] rollouts labeled, **TBD** exploits found and patched.
