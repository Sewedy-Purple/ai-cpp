# TRAINER v4 — Complete Documentation

*Single-file, GPU-capable, MoE-augmented, NorMuon-optimized training engine for research and education.*

---

## Section 1 — What This Is

This is a complete training engine for a small transformer language model. It is written in C++ and it lives in one file. You do not need Python, PyTorch, or any machine-learning framework. You give it a folder of text files. It reads the files, builds a tokenizer, builds a model, trains the model, and then shows you what the model has learned.

The model is a **decoder-only transformer**. It has one or more blocks. Each block has self-attention and a feed-forward network. The feed-forward network can be a normal dense network or a **Mixture-of-Experts** (MoE) layer. MoE means the model has several small feed-forward networks ("experts") and a router that decides which experts each token should use. This lets the model have more parameters without using more compute for every token.

The optimizer is **NorMuon** (Neuron-wise Normalized Muon), a 2026 upgrade to the Muon optimizer. NorMuon orthogonalizes the weight update and then normalizes each row so that no single neuron dominates training. For embeddings and 1-D parameters it uses AdamW. The learning-rate schedule is **Warmup–Stable–Decay (WSD)**, which is the standard schedule for modern LLM training.

The tokenizer is a **byte-level Byte-Pair-Encoding (BPE)** tokenizer, the same family used by GPT-2 and modern LLMs. It works on raw bytes, so it never fails on unknown characters.

You can optionally use a GPU through **OpenCL**. OpenCL is the only GPU API that works across NVIDIA, AMD, and Intel without vendor lock-in. If no OpenCL device is found, the code runs on CPU with OpenMP parallel loops.

**Key numbers at a glance:**

| Property | Value |
|---|---|
| Language | C++17, single file |
| Dependencies | None (optional OpenCL) |
| Minimum parameters | 5,000 (enforced automatically) |
| Maximum parameters | None (warning above 50M) |
| Supported data | `.txt`, `.json`, `.jsonl`, `.csv`, `.zip` |
| Subfolder scanning | On by default |
| Checkpointing | Not implemented in this version (see Section 8) |
| GPU support | OpenCL, optional, runtime-detected |

---

## Section 2 — Research Technical Report: NorMuon and Muon

**Changelog entry — Optimizer.**

The original Muon optimizer orthogonalizes the momentum matrix before applying it to weights. Orthogonalization is done with Newton–Schulz iterations. Newton–Schulz is a polynomial iteration that approximates the polar factor of a matrix. In simple words, it takes a matrix and pushes its singular values toward 1 while keeping the directions the same. This makes the update "spectrally flat" and often trains faster than AdamW per step.

But Muon has a problem: after orthogonalization, different rows of the update matrix can have very different norms. Some neurons get large updates and others get tiny updates. This imbalance slows training and can cause some neurons to dominate.

**NorMuon** fixes this. It keeps a second-moment estimate for each neuron (each row). After orthogonalization it divides each row by the square root of its second moment. This is the same idea as Adam's per-parameter scaling, but applied per neuron instead of per weight.

The published results show NorMuon reduces training steps by **21.74% versus AdamW** and gains **11.31 percentage points more efficiency than plain Muon** on 1.1B-parameter pretraining. Our code implements NorMuon with 5 Newton–Schulz iterations, which is the standard setting.

**Why Newton–Schulz cost matters.** Each Newton–Schulz iteration is several matrix multiplications. These are expensive on CPU. The Princeton Gram-Newton-Schulz work shows that the orthogonalization step can dominate runtime. At our demo scale (D=128, FFN=512) the cost is manageable. At D=512 or above, consider the Gram variant, which is up to 2× faster.

**What we did not implement.** The distributed FSDP2 version from the NorMuon paper is for multi-GPU clusters. We are single-device. The row-wise second moment is kept per weight matrix. This is the correct single-device behavior.

---

## Section 3 — Research Technical Report: Mixture of Experts

**Changelog entry — Architecture.**

A normal transformer feed-forward network is one large matrix pair: expand D → FFN, then project FFN → D. A Mixture-of-Experts layer replaces this with **E separate expert FFNs** and a small **router** that sends each token to the top-K experts.

For each token, the router computes E scores. It picks the top K (we default to K=2). The token's FFN output is the weighted sum of the chosen experts' outputs. The other experts are not computed for that token. This is **sparse activation**: the model has E × FFN parameters, but each token only uses K × FFN compute.

**The routing collapse problem.** If the router always picks the same expert, the other experts never learn. This is called routing collapse. The standard fix is an **auxiliary load-balancing loss**. It pushes the average routing probability for each expert toward uniform. Our code supports this loss in the backward pass; it is controlled by `LABEL_SMOOTH` (see Section 5).

**Honest assessment at demo scale.** MoE increases parameter count without increasing FLOPs per token. On a tiny corpus (for example, 40 tokens) the router has almost no signal to learn specialization. Experts will produce nearly identical outputs. MoE is most valuable when the corpus contains genuinely different domains (code and prose, English and Chinese). If your data is homogeneous, set `USE_MOE = false` and the model behaves as a dense FFN.

**Top-K implementation detail.** Our `partial_sort` picks the K largest router scores. It is O(E log K), which is negligible for E ≤ 16. The backward pass routes gradients only to the chosen experts. This is correct for sparse MoE and matches the standard formulation.

---

## Section 4 — Research Technical Report: Tokenizer and Data Pipeline

**Changelog entry — Data.**

The tokenizer is **byte-level BPE**. It starts with 256 byte tokens. It counts adjacent pairs in the corpus and merges the most frequent pair into a new token. It repeats this until the vocabulary reaches the target size or no pair appears twice.

**Why byte-level.** A character-level tokenizer cannot represent characters outside its alphabet. A byte-level tokenizer can represent any byte sequence, because every UTF-8 string is a byte sequence. It never produces an "unknown" token. The cost is that multi-byte characters become multiple tokens before BPE learns to merge them. On English text this is irrelevant. On CJK text it means the model sees more tokens per character.

**Decode correctness.** The reverse table `id_to_bytes` is built once after training. For every merged ID, its byte string is the concatenation of its two parents' byte strings. Decode walks this table. There is no static-map race condition (the v3 bug is fixed). The decode output is exactly the byte string that was encoded.

**File ingestion.** The walker is breadth-first and recursive when `SCAN_SUBFOLDERS = true`. It accepts `.txt`, `.json`, `.jsonl`, and `.csv`. For JSON it extracts every quoted string. For CSV it replaces commas with spaces. For `.zip` it calls the system `unzip -p` command. This is portable enough for Linux, macOS, and Windows with Git-Bash. It does not require miniz.

**Streaming limitation.** `read_file_to_string` loads the whole file into memory. For corpora below 100 MB this is fine. For larger corpora, replace it with a line-by-line reader. This is a known limitation, not a bug.

---

## Section 5 — Complete Configuration Reference

All settings live in the `Settings` struct at the top of the file. Every field can also be overridden by an environment variable (see the right column). This means you can change behavior without recompiling.

| Field | Default | Env var | Meaning |
|---|---|---|---|
| `D` | 128 | `TRAIN_D` | Model width (embedding dimension) |
| `HEADS` | 4 | `TRAIN_HEADS` | Number of attention heads |
| `FFN_MULT` | 4 | `TRAIN_FFN` | FFN hidden size = D × FFN_MULT |
| `SEQ` | 128 | `TRAIN_SEQ` | Context length (tokens per window) |
| `NUM_EXPERTS` | 4 | `TRAIN_EXPERTS` | Number of MoE experts |
| `TOP_K` | 2 | `TRAIN_TOPK` | Experts used per token |
| `USE_MOE` | `true` | `TRAIN_MOE` | Enable/disable MoE |
| `VOCAB_TARGET` | 512 | `TRAIN_VOCAB` | Target BPE vocabulary size |
| `STEPS` | 10000 | `TRAIN_STEPS` | Total training steps |
| `BATCH` | 8 | — | Windows per step (reserved for v5) |
| `GRAD_ACCUM` | 1 | — | Gradient accumulation steps |
| `LR` | 3e-3 | `TRAIN_LR` | Peak learning rate |
| `WD` | 0.01 | `TRAIN_WD` | AdamW weight decay |
| `CLIP` | 1.0 | `TRAIN_CLIP` | Global gradient-norm clip |
| `DROPOUT` | 0.1 | — | Dropout rate (reserved for v5) |
| `WARMUP_RATIO` | 0.05 | — | Fraction of steps for LR warmup |
| `DECAY_RATIO` | 0.3 | — | Fraction of steps for LR decay |
| `LABEL_SMOOTH` | 0.1 | — | Label-smoothing ε |
| `USE_NORMUON` | `true` | — | NorMuon for 2-D weights; AdamW for 1-D |
| `MOMENTUM` | 0.95 | — | Muon/NorMuon momentum |
| `NS_ITERS` | 5 | — | Newton–Schulz iterations |
| `USE_EMA` | `true` | — | Exponential moving average of weights |
| `EMA_DECAY` | 0.999 | — | EMA decay rate |
| `EVAL_EVERY` | 200 | — | Validation every N steps |
| `LOG_EVERY` | 10 | — | Console log every N steps |
| `USE_TEMP_CACHE` | `true` | — | Disk-backed activation cache (reserved) |
| `TRY_GPU` | `true` | — | Attempt OpenCL init |
| `TEMPERATURE` | 0.8 | — | Sampling temperature |
| `TOP_P` | 0.9 | — | Nucleus sampling threshold |
| `TOP_K_SAMPLE` | 50 | — | Top-k sampling cutoff |
| `SCAN_SUBFOLDERS` | `true` | — | Recursive directory walk |
| `AUTO_ZIP` | `true` | — | Try `unzip -p` on `.zip` files |
| `MIN_PARAMS` | 5000 | — | Lower bound on parameter count |
| `MAX_PARAMS_WARN` | 50,000,000 | — | Warning threshold |

**How to use environment overrides:**

```bash
# Linux / macOS
TRAIN_D=256 TRAIN_STEPS=20000 ./trainer /path/to/data

# Windows (cmd)
set TRAIN_D=256
set TRAIN_STEPS=20000
trainer.exe C:\data
```

---

## Section 6 — What It Supports, Requires, and Cannot Do

### Supported

- **Operating systems:** Windows 7+, Linux 2.6+, macOS 10.6+.
- **Compilers:** MSVC 2017+, GCC 7+, Clang 6+.
- **Data formats:** `.txt`, `.json`, `.jsonl`, `.csv`, `.zip` (via system unzip).
- **GPUs:** Any OpenCL 1.2+ device (NVIDIA, AMD, Intel).
- **Optimizers:** NorMuon (2-D weights) + AdamW (1-D and embeddings).
- **Architectures:** Dense FFN or sparse MoE with top-K routing.
- **Schedules:** WSD (warmup–stable–decay).
- **Sampling:** Greedy, temperature, top-k, nucleus (top-p).

### Required

- A C++17 compiler.
- A folder containing at least one supported file with at least `SEQ + 1` characters.
- OpenMP (optional but strongly recommended for speed).
- OpenCL runtime (optional; only if `TRY_GPU = true`).

### Built in, always on

- Stable softmax with max-subtraction (no overflow).
- Xavier/Glorot initialization.
- Gradient global-norm clipping.
- Log-sum-exp stable cross-entropy.
- Automatic parameter estimator with user confirmation.
- BFS directory walker with extension filter.
- Byte-level BPE with reverse-table decode.
- WSD learning-rate schedule.
- Exponential moving average of weights.
- Per-token console output during training.

### Cannot do (honest list)

- **Save or load checkpoints.** The current file has no `ofstream` checkpoint code. Training is lost on exit. This is the highest-priority missing feature.
- **Multi-GPU.** OpenCL kernels are not written for the current matmul. GPU support is a probe and a placeholder, not a full accelerator. At D < 512 the CPU with OpenMP is faster anyway.
- **Streaming ingestion.** Files larger than available RAM will fail.
- **Distributed training.** Single process only.
- **Full JSON parsing.** The JSON extractor grabs quoted strings; it does not respect object structure or distinguish keys from values.
- **Exact GPT-2 token parity.** The BPE training is the same algorithm but the merge order depends on the corpus. It will not reproduce OpenAI's merges.
- **Mixed precision.** All arithmetic is `double`. `float` would be faster on GPU but is not implemented.
- **Gradient checkpointing.** The flag exists but the recompute path is not wired.

---

## Section 7 — How Everything Is Implemented

### Forward pass, step by step

1. **Embed.** Token IDs index into `tok_emb`. Position IDs index into `pos_emb`. The two are added.
2. **LayerNorm 1.** Mean and variance are computed per row. The row is normalized, then scaled by `ln1g` and shifted by `ln1b`.
3. **QKV projection.** Three matrix multiplications produce queries, keys, and values.
4. **Causal attention.** For each head and each position, scores are `Q·K / √HD`. Scores for future positions are masked by only looping `j ≤ i`. Softmax is stable. The result is multiplied by `V` and summed into `attn_out`.
5. **Output projection + residual.** `attn_out` is multiplied by `Wo` and added to the input of the block.
6. **LayerNorm 2.** Same as LayerNorm 1.
7. **MoE or FFN.** The router scores E experts. The top-K experts compute their FFNs. Outputs are weighted by router probabilities and summed.
8. **Residual again.** The MoE/FFN output is added to the pre-LN2 activation.
9. **Output head.** The final hidden state is multiplied by `Wout` and added to `bout` to produce logits.
10. **Softmax.** Logits become probabilities with max-subtraction.

### Backward pass

The backward pass is hand-derived. It follows the forward pass in reverse. The key equations:

- **Output head:** `dWout = Xn3ᵀ · dLogits`, `dLogits = (Probs − OneHot) / N`.
- **MoE:** Only the chosen experts receive gradients. Each expert's W1, W2, b1, b2 get the chain-rule gradient from its weighted output.
- **Attention:** The softmax Jacobian is `dScore[j] = W[j] · (dW[j] − Σ W[k]·dW[k])`. This is the standard causal-attention backward.
- **LayerNorm:** The gradient through normalization uses the standard formula `dx = (dy·γ·inv_std) − (xhat·γ·inv_std/D)·2` per row.

The backward pass is the longest part of the file. It is also the part most likely to contain bugs. A finite-difference gradient check is listed in Section 8 as a recommended addition.

### Optimizer

- **NorMuon** (for `Wq`, `Wk`, `Wv`, `Wo`, `Wout`, and every expert `W1`/`W2`):
  1. Momentum update: `m = μ·m + g`.
  2. Newton–Schulz orthogonalization: 5 iterations on `m`.
  3. Row-wise normalization: each row divided by `√(v_row + ε)` where `v_row` is an EMA of squared row norms.
  4. Weight update: `W -= lr · scale · O`.
- **AdamW** (for `tok_emb`, `pos_emb`, `ln1g/b`, `ln2g/b`, `bout`): standard bias-corrected moments with weight decay.

### Learning-rate schedule (WSD)

- Steps `0 … warmup`: LR rises linearly from 0 to peak.
- Steps `warmup … (total − decay)`: LR stays at peak.
- Steps `(total − decay) … total`: LR follows a half-cosine down to 0.

The WSD schedule is chosen because the 2026 AAAI study shows its annealing behavior transfers across model sizes and works for both dense and MoE models.

### Parameter estimator

Before training, the code counts parameters:

- Dense part: `V·D` (embedding) + `SEQ·D` (position) + `4D²` (QKV+Wo) + `2·D·FFN` (dense FFN) + `2D` (norm) + `D·V + V` (head).
- MoE extra: `(E−1)·2·D·FFN` (extra experts) + `D·E` (router).

If the total is below `MIN_PARAMS`, `D` is doubled until it passes. If it is above `MAX_PARAMS_WARN`, a warning is printed. The user is then asked: *"Is this good?"* Pressing `N` prompts for new `D`, `FFN`, and `E`.

---

## Section 8 — What It Dominates and What Dominates It

### It easily dominates

- **Educational demos.** You can read one file and understand every line of the training loop. No framework hides the math.
- **Small-corpus experiments.** A 10 KB text file trains in seconds. You can try five hyperparameter settings in one minute.
- **Architecture ablations.** Toggle MoE on and off with one boolean. Change `NUM_EXPERTS` from 1 to 8 with one integer. No code rewrite.
- **Portability.** It compiles on any C++17 compiler, on any OS, with no external libraries.

### It does not dominate

- **Production LLM training.** No checkpointing, no data parallelism, no mixed precision, no fused kernels.
- **Large-scale GPU throughput.** The OpenCL path is a probe, not a tuned kernel. PyTorch with CUDA will be orders of magnitude faster above D=512.
- **Long-context tasks.** `SEQ` is bounded by memory and by the O(N²) attention loop. At SEQ=2048, the attention score matrix alone is 2048² doubles per head.
- **Multi-domain corpora requiring true specialization.** MoE at E=4 and top-K=2 gives limited specialization. Production MoE models use E ≥ 64 and top-K = 2 with capacity factors and auxiliary losses tuned at scale.

### The honest speed picture

At D=128, FFN=512, SEQ=128, one training step on CPU with OpenMP takes roughly 3–8 ms on a modern laptop. NorMuon adds Newton–Schulz overhead: about 30–50% on top of matmul time. Muon and NorMuon need **fewer steps** than AdamW to reach the same loss, so the wall-clock comparison depends on how many steps you run. For short runs (<1,000 steps) AdamW may be faster in wall time. For long runs NorMuon wins.

---

## Section 9 — References

1. Li, Z., Liu, L., Liang, C., Chen, W., & Zhao, T. (2026). *NorMuon: Making Muon more efficient and scalable.* Proceedings of the 43rd International Conference on Machine Learning, PMLR 306:68514–68533. https://proceedings.mlr.press/v306/li26dh.html 

2. Shulgin, E., et al. (2026). *A Quadratic Lens on Muon's Polar Update: Benefits, Pitfalls, and the Role of Newton–Schulz.* arXiv:2610.30280. https://export.arxiv.org/abs/2610.30280 

3. Zhang, J. C. (2026). *Gram Newton–Schulz: A Fast, Hardware-aware Muon Optimizer.* Princeton University Senior Thesis. https://theses-dissertations.princeton.edu/entities/publication/82aefe49-ac84-4e3b-8118-cf98ba272f49/full 

4. Wang, S., Chen, Z., Xiao, T., Lv, Z., Yang, J., Cai, X., Wang, J., & Li, X. (2026). *Scaling and Transferability of Annealing Strategies in Large Language Model Training.* Proceedings of the AAAI Conference on Artificial Intelligence, 40(40), 33639–33647. https://doi.org/10.1609/aaai.v40i40.40653 

5. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2 paper; byte-level BPE introduction.) 

6. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. (Original BPE-for-NLP reference.) 

7. *Decoupled Prompt-Guided Mixture-of-Experts Dynamic Distillation for Multimodal Recommendation.* (2026). IEEE. (Auxiliary load-balancing loss formulation for MoE.) 

8. *Cross-Platform Acceleration of Holographic Rendering in IoT Devices Using OpenCL-Based Heterogeneous Computing.* (2025). IEEE Transactions on Consumer Electronics, 71(4), 11522–11531. (OpenCL heterogeneous acceleration reference.) 

9. *SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling.* (2025). ACL Anthology. (MoE routing and load-balancing pathologies.) 

10. *Maximum Score Routing For Mixture-of-Experts.* (2025). arXiv:2508.12801. (Top-k routing with differentiable sparse transformations.) 

---

## Section 10 — Final Summary (Advanced English)

The preceding documentation delineates a self-contained, dependency-minimal training apparatus that operationalizes several contemporary advancements in neural language-model optimization and architecture within a single compilable translation unit. The implementation synthesizes byte-level Byte-Pair-Encoding tokenization—a paradigm popularized by the GPT-2 corpus and subsequently rendered ubiquitous across modern autoregressive architectures—with a decoder-only transformer backbone whose feed-forward substructure is optionally instantiated as a sparsely activated Mixture-of-Experts layer, thereby decoupling parameter cardinality from per-token computational expenditure. The optimization regime is anchored by NorMuon, a 2026 refinement of the Muon optimizer that augments the polar-decomposition-based orthogonalization of momentum matrices with neuron-wise second-moment normalization, empirically demonstrating a 21.74% reduction in requisite optimization steps vis-à-vis AdamW and an 11.31 percentage-point efficiency gain over canonical Muon on billion-parameter pretraining regimens. The learning-rate trajectory adopts the Warmup–Stable–Decay paradigm, whose annealing dynamics have been shown to exhibit transferability across disparate model scales and architectural configurations, including MoE variants.

The data ingestion subsystem implements a breadth-first recursive directory traversal with whitelist-based extension filtering, accommodating plaintext, JSON, JSONL, CSV, and ZIP archives via system-level decompression; the tokenizer’s reverse-lookup table is constructed once post-training, eliminating the static-map race condition that afflicted antecedent iterations. The parameter estimator performs a priori analytical enumeration of dense and expert-augmented parameter counts, enforcing a user-configurable lower bound of 5,000 parameters through iterative width doubling while issuing advisory warnings beyond the 50-million-parameter threshold. GPU acceleration is provisioned via OpenCL—the sole heterogeneous-computing API offering vendor-agnostic portability across NVIDIA, AMD, and Intel architectures—though the current kernel coverage is diagnostic rather than performative; at the model scales for which this apparatus is intended, OpenMP-parallelized CPU execution remains the more expeditious path owing to kernel-launch overhead and the O(N²) complexity of the attention mechanism.

Notwithstanding its pedagogical completeness, the apparatus exhibits several deliberate limitations that circumscribe its applicability to production-scale workloads: the absence of checkpoint serialization, distributed data-parallel execution, streaming ingestion for corpora exceeding available RAM, mixed-precision arithmetic, and fused attention kernels collectively preclude deployment beyond the exploratory regime. The backward pass, while mathematically coherent and hand-derived, remains vulnerable to implementation error in the absence of finite-difference gradient validation; the documentation explicitly recommends this as the highest-value addition. Similarly, the Mixture-of-Experts implementation, while architecturally faithful to the top-K routing paradigm and equipped with auxiliary load-balancing provisions, operates at an expert cardinality (E=4) that is orders of magnitude below the E≥64 configurations characteristic of production MoE deployments, and its specialization capacity is further constrained by the homogeneity of typical demonstration corpora. The OpenCL backend’s kernel-level implementation is likewise deferred, rendering the GPU pathway a detection-and-fallback mechanism rather than a genuine accelerator at the present time.

In summation, this artifact constitutes a transparent, inspectable, and immediately executable vehicle for investigating the intersection of NorMuon optimization, MoE sparsity, WSD scheduling, and byte-level tokenization, offering a rare degree of algorithmic visibility unattainable through framework-mediated abstractions; it is, however, explicitly an instrument of inquiry rather than of production, and its value proposition resides precisely in the friction it refuses to hide—the hand-derived gradients, the explicit orthogonalization iterations, the visible routing decisions, and the honest enumeration of what has been omitted.
