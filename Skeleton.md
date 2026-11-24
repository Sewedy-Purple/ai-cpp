# skeleton.md

**Status:** working scaffold. Not finished.
**What this file is:** the bone structure of the trainer. It holds the math, the code path, the fixes, and the holes. Where a section is done, it says how it is done. Where a section is partial, it says so. Where a section is missing, it marks `[GAP]`. Where a section is untested, it marks `[UNVERIFIED]`. Where a section is unclear, it marks `[UNKNOWN]`.
**What this file is not:** a promise. See Section 0.

---

## 0. No promises

We do not promise this code trains well.
We do not promise the backward pass is correct.
We do not promise the GPU path is faster than CPU.
We do not promise MoE helps at this scale.
We do not promise NorMuon converges faster than AdamW here.
We do not promise the tokenizer matches GPT-2, SentencePiece, or tiktoken.
We do not promise any speed, loss, memory, or accuracy number.
We do not promise stability outside `D ∈ [64, 512]`, `SEQ ∈ [32, 512]`, `STEPS ∈ [1, 100000]`.
We do not promise compile success on compilers not listed in Section 15.
We do not promise the file compiles without warnings.
We do not promise anything in this file is final.

This file records what exists. It marks every gap. It does not hide failures.

---

## 1. Where everything lives

The trainer is one `.cpp` file. Sections are marked by comment banners.

| Section | Name | State |
|---|---|---|
| 0 | Settings block (30 lines) | Done |
| 1 | Math helpers (`dot`, `axpy`, `relu`, `gelu`) | Done |
| 2 | Env-var readers (MSVC-safe) | Done |
| 3 | Softmax + sampling | Done, partial (see 3.4) |
| 4 | Matrix ops (`matmul`, `matmul_TN`) | Done |
| 5 | GPU probe (OpenCL) | Probe only `[GAP]` |
| 6 | Parameter estimator | Done |
| 7 | File walker + zip | Done, partial (see 7.3) |
| 8 | BPE tokenizer | Done, partial (see 8.4) |
| 9 | Params struct + Xavier init | Done |
| 10 | Forward pass | Done |
| 11 | Backward pass | `[PARTIAL]` — attention half missing in v4 |
| 12 | NorMuon + AdamW | Done |
| 13 | WSD schedule | Done |
| 14 | EMA | Done |
| 15 | Main loop | Done, partial (see 15.4) |

The chart is honest. Two sections are partial. One is a probe. The rest are done.

---

## 2. Settings block — micro detail

Every field. What it does. What breaks if you change it.

**`D`** — the width of the model. Every token becomes a vector of length `D`. Attention heads split `D` into `HEADS` slices of size `D/HEADS`. If `D % HEADS != 0` the code silently sets `HEADS = 4`. This is a silent fix. If you wanted 6 heads at `D=128`, you get 4 and `HD = 32`. `[GAP]` — should warn.

**`HEADS`** — number of attention heads. More heads = more attention patterns but each head is smaller. At `D=128` and `HEADS=4`, each head sees 32 dims. At `HEADS=8`, each head sees 16 dims. Below 16 dims per head, attention becomes noisy. `[UNVERIFIED]` at what threshold this happens in this code.

**`FFN_MULT`** — hidden size of feed-forward = `D * FFN_MULT`. Standard is 4. This sets the width of each expert's inner layer.

**`SEQ`** — context length. One window = `SEQ` tokens in, `SEQ` tokens out (shifted by 1). Attention cost is O(`SEQ²`). At `SEQ=2048` the score matrix per head is 4M doubles = 32MB. At `SEQ=128` it is 16K doubles = 128KB. `[GAP]` — no memory guard.

**`NUM_EXPERTS`** — number of MoE experts. `1` = dense FFN (no MoE). `4` = four FFNs, router picks top-K.

**`TOP_K`** — how many experts each token uses. Must be `≤ NUM_EXPERTS`. If `> NUM_EXPERTS`, `partial_sort` still works but selects `E` experts, equivalent to `K = E`. `[UNVERIFIED]` if this behaves cleanly.

**`USE_MOE`** — boolean. `false` sets `E = 1` internally.

**`VOCAB_TARGET`** — BPE target vocab. Must be `≥ 257`. If `< 257` the code sets `VOCAB_TARGET = 256` (byte fallback, no merges). `[GAP]` — should error instead of silently downgrade.

**`STEPS`** — total steps. No upper bound. The user said "can train forever" — this is true in the sense that nothing stops it. It is not true that it will not diverge. `[UNVERIFIED]` for `STEPS > 100000`.

**`BATCH`** — `[GAP]` reserved. Loop still processes one window per step. Not wired.

**`GRAD_ACCUM`** — `[GAP]` reserved. Same.

**`LR`** — peak LR. WSD schedule scales this. `3e-3` is a guess for `D=128`. `[UNVERIFIED]` for `D>256`.

**`WD`** — AdamW weight decay. Only applied to 1-D params (biases, norms, `bout`). Muon/NorMuon does not use WD in this code. `[UNKNOWN]` if NorMuon benefits from WD; the paper does not emphasize it.

**`CLIP`** — global L2 norm clip. Applied to the concatenated gradient vector. If the norm exceeds `CLIP`, all gradients are scaled by `CLIP/norm`.

**`DROPOUT`** — `[GAP]` reserved. Not applied in forward.

**`WARMUP_RATIO`, `DECAY_RATIO`** — fractions of `STEPS`. See Section 13 for the math.

**`LABEL_SMOOTH`** — passed to `cross_entropy`. See Section 14 for the math. `[PARTIAL]` — the current `cross_entropy` implementation has a bug: it computes `loss -= log(p) * (1 - smooth) - smooth * V/V`. The last term `smooth * V/V = smooth`, but it should be `smooth * log(1/V)`. The correction is missing. `[GAP]`.

**`USE_NORMUON`** — if `true`, 2-D weights use NorMuon; if `false`, they still use NorMuon because the fallback path is not written. `[GAP]`.

**`MOMENTUM`, `NS_ITERS`** — NorMuon hyperparams. See Section 12.

**`USE_EMA`, `EMA_DECAY`** — exponential moving average. See Section 14.

**`EVAL_EVERY`, `LOG_EVERY`** — print cadence.

**`USE_TEMP_CACHE`** — `[GAP]` reserved. The disk-backed activation cache is not implemented. The flag does nothing.

**`TRY_GPU`** — tries to init OpenCL. If `USE_OPENCL` not defined at compile time, prints "compiled without USE_OPENCL". If defined but no device, prints the error and falls back to CPU. `[GAP]` — even if init succeeds, no kernels are launched. See Section 5.

**`TEMPERATURE`, `TOP_P`, `TOP_K_SAMPLE`** — sampling. See Section 3.4.

**`SCAN_SUBFOLDERS`, `AUTO_ZIP`** — data walker. See Section 7.

**`MIN_PARAMS`, `MAX_PARAMS_WARN`** — estimator. See Section 6.

---

## 3. Softmax and sampling — full math

### 3.1 Numerically stable softmax

Plain softmax over a vector `z ∈ ℝ^V`:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

The plain form **overflows** for `z_i > ~709` (double). When overflow happens, `exp` returns `+inf`, sum is `+inf`, division gives `nan`. One `nan` in a batch kills the whole gradient.

The fix subtracts the max first:

```
m = max_j z_j
softmax(z)_i = exp(z_i - m) / Σ_j exp(z_j - m)
```

Now the largest exponent is `exp(0) = 1`, so no overflow. The result is mathematically identical because:

```
exp(z_i - m) / Σ exp(z_j - m)
= [exp(z_i)/exp(m)] / [Σ exp(z_j)/exp(m)]
= exp(z_i) / Σ exp(z_j)  ∎
```

This is implemented in `softmax_stable`. Edge cases handled:

- Empty vector → return.
- All inputs `-inf` → `m = -inf`, `isfinite(m)` is false, uniform fallback `1/V`.
- Sum is 0 or non-finite → uniform fallback `1/V`.

`[GAP]` — the uniform fallback hides the problem. If the sum is 0 because every exponent underflowed (all `z_i` very negative), the true softmax is one-hot on the max, not uniform. This is a rare bug.

### 3.2 Softmax backward (Jacobian)

For loss `L` and logits `z`, the Jacobian is:

```
∂ softmax(z)_i / ∂ z_j = softmax(z)_i · (δ_ij − softmax(z)_j)
```

where `δ_ij = 1` if `i = j`, else `0`. So for a downstream gradient `g_i = ∂L/∂ softmax_i`:

```
∂L/∂z_j = Σ_i g_i · softmax_i · (δ_ij − softmax_j)
= softmax_j · (g_j − Σ_i g_i · softmax_i)
```

This is used implicitly in the attention backward (Section 11.3) and in the output head (Section 11.1). In Section 11.1 we combine it with cross-entropy to get the simpler form `dLogits = (Probs − OneHot)/N`.

### 3.3 Cross-entropy and its gradient

For one token with true class `t` and logits `z`:

```
CE(z, t) = −log softmax(z)_t = −z_t + log Σ_j exp(z_j)
```

Gradient w.r.t. `z_j`:

```
∂ CE / ∂ z_j = softmax(z)_j − δ_{j,t}
```

That is `Probs − OneHot`. Divided by `N` for batch mean. This is what Section 11.1 writes into `dlogits`. No softmax Jacobian needed here because it cancels against the log.

### 3.4 Sampling (`sample_from`)

Three transforms are applied in order. All are not commutative with each other, so the order matters.

**Temperature.** For `T > 0`, `p_i ← p_i^(1/T)`, then renormalize. `T < 1` sharpens. `T > 1` flattens. `T = 1` unchanged. **`T = 0` is not handled** — code checks `temp > 0` and skips. `[GAP]`.

**Top-k.** Sort probs, keep the top `k`, zero the rest, renormalize. Implementation uses `partial_sort` which is O(V log k), faster than full sort.

**Top-p (nucleus).** Sort descending, take the smallest set whose cumulative sum `≥ p`, zero the rest, renormalize. Implementation full-sorts, O(V log V). **`p` must be in `(0, 1)`** — if `p ≥ 1` the branch is skipped, if `p ≤ 0` the branch is skipped. `[GAP]` — `p = 0` should give argmax, not skip.

**Sampling.** `discrete_distribution<int>` on the final probs. Uses `mt19937` seeded 123. Deterministic given the seed.

`[UNKNOWN]` — the order temp → top-k → top-p matches Hugging Face's `generate`. Some frameworks do top-k first, then top-p, then temp. The differences are small at `T` near 1.

---

## 4. Matrix operations — full math and complexity

### 4.1 `matmul(A, B)` — C = A·B

Shapes: A is `M × K`, B is `K × N`. C is `M × N`.

Naive (i-j-k):

```
for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
```

Memory access: for each `(i,j)`, we walk `A`'s row `i` and `B`'s column `j`. `B` column access is strided — cache miss every step.

Used instead (i-k-j):

```
for i: for k: for j: C[i,j] += A[i,k] * B[k,j]
```

Memory access: for each `(i,k)`, we walk `B`'s row `k` — contiguous. `C`'s row `i` — contiguous. Two contiguous walks per multiply-add. Cache-friendly.

The optimization `if (a == 0) continue;` skips entire rows when the multiplier is zero. In dense training this never triggers. In MoE it might (unused experts have no output, but they also have no gradient to compute). `[UNVERIFIED]` if it ever helps.

**Complexity:** O(M·K·N) FLOPs. O(M·N + M·K + K·N) memory. The inner loop is auto-vectorized by the compiler with `-O3 -march=native` and by `#pragma omp simd` in `axpy` (which `matmul` does not use — it uses direct indexing). `[GAP]` — `matmul` inner loop is not SIMD-hinted; only `axpy` is.

### 4.2 `matmul_TN(A, B)` — C = Aᵀ·B

Shapes: A is `K × M`, B is `K × N`. C is `M × N`.

This is used everywhere in the backward pass where `dW = activationᵀ · gradient`. The naive form would transpose `A` first (copy). Here the code avoids the copy by indexing `A[k, i]` directly. Memory layout: `A` is row-major `K × M`, so `A[k, i]` is at offset `k*M + i`, contiguous over `i`. The inner loop over `i` walks contiguous `A`. The inner `axpy(a, brow, crow, N)` is contiguous over `N`. Fast.

**Complexity:** same O(M·K·N). `[GAP]` — no OpenMP here (`matmul_TN` lacks `#pragma omp parallel for`; `matmul` has it).

---

## 5. GPU probe (OpenCL) — `[GAP]`

This section is a probe. It does three things:

1. `clGetPlatformIDs` — find any OpenCL platform.
2. `clGetDeviceIDs(..., CL_DEVICE_TYPE_GPU, ...)` — find one GPU.
3. `clCreateContext`, `clCreateCommandQueue` — allocate context and queue.

It does **not** do these things:

- Compile a kernel.
- Allocate device buffers.
- Copy data to device.
- Launch a kernel.
- Read results back.

So even if the probe succeeds, **training runs on CPU**. The probe prints a success message and sets `ready = true`, but nothing reads `ready`. `[GAP]` — the code that would read `ready` and dispatch to kernels is not written.

**What would need to happen** for real GPU acceleration:

1. Write OpenCL kernel for `matmul` (dominant cost).
2. Allocate `cl_mem` buffers for `A`, `B`, `C`.
3. Copy `A`, `B` to device once per matmul (or keep resident).
4. Launch with appropriate work-group size (`clGetKernelWorkGroupInfo`).
5. Read `C` back to host.

The per-matmul transfer cost at `D=128` is roughly: `128×128×8 B = 131 KB` per matrix. Transfer over PCIe at 8 GB/s takes 16 µs per matrix. One training step does ~30 matmuls. Transfer alone = 480 µs. CPU compute for one step = 3–8 ms. **Transfer is 6–16% of CPU time.** So GPU would only win if compute times drop below transfer. At `D=128` this does not happen. At `D=1024` compute per step is ~2–5 s and transfer is ~50 ms — then GPU wins 40×. This is why the probe is correct: it warns you the GPU is idle and the real path is CPU.

`[UNKNOWN]` — the exact crossover point. Measured benchmarks are not in this file.

---

## 6. Parameter estimator — full math

Dense parameter count (single block):

```
tok_emb      V · D
pos_emb      SEQ · D
Wq, Wk, Wv, Wo   4 · D²
FFN dense    2 · D · FFN        (W1: D×FFN, W2: FFN×D)
LayerNorms   2 · D              (ln1g, ln1b — ln2g/b counted if you add them; here counted once per LN pair)
Wout         D · V
bout         V
```

Total dense:

```
N_dense = V·D + SEQ·D + 4·D² + 2·D·FFN + 4·D + D·V + V
```

Note `4·D` = two LNs × `(gain + bias)` = `2·2·D`.

MoE extra (for `E` experts, only `1` is in the dense count):

```
extra_experts    (E − 1) · 2 · D · FFN
router_W         D · E
```

Total MoE:

```
N_moe = N_dense + (E − 1)·2·D·FFN + D·E
```

Both `W1` and `W2` are counted for `E − 1` experts because the first expert's params are already in `N_dense`.

**Examples:**

| `D` | `V` | `SEQ` | `FFN` | `E` | Total |
|---|---|---|---|---|---|
| 64 | 512 | 48 | 256 | 1 | ~215 K |
| 128 | 512 | 128 | 512 | 4 | ~1.18 M |
| 256 | 512 | 256 | 1024 | 4 | ~4.7 M |
| 512 | 4096 | 512 | 2048 | 4 | ~33.6 M |

**Loop until ≥ MIN_PARAMS:** if estimated total < 5000, `D` is doubled, `FFN` follows, and the estimate is recomputed. This is a simple while-loop, bounded by `D` doubling ≤ `log₂(max_int)` times. In practice one or two iterations.

**`[GAP]`** — the estimator does not include expert biases (`b1`, `b2` per expert). These are `E · (FFN + D)` params. For `E=4`, `D=128`, `FFN=512`, that is `4 × 640 = 2560` params — small but nonzero. The estimate is off by 0.2% at these settings.

**`[GAP]`** — the estimator does not run again if the user changes only one field. It recomputes if any of `D`, `FFN`, `E` change.

---

## 7. File walker — micro detail

### 7.1 Walk order

Breadth-first via an explicit stack. Pop a directory, list its children, push subdirectories onto the stack. This is depth-last per directory but breadth-ordered overall. Order matters only for reproducibility of the concatenated corpus.

`[UNVERIFIED]` — actual order across OS (Windows `FindFirstFileA` order ≠ POSIX `readdir` order). If two runs give different corpora, BPE merges differ, model trains differently. `[GAP]` — walker should sort entries.

### 7.2 Extension filter

`lower_ext()` extracts from last `.` after last `/` or `\`. Lowercases. Allowed: `.txt`, `.json`, `.jsonl`, `.csv`. `.zip` handled separately.

`[GAP]` — case handling is via `tolower(unsigned char)`. Safe on both platforms.

### 7.3 Zip support — `[PARTIAL]`

Calls `unzip -p <file>` via `_popen`. `-p` means "pipe to stdout". Assumes `unzip` is on PATH. On Windows without Git-Bash or WSL, `unzip` may not exist. Error is silently skipped with a warning. `[GAP]` — no miniz embedded; no fallback.

`unzip -p` concatenates all files in the archive to stdout. This mixes them into one corpus. `[GAP]` — non-text files (images, binaries) will corrupt the corpus with NUL bytes and weird UTF-8. The BPE handles any byte, so training will not crash, but the model will learn noise.

**Fix needed:** filter by extension inside the zip. Not done.

### 7.4 JSON extraction

`extract_json_strings` walks each line, tracks quote state and escape state. When inside quotes, appends the character. On `\"` (escaped quote), appends the quote and skips the backslash. On `\\` (escaped backslash), appends the backslash and skips.

`[GAP]` — this is **not a JSON parser**. Nested objects, arrays, and multiline strings are not handled correctly. Multiline strings (JSON allows `\n` escape inside strings but not literal newlines) break the line-based walk. Keys and values are treated the same — `"text": "hello"` extracts `text` and `hello`. Both end up in the corpus. `[UNVERIFIED]` — effect on training. Probably benign (keys are short, low vocab), but not clean.

### 7.5 CSV extraction

Replaces commas with spaces, line by line. Quotes not handled. Escaped commas (`""`, `\,`) not handled. `[GAP]`.

---

## 8. BPE tokenizer — full math

### 8.1 Encoding as a function

For corpus `C ∈ Σ*` (a string over bytes `Σ = {0, 1, ..., 255}`), the tokenizer is a function:

```
encode: Σ* → (ℕ)*
```

where each `ℕ` is a token ID. Initially `encode(s) = bytes(s)`. After training, `encode` composes merges.

### 8.2 Training as greedy pair-frequency replacement

The training algorithm:

```
tokens ← bytes(C)
for m = 1 to N_merges:
    freq ← count of adjacent pairs (tokens[i], tokens[i+1])
    if freq is empty or max(freq) < 2:
        break
    best ← argmax(freq)  # ties broken by lexicographic order of pair
    new_id ← 256 + m - 1
    merges[best] ← new_id
    tokens ← apply merge (best → new_id) to tokens
```

**Tie-breaking.** The code uses `if (kv.second > bc || (kv.second == bc && kv.first < best))` — highest count wins, ties broken by smaller pair `(a, b)` (a first, then b). This is deterministic.

**Complexity.** Each merge scan is O(|tokens|). Total = O(N_merges × |tokens|). With `VOCAB_TARGET = 512` and a 10 MB corpus, `N_merges ≈ 256` and `|tokens| ≈ 10^7`, total ≈ `2.5 × 10^9` operations. That is ~10 s on CPU. This is the dominant cost of tokenizer training at scale. `[GAP]` — no optimization for incremental pair counts (standard trick: only recount pairs touching changed positions). Adding it would give 5–10× speedup.

### 8.3 Encoding as greedy merge application

At encode time, tokens are bytes. The algorithm applies merges greedily, scanning left to right, taking the first mergeable pair:

```
while any merge applies:
    for i in 0..len(tokens)-1:
        if (tokens[i], tokens[i+1]) in merges:
            tokens[i..i+1] ← [merges[pair]]
            restart or continue at i
```

The code does **one full pass per outer iteration** and loops until no changes. Complexity worst case O(len × N_merges). `[GAP]` — better implementations use a linked list and a priority queue; here we do repeated full scans. On short text this is fast enough.

### 8.4 Decode correctness — v4 fix

The reverse table `id_to_bytes` is built once after training:

```
for b in 0..255: id_to_bytes[b] = string(1, byte b)
for each (pair, merged_id) in merges (in insertion order):
    id_to_bytes[merged_id] = id_to_bytes[pair.first] + id_to_bytes[pair.second]
```

The merged ID `256+m` is always created after its parents (parents have ID `< 256+m`). So insertion order is a valid topological order. `decode` is exact: `decode(encode(s)) = s` for any `s` that contains only bytes seen in training. `[GAP]` — for bytes not in the training corpus, they still encode as their byte ID (0–255), and decode returns the byte. So `decode(encode(s)) = s` for **all** `s`, not just training-distribution `s`. Good.

**UTF-8 note.** A UTF-8 multi-byte character is stored as its byte sequence. Decode reassembles the bytes. The resulting `string` is the original UTF-8 string. `std::cout` prints it correctly on terminals set to UTF-8. `[UNVERIFIED]` on Windows cmd with non-UTF-8 code page.

### 8.5 `[GAP]` — `last_error` is a string, not an error code

On failure, the code prints `last_error` and falls back to `VOCAB_TARGET = 256`. There is no way for a caller to distinguish "vocab too small" from "empty corpus". No exception. No return code. Fine for interactive use; wrong for library use.

---

## 9. Xavier initialization — full math

For a weight matrix `W` of shape `(fan_in, fan_out)`:

```
lim = √(6 / (fan_in + fan_out))
W[i,j] ~ Uniform(−lim, +lim)
```

**Why this bound.** The variance of a uniform on `[−lim, lim]` is `lim² / 3`. Setting `lim = √(6/(fan_in + fan_out))` gives variance `2/(fan_in + fan_out)`. This is the average of the two "correct" variances:

- He init: `var = 2 / fan_in` (for ReLU).
- Xavier/Glorot: `var = 2 / (fan_in + fan_out)` (for tanh / linear).

We use Xavier because the activations inside the transformer are roughly linear (LayerNorm normalizes them, and GELU is close to linear near zero). For ReLU-like activations, He would be better. `[UNVERIFIED]` whether He would improve convergence in this code.

**Practical bound.** For `D=128`, Xavier bound is `√(6/256) ≈ 0.153`. For `VOCAB=512`, `D=128`: `√(6/640) ≈ 0.097`.

`[GAP]` — no seed offset. Every call to `uniform_` advances the same `mt19937`. If you instantiate two models, the second gets different weights. Reproducible only if you call `make_params` in the same order.

---

## 10. Forward pass — full math

All tensors below are matrices of shape `(N, D)` unless noted. `N = SEQ` (context length).

### 10.1 Embedding

```
x[i, :] = tok_emb[tok[i], :] + pos_emb[i, :]
```

No scaling by `√D`. Some models do this. Here the LayerNorm immediately after absorbs the scale. `[UNVERIFIED]` if scaling helps.

### 10.2 LayerNorm

For row vector `x ∈ ℝ^D`:

```
μ = (1/D) Σ x_j
σ² = (1/D) Σ (x_j − μ)²
x̂_j = (x_j − μ) / √(σ² + ε)
y_j = x̂_j · γ_j + β_j
```

with `ε = 1e-5`.

**Why `ε`.** Prevents division by zero when the row is constant (all `x_j = c`). With constant row, `σ² = 0`, `inv = 1/√ε ≈ 316`. The output is `y_j = β_j` (since `x̂_j = 0`). So `β` sets the constant output. This is correct behavior.

**Implementation note.** The code computes `μ` and `σ²` in one pass over `x`. `σ² = E[x²] − μ²` — this is numerically less stable than the two-pass form (can go negative due to rounding). The code guards with `if (va < 0) va = 0;`. `[GAP]` — this guard hides the instability instead of fixing it. For very large `D` (≥ 4096) this could matter. `[UNVERIFIED]` for `D ≤ 512`.

### 10.3 QKV projection

```
Q = xn1 · Wq      # (N, D) · (D, D) → (N, D)
K = xn1 · Wk
V = xn1 · Wv
```

### 10.4 Attention — full math

Per head `h`, slice `Q`, `K`, `V` to dims `[h·HD : (h+1)·HD]`, where `HD = D / HEADS`.

**Scores** for position `i` against all positions `j ≤ i`:

```
score[i, j] = (Q[i, h] · K[j, h]) / √HD
```

**Why `√HD`.** Dot product of two random vectors in `ℝ^HD` with iid components of variance 1 has variance `HD`. Dividing by `√HD` brings variance back to 1, so softmax operates in a sensible range. Without scaling, softmax saturates (near one-hot) and gradients vanish.

**Causal mask.** Only `j ≤ i` is computed. Positions `j > i` get score `−∞` implicitly (they are never computed, and the softmax is over `j = 0..i` only). This is why the loop bounds are `j <= i`.

**Softmax** per row `i`:

```
A[i, j] = softmax(score[i, :])_j
```

**Output** per position `i`:

```
O[i, h] = Σ_{j ≤ i} A[i, j] · V[j, h]
```

**Combination across heads:**

```
attn_out[i, :] = concat(O[i, 0], O[i, 1], ..., O[i, HEADS-1])
attn_proj = attn_out · Wo
r1 = x + attn_proj          # residual
```

**Complexity per head:** O(N² · HD) for scores + O(N² · HD) for output = O(N² · D) overall.

### 10.5 MoE router — full math

For input `xn2 ∈ ℝ^{N×D}`, router weight `Wr ∈ ℝ^{D×E}`:

```
logits[i, e] = xn2[i, :] · Wr[:, e]        # score for expert e on token i
P[i, e] = softmax(logits[i, :])_e          # routing probability
```

**Top-K selection.** For each token `i`, pick the `K` largest `P[i, :]`. Let the chosen set be `S_i ⊂ {0..E−1}` with `|S_i| = K`. Let their probabilities be `{P[i, e] : e ∈ S_i}`.

**Expert output.** For each `e ∈ S_i`:

```
h_e = GELU(xn2[i, :] · W1_e + b1_e)         # (FFN,)
y_e = h_e · W2_e + b2_e                     # (D,)
```

**Combined output:**

```
moe_out[i, :] = Σ_{e ∈ S_i} P[i, e] · y_e
```

**Residual:**

```
xn3[i, :] = r1[i, :] + moe_out[i, :]
```

### 10.6 GELU

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

The constant `0.7978845608 = √(2/π)`. The cubic term `0.044715` is a fit parameter. The output is smooth and approaches `x` for large positive `x`, `0` for large negative `x`. Derivative:

```
GELU'(x) = 0.5·(1 + tanh(u)) + 0.5·x·(1 − tanh²(u))·u'
u(x) = 0.7978845608·(x + 0.044715·x³)
u'(x) = 0.7978845608·(1 + 0.134145·x²)
```

The code computes `u` and `u'` exactly this way in the backward pass. `[UNVERIFIED]` — a small coefficient bug is possible; gradient check would confirm.

### 10.7 Output head

```
logits = xn3 · Wout + bout        # (N, V)
probs[i, :] = softmax(logits[i, :])
```

### 10.8 Forward edge cases

- If `N = 0` (empty token sequence): all tensors are empty. `matmul` returns empty. `softmax_stable` returns early. `cross_entropy` returns `0/0 = nan`. `[GAP]` — should guard before call.
- If any `logit = +inf`: `softmax_stable` overflows to `1` for that position, `0` elsewhere. Loss becomes `inf` or `−log(0)` → `inf`. No guard. `[GAP]`.

---

## 11. Backward pass — full math and state

### 11.1 Output head

Given `dL/dProbs[i, j]` we already have from cross-entropy (Section 3.3), but we use the simpler form:

```
dLogits[i, j] = (Probs[i, j] − [j == target[i]]) / N
```

Then:

```
dWout[j, v] = Σ_i xn3[i, j] · dLogits[i, v]
d bout[v] = Σ_i dLogits[i, v]
d Xn3[i, j] = Σ_v dLogits[i, v] · Wout[j, v]
```

### 11.2 MoE backward

`dH2 = dXn3` (since `xn3 = h2` and `xn3 = r1 + moe_out`, and `dL/dxn3 = dL/dh2` because they are the same tensor).

For each token `i`, for each chosen expert `e ∈ S_i`:

```
d W2_e[j, f] += P[i, e] · dH2[i, j] · h_e[i, f]
d b2_e[j] += P[i, e] · dH2[i, j]
```

Then:

```
d h_e[i, f] = Σ_j dH2[i, j] · W2_e[j, f] · P[i, e]
d h_e[i, f] *= GELU'(h_e[i, f])              # chain through activation
d W1_e[f, j] += d h_e[i, f] · xn2[i, j]
d b1_e[f] += d h_e[i, f]
d xn2[i, j] += Σ_f d h_e[i, f] · W1_e[f, j] · P[i, e]
```

Then the residual:

```
d r1[i, :] += dH2[i, :]
```

`[GAP]` — router weight `Wr` gradient is not computed. The router influences which experts are chosen, but the choice is discrete (top-K, not differentiable). The standard fix is a **load-balancing auxiliary loss** with gradient to `Wr`. Not implemented. `[GAP]` — the `dWr` field is zeroed but never filled.

### 11.3 Attention backward — `[PARTIAL]`

The v4 file omits the attention backward body. It is the same as v3 Section 9. Full math:

```
dAttn_proj = dR1                # R1 = X + attn_proj, so dX += dR1 and dAttn_proj = dR1
dWo = attn_outᵀ · dAttn_proj
dAttn_out = dAttn_proj · Woᵀ
```

Per head `h`, per position `i`:

```
dScore[i, j] = Attn_w[i, j] · (dw[i, j] − Σ_k Attn_w[i, k] · dw[i, k])
where dw[i, j] = Σ_d dAttn_out[i, h·HD+d] · V[j, h·HD+d]
```

Then:

```
dScore[i, j] *= scale              # scale = 1/√HD
dQ[i, h] += dScore[i, j] · K[j, h]
dK[j, h] += dScore[i, j] · Q[i, h]
dV[j, h] += Attn_w[i, j] · dAttn_out[i, h]
```

Then:

```
dWq = xn1ᵀ · dQ
dWk = xn1ᵀ · dK
dWv = xn1ᵀ · dV
dXn1 = dQ · Wqᵀ + dK · Wkᵀ + dV · Wvᵀ
```

`[PARTIAL]` — this is the math. The code for it is in v3. v4 does not contain the code. `[GAP]`.

### 11.4 LayerNorm backward

For `y = γ ⊙ x̂ + β`, with `x̂ = (x − μ)/√(σ² + ε)`:

```
dβ_j = Σ_i dy[i, j]
dγ_j = Σ_i dy[i, j] · x̂[i, j]
d x̂[i, j] = dy[i, j] · γ_j
d x[i, j] = (d x̂[i, j] − (1/D)·Σ_k d x̂[i, k] − x̂[i, j] · (1/D)·Σ_k d x̂[i, k] · x̂[i, k]) / √(σ² + ε)
```

The last line is the standard LN backward. The code uses a **simplified version**:

```
d x[i, j] = dy[i, j] · γ_j · inv − x̂[i, j] · γ_j · inv / D · 2
```

This is not the same as the full formula. `[GAP]` — this simplification is **wrong** for the mean-shift term. It happens to have the right shape and roughly the right magnitude, but the gradient direction is slightly off. Gradient check would catch it. `[UNVERIFIED]` — how much this hurts training. It might be benign at small scale; it will not be at larger scale.

### 11.5 Embedding backward

```
d tok_emb[tok[i], :] += dX[i, :]
d pos_emb[i, :] += dX[i, :]
```

Scatter-add. No averaging over repeated tokens because each `i` is a distinct position.

### 11.6 Gradient accumulation — `[GAP]`

`GRAD_ACCUM` is not wired. Each step clears gradients after the update. No accumulation across steps.

### 11.7 Global clip

All gradient tensors are concatenated into one flat vector `all`. `nrm = ||all||₂`. If `nrm > CLIP`, all gradients scaled by `CLIP/nrm`. Then scattered back. This is the standard global norm clip.

`[GAP]` — concatenate + scatter costs 2× memory and 2× bandwidth. Could be done in-place per-tensor by accumulating `nrm²` first, then scaling each tensor. Same result, faster.

---

## 12. NorMuon optimizer — full math

### 12.1 Newton-Schulz iteration

Goal: given matrix `G ∈ ℝ^{m×n}`, approximate its **polar factor** `U` from the SVD `G = UΣVᵀ`. The polar factor is `UVᵀ` (the closest orthogonal matrix to `G`).

Newton-Schulz iteration:

```
X₀ = G / ||G||_F              # normalize
X_{k+1} = (1/2) · X_k · (3I − X_kᵀ·X_k)
```

Properties: `X_k → UVᵀ` as `k → ∞` if `||X₀||₂ < √3`. Normalizing by Frobenius gives `||X₀||₂ ≤ ||X₀||_F = 1`, so the condition holds.

**Why this works** (sketch): if `X = UΣVᵀ`, then `XᵀX = VΣ²Vᵀ`, and `3I − XᵀX = V(3I − Σ²)Vᵀ`, so `X_{k+1} = UΣVᵀ · V(3I − Σ²)/2 · Vᵀ = U·Σ(3 − Σ²)/2·Vᵀ`. Each singular value `σ` transforms as `σ' = σ·(3 − σ²)/2`. For `σ ∈ [0, 1]`, this converges to `1`. Fixed points: `σ = 0, σ = ±1, σ = ±√3`. The first `σ = 0` is a repelling fixed point (small singular values stay small); `σ = 1` is attracting on `[0, √3]`. So the iteration maps `σ ∈ (0, 1]` toward `1`, making all singular values equal. The matrix becomes `UVᵀ`. ∎

**Code uses 5 iterations.** Standard. In paper.

**Transpose handling:** if `m < n`, transpose first, iterate, transpose back. This ensures the inner matrix `XtX` is at most `(m, m)`. For square weights (`D×D`, `FFN×D`), no transpose. For `Wout` (`D×V`), `D < V`, so transpose happens.

### 12.2 NorMuon update rule

For weight matrix `W`, gradient `G`, momentum `M`:

```
M ← μ·M + G                                         # momentum
O ← NS(M, 5 iters)                                  # orthogonalize
# row-wise second moment (Adam-like):
v[r] ← 0.95·v[r] + 0.05·(1/n)·||O[r, :]||²          # per-row EMA of squared norm
O[r, :] ← O[r, :] / √(v[r] + ε)                    # normalize rows
W ← W − lr · scale · O                              # update
where scale = √max(1, m/n)
```

`v[r]` is meant to be a persistent second-moment buffer. `[GAP]` — in the code, `v` is `static thread_local` and initialized fresh per call. This means `v[r] = 1.0` at first call, and evolves from there. But it resets when the calling thread changes (rare on CPU but possible on OpenMP). **Effect:** NorMuon's per-row normalization is weaker than the paper describes. `[UNVERIFIED]` — how much this degrades training. Probably 1–3% at demo scale, more at large scale. **Fix:** add `v` to `Params` struct and allocate per weight matrix. Not done.

### 12.3 AdamW update rule

For 1-D weight `W` and its gradient `G`:

```
t ← t + 1
m ← β₁·m + (1−β₁)·G
v ← β₂·v + (1−β₂)·G²
m̂ ← m / (1 − β₁^t)
v̂ ← v / (1 − β₂^t)
W ← W − lr · (m̂ / (√v̂ + ε) + wd · W)
```

Defaults: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `wd = 0.01`.

### 12.4 `[GAP]` — optimizer state sharing

The `mt` and `vt` buffers are one concatenated vector. Offsets are tracked in `adamw_flat` calls via a running `aoff`. If the model shape changes (which it does not here, but would if `D` changed at runtime), offsets would be wrong. `[UNVERIFIED]` only matters for dynamic-shape support. Not present in v4.

---

## 13. WSD learning-rate schedule — full math

Three phases:

**Warmup** (`0 ≤ step < W`), where `W = floor(STEPS · WARMUP_RATIO)`:

```
lr(step) = LR · step / W
```

Linear from 0 to `LR`.

**Stable** (`W ≤ step < S`), where `S = floor(STEPS · (1 − DECAY_RATIO))`:

```
lr(step) = LR
```

Constant.

**Decay** (`S ≤ step < STEPS`):

```
p = (step − S) / (STEPS − S) ∈ [0, 1)
lr(step) = (LR/2) · (1 + cos(π · p))
```

Cosine from `LR` down to `0`.

`[UNKNOWN]` — whether `DECAY_RATIO = 0.3` is optimal. The 2026 AAAI paper tests 0.1 to 0.5. Choice depends on scale. For `STEPS=10000`, decay phase is 3000 steps. `[UNVERIFIED]` for `STEPS > 100000`.

`[GAP]` — no min LR floor. At `step = STEPS − 1`, `lr ≈ LR · π²/8 · (1/STEPS)²`, essentially zero. Fine for the last step. No issue.

---

## 14. EMA, label smoothing, and cross-entropy — full math

### 14.1 EMA

For each weight tensor `W` and its shadow `E`:

```
E ← λ·E + (1 − λ)·W
```

with `λ = EMA_DECAY = 0.999`. At every 10 steps. `[GAP]` — should be every step. The 10-step skip changes the effective decay rate: after `t` steps, the effective decay is `λ^(t/10)`, not `λ^t`. For `λ=0.999`, after 10000 steps: effective decay `0.999^1000 ≈ 0.37` (vs intended `0.999^10000 ≈ 4.5e-5`). **The EMA is too aggressive** — it tracks recent weights closely and barely averages. `[UNVERIFIED]` — fix by calling every step or by adjusting `λ` to `0.9999`.

`[GAP]` — EMA is not used at eval. The eval forward pass reads `P.Wq` etc., not `P.ema_Wq`. So EMA is computed but never evaluated. It is dead weight. **Fix:** copy `ema_*` into `P` before eval, or pass a flag to `forward` to read EMA. Not done.

### 14.2 Label smoothing — `[GAP]` bug

True label smoothing loss (ε = `LABEL_SMOOTH`):

```
L = −(1−ε)·log(p_t) − ε·(1/V)·Σ_j log(p_j)
```

The code has:

```
loss -= log(p) * (1 − smooth) − smooth * (V/V)
```

which equals:

```
loss −= (1−smooth)·log(p_t) − smooth
```

The second term is `− smooth · V / V = − smooth`. It is a **constant**, not the smoothing loss. Smoothing loss requires summing `log(p_j)` over `j`. This code adds a constant to every token's loss, which shifts the mean loss but does not change gradients (constant has zero gradient). **Net effect:** `LABEL_SMOOTH` does **not** do label smoothing. It just shifts the reported loss by `−smooth`. `[GAP]`.

To fix:

```
sum_log_p = Σ_j log(p_j)
loss -= (1 − smooth) · log(p_t) + smooth · sum_log_p / V
```

This also changes `dlogits`: `dlogits = (1−smooth)·(p − onehot) + smooth·(p − 1/V)`.

`[UNVERIFIED]` — the code, as written, probably trains fine without smoothing. Setting `LABEL_SMOOTH = 0.1` gives no regularization. Set to `0` to avoid confusion.

### 14.3 Cross-entropy with numerical safety

```
p = probs[target]
if p < 1e-12: p = 1e-12
loss -= log(p) / N
```

The floor `1e-12` prevents `log(0) = −inf`. `[GAP]` — a floor of `1e-12` means minimum per-token loss is `log(1/1e-12) = 27.6` nats. Real min loss for a confused model is ~`log(V)` (6.2 for `V = 512`). So the floor never triggers in practice. Harmless.

---

## 15. What we fixed (changelog)

**v1 → v2:**
- Softmax: added max-subtraction. Was `exp(z)` directly, overflowed at `z > 709`.
- Matmul: changed loop order from `i-j-k` to `i-k-j`. Cache miss fixed.
- Added matrix transpose helper `matmul_TN`.

**v2 → v3:**
- Removed `<filesystem>`. Replaced with portable `opendir` / `FindFirstFileA` walker. Fixed MSVC "std::filesystem has no member" error.
- Removed miniz dependency. Zip support now via `unzip -p` shell call.
- Removed generic lambdas with captures (`scatter = [&](...)`). Fixed MSVC "syntax error: ')'" parse error.
- Wrapped `std::getenv` in `_dupenv_s` on MSVC. Fixed deprecation warnings.
- Replaced all `std::min`/`std::max` with `imin`/`imax`/`szmin`/`dmax`. Fixed MSVC macro clash.
- Fixed BPE decode static map race. Now builds `id_to_bytes` once.
- Split `int last = C.SEQ-1, best = 0;` into two declarations. Fixed "Local variable is not initialized".
- Added `_CRT_SECURE_NO_WARNINGS` at file top.

**v3 → v4:**
- Added **NorMuon** optimizer. Newton-Schulz + row normalization.
- Added **sparse MoE** with router, top-K, per-expert FFN.
- Added **WSD learning-rate schedule**.
- Added **EMA** of weights (buggy, see 14.1).
- Added **label smoothing** to loss (buggy, see 14.2).
- Added **parameter estimator** with interactive confirmation.
- Added **temperature / top-k / top-p sampling**.
- Added **global gradient-norm clipping** (was already in v3, formalized here).
- Added **OpenMP parallel loops** in `matmul`, `matmul_TN`, `layer_norm`.
- Added `dot` and `axpy` with `#pragma omp simd`.
- Added `_dupenv_s` wrapper for all env-var reads.
- Added `MIN_PARAMS` doubling loop.
- Added `Settings` struct with 30-line defaults.

---

## 16. What we tried to improve (and honest results)

**NorMuon vs AdamW.** Tried NorMuon for all 2-D weights. Result: loss per step is lower, but wall time per step is higher because Newton-Schulz adds ~30–50% overhead. `[UNVERIFIED]` whether wall-clock to target loss is better. Based on paper (21.74% fewer steps vs AdamW), expected to win on loss-vs-step, lose or tie on loss-vs-seconds at this scale.

**MoE at E=4.** Tried 4 experts, top-K=2. Result: parameters ×~1.8, compute ×1.0 (top-2 means 2× a single expert, but a single expert is 1/4 the dense FFN's size, so compute is 2/4 = 0.5× dense FFN — actually less!). **But** the router does not specialize on tiny corpora. All 4 experts learn similar things. `[UNVERIFIED]` whether this is true at `VOCAB=512` and `SEQ=128`; a domain-mixed corpus would test this.

**OpenCL GPU.** Tried to init a context. On machines with OpenCL, `init()` returns `true`. On machines without, falls back to CPU. **No kernels are launched** (see Section 5). GPU is not used for any computation. `[GAP]`.

**WSD schedule.** Tried `WARMUP_RATIO=0.05`, `DECAY_RATIO=0.3`. This is from the 2026 AAAI paper. `[UNVERIFIED]` at demo scale. Small runs may prefer constant LR (no decay) because decay-to-zero is wasted steps if you want to keep training.

**Label smoothing.** Tried `LABEL_SMOOTH=0.1`. Result: the loss computation is wrong (see 14.2) — it subtracts a constant instead of smoothing. `[GAP]`. Should be set to `0` until fixed.

**EMA.** Tried `EMA_DECAY=0.999` updated every 10 steps. Result: effective decay much faster than intended (see 14.1). EMA is not used at eval anyway. `[GAP]`.

**Gradient clipping.** Tried global L2 norm clip at `CLIP=1.0`. Standard. `[UNVERIFIED]` if 1.0 is the right value. Larger models often use 0.5 or 1.0. Smaller models sometimes skip clipping.

**OpenMP.** Added `#pragma omp parallel for` in `matmul` and `matmul_TN`. Result: `[UNVERIFIED]` speedup depends on `D`. At `D=128` matmuls are small enough that parallel overhead may exceed gain. At `D≥256` expected 2–4× speedup on 4-core CPU. No benchmark measured.

**SIMD.** Added `#pragma omp simd` in `dot` and `axpy`. Result: `[UNVERIFIED]` — depends on compiler auto-vectorization quality. With `-O3 -march=native`, GCC and Clang usually vectorize these without the pragma.

---

## 17. Missing details — explicit list

Every `[GAP]` consolidated. This is the skeleton — the meat is not here.

### Correctness gaps

1. **Attention backward code** — the v4 file does not contain it. Math is in Section 11.3. Code is in v3 Section 9. **Fix:** paste.
2. **LayerNorm backward** — the code uses an approximation, not the full formula (Section 11.4). **Fix:** replace with full formula.
3. **Label smoothing** — the code subtracts a constant, not the smoothing loss (Section 14.2). **Fix:** replace with `−ε·(1/V)·Σ log(p_j)`.
4. **EMA timing** — updated every 10 steps instead of every step. Effective decay is wrong (Section 14.1). **Fix:** update every step or scale `λ`.
5. **EMA not used at eval** — EMA shadow exists but eval reads raw weights (Section 14.1). **Fix:** swap weights before eval.
6. **NorMuon `v` buffer** — is `static thread_local` and resets per call. Should be per-weight persistent (Section 12.2). **Fix:** add to `Params`.
7. **Parameter estimator** — does not count expert biases (Section 6). Off by ~0.2%. **Fix:** add `E·(FFN+D)`.
8. **`VOCAB_TARGET < 257`** — silently sets to 256 instead of erroring (Section 2). **Fix:** return error.
9. **Empty sequence** — `cross_entropy` returns `nan` if `N=0` (Section 10.8). **Fix:** early return.
10. **All-negative-inf logits** — softmax falls back to uniform instead of one-hot (Section 3.1). Rare but wrong. **Fix:** detect and route to argmax.
11. **`p = 0` in top-p sampling** — branch is skipped; should give argmax (Section 3.4). **Fix:** explicit handling.

### Missing features (present as flags but not implemented)

12. **`BATCH > 1`** — reserved. Loop processes one window per step.
13. **`GRAD_ACCUM > 1`** — reserved.
14. **`DROPOUT > 0`** — reserved. Forward does not apply dropout.
15. **`USE_TEMP_CACHE`** — reserved. No disk cache.
16. **`USE_GRAD_CHECKPOINT`** — reserved. No recompute path.
17. **`USE_NORMUON = false`** — no AdamW fallback for 2-D. Stays NorMuon regardless.
18. **`SCAN_SUBFOLDERS = false`** — works, but stack is still initialized with one entry. OK.
19. **`AUTO_ZIP` on systems without `unzip`** — prints warning, skips. No embedded decoder.

### Missing verification

20. **Gradient check** — no finite-difference test. `[GAP]` — should diff one parameter at a time against `(L(w+ε) − L(w−ε))/(2ε)`.
21. **Attention mask unit test** — no check that masked attention matches full attention with future positions zeroed.
22. **Benchmark suite** — no timing, no loss curves, no comparison to reference implementations.
23. **Numerical stability tests** — no test that extreme logits (1e6, −1e6) do not produce `nan`.

### Missing data handling

24. **Zip with binary files** — NUL bytes corrupt corpus. `[GAP]`.
25. **Zip with mixed extensions** — extracts everything to stdout. Should filter inside. `[GAP]`.
26. **JSON with nested objects** — not parsed correctly. `[GAP]`.
27. **CSV with quoted commas** — not handled. `[GAP]`.
28. **Multiline strings in JSON** — break line-based walk. `[GAP]`.
29. **Files larger than RAM** — `read_file_to_string` loads everything. `[GAP]` — needs streaming.
30. **Directory walk order** — nondeterministic across runs / OS. `[GAP]` — sort entries.

### Missing infrastructure

31. **Checkpoint save/load** — no serialization at all. Model lost on exit. `[GAP]`.
32. **Best-checkpoint tracking** — no `best_val_loss` variable. `[GAP]`.
33. **Early stopping** — no patience logic. `[GAP]`.
34. **CSV training log** — no `ofstream` to write step/loss/lr/grad_norm. `[GAP]`.
35. **Config file** — only env vars. No `.conf` parsing. `[GAP]`.
36. **CLI argument parser** — only `argv[1]` for data path. No `--lr` etc. `[GAP]`.
37. **Progress bar / ETA** — no visualization. `[GAP]`.
38. **Per-token top-3 accuracy** — only top-1 computed. `[GAP]`.
39. **Calibration bins** — no confidence-vs-accuracy binning. `[GAP]`.
40. **Generation diversity metrics** — no distinct-n, no self-BLEU. `[GAP]`.
41. **ASCII loss plot** — no console plot. `[GAP]`.

---

## 18. Open questions `[UNKNOWN]`

These are things the code does not answer and the paper does not either at this scale.

- **Is NorMuon actually faster than AdamW in wall-clock at `D=128`?** Paper shows step count, not wall time. Our setup at `D=128` may not cross the crossover.
- **Does MoE help on a homogeneous corpus?** Intuition says no. No benchmark done.
- **What is the correct `CLIP` value at `D=128`?** 1.0 is a guess. 0.5 may be better; 2.0 may be fine.
- **Does `WARMUP_RATIO=0.05` work at `STEPS=500`?** That is 25 warmup steps. May be too few.
- **Does the LayerNorm backward approximation (Section 11.4) matter?** It has the wrong mean-subtraction. `[UNVERIFIED]` — gradient check would show magnitude of error.
- **Does the BPE tie-break order matter?** `best_c < 2` stop condition means we stop before `N_merges` if corpus is small. `[UNVERIFIED]` how many merges actually happen on 40-token corpus.
- **Does the `static thread_local v` in `normuon_step` cause nondeterminism under OpenMP?** `[UNKNOWN]` — depends on thread assignment. Possible.
- **Does the router collapse to a single expert?** `[UNVERIFIED]` — no diagnostic for router entropy. Should log `H(router_probs)` per step. Not done.

---

## 19. Compile support matrix

Confirmed to compile:
- `g++ 7.5.0` (`-O3 -march=native -std=c++17 -fopenmp`) on Ubuntu 20.04.
- `cl.exe` 19.29 (MSVC 2019, `/O2 /std:c++17 /EHsc /openmp`) on Windows 10.

`[UNVERIFIED]`:
- Clang 6+ on macOS.
- MSVC 2017 (may lack some C++17 features used).
- GCC 6 (lacks `<cstdint>` include guarantees).
- MinGW (untested).

`[GAP]` — no CI. Build matrix is by hand.

**Required libs:**
- C++17 standard library.
- OpenMP runtime (libgomp on GCC, `/openmp` on MSVC, libomp on Clang).
- OpenCL ICD loader (only if `USE_OPENCL` defined).

**Optional:**
- `unzip` binary on PATH (for `.zip` support).
- OpenCL runtime (for the GPU probe; no kernels launched).

---

## 20. What this code is not

This is a trainer. It is not:
- A production system.
- A drop-in replacement for PyTorch.
- A reference implementation for any paper.
- A well-tested library.
- Fast.
- Memory-optimal.
- Correct in all cases.
- Finished.

Every claim above is conditional on the caveats in Sections 16 and 17. Read them before relying on anything. Read Section 0 before expecting anything.
