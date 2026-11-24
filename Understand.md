# Understand.md

**What this file is:** A patient teacher. It walks through every line of code in the trainer, one section at a time. It assumes you know what a variable is and what a `for` loop does. It does not assume you know what a transformer is, what a gradient is, or what MoE means. If you have ever written a "hello world," you can follow this.

**How to read this file:** Read it top to bottom. Do not skip. Each section builds on the last. When you see a block of code, read it, then read the explanation below it. If something does not click, read it again. Some ideas (attention, backpropagation) take three or four reads to feel natural. That is normal.

**How the code blocks work:** Every time you see a block of C++, it is a copy of a real line from the trainer. The explanation below it tells you:

1. What the line does, in plain words.
2. Why it is there.
3. What would break without it.

At the end of each section, there is a **"Pattern to remember"** — one idea worth carrying forward.

---

# Part 1 — The Big Picture

## What is this program trying to do?

Imagine you want to teach a very patient parrot to finish sentences. You show the parrot a thousand sentences. The parrot does not understand meaning. But it notices patterns. After enough examples, when you say "the cat sat on the," the parrot might squawk "mat."

This program teaches a **mathematical parrot** to do that. The parrot is called a **language model**. The parrot is a pile of numbers, called **weights**. The program shows the parrot sentences from your files. The parrot guesses the next character. The program checks the guess. If wrong, the program nudges the parrot's weights so the parrot guesses better next time.

After many rounds of showing sentences and nudging weights, the parrot gets good at continuing text. It does not "understand." It just guesses well.

## The four big steps

Every training program, no matter how fancy, does the same four things:

1. **Read data** — get sentences from files. (Sections 7 and 8.)
2. **Make a guess** — pass the sentence through the model to get predictions. (Section 10, called the *forward pass*.)
3. **Measure the mistake** — compare guess to the true next character. The size of the mistake is called **loss**. (Sections 3 and 14.)
4. **Fix the weights** — nudge every weight in the direction that makes the guess less wrong next time. (Sections 11 and 12, called the *backward pass* and the *optimizer*.)

Everything else — GPU code, MoE, NorMuon, EMA, label smoothing — is an optimization or a refinement of those four steps. The core never changes.

## The pieces you will meet

| Name | What it is | Where it lives |
|---|---|---|
| **Weights** | The parrot's numbers. Hundreds of thousands of `double`s. | `Params` struct, Section 9 |
| **Token** | A chunk of text. Usually a character or a piece of a word. | `BPETokenizer`, Section 8 |
| **Embedding** | A vector of numbers that represents a token. | `tok_emb`, Section 9 |
| **Attention** | A way for the model to look at earlier tokens before guessing the next one. | Section 10.4 |
| **FFN** | A big matrix pair that transforms each token's vector. | Section 10.5 |
| **MoE** | Several small FFNs, and a router that picks which ones to use. | Section 10.5 |
| **Loss** | A number that says how wrong the guess was. Lower is better. | Section 14 |
| **Gradient** | A vector that says which way to move each weight to lower the loss. | Section 11 |
| **Optimizer** | A recipe for using gradients to update weights. | Section 12 |

---

# Part 2 — Section 0: The Settings Block

## What it looks like

```cpp
struct Settings {
    int D = 128, HEADS = 4, FFN_MULT = 4, SEQ = 128;
    int NUM_EXPERTS = 4, TOP_K = 2;
    bool USE_MOE = true;
    int VOCAB_TARGET = 512;
    int STEPS = 10000, BATCH = 8, GRAD_ACCUM = 1;
    double LR = 3e-3, WD = 0.01, CLIP = 1.0, DROPOUT = 0.1;
    double WARMUP_RATIO = 0.05, DECAY_RATIO = 0.3;
    double LABEL_SMOOTH = 0.1;
    bool USE_NORMUON = true;
    double MOMENTUM = 0.95, NS_ITERS = 5;
    bool USE_EMA = true; double EMA_DECAY = 0.999;
    int EVAL_EVERY = 200, LOG_EVERY = 10;
    bool USE_GRAD_CHECKPOINT = false;
    bool USE_TEMP_CACHE = true;
    bool TRY_GPU = true;
    double TEMPERATURE = 0.8, TOP_P = 0.9; int TOP_K_SAMPLE = 50;
    bool SCAN_SUBFOLDERS = true, AUTO_ZIP = true;
    int MIN_PARAMS = 5000, MAX_PARAMS_WARN = 50'000'000;
};
```

## What is `struct`?

A `struct` is a way to group related variables under one name. Think of a `struct` like a paper form with fields. `Settings` is the form that says how the whole program should behave. Every field is a knob you can turn.

## Line by line

**`int D = 128, HEADS = 4, ...`**

`int` means a whole number (no decimals). `D` is the width of the model. Every token turns into a list of `D` numbers. `128` is the default width. Bigger `D` means a smarter but slower model.

`HEADS = 4` means the attention mechanism (explained later) is split into 4 parallel "heads." Each head sees `D / 4 = 32` numbers per token. More heads = more patterns noticed, but each head is smaller.

`FFN_MULT = 4` means the feed-forward network's hidden size is `D × 4 = 512`. `SEQ = 128` means the model looks at 128 tokens at a time. This is the "context window."

**`int NUM_EXPERTS = 4, TOP_K = 2;`**

MoE uses several small networks ("experts"). `NUM_EXPERTS = 4` means 4 experts. `TOP_K = 2` means each token only uses 2 of them. This is the point of MoE: more total parameters, but only a fraction used per token.

**`bool USE_MOE = true;`**

`bool` is `true` or `false`. If `false`, MoE is turned off and you get a plain dense model.

**`int VOCAB_TARGET = 512;`**

The tokenizer will have at most 512 possible tokens. Bigger vocabulary = fewer tokens per sentence, but each token is harder to learn.

**`int STEPS = 10000, BATCH = 8, GRAD_ACCUM = 1;`**

`STEPS` is how many training rounds to run. `BATCH` is how many examples per round (currently reserved — not wired up). `GRAD_ACCUM` is for accumulating gradients over several mini-batches (currently reserved too).

**`double LR = 3e-3, WD = 0.01, CLIP = 1.0, DROPOUT = 0.1;`**

`double` is a decimal number. `LR = 3e-3` means learning rate `0.003`. This is how big each weight update is. Too big = training explodes. Too small = training crawls. `WD` is weight decay (pushes weights gently toward zero, prevents overfitting). `CLIP` limits how big any gradient can be in a single step. `DROPOUT` is reserved (not wired).

**`double WARMUP_RATIO = 0.05, DECAY_RATIO = 0.3;`**

The learning rate rises for the first 5% of training, stays flat, then drops for the last 30%. Explained in Section 13.

**`double LABEL_SMOOTH = 0.1;`**

Instead of forcing the model to be 100% sure about the true next token, we ask it to be 90% sure and spread 10% among all tokens. This prevents overconfidence.

**`bool USE_NORMUON = true;`**

Turn on NorMuon, the special optimizer.

**`double MOMENTUM = 0.95, NS_ITERS = 5;`**

Muon remembers previous gradients (momentum). `MOMENTUM = 0.95` means each step keeps 95% of the old direction and adds 5% of the new one. `NS_ITERS = 5` means the Newton-Schulz matrix trick runs 5 times (more iterations = more accurate but slower).

**`bool USE_EMA = true; double EMA_DECAY = 0.999;`**

EMA = Exponential Moving Average. We keep a slow-updating copy of the weights. 0.999 means each update only takes 0.1% of the new value.

**`int EVAL_EVERY = 200, LOG_EVERY = 10;`**

Every 200 steps, run validation. Every 10 steps, print progress.

**`bool USE_GRAD_CHECKPOINT = false; bool USE_TEMP_CACHE = true;`**

Both are reserved — the flags exist but the features behind them are not implemented.

**`bool TRY_GPU = true;`**

Try to use a GPU. If none found, fall back to CPU.

**`double TEMPERATURE = 0.8, TOP_P = 0.9; int TOP_K_SAMPLE = 50;`**

Sampling settings for generation. Explained in Section 3.

**`bool SCAN_SUBFOLDERS = true, AUTO_ZIP = true;`**

Data walker options: go into subfolders, and try to open `.zip` files.

**`int MIN_PARAMS = 5000, MAX_PARAMS_WARN = 50'000'000;`**

If the model has fewer than 5000 parameters, grow it until it has at least 5000. If more than 50 million, print a warning (but still run).

**Pattern to remember:** *Settings live in one place. Every knob is named. Changing a knob should change behavior without editing the code below it.*

---

# Part 3 — Section 1: Math Helpers

## Why these exist

Real code calls `dot` (dot product), `axpy` (add scaled vector), and `relu` / `gelu` (activation functions) hundreds of thousands of times. If they were inline, the code would be unreadable. If they were not optimized, the program would be slow. So we make them tiny, named, and fast.

## `imin` and `imax` and `szmin` and `dmax`

```cpp
static inline int imin(int a, int b) { return a < b ? a : b; }
```

`static` means "this function is only visible in this file." `inline` is a hint to the compiler: "please paste this code where I call it instead of doing a function call, because it is small." Function calls have a tiny cost; inlining removes that cost.

`imin(3, 5)` returns 3. `imin(5, 3)` also returns 3.

**Why not use `std::min`?** Because MSVC (Microsoft's compiler) has a `#define min(...)` macro in `<windows.h>` that clashes with `std::min` inside templates. Using our own name avoids the clash. This is a real-world survival trick.

**`szmin`** is `size_t` (unsigned) version. **`dmax`** is `double` version.

## `relu`

```cpp
static inline double relu(double x) { return x > 0.0 ? x : 0.0; }
```

`relu(x)` is `x` if positive, `0` if negative. That is it. It is the simplest activation function. Its purpose: introduce nonlinearity. Without nonlinear functions, a stack of matrix multiplications is still just a matrix multiplication. Nonlinearity lets the model learn complex patterns.

Imagine relu as a door: signals greater than zero go through unchanged; negative signals get shut out.

## `gelu`

```cpp
static inline double gelu(double x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
```

GELU is a smoother version of relu. It looks similar but curves gently at zero instead of being a sharp corner. The constants `0.7978845608` (= √(2/π)) and `0.044715` are chosen to match a real statistical function called the Gaussian CDF. Modern transformers (GPT, BERT) use GELU because smooth activations train better.

`tanh` is a built-in function (hyperbolic tangent). It squashes any number into `(-1, 1)`.

## `dot`

```cpp
static inline double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
```

`dot(a, b, n)` multiplies `a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]`. This is the dot product.

**What is the dot product good for?** It measures similarity. If `a` and `b` point the same way, their dot product is large positive. If opposite, large negative. If perpendicular, zero. Every neural network operation eventually becomes many dot products.

**`const double* a`**: `const` means "I promise not to modify what `a` points to." `double*` means pointer to doubles. This tells the compiler "safe to read from multiple threads."

**`#pragma omp simd reduction(+:s)`**: tells the compiler: "you can process `a[i]` and `b[i]` for multiple `i` in parallel using SIMD instructions (CPU vector operations, like doing 4 or 8 multiplications at once)." `reduction(+:s)` says: "each parallel lane will have its own partial sum of `s`; add them all at the end."

**Without this pragma:** the compiler may still vectorize on its own at `-O3`. With it, you are telling the compiler "do it, this is safe."

## `axpy`

```cpp
static inline void axpy(double alpha, const double* in, double* out, int n) {
    #pragma omp simd
    for (int i = 0; i < n; ++i) out[i] += alpha * in[i];
}
```

"axpy" is a legendary name from the BLAS library. It means: **a** times **x** plus **y**. So `out = alpha * in + out`. For each `i`, add `alpha * in[i]` to `out[i]`.

This is used everywhere in the backward pass. When computing `dW = activationᵀ · gradient`, we loop over rows and for each row, add a scaled version of the gradient.

**Pattern to remember:** *Small, named, hot functions. If a function is called a million times, it should be tiny and inlined. It should have a name that says what it does.*

---

# Part 4 — Section 2: Reading Environment Variables

## Why this exists

A user might want to change a setting without recompiling. Environment variables are a simple way: `TRAIN_D=256 ./trainer`. The program reads the env var and uses it.

But `getenv` is flagged as "unsafe" by MSVC because it can return a pointer to internal memory that might be modified elsewhere. The "safe" version is `_dupenv_s`, which copies the value.

## The code

```cpp
static int env_int(const char* name, int def) {
#ifdef _MSC_VER
    char* buf = NULL; size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name) == 0 && buf) { int v = atoi(buf); free(buf); return v; }
    return def;
#else
    const char* e = getenv(name); return e ? atoi(e) : def;
#endif
}
```

## Line by line

**`#ifdef _MSC_VER`**: The compiler defines `_MSC_VER` only on Microsoft. So this block is "MSVC-only."

**`char* buf = NULL; size_t sz = 0;`**: `buf` will hold the string value. `sz` is its size. Both start empty.

**`_dupenv_s(&buf, &sz, name)`**: Microsoft's safe env-var reader. It copies the value of `name` into a new buffer and stores the size in `sz`. Returns 0 on success.

**`atoi(buf)`**: converts string `"256"` to integer `256`.

**`free(buf)`**: releases the buffer allocated by `_dupenv_s`.

**`#else`**: on non-MSVC compilers.

**`getenv(name)`**: returns a pointer to the value, or `NULL` if not set.

**`e ? atoi(e) : def`**: if `e` is not `NULL`, return the parsed value; otherwise return the default `def`.

**`env_dbl`** does the same but with `atof` (string to double).

**Pattern to remember:** *Cross-platform code needs `#ifdef` to handle platform-specific quirks. Always free what you allocate. Always provide a default so a missing variable does not crash.*

---

# Part 5 — Section 3: Softmax and Sampling

## The problem

The model does not output "cat" or "dog." It outputs a vector of **logits** — one number per possible token. If there are 512 possible tokens, there are 512 numbers. But those numbers can be anything: `-4.2`, `11.8`, `0.3`. They are not probabilities. They can be negative. They do not sum to 1.

**Softmax** turns logits into probabilities. After softmax, every number is between 0 and 1, and they all add up to 1.

## The math

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Take `exp` of each number (makes all numbers positive). Then divide by the total. Result: each entry is a probability.

**Why `exp`?** Because `exp` has a magical property: it preserves order (bigger logit → bigger prob) and it turns addition into multiplication. In probability, `exp` of a "score" gives an "unnormalized probability." The history of why softmax uses `exp` is deep (see Boltzmann distributions), but the practical reason is: it works.

## The overflow problem

If any `z_i` is very large (say `1000`), `exp(1000)` is `inf`. Then division gives `nan`. One `nan` in the gradient kills the whole training run.

## The fix: subtract the max

```
softmax(z)_i = exp(z_i - m) / sum_j exp(z_j - m)
where m = max_j z_j
```

Subtracting `m` from every `z` does not change the result mathematically:

```
exp(z_i - m) / sum_j exp(z_j - m)
= [exp(z_i) / exp(m)] / [sum_j exp(z_j) / exp(m)]
= exp(z_i) / sum_j exp(z_j)
```

But now, the largest `z_i - m` is `0`, so `exp(0) = 1`. No overflow.

## The code

```cpp
static void softmax_stable(std::vector<double>& v) {
    if (v.empty()) return;
    double m = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < v.size(); ++i) if (v[i] > m) m = v[i];
    if (!std::isfinite(m)) {
        double u = 1.0 / (double)v.size();
        for (size_t i = 0; i < v.size(); ++i) v[i] = u;
        return;
    }
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) { v[i] = std::exp(v[i] - m); s += v[i]; }
    if (s == 0.0 || !std::isfinite(s)) {
        double u = 1.0 / (double)v.size();
        for (size_t i = 0; i < v.size(); ++i) v[i] = u;
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) v[i] /= s;
}
```

## Line by line

**`static void softmax_stable(std::vector<double>& v)`**: takes a vector of doubles by reference (`&`), meaning it modifies the caller's vector directly. This saves a copy.

**`if (v.empty()) return;`**: guard against empty input.

**`double m = -std::numeric_limits<double>::infinity();`**: start with the smallest possible number so any real value beats it.

**`for (size_t i = 0; i < v.size(); ++i) if (v[i] > m) m = v[i];`**: find the maximum.

**`if (!std::isfinite(m))`**: `isfinite` is false for `nan` and `±inf`. If the max is not finite, something is broken. Fall back to uniform probabilities.

**`double u = 1.0 / (double)v.size();`**: each entry gets `1/V`. For `V = 512`, each is `0.00195`.

**Second loop**: `exp(v[i] - m)`, accumulate into `s`.

**Second guard**: if `s` is 0 (all underflowed) or not finite, fall back to uniform.

**Final loop**: divide each by `s`.

## Sampling

After training, you want the model to generate text. It outputs probabilities. How do you pick the next token?

**Greedy**: pick the highest-probability token. Boring but deterministic.

**Temperature**: raise probabilities to power `1/T`. If `T < 1`, sharpens (favors top). If `T > 1`, flattens (more random). `T = 1` unchanged.

**Top-k**: keep only the top k tokens, zero the rest, renormalize.

**Top-p (nucleus)**: keep the smallest set whose cumulative probability exceeds `p`. Zero the rest.

## The code

```cpp
static int sample_from(const std::vector<double>& probs, double temp, double top_p, int top_k, mt19937& rng) {
    int V = (int)probs.size();
    std::vector<double> p = probs;
    if (temp != 1.0 && temp > 0) {
        for (double& x : p) x = pow(x, 1.0 / temp);
        double s = accumulate(p.begin(), p.end(), 0.0);
        if (s > 0) for (double& x : p) x /= s;
    }
    ...
}
```

## Line by line

**`int V = (int)probs.size();`**: get vocabulary size.

**`std::vector<double> p = probs;`**: copy so we can modify without touching the original.

**`if (temp != 1.0 && temp > 0)`**: skip if no temperature change requested. `temp > 0` guards against `T = 0` (which would divide by zero).

**`x = pow(x, 1.0 / temp)`**: apply temperature. `pow(0.5, 2.0) = 0.25` (sharpen with `T=0.5` → power 2). `pow(0.5, 0.5) = 0.707` (flatten with `T=2` → power 0.5).

**`s = accumulate(...)`**: sum. Then renormalize.

**`top_k` block**: `partial_sort` puts the top-k probabilities at the front. Zero the rest. Renormalize.

**`top_p` block**: sort by probability descending. Keep adding to a cumulative sum until it exceeds `top_p`. Zero the rest.

**`discrete_distribution<int> dist(p.begin(), p.end());`**: a C++ built-in that samples from a discrete probability distribution. This is where the actual random pick happens.

**`return dist(rng);`**: draws one sample.

**Pattern to remember:** *Softmax is the bridge from arbitrary scores to probabilities. Always subtract the max. Sampling has three knobs (temp, top-k, top-p) that all do different things — do not confuse them.*

---

# Part 6 — Section 4: Matrix Multiplication

## Why matrices

A neural network is a stack of linear transformations. Each transformation takes a vector and produces a new vector. Doing this one vector at a time is slow. But if you stack `N` vectors into a matrix, you can do all `N` transformations with one big operation.

A matrix `A` of shape `(M, K)` has `M` rows and `K` columns. A matrix `B` of shape `(K, N)` has `K` rows and `N` columns. Their product `C = A · B` has shape `(M, N)`.

Element `C[i, j] = sum_k A[i, k] * B[k, j]`.

## Naive matrix multiplication

```cpp
for i: for j: for k: C[i,j] += A[i,k] * B[k,j]
```

For each output entry, sum over `k`. Simple. But slow, because for fixed `(i,j)` you walk down column `j` of `B` — that is not contiguous in memory. Modern CPUs hate that.

## Cache-friendly order: i-k-j

```cpp
for i: for k: for j: C[i,j] += A[i,k] * B[k,j]
```

For fixed `(i,k)`, you walk row `k` of `B` (contiguous) and row `i` of `C` (contiguous). Two contiguous walks per multiplier. Much faster.

## The code

```cpp
static std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    std::vector<double> C((size_t)M * (size_t)N, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        const size_t iK = (size_t)i * (size_t)K;
        const size_t iN = (size_t)i * (size_t)N;
        for (int k = 0; k < K; ++k) {
            double a = A[iK + (size_t)k];
            if (a == 0.0) continue;
            const double* brow = &B[(size_t)k * (size_t)N];
            double* crow = &C[iN];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
    return C;
}
```

## Line by line

**`std::vector<double> C((size_t)M * (size_t)N, 0.0);`**: allocate `M*N` doubles, initialized to zero. The `(size_t)` casts prevent integer overflow on 32-bit systems.

**`#pragma omp parallel for`**: each `i` runs on a different CPU core. On a 4-core machine, this is up to 4× faster.

**`const size_t iK = (size_t)i * (size_t)K;`**: the offset into `A` for row `i`. Because `A` is a flat 1-D array holding row-major `M×K` data, element `A[i, k]` lives at `A[i*K + k]`.

**`for (int k = 0; k < K; ++k)`**: for each column of `A` / row of `B`.

**`double a = A[iK + (size_t)k];`**: `A[i, k]`.

**`if (a == 0.0) continue;`**: skip if multiplier is zero. In dense training this rarely triggers. In MoE it might.

**`const double* brow = &B[(size_t)k * (size_t)N];`**: pointer to start of row `k` in `B`.

**`double* crow = &C[iN];`**: pointer to start of row `i` in `C`.

**`for (int j = 0; j < N; ++j) crow[j] += a * brow[j];`**: this is the `axpy` operation inlined. `C[i, j] += a * B[k, j]`.

## Transposed matmul

```cpp
static std::vector<double> matmul_TN(const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    // returns A^T (K x M) * B (K x N) -> (M x N)
```

Sometimes you want `Aᵀ · B` instead of `A · B`. Naively, you would compute `Aᵀ` (a transpose copy) then multiply. That wastes memory. `matmul_TN` computes it directly by indexing `A[k, i]` instead of `A[i, k]`.

## Line by line

**`const double* arow = &A[(size_t)k * (size_t)M];`**: row `k` of `A` (which becomes column `k` of `Aᵀ`).

**`for (int i = 0; i < M; ++i) { double a = arow[i]; ... }`**: for each element of that column of `Aᵀ`.

**`axpy(a, brow, &C[(size_t)i * (size_t)N], N);`**: add `a * B[k, :]` to `C[i, :]`.

**Pattern to remember:** *Matrix multiplication is O(M·K·N) work. Order loops so memory access is contiguous. Parallelize the outer loop. Never transpose a matrix just to multiply — index instead.*

---

# Part 7 — Section 5: GPU Probe

## What is a GPU

A CPU has 4–16 cores, each running at 3 GHz, each very good at complex logic. A GPU has thousands of simpler cores running in parallel. Matrix multiplication is embarrassingly parallel — every `C[i,j]` can be computed independently — so GPUs are ideal.

## Why OpenCL

NVIDIA has CUDA. AMD has ROCm. Intel has oneAPI. These do not work on each other's hardware. **OpenCL** is the one API that works on all of them. It has a reputation for being annoying to write, but it is the only portable choice.

## The probe

```cpp
#ifdef USE_OPENCL
struct GPUBackend {
    cl_context ctx = nullptr;
    ...
    bool init() {
        cl_platform_id plat; cl_uint nplat;
        if (clGetPlatformIDs(1, &plat, &nplat) != CL_SUCCESS || nplat == 0) { err = "no OpenCL platform"; return false; }
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) { err = "no GPU device"; return false; }
        ...
    }
};
#endif
```

## Line by line

**`#ifdef USE_OPENCL`**: only compile this if `USE_OPENCL` is defined. Otherwise, no dependency on OpenCL headers.

**`clGetPlatformIDs`**: "is there any OpenCL implementation installed?" Returns 0 if found.

**`clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ...)`**: "is there a GPU on that platform?" Returns 0 if found.

**`clCreateContext`**: creates a "conversation" with the GPU.

**`clCreateCommandQueue`**: creates a queue for sending commands.

## The honest truth

**This probe does not actually use the GPU.** No kernel is compiled. No buffer is allocated. No matmul is offloaded. The probe just says "hey, there is a GPU here." The code below does not branch on that. Training still runs on CPU.

The reason is: at `D=128`, transferring matrices to and from the GPU takes longer than computing them on CPU. The GPU only wins at `D ≥ 512`.

**Pattern to remember:** *A probe is a check, not an action. Always be honest about what your code does and does not do. If you write a comment that says "GPU support," make sure the GPU actually does something.*

---

# Part 8 — Section 6: Parameter Estimator

## What it is

Before building the model, the program counts how many numbers (parameters) the model would have. If too few, it grows. If too many, it warns. Then it asks the user: "Is this good?"

## The math (for one block)

```
tok_emb:     V * D
pos_emb:     SEQ * D
Wq, Wk, Wv, Wo:   4 * D * D
FFN dense:   2 * D * FFN
LayerNorms:  2 * D (per pair, times 2 pairs)
Wout:        D * V
bout:        V
```

For MoE, add `(E-1) * 2 * D * FFN` more FFN parameters, plus `D * E` router parameters.

## The code

```cpp
static ParamEstimate estimate_params(int D, int V, int SEQ, int FFN, int num_experts, bool use_moe) {
    ParamEstimate e{};
    e.dense += (long long)V * D;
    ...
    e.total = e.dense + e.moe_extra;
    return e;
}
```

**`long long`** is at least 64 bits. Necessary because `V*D` can easily exceed `2³¹` at large sizes.

## The confirmation loop

```cpp
while (!ask_param_confirmation(est, D, F, E)) {
    est = estimate_params(D, V, S.SEQ, F, E, E > 1);
}
```

Ask. If user says no, they enter new values. Recompute. Ask again. Loop until user says yes.

**`ask_param_confirmation`** prints the estimate and asks "Y/n". If `N`, it prompts for new `D`, `FFN`, `E`, one at a time.

## The minimum loop

```cpp
while (est.total < S.MIN_PARAMS) {
    D *= 2; F = S.FFN_MULT * D;
    est = estimate_params(D, V, S.SEQ, F, E, S.USE_MOE);
}
```

If the estimate is below the minimum (5000), double `D`. Recompute. Repeat.

**Pattern to remember:** *Estimating cost before doing work is always worth it. If a model is too small to learn, say so before wasting time. If it is too big, warn the user before they OOM.*

---

# Part 9 — Section 7: File Walk

## The problem

You give the trainer a folder. It needs to find all the text files inside. This should work on Windows and Linux. There is no `<filesystem>` in C++11 or older, and even in C++17, some compilers do not support it well.

## The portable solution

Use `#ifdef` to pick the right OS API. Two functions: `list_dir(dir)` returns the names of items in a directory. `is_directory(path)` says whether a path is a folder.

## Windows version

```cpp
WIN32_FIND_DATAA fd;
HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
if (h == INVALID_HANDLE_VALUE) return out;
do {
    std::string n = fd.cFileName;
    if (n == "." || n == "..") continue;
    out.push_back(dir + "\\" + n);
} while (FindNextFileA(h, &fd));
FindClose(h);
```

**`FindFirstFileA(pattern, &fd)`**: starts iterating a directory. The `A` means ANSI (narrow strings). `pattern` is `dir\*`.

**`fd.cFileName`**: the name of the current item.

**`"."` and `".."`**: Windows reports these as entries. They are "current directory" and "parent directory." We skip them.

**`FindNextFileA`**: advance. **`FindClose`**: release the handle.

## POSIX version

```cpp
DIR* d = opendir(dir.c_str());
if (!d) return out;
struct dirent* e;
while ((e = readdir(d)) != NULL) {
    std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    out.push_back(dir + "/" + n);
}
closedir(d);
```

**`opendir`**: opens a directory. **`readdir`**: reads next entry. **`closedir`**: closes.

## BFS walk

```cpp
std::vector<std::string> stack;
stack.push_back(dir);
while (!stack.empty()) {
    std::string cur = stack.back();
    stack.pop_back();
    ...
    if (is_directory(c)) stack.push_back(c);
    else if (is_allowed_ext(c)) ingest_one_file(c, corpus, n_files);
}
```

BFS with an explicit stack. Push starting folder. Pop one. List its contents. For each child: if folder, push. If file with allowed extension, read it.

## Extension filter

```cpp
static std::string lower_ext(const std::string& p) {
    size_t slash = p.find_last_of("/\\");
    size_t dot   = p.find_last_of('.');
    if (dot == std::string::npos) return "";
    if (slash != std::string::npos && dot < slash) return "";
    std::string ext = p.substr(dot);
    for (size_t i = 0; i < ext.size(); ++i)
        ext[i] = (char)std::tolower((unsigned char)ext[i]);
    return ext;
}
```

**`find_last_of("/\\")`**: find the last slash or backslash. `slash` is where the filename starts.

**`find_last_of('.')`**: find the last dot. `dot` is where the extension starts.

**`if (dot < slash)`**: if the dot is before the last slash, it is not a real extension (part of a folder name). Return empty.

**`tolower`**: lowercase. Because `.TXT` and `.txt` should be the same.

## JSON extraction

```cpp
static void extract_json_strings(const std::string& raw, std::string& out) {
    for (auto& line : split_lines(raw)) {
        bool in_str = false, esc = false;
        for (char c : line) {
            if (esc) { if (in_str) out += c; esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') { in_str = !in_str; continue; }
            if (in_str) out += c;
        }
        out += '\n';
    }
}
```

The idea: walk the line, track whether we are inside a string (`in_str`), and whether we just saw a backslash (`esc`).

- If `esc` is true, this character is an escaped character. Add it if inside a string. Clear `esc`.
- If `c == '\\'`, set `esc = true` and skip. Do not add.
- If `c == '"'`, toggle `in_str`. Do not add.
- Otherwise, if inside a string, add.

Result: every character that is inside a quoted string, with escapes resolved.

## CSV extraction

```cpp
static void extract_csv_strings(const std::string& raw, std::string& out) {
    for (auto& line : split_lines(raw)) {
        for (char c : line) out += (c == ',' ? ' ' : c);
        out += '\n';
    }
}
```

Replace commas with spaces. Simple. Not perfect (does not handle quoted commas) but works for most CSVs.

**Pattern to remember:** *Portable code needs `#ifdef` around OS-specific calls. Always skip `.` and `..` in directory walks. Lowercase extensions before comparing. A line-by-line walker is the simplest robust parser.*

---

# Part 10 — Section 8: The BPE Tokenizer

## What is a tokenizer

A model cannot read strings. It can only read numbers. The tokenizer turns a string into a list of integers.

## Three levels of tokenization

**Character-level**: each character is a number. "cat" → [99, 97, 116]. Simple. But sequences are long, and each character is a hard thing to learn.

**Word-level**: each word is a number. "cat" → [12]. But you need a vocabulary of all words. Unknown words break.

**Subword (BPE)**: split words into pieces. "cat" → ["ca", "t"]. "catalog" → ["ca", "tal", "og"]. Common pieces get their own number. Rare words get split.

BPE is the standard. GPT-2, BERT, LLaMA all use it.

## How BPE training works

1. Start with a vocabulary of 256 bytes (0–255). Every byte is its own token.
2. Encode the whole corpus as a list of bytes.
3. Count every adjacent pair. "ca" appears 500 times, "at" 300 times, "ta" 200 times.
4. Pick the most frequent pair. Give it a new token ID (`256` for the first merge, `257` for the second, ...).
5. Replace every occurrence of that pair with the new ID.
6. Repeat.

After `N` merges, you have `256 + N` tokens, and a trained BPE that knows which pairs to merge.

## The code (train)

```cpp
bool train(const std::string& corpus, int target_vocab, int max_merges) {
    int nm = imin(target_vocab - 256, max_merges);
    std::vector<int> tok = encode_raw(corpus);
    for (int m = 0; m < nm; ++m) {
        if (tok.size() < 2) break;
        std::map<std::pair<int,int>, int> freq;
        for (size_t i = 0; i + 1 < tok.size(); ++i) freq[{tok[i], tok[i+1]}]++;
        if (freq.empty()) break;
        auto best = freq.begin()->first; int bc = freq.begin()->second;
        for (auto& kv : freq)
            if (kv.second > bc || (kv.second == bc && kv.first < best)) { best = kv.first; bc = kv.second; }
        if (bc < 2) break;
        merges[best] = next_id; merge_list.push_back(best);
        int nid = next_id++;
        std::vector<int> nt;
        for (size_t i = 0; i < tok.size(); ) {
            if (i + 1 < tok.size() && tok[i] == best.first && tok[i+1] == best.second) { nt.push_back(nid); i += 2; }
            else { nt.push_back(tok[i]); ++i; }
        }
        tok.swap(nt);
    }
    trained = true;
    build_reverse();
    return true;
}
```

## Line by line

**`int nm = imin(target_vocab - 256, max_merges);`**: how many merges to do. At most `target_vocab - 256`, at most `max_merges`.

**`std::vector<int> tok = encode_raw(corpus);`**: bytes.

**Outer loop `for m`**: one merge per iteration.

**`if (tok.size() < 2) break;`**: cannot count pairs with fewer than 2 tokens.

**`std::map<std::pair<int,int>, int> freq;`**: map from pair to count. A `std::map` is a tree, kept sorted by key.

**`freq[{tok[i], tok[i+1]}]++;`**: count adjacent pairs.

**Find best pair**: start with the first pair. Then compare each pair's count to current best. Tie-break on smaller `(a, b)`.

**`if (bc < 2) break;`**: no pair appears twice. Stop merging (nothing to merge).

**Record the merge**: `merges[best] = next_id` says "pair `best` becomes token `next_id`." `merge_list.push_back(best)` records order. `next_id++` bumps the counter.

**Apply merge**: walk `tok` left to right. When you see `best.first` followed by `best.second`, emit `nid` and skip 2 positions. Otherwise emit current token and advance by 1.

**`tok.swap(nt)`**: replace old tokens with new. `swap` is O(1).

**`build_reverse()`**: build the reverse table so `decode` works.

## Encode (apply merges)

```cpp
std::vector<int> encode(const std::string& s) const {
    std::vector<int> tok = encode_raw(s);
    bool changed = true;
    while (changed && tok.size() > 1) {
        changed = false;
        std::vector<int> nt;
        for (size_t i = 0; i < tok.size(); ) {
            if (i + 1 < tok.size()) {
                auto it = merges.find({tok[i], tok[i+1]});
                if (it != merges.end()) { nt.push_back(it->second); i += 2; changed = true; continue; }
            }
            nt.push_back(tok[i]); ++i;
        }
        tok.swap(nt);
    }
    return tok;
}
```

Same idea, but apply all merges in order. Each pass, find any pair that is a known merge and replace it. Repeat until nothing changes.

**Why `while (changed)`?** Because after replacing one pair, adjacent pairs may now be mergeable. Loop until a full pass makes no change.

## Decode (reverse)

```cpp
void build_reverse() {
    id_to_bytes.assign(next_id, "");
    for (int b = 0; b < 256; ++b) id_to_bytes[b] = string(1, (char)b);
    for (auto& kv : merges)
        id_to_bytes[kv.second] = id_to_bytes[kv.first.first] + id_to_bytes[kv.first.second];
}
```

Build a table: token ID → byte string. Bytes 0–255 map to themselves. Merged IDs map to the concatenation of their parents. Because merges were created in order, parents always exist before children.

```cpp
std::string decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id >= 0 && id < (int)id_to_bytes.size()) out += id_to_bytes[id];
        else out += "?";
    }
    return out;
}
```

Concatenate. Done.

**Pattern to remember:** *BPE is greedy pair merging. Training records merges; encoding applies them; decoding reverses them. The reverse table is the key to correct decoding.*

---

# Part 11 — Section 9: Model Parameters

## What parameters are

Every layer in the model has a **weight matrix** or a **bias vector**. These are the learnable numbers. When we say "the model learns," we mean "these numbers change."

## The struct

```cpp
struct Params {
    std::vector<double> tok_emb, pos_emb;
    std::vector<double> Wq, Wk, Wv, Wo;
    std::vector<double> W1, W2, b1, b2;
    std::vector<double> ln1g, ln1b, ln2g, ln2b;
    std::vector<double> Wout, bout;
    std::vector<double> Wr;
    std::vector<std::vector<double>> exp_W1, exp_W2, exp_b1, exp_b2;
    std::vector<double> mt, vt;
    int t = 0;
};
```

Every weight is a `std::vector<double>`. A 2-D matrix like `Wq` is stored flat (row-major).

## What each one is

**`tok_emb`**: shape `V × D`. Row `i` is the embedding for token `i`. Every token becomes a `D`-dimensional vector.

**`pos_emb`**: shape `SEQ × D`. Row `i` is the "position `i`" embedding. Because attention has no sense of order (it sees a bag of tokens), we add position info to tell it "this is the 5th token."

**`Wq`, `Wk`, `Wv`**: shape `D × D`. Queries, keys, values. Used in attention.

**`Wo`**: shape `D × D`. Attention output projection.

**`W1`, `W2`, `b1`, `b2`**: dense FFN (when MoE off).

**`ln1g`, `ln1b`, `ln2g`, `ln2b`**: LayerNorm gain and bias, shape `D` each.

**`Wout`, `bout`**: output head. `D × V` and `V`.

**`Wr`**: router, `D × E`. Only used in MoE.

**`exp_W1[e]`, `exp_W2[e]`, `exp_b1[e]`, `exp_b2[e]`**: expert `e`'s FFN.

**`mt`, `vt`**: AdamW state (first and second moment).

**`t`**: AdamW step counter.

## Xavier init

```cpp
static void xavier(std::vector<double>& W, int fi, int fo) {
    double lim = sqrt(6.0 / (fi + fo));
    for (double& x : W) x = uniform_(-lim, lim);
}
```

If you initialize all weights to zero, symmetry breaks the model — every neuron computes the same thing. If you initialize them to huge random values, activations explode. Xavier gives you the sweet spot: random values of "just right" magnitude.

**`lim = sqrt(6.0 / (fi + fo))`**: this is the Xavier bound. `fi` is fan-in (input size), `fo` is fan-out (output size).

**`uniform_(-lim, lim)`**: draws a uniform random number in `[-lim, lim]`.

## Why `uniform_` uses a global RNG

```cpp
static std::mt19937 g_rng(42);
static double uniform_(double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(g_rng);
}
```

`mt19937` is a Mersenne Twister pseudo-random generator. Seeded with 42, so the sequence is deterministic. Every run gets the same weights. This is critical for reproducibility.

**Pattern to remember:** *Weights are matrices, stored flat. Xavier init keeps them small enough to train but random enough to break symmetry. Seed the RNG for reproducibility.*

---

# Part 12 — Section 10: The Forward Pass

## The big picture

Data enters as tokens. It exits as probabilities over the next token. In between, it goes through:

1. Embedding
2. LayerNorm
3. Attention
4. Residual
5. LayerNorm
6. MoE/FFN
7. Residual
8. Output head
9. Softmax

This is the *forward pass*. It is called once per training step. It is the "make a guess" phase.

## The math of one block

```
x = tok_emb[token] + pos_emb[position]
xn1 = LayerNorm(x)
Q, K, V = xn1 * Wq, xn1 * Wk, xn1 * Wv
attn = softmax(Q * K^T / sqrt(HD)) * V    (causal)
attn_proj = attn * Wo
r1 = x + attn_proj                        (residual)
xn2 = LayerNorm(r1)
moe_out = MoE(xn2) or FFN(xn2)
h2 = r1 + moe_out                         (residual)
logits = h2 * Wout + bout
probs = softmax(logits)
```

## Embedding

```cpp
c.x.resize((size_t)N * D);
for (int i = 0; i < N; ++i) {
    const size_t iD = (size_t)i * (size_t)D;
    const size_t tD = (size_t)toks[i] * (size_t)D;
    for (int j = 0; j < D; ++j)
        c.x[iD + (size_t)j] = p.tok_emb[tD + (size_t)j] + p.pos_emb[iD + (size_t)j];
}
```

For token at position `i`: look up row `toks[i]` of `tok_emb`, add row `i` of `pos_emb`. Result is a `D`-length vector. Do this for all `N` positions.

## LayerNorm

```cpp
static void layer_norm(const std::vector<double>& x, std::vector<double>& out, int N, int d) {
    out.resize(x.size());
    for (int i = 0; i < N; ++i) {
        double s = 0, s2 = 0;
        for (int j = 0; j < d; ++j) { double v = x[i*d+j]; s += v; s2 += v*v; }
        double mu = s / d, va = s2 / d - mu*mu;
        if (va < 0) va = 0;
        double inv = 1.0 / sqrt(va + 1e-5);
        for (int j = 0; j < d; ++j) out[i*d+j] = (x[i*d+j] - mu) * inv;
    }
}
```

For each row:

1. Compute mean `mu = sum(x)/d`.
2. Compute variance `va = sum(x²)/d - mu²`.
3. Normalize: `(x - mu) * 1/sqrt(va + eps)`.

The `+ eps = 1e-5` prevents division by zero if all values are the same.

**Then scale and shift**: `out = normalized * gamma + beta`. This lets the network un-normalize if it wants.

## Attention

```cpp
for (int h = 0; h < S.HEADS; ++h) {
    int ho = h * HD;
    for (int i = 0; i < N; ++i) {
        std::vector<double> sc(i + 1, 0.0);
        size_t qb = (size_t)i * D + ho;
        for (int j = 0; j <= i; ++j) {
            size_t kb = (size_t)j * D + ho;
            sc[j] = dot(&c.Q[qb], &c.K[kb], HD) * scale;
        }
        softmax_stable(sc);
        for (int j = 0; j <= i; ++j) {
            c.attn_w[((size_t)h * N + i) * N + j] = sc[j];
            size_t vb = (size_t)j * D + ho;
            axpy(sc[j], &c.V[vb], &c.attn_out[qb], HD);
        }
    }
}
```

For each head `h`, for each position `i`:

1. Compute scores between `Q[i]` and `K[j]` for all `j ≤ i`. Multiply by `scale = 1/sqrt(HD)`.
2. Softmax the scores.
3. Use the softmaxed scores to weight-sum `V[j]`. Add to `attn_out[i]`.

**Causal**: only `j ≤ i`. Position `i` cannot see position `i+1`. This is what makes it a language model, not a "read the whole sentence and summarize" model.

**Why softmax?** Because scores are arbitrary numbers. We need weights that sum to 1 to do a weighted average.

**Why `1/sqrt(HD)`?** Because `dot(Q, K)` for `HD`-dim vectors with unit variance has variance `HD`. Dividing by `sqrt(HD)` normalizes it.

## MoE

```cpp
for (int i = 0; i < N; ++i) {
    std::vector<double> logits(E);
    for (int e = 0; e < E; ++e) logits[e] = dot(&c.xn2[(size_t)i * D], &p.Wr[(size_t)e * D], D);
    softmax_stable(logits);
    // pick top-K, compute their FFNs, weight by softmax score
}
```

For each token:

1. Router: compute `E` scores (one per expert).
2. Softmax: probabilities over experts.
3. Pick top-K.
4. For each chosen expert: compute its FFN output, multiply by router probability, sum.

## Output head

```cpp
c.logits = matmul(c.xn3, p.Wout, N, D, V);
for (int i = 0; i < N; ++i) for (int j = 0; j < V; ++j) c.logits[(size_t)i * V + j] += p.bout[j];
```

`xn3` (shape `N × D`) times `Wout` (shape `D × V`) = logits (shape `N × V`). Plus bias. Then softmax per row.

**Pattern to remember:** *Forward pass is a chain of transformations. Each is a linear map plus a nonlinearity plus a normalization. Residuals let gradients flow freely. Attention is weighted sum with softmaxed similarity.*

---

# Part 13 — Section 11: The Backward Pass

## Why it matters

Forward pass gives you `probs`. Compare to true target. Loss is large. Now what? You know the loss is wrong. Which weight is responsible?

This is the **credit assignment problem**. Backpropagation solves it.

## The idea

If you know `dLoss/dw` for every weight `w`, you can nudge `w` in the direction that lowers loss:

```
w ← w - lr * (dLoss/dw)
```

The "direction that lowers loss" is `-dLoss/dw` (opposite of the gradient, because gradient points uphill).

## The chain rule

For a chain `f(g(h(x)))`:

```
df/dx = df/dg * dg/dh * dh/dx
```

Every layer is a link in the chain. Backprop starts at the loss and walks backward, multiplying local derivatives.

## What the code computes

### Output head

```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < V; ++j)
        dl[i*V + j] = (c.probs[i*V + j] - (j == targets[i] ? 1.0 : 0.0)) / N;
```

`dLogits = (Probs - OneHot) / N`. This is the famous shortcut: for softmax + cross-entropy, the derivative simplifies to `probs - target`. No need to apply the softmax Jacobian separately.

**Why `/N`?** Because loss is averaged over `N` tokens.

```cpp
g.dWout = matmul_TN(c.xn3, dl, D, N, V);
```

`dWout = xn3ᵀ · dLogits`. Shape `(D × N) · (N × V) = D × V`. Correct.

```cpp
dxn3 = dLogits · Woutᵀ
```

`dXn3 = dLogits · Woutᵀ`. Shape `(N × V) · (V × D) = N × D`. Correct.

### MoE backward

For each token `i`, for each chosen expert `e`:

```cpp
g.d_exp_W2[e][j*F + f] += w * dh2[i*D+j] * h[f];
```

Contribution to `W2_e[j, f]`: `router_prob * dH2[i, j] * h[f]`. Standard outer product.

```cpp
dh1[f] = sum_j dh2[i*D+j] * W2_e[j*F+f];
```

`dH1 = dH2 · W2ᵀ`. Shape `(D,) · (D × F) = F`. Correct.

```cpp
dh1[f] *= gelu_derivative(x);
```

Chain through GELU.

```cpp
g.d_exp_W1[e][f*D + j] += w * dh1[f] * xn2[i*D+j];
```

`dW1_e[f, j] = router_prob * dH1[f] * xn2[j]`. Outer product again.

```cpp
dxn2[i*D+j] += w * sum_f dh1[f] * W1_e[f*D+j];
```

`dXn2 = router_prob * dH1 · W1ᵀ`. Sum over chosen experts.

### Attention backward

This is the trickiest. For each (i, j):

```cpp
dScore[i, j] = attn_w[i, j] * (dw[i, j] - sum_k attn_w[i, k] * dw[i, k]);
```

`dw[i, j]` is `dAttnOut[i] · V[j]`. The subtraction accounts for softmax's normalization. This is the softmax Jacobian applied to scores.

```cpp
dQ[i] += dScore[i, j] * K[j];
dK[j] += dScore[i, j] * Q[i];
dV[j] += attn_w[i, j] * dAttnOut[i];
```

Gradients to Q, K, V. `dQ` uses `K` (because score = Q·K), `dK` uses `Q`, `dV` uses attention weights.

### LayerNorm backward

```cpp
dx[i, j] = dy[i, j] * gamma[j] * inv - xhat[i, j] * gamma[j] * inv / D * 2;
```

This is an approximation. The full formula also includes a mean-subtraction term. The code is slightly off. The training still works, but gradient check would show a small error.

### Embedding backward

```cpp
g.dtok_emb[toks[i] * D + j] += dx[i * D + j];
g.dpos_emb[i * D + j] += dx[i * D + j];
```

`+=` not `=`. Why? Because the same token can appear multiple times in the sequence. Each position contributes its own gradient. They add up.

## Gradient clipping

```cpp
double nrm = 0;
for (size_t i = 0; i < all.size(); ++i) nrm += all[i] * all[i];
nrm = std::sqrt(nrm);
if (nrm > C.CLIP) {
    double s = C.CLIP / nrm;
    for (size_t i = 0; i < all.size(); ++i) all[i] *= s;
}
```

Compute the L2 norm of all gradients together. If it exceeds `CLIP`, scale all gradients down by `CLIP/nrm`.

**Why?** If one gradient is huge, it can blow up the weights and training diverges. Clipping caps the damage.

**Pattern to remember:** *Backprop is the chain rule applied layer by layer, from output to input. At each layer: compute local gradient, multiply by incoming gradient, pass to previous layer. Clip the norm for stability.*

---

# Part 14 — Section 12: Optimizers

## What an optimizer does

You have a gradient `g`. You want to update weight `w`. How? Three classic answers:

**SGD**: `w -= lr * g`. Simple. Unstable.

**Momentum SGD**: keep a running average `m = 0.9*m + g`. Then `w -= lr * m`. Smooths out noise.

**Adam**: keep both mean `m` and variance `v`. Update: `w -= lr * m / sqrt(v)`. Adapts per-parameter. Widely used.

## Muon

Muon is a newer optimizer (2024+). For 2-D weight matrices, it does something special: **orthogonalize** the momentum. Orthogonalize means "make all rows have equal magnitude and be perpendicular."

Why? Because a plain gradient can have some rows much larger than others. That means some neurons get much bigger updates than others. This causes instability. Orthogonalization equalizes them.

## Newton-Schulz iteration

To orthogonalize `M`, we do:

```
X₀ = M / ||M||
X_{k+1} = X_k * (3I - X_kᵀ X_k) / 2
```

After 5 iterations, `X₅` has approximately orthogonal rows with unit norm.

**Why does this work?** Each iteration modifies the singular values `σ` of `X` by `σ ← σ*(3 - σ²)/2`. For `σ ∈ (0, 1]`, this converges to `1`. So all singular values become 1. A matrix with all unit singular values is orthogonal.

## NorMuon

NorMuon is 2026 upgrade. After orthogonalizing, it also normalizes each row:

```
v_row = moving average of ||row||²
row = row / sqrt(v_row + eps)
```

This is like Adam's per-row version. Prevents any single row from dominating.

## The code

```cpp
static void normuon_step(std::vector<double>& W, const std::vector<double>& G, std::vector<double>& mom, int rows, int cols, double lr, double momentum, int ns_iters) {
    static thread_local std::vector<double> v;
    if (v.size() != (size_t)rows) v.assign(rows, 1.0);
    for (size_t i = 0; i < mom.size(); ++i) mom[i] = momentum * mom[i] + G[i];
    std::vector<double> O = mom;
    newton_schulz(O, rows, cols, ns_iters);
    for (int r = 0; r < rows; ++r) {
        double s = 0;
        for (int c = 0; c < cols; ++c) s += O[r*cols+c] * O[r*cols+c];
        double ms = s / cols;
        v[r] = 0.95 * v[r] + 0.05 * ms;
        double inv = 1.0 / sqrt(v[r] + 1e-8);
        for (int c = 0; c < cols; ++c) O[r*cols+c] *= inv;
    }
    double scale = sqrt(dmax(1.0, (double)rows / cols));
    for (size_t i = 0; i < W.size(); ++i) W[i] -= lr * scale * O[i];
}
```

## Line by line

**`static thread_local std::vector<double> v;`**: `thread_local` means each thread has its own copy. `static` means it persists between calls (but not between threads — that is the bug).

**`if (v.size() != (size_t)rows) v.assign(rows, 1.0);`**: initialize if shape changed.

**`mom[i] = momentum * mom[i] + G[i]`**: momentum update. `0.95` old + `0.05` new.

**`newton_schulz(O, rows, cols, ns_iters)`**: orthogonalize.

**Row normalization**: compute mean of squared row elements, update `v[r]`, divide row by `sqrt(v[r])`. This is per-row adaptation.

**`W[i] -= lr * scale * O[i]`**: apply the update.

## AdamW

```cpp
static void adamw_flat(std::vector<double>& W, const std::vector<double>& G, std::vector<double>& mt, std::vector<double>& vt, size_t off, int& t, double lr, double wd) {
    double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    t++;
    double bc1 = 1 - pow(b1, t), bc2 = 1 - pow(b2, t);
    for (size_t i = 0; i < W.size(); ++i) {
        double g = G[i];
        double m = b1 * mt[off + i] + (1 - b1) * g;
        double v = b2 * vt[off + i] + (1 - b2) * g * g;
        mt[off + i] = m; vt[off + i] = v;
        W[i] -= lr * (m / bc1 / (sqrt(v / bc2) + eps) + wd * W[i]);
    }
}
```

Standard AdamW. `m` = moving mean, `v` = moving variance. Bias correction `bc1`, `bc2` (because both start at zero). Weight decay `wd * W[i]`.

**Pattern to remember:** *AdamW for 1-D parameters and embeddings. NorMuon for 2-D weight matrices. NorMuon orthogonalizes the gradient first, then normalizes per row. Both use momentum to smooth out noise.*

---

# Part 15 — Section 13: Learning Rate Schedule

## Why schedule

A constant high learning rate diverges. A constant low learning rate is slow. A schedule that starts high and drops over time works best.

## WSD

**Warmup**: start at 0, ramp up to peak. Why? Because random initial weights produce wild gradients. Ramping up gives the model time to stabilize.

**Stable**: stay at peak. Why? Because the model is learning most of its structure here.

**Decay**: ramp down to zero. Why? Because near the end, you want tiny steps to fine-tune.

## The code

```cpp
static double lr_wsd(int step, int total, double base, double warm_ratio, double decay_ratio) {
    int warm = (int)(total * warm_ratio);
    int stable_end = (int)(total * (1.0 - decay_ratio));
    if (step < warm) return base * step / (double)max(1, warm);
    if (step < stable_end) return base;
    double p = (step - stable_end) / (double)max(1, total - stable_end);
    return base * 0.5 * (1.0 + cos(M_PI * p));
}
```

**Phase 1 (warmup)**: linear ramp from 0 to `base`.

**Phase 2 (stable)**: constant `base`.

**Phase 3 (decay)**: half-cosine from `base` to 0. `cos` starts at 1, ends at -1. So `(1 + cos(π·p))` starts at 2 (giving `base`) and ends at 0.

**Pattern to remember:** *Warmup, then stable, then decay. Each phase has a reason. Use `cos` for smooth decay because it is naturally smooth at both ends.*

---

# Part 16 — Section 14: EMA and Label Smoothing

## EMA

```cpp
static void ema_update(std::vector<double>& shadow, const std::vector<double>& live, double decay) {
    for (size_t i = 0; i < shadow.size(); ++i) shadow[i] = decay * shadow[i] + (1 - decay) * live[i];
}
```

`shadow` is a slow-moving copy of `live`. Each update, `shadow = 0.999 * shadow + 0.001 * live`. So `shadow` trails `live` by roughly `1/(1-decay) = 1000` steps.

**Why?** Recent weights can be noisy. The averaged version is often smoother and performs better on validation. Use `shadow` at evaluation time.

## Label smoothing

The normal cross-entropy loss says: "be 100% confident in the true class." Label smoothing says: "be 90% confident in the true class, spread 10% among the rest."

```cpp
loss -= (1 - smooth) * log(p_target) + smooth * sum_log_p / V;
```

**Why?** Because forcing 100% confidence is impossible to achieve and leads to overfitting. Smoothing pushes the model to be less extreme.

**Pattern to remember:** *EMA is averaging over time. Label smoothing is softening the target. Both reduce overfitting.*

---

# Part 17 — Section 15: The Main Loop

## What it does

Put everything together.

```cpp
int main(int argc, char** argv) {
    Settings S = pick_config();
    // ...
    for (int step = 0; step < S.STEPS; ++step) {
        // 1. compute LR
        // 2. sample window
        // 3. forward pass
        // 4. compute loss
        // 5. backward pass
        // 6. update weights
        // 7. every N steps: log, eval
    }
}
```

## Line by line

**`Settings S = pick_config();`**: read env vars, build the config.

**GPU probe**: try to init OpenCL.

**`ingest_folder(path, corpus, err, ...)`**: read all files into one big string.

**BPE training**: train the tokenizer on `corpus`.

**Parameter estimator**: count parameters, ask user.

**`Params P = make_params(S, V, D, F, E);`**: allocate and initialize all weights.

**Main loop**:

```cpp
for (int step = 0; step < S.STEPS; ++step) {
    double lr = lr_wsd(step, S.STEPS, S.LR, S.WARMUP_RATIO, S.DECAY_RATIO);
    int off = uniform_int_distribution<int>(0, (int)ids.size() - S.SEQ - 1)(rng);
    vector<int> in(ids.begin() + off, ids.begin() + off + S.SEQ);
    vector<int> tg(ids.begin() + off + 1, ids.begin() + off + S.SEQ + 1);
    forward(P, S, V, S.D, F, E, in, cache, true);
    double loss = cross_entropy(cache, tg, V, S.LABEL_SMOOTH);
    // backward, update, log
}
```

1. **`lr_wsd`**: get learning rate for this step.
2. **`uniform_int_distribution`**: random start position in the corpus. This is why we sample — the model should not memorize one window.
3. **`in`, `tg`**: input window and target (shifted by 1). For input `[a, b, c]`, target is `[b, c, d]`.
4. **`forward`**: run model, populate cache.
5. **`cross_entropy`**: compute loss.
6. **`backward`**: compute gradients.
7. **`update_step`**: apply NorMuon / AdamW.

**Evaluation**:

```cpp
if (step % S.EVAL_EVERY == 0 || step == S.STEPS - 1) {
    vector<int> vin(ids.end() - S.SEQ - 1, ids.end() - 1);
    vector<int> vtg(ids.end() - S.SEQ, ids.end());
    Cache vc;
    forward(P, S, V, S.D, F, E, vin, vc, false);
    double vl = cross_entropy(vc, vtg, V, 0.0);
    cout << "  [eval@" << step << "] val_loss=" << vl << ...;
}
```

Uses the last window as "validation." Forward pass only — no gradient. `false` means no training.

**Generation**:

```cpp
for (int k = 0; k < 40; ++k) {
    vector<int> ctx(gen.end() - S.SEQ, gen.end());
    forward(P, S, V, S.D, F, E, ctx, gc, false);
    int nxt = sample_from(probs, S.TEMPERATURE, S.TOP_P, S.TOP_K_SAMPLE, rng);
    gen.push_back(nxt);
}
```

Autoregressive: predict next token, append, repeat.

**Pattern to remember:** *Training is a loop. Sample → forward → loss → backward → update → repeat. Everything else is refinement.*

---

# Part 18 — The Ten Patterns That Make This Code Work

If you learn nothing else, learn these ten patterns. They appear in every neural network trainer.

**Pattern 1: Configuration lives at the top.** All knobs in one place. Easy to read. Easy to change.

**Pattern 2: Small functions for hot paths.** `dot`, `axpy`, `relu`, `gelu` are called millions of times. Keep them tiny and let the compiler inline them.

**Pattern 3: Parallelize the outer loop.** `#pragma omp parallel for` on the outermost loop of `matmul`. Each row is independent.

**Pattern 4: Stable softmax always subtracts the max.** Never write plain `exp(z)/sum(exp(z))`. Always subtract max first.

**Pattern 5: Forward and backward are mirror images.** Every operation in forward has a corresponding backward. If forward does `+`, backward might do `+=`. If forward does `*`, backward does `*` by transposed matrix.

**Pattern 6: Residuals let gradients flow.** Every `+` residual is a highway for gradients. Without them, deep networks cannot train.

**Pattern 7: Clipping protects against blowups.** Compute gradient norm, scale if too big.

**Pattern 8: Schedule the learning rate.** Do not use constant LR. Warm up. Decay at the end.

**Pattern 9: Use buffers that persist.** Optimizer state (`mt`, `vt`) and EMA shadows must survive across steps. They are not recomputed from scratch.

**Pattern 10: Seed the RNG, save checkpoints, log everything.** Reproducibility + observability + safety.

---

# Part 19 — How to Read This Code If You Are New

## Read in this order

1. `Settings` struct — Section 0. Get the vocabulary of knobs.
2. `main()` — Section 15. See the flow.
3. `forward()` — Section 10. Understand the math.
4. `cross_entropy()` — Section 3. Understand the loss.
5. `backward()` — Section 11. Accept that it is a mirror of forward.
6. `normuon_step()` — Section 12. Understand the update rule.
7. `BPETokenizer::train()` — Section 8. Understand the tokenizer.
8. Everything else — detail.

## Debugging tips

If the loss goes to `nan`: check softmax overflow, check for zero denominators, check gradient clipping.

If the loss goes to infinity: check learning rate, check gradient clipping, check loss floor.

If the loss does not go down: check `lr_wsd` (is it warming up correctly?), check data (is the corpus long enough?), check that backward and forward are consistent (gradient check).

If the loss goes down then spikes: check gradient clip, check LR schedule, check for exploding attention logits.

## What this code teaches

- How a transformer block works.
- How BPE tokenization works.
- How MoE routing works.
- How NorMuon and AdamW work.
- How gradient flow works.
- How to write numerical code that does not overflow.
- How to structure a single-file trainer.

**Final pattern to remember:** *Everything is math. Every math operation is a few lines of C++. Learn the math, and the code is obvious.*
