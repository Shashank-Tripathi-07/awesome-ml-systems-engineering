# 0 to ML Systems Engineer

This is not a comfort document. It is a map of what the job actually requires,
how long it actually takes, and what separates engineers who can do it from
engineers who think they can.

ML Systems Engineering is the discipline of building the infrastructure that
trains, serves, and operates machine learning models at scale. It is not data
science. It is not prompt engineering. It is systems engineering applied to ML
workloads -- which means every hard problem in distributed systems, compilers,
and hardware shows up, plus the additional complexity that your workload is
non-deterministic and your correctness criteria are probabilistic.

The bar is high and it is not going down. This roadmap tells you exactly what
to build, in what order, and what it proves when you can do it.

---

## The Brutal Truth First

Most people who say they want to be ML Systems Engineers mean they want to call
PyTorch APIs and push models to HuggingFace. That is not the job.

The job is: your training run is silently corrupting weights at step 40,000 of
an 80,000-step job on a 512-GPU cluster. Find it. Fix it. Prevent it from
happening again. Do this while 40 engineers are waiting for the checkpoint.

The job is: your inference service is handling 50,000 requests per second and
p99 latency just spiked to 800ms. Locate the bottleneck. Is it the attention
kernel, the KV cache eviction policy, the load balancer, or the NCCL ring?
You have 20 minutes before the SLA breach.

The job is: your 70B model fits in 140GB of HBM across 4 H100s in BF16. The
team wants to serve it on 2 H100s. Make it work without a quality regression.

If those problems sound interesting, keep reading. If they sound like someone
else's problem, this career path is not for you and you should find that out now
rather than after 2 years of wasted effort.

---

## What the Role Actually Is

ML Systems Engineering has several subspecialties. Know which one you are
targeting before you start because the required skills diverge significantly.

**Training Infrastructure** -- The systems that run pre-training and fine-tuning
jobs. Distributed training frameworks, gradient communication, checkpointing,
fault tolerance, job scheduling. You spend most of your time making large jobs
not crash, and when they do crash, making them resume correctly.

**Inference Infrastructure** -- The systems that serve model predictions to users.
Batching, KV caching, speculative decoding, quantization, hardware-efficient
kernels, SLO management. You spend most of your time chasing latency and cost.

**ML Platform** -- The internal tooling layer: experiment tracking, feature
stores, model registries, data pipelines, evaluation frameworks. Closer to
software engineering than systems, but requires understanding the whole stack.

**Compiler and Kernel Engineering** -- Writing CUDA/Triton kernels and working
on XLA/MLIR compilers to make specific operations faster on specific hardware.
Deepest technical role. Requires serious CS fundamentals.

**Hardware and Networking** -- Designing the physical infrastructure: GPU cluster
topology, InfiniBand networking, storage systems, power and cooling. Rare role,
extremely high impact.

This roadmap covers the foundations common to all of them, then branches.

---

## Phase 0: Prerequisites (Non-Negotiable)

Before you touch anything ML-specific, you need these. If you are missing any
of them, fix that first. There is no shortcut.

### Systems fundamentals

You must understand how a computer actually works. Not abstractly. Concretely.

**Memory hierarchy**: L1/L2/L3 cache, DRAM, NVMe. Know the latency and bandwidth
numbers from memory. Know what cache misses cost. Know why memory-bound vs
compute-bound matters for every operation you write.

```
L1 cache:   ~4 cycles     ~1 ns
L2 cache:   ~12 cycles    ~4 ns
L3 cache:   ~40 cycles    ~15 ns
DRAM:       ~200 cycles   ~60 ns
NVMe SSD:   ~100,000 ns
```

If you cannot explain why these numbers matter for matrix multiplication
performance, you are not ready.

**Concurrency**: threads, processes, locks, atomics, memory ordering. Know the
difference between data parallelism and task parallelism. Understand why
Python's GIL exists and what it prevents.

**Networking**: TCP/IP stack, bandwidth vs latency, what a socket is, why
RDMA exists and what it solves, why InfiniBand is different from Ethernet.
Distributed training is fundamentally a networking problem.

**Operating systems**: virtual memory, how `mmap` works, what a page fault costs,
file descriptors, process scheduling. ML data pipelines break in ways that are
impossible to debug without this.

Resources that are not optional:
- *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron) -- read it cover to cover
- *Operating Systems: Three Easy Pieces* -- free online, do the exercises
- *Computer Networking: A Top-Down Approach* (Kurose & Ross) -- first 4 chapters minimum

### Mathematics

**Linear algebra**: matrix multiplication, eigendecomposition, SVD, vector
spaces, norms. Not "I know what a matrix is." You need to be able to derive
why matrix multiply is O(n^3) and why blocking/tiling makes it cache-efficient.

**Calculus and optimization**: partial derivatives, chain rule, gradient descent,
Lagrange multipliers, convexity. You need to understand backpropagation by
deriving it yourself, not by calling `.backward()`.

**Probability and statistics**: expectation, variance, distributions, MLE,
Bayesian inference basics, concentration inequalities. Required for understanding
training dynamics, evaluation, and anything involving uncertainty.

**Numerical methods**: floating point representation, catastrophic cancellation,
condition numbers, numerical stability. Non-negotiable for anyone writing kernels
or debugging training instability.

### Programming

You need to be genuinely proficient at Python. Not "I can write scripts."
Proficient means: you understand the data model, the memory model, the import
system, generators, context managers, decorators, and the GIL. You can profile
a Python program and find the bottleneck.

You need working knowledge of C/C++. Not expert level, but you must be able to
read CUDA kernel code, understand pointer arithmetic, and modify existing C
extensions without causing memory corruption.

You need to be comfortable with the command line, `git`, `ssh`, and Linux
administration. You will spend significant time on remote machines with no GUI.

---

## Phase 1: ML Fundamentals (2-3 months)

Do not skip this phase because you think you already know it. The ML Systems
engineer who cannot explain why cross-entropy loss works, what a gradient is,
or why batch normalization helps training will make wrong decisions at every
level of the stack. The systems are in service of the ML. You have to understand
what you are serving.

### What you must actually understand

**Neural networks from scratch**: implement a 2-layer MLP in pure NumPy.
Forward pass, backward pass, weight update. No frameworks. If you cannot do this,
you do not understand backpropagation. Stop and do it now.

**The transformer**: implement attention from scratch in NumPy. Query, key, value
projections, scaled dot-product attention, causal masking, multi-head. Then
implement a full decoder-only transformer block. Train it on something small
(character-level Shakespeare, or the addition task). It should overfit the
training set -- if it does not, your implementation is wrong.

**Training dynamics**: learn what loss curves look like when things are working
and when they are not. Understand learning rate warmup, why gradient clipping
is necessary, what a loss spike looks like vs divergence, what good vs bad
gradient norm behavior is.

**Standard architectures**: not just transformers. Know CNNs well enough to
explain the receptive field and why depthwise separable convolutions exist.
Know why LSTMs were the pre-transformer standard and what their fundamental
limitation was. Know what ResNets solved with residual connections and why
that matters for transformers too.

### The projects that prove you did this phase

1. Implement backpropagation for a 2-layer MLP without autograd. Get it to
   classify MNIST to >97% accuracy.

2. Implement multi-head self-attention from scratch in NumPy. Verify it matches
   PyTorch's output on the same inputs.

3. Train nanoGPT on Shakespeare character-level. Read every line of the 800-line
   codebase. Understand every design decision.

4. Reproduce one result from a foundational paper: either the original Transformer
   on WMT translation, or GPT-2 small language modeling perplexity. Exact match
   not required, within 2 PPL is sufficient.

If you cannot do all four, you are not done with Phase 1.

---

## Phase 2: The GPU Programming Model (2-3 months)

This is where most people quit. The GPU programming model is genuinely hard.
It is also non-negotiable for anyone who wants to work on inference kernels,
training efficiency, or compiler backends.

### The mental model you must build

A GPU is not a fast CPU. It is a massively parallel processor designed for
throughput, not latency. Understanding it requires building an accurate mental
model of:

**Thread hierarchy**: threads, warps (32 threads), thread blocks, grids.
Every 32 threads in a warp execute the same instruction in lockstep (SIMT).
Divergent branches within a warp serialize. This is why branchy code is
catastrophically slow on GPU.

**Memory hierarchy**:
```
Registers:     per-thread, ~255 per thread, fastest
Shared memory: per-block, ~48-100KB, fast, explicitly managed
L2 cache:      chip-wide, ~40MB on H100
HBM:           off-chip DRAM, 80GB on H100, ~3.3 TB/s bandwidth
```

Shared memory is the key. Efficient GPU kernels tile computation so that data
is loaded from HBM into shared memory once, reused many times, and results
written back once. This is exactly what Flash Attention does.

**Compute vs memory bound**: operations that reuse data heavily (matrix multiply)
are compute-bound -- the bottleneck is the number of FLOPs the hardware can
execute. Operations that touch memory without reusing it (elementwise ops) are
memory-bound -- the bottleneck is how fast data moves between HBM and compute
units. You cannot optimize a memory-bound kernel by adding more compute.

**Occupancy**: the fraction of the maximum possible warps that are resident on
an SM at once. High occupancy lets the hardware hide memory latency by switching
to other warps while one waits for a memory load. Low occupancy means idle
compute units.

### What to study

Read the CUDA Programming Guide. Not skim. Read it. Chapters 1-6 minimum.
Every concept in it will matter when you debug a slow kernel.

*Programming Massively Parallel Processors* (Kirk & Hwu): the textbook. Work
through the exercises.

Lec 1-10 of Stanford CS149 (Parallel Computing): the clearest explanation of
the GPU programming model in existence. Free on YouTube.

### The projects that prove you did this phase

1. Write a CUDA matrix multiply from scratch. Start naive (one thread per output
   element). Then tile it to use shared memory. Measure the speedup. You should
   see 10-50x improvement from tiling alone. If you do not, your tiling is wrong.

2. Implement a fused CUDA kernel for `LayerNorm + Dropout + Residual Add`. Verify
   correctness against PyTorch's output. Measure memory bandwidth utilization
   using Nsight Compute. Understand why fusing these operations saves memory
   bandwidth.

3. Implement Flash Attention v1 in CUDA (or Triton). You do not need to match
   FlashAttention2's performance. You need to understand every line of the
   tiling algorithm and why it never materializes the full attention matrix.
   Read the paper, implement it, verify it produces identical outputs to naive
   attention.

4. Profile a PyTorch training loop with Nsight Systems or PyTorch Profiler.
   Identify the three most expensive operations. Explain why each one is
   expensive (is it compute bound? Memory bound? What is its arithmetic intensity?).

If you cannot complete all four, you are not ready for Phase 3 of the inference
or compiler tracks.

---

## Phase 3: Distributed Training (2-3 months)

Single-machine training does not scale past ~1B parameters on a single 8-GPU
node. Everything interesting in ML today requires understanding how to split work
across many machines and manage the resulting communication.

### Data Parallelism

The simplest form. Replicate the model on N devices. Split the batch N ways.
Each device computes gradients for its shard. All-reduce the gradients across
devices. Each device updates its local copy of parameters identically.

The key operation is **all-reduce**: every device contributes a gradient tensor,
every device receives the sum. The dominant algorithm is the ring all-reduce:

```
Ring all-reduce for N devices:
  Phase 1 (reduce-scatter): N-1 steps, each device sends to its neighbor
  Phase 2 (all-gather):     N-1 steps, each device sends to its neighbor
  Total: 2*(N-1)/N * data per device
  Bus utilization approaches 100% as N grows
```

Implementation: `torch.nn.parallel.DistributedDataParallel` (DDP) or `jax.pmap`.
Use `torchrun` or `mpirun` to launch. Know how gradient averaging and optimizer
state are handled across ranks.

**The critical bug**: forgetting to synchronize random seeds across ranks. Your
dropout masks must be different across ranks (different data, different masks),
but your weight initialization must be identical. Get this wrong and your model
trains inconsistently.

### Tensor Parallelism

When a single weight matrix does not fit on one device, you partition it.
Megatron-LM style tensor parallelism splits each weight matrix column-wise or
row-wise across devices:

```
Y = XA   where A is [d_model, d_ff]

Column-parallel:  A1 = A[:, :d_ff/N],  A2 = A[:, d_ff/N:]
                  Y1 = XA1,  Y2 = XA2
                  Collect: all-gather(Y1, Y2) -> Y

Row-parallel:     A1 = A[:d_model/N, :],  A2 = A[d_model/N:, :]
                  X1 = X[:, :d_model/N],  X2 = X[:, d_model/N:]
                  Y = reduce_scatter(X1@A1 + X2@A2)
```

Each forward and backward pass requires an all-reduce across the tensor parallel
group. Communication is synchronous and blocks the forward pass.

### Pipeline Parallelism

Assign different layers to different devices. Device 0 runs layers 0-9, device 1
runs layers 10-19, etc. Split the batch into microbatches. Device 0 processes
microbatch 1 and passes activations to device 1, then immediately starts
microbatch 2 while device 1 is processing microbatch 1.

The challenge: the pipeline bubble. At startup and teardown, some devices are
idle. The bubble fraction is `(p-1)/(m+p-1)` where p is pipeline depth and m
is number of microbatches. Minimize it by using many microbatches.

### ZeRO / FSDP

Fully Sharded Data Parallelism. Instead of replicating the entire model on each
device, shard the optimizer state, gradients, and parameters across devices.
Each device holds only `1/N` of each.

ZeRO has three stages:
- **Stage 1**: shard optimizer state only. 4x memory reduction.
- **Stage 2**: shard optimizer state + gradients. 8x memory reduction.
- **Stage 3**: shard optimizer state + gradients + parameters. Nx memory reduction.

Stage 3 requires an all-gather before each forward operation (to reconstruct
the full layer) and a reduce-scatter after each backward. More communication
than DDP but allows training models far larger than a single device's memory.

PyTorch FSDP implements ZeRO Stage 3. This is the standard for large model
training on GPU clusters.

### Communication libraries

**NCCL** (NVIDIA Collective Communications Library): the low-level communication
library underlying PyTorch DDP, FSDP, and most distributed training. Know its
primitives (all-reduce, all-gather, reduce-scatter, broadcast, barrier). Know
that NCCL uses ring algorithms over NVLink (intra-node) and InfiniBand/RDMA
(inter-node).

**Gloo**: CPU fallback for NCCL. Slower. Used for CPU-based collectives.

**NCCL tuning**: `NCCL_ALGO` (ring vs tree), `NCCL_PROTO` (simple vs LL vs
LL128), socket buffer sizes. Getting these wrong can halve your training
throughput.

### The projects that prove you did this phase

1. Train a model with DDP across 4 GPUs using `torchrun`. Verify that loss
   matches single-GPU training. Measure the communication overhead using
   PyTorch Profiler. Identify the all-reduce calls in the trace.

2. Implement gradient accumulation correctly with DDP. Specifically: use
   `model.no_sync()` for all but the last microbatch to avoid spurious all-reduces.
   Verify that the effective batch size is correct.

3. Train a model that does not fit on one GPU using FSDP. Measure peak memory
   per device with and without FSDP. Understand why the memory savings are less
   than theoretical maximum.

4. Diagnose a distributed training bug: given a training run where loss matches
   single-GPU for the first 100 steps then diverges, identify the cause.
   (Common causes: un-synced dropout, wrong gradient averaging, data loader
   returning different sequences on different ranks due to missing seed sync.)

---

## Phase 4: Inference Systems (2-3 months)

Training a model is the beginning. Serving it to millions of users is where
the engineering gets interesting. Inference is where latency, cost, and
correctness intersect, and where most ML systems engineering time is spent
in production.

### The KV cache

Autoregressive generation is sequential: each new token requires a forward pass.
During that forward pass, the attention keys and values for all previous tokens
must be recomputed -- unless you cache them.

The KV cache stores K and V tensors for every generated token, for every layer,
for every attention head. Memory requirement:

```
per token, per layer: 2 * n_kv_heads * head_dim * bytes_per_element
for a 7B model with 32 layers, 8 KV heads, head_dim=128, BF16:
  = 2 * 8 * 128 * 2 = 4096 bytes = 4KB per token per layer... wait
  = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128KB per token total
```

For 2048-token sequences with a batch of 32: 8GB just for KV cache. This is why
GQA (fewer KV heads) is standard -- it reduces KV cache proportionally.

**PagedAttention** (vLLM): analogous to virtual memory. The KV cache is divided
into fixed-size blocks. Each sequence gets blocks allocated on demand. Blocks
from different sequences can be non-contiguous in memory. This eliminates
fragmentation and allows much higher batch sizes. The key insight is that
sequences rarely fill their pre-allocated contiguous memory, wasting 60-80% of
KV cache under naive schemes.

### Continuous batching

Naive batching: collect a batch, run a full generation for all sequences, return
all results. Problem: short sequences finish early and their GPU is idle while
long sequences complete.

Continuous batching (also: iteration-level scheduling): at every iteration, swap
in new requests as old ones finish. No sequence waits for others to complete.
Throughput increase of 5-10x over naive batching.

Implemented by: vLLM, TensorRT-LLM, Ollama. Understand the mechanism, not just
that it exists.

### Speculative decoding

The bottleneck in autoregressive generation: each token requires a full forward
pass through the large model. The forward pass is memory-bandwidth bound (all
weights must be read from HBM for each token, even though most are not the
bottleneck).

Speculative decoding uses a small draft model to generate K tokens quickly,
then the large model verifies all K tokens in a single forward pass (which is
possible because verification is parallel, unlike generation which is sequential).
If the large model agrees with all K draft tokens, you get K tokens for the price
of one verification pass.

Acceptance rate depends on how well the draft model matches the large model.
Typical acceptance rates: 0.6-0.9 per token. Typical speedup: 1.5-3x.

### Quantization

**Post-Training Quantization (PTQ)**: after training, reduce the precision of
weights from BF16/FP32 to INT8 or INT4. No retraining. Some quality loss.

**INT8 quantization**: weights stored as INT8, dequantized to FP16 for compute.
Halves weight storage (and HBM reads), roughly equivalent throughput on hardware
with INT8 matrix units. LLM.int8() (bitsandbytes) is the standard for single-GPU.

**GPTQ**: quantize weights to 4-bit using a second-order optimization method
(approximate the Hessian, compensate for quantization error). Better quality
than naive INT4 rounding. Standard for 4-bit weight-only quantization.

**AWQ** (Activation-Aware Weight Quantization): quantizes weights by scaling
channels based on activation magnitude. Outperforms GPTQ at low bit widths.
The current state of the art for 4-bit.

**FP8 training and inference**: NVIDIA H100 has native FP8 matrix units. FP8
is emerging as the precision for both training (replacing BF16) and inference
(replacing INT8). 2x throughput vs BF16, better quality than INT8.

**The quantization tradeoff**: every bit you remove reduces model size and
increases throughput but decreases quality. The exact quality degradation
depends on the model, the quantization method, and the task. Always measure.

### The projects that prove you did this phase

1. Deploy a 7B model using vLLM. Measure throughput (tokens/second) and latency
   (TTFT and TPOT) under different concurrency levels. Find the saturation point.
   Understand what saturates first.

2. Implement a KV cache from scratch for a small transformer. Measure the speedup
   vs recomputing keys and values every step. Verify that outputs are identical.

3. Quantize a model to INT8 using bitsandbytes and to 4-bit using GPTQ. Run the
   same benchmark (e.g. MMLU subset) on all three (FP16, INT8, GPTQ-4bit).
   Measure quality vs memory vs throughput tradeoffs.

4. Implement speculative decoding for a small model pair. Measure acceptance
   rate and actual throughput speedup. Understand why the speedup is less than
   `1/(1-acceptance_rate)`.

---

## Phase 5: ML Platform and Operations (2 months)

Building the infrastructure that ML engineers use every day: experiment tracking,
data pipelines, model registries, evaluation frameworks, and monitoring. Less
glamorous than kernel engineering. The reason most ML projects succeed or fail.

### Experiment tracking

Every training run should record: hyperparameters, hardware, code commit,
dataset version, training metrics over time, evaluation metrics at the end.

This is not optional. Without it, you cannot reproduce your best result, you
cannot compare experiments, and you cannot debug regressions.

Tools: Weights & Biases (standard), MLflow (self-hosted), Neptune, CometML.
Know how to log metrics, artifacts, and media. Know how to query runs
programmatically to compare experiments.

**The discipline**: every experiment must be reproducible from the logged
information alone. If you cannot recreate a run from what you logged, you logged
the wrong things.

### Data versioning and pipelines

Training data is not static. New data arrives. Data quality changes. A model
trained on version 1.2 of the dataset must be distinguishable from one trained
on version 1.3.

**DVC** (Data Version Control): Git for datasets and models. Store large files
in S3/GCS/Azure, version them with DVC, commit the `.dvc` pointer files to git.
Every model can be traced back to the exact dataset and code that produced it.

**Data pipeline design**: the pipeline from raw data to training-ready shards
must be deterministic and idempotent. Given the same raw data and the same
pipeline code, you must get the same output shards. Use checksums to verify.

**The data quality problem**: garbage in, garbage out, always. Build validation
checks at every stage of your pipeline. Check for: duplicates, malformed examples,
label errors, distribution shift from the expected data. A single bad shard can
corrupt an entire training run.

### Model registry

A versioned store of trained model artifacts. Associates each model version with:
the training job that produced it, the data it was trained on, its evaluation
metrics, its intended use, and its deployment status.

Models in the registry should have explicit lifecycle states: staging, production,
deprecated. Promotion from staging to production should be gated on evaluation
metrics. Rollback to a previous version should be a one-command operation.

### Evaluation

**Offline evaluation**: run the model on a held-out test set and compute metrics
before deployment. Necessary but not sufficient. Offline metrics often do not
predict online performance.

**Online evaluation** (A/B testing): route a fraction of production traffic to
the new model. Measure the metric you actually care about (revenue, user
retention, task success rate). This is the only way to know if the model is
actually better.

**Shadow mode**: route production traffic to both models but only serve the old
model's output. Log both outputs. Analyze the new model's behavior without
exposing users to it. Good for catching catastrophic failures before they affect
users.

**Evaluation harness**: build a standard evaluation pipeline that can be run
against any model version, produces the same metrics, and stores results in the
experiment tracker. This is infrastructure work that pays off in every future
model iteration.

### Monitoring and alerting

A model in production degrades. Data distribution shifts. The model was trained
on winter data and you are serving summer data. User behavior changes. New
product features change the input distribution.

What to monitor:
- **Data drift**: is the distribution of inputs changing? (feature means,
  variances, category frequencies)
- **Concept drift**: given the same inputs, are the correct labels changing?
- **Model performance**: is your metric (accuracy, AUC, precision/recall)
  declining? (requires ground truth labels, which often arrive with delay)
- **Infrastructure metrics**: latency, error rate, throughput, GPU utilization

Set up alerts for: error rate spikes, latency SLA breaches, significant data
drift detected, model performance below threshold.

Tools: Evidently AI, Arize, Fiddler, WhyLabs, or build your own on top of
Prometheus and Grafana.

---

## Phase 6: Specialization (3-6 months)

By this point you have the foundations. Now you pick your lane and go deep.

### Kernel Engineering

This is the deepest technical path. You write CUDA/Triton kernels to make
specific operations faster than the generic implementations.

What you need beyond Phase 2:
- Triton: Python-based kernel language that compiles to GPU. Faster to iterate
  than raw CUDA. The standard for writing custom attention variants, activation
  functions, and fused operations. Read the Triton paper. Implement Flash
  Attention in Triton.
- CUTLASS: NVIDIA's template library for building high-performance GEMM kernels.
  Required if you want to push the hardware to its limits.
- Profiling at the hardware level: Nsight Compute, roofline analysis, memory
  traffic analysis. You need to know exactly why your kernel is not hitting
  the hardware ceiling.

The target: write a kernel that achieves >70% of theoretical peak FLOPS or
bandwidth utilization. If you cannot get there, understand why.

### Compiler Engineering

Working on XLA, MLIR, TVM, or similar compiler stacks. You work on the layer
between the user-facing framework (PyTorch, JAX) and the hardware kernels.

What you need beyond Phase 2:
- MLIR: the compiler infrastructure used by XLA, TVM, and most modern ML
  compilers. Understand dialects, transformations, and lowering passes.
- XLA internals: how JAX's JIT works, what the HLO (High Level Optimizer)
  IR looks like, how XLA fuses and tiles operations.
- Polyhedral compilation: the mathematical framework for loop transformations.
  Required for understanding how compilers optimize affine loop nests.

### Large-Scale Training Systems

Designing and operating training infrastructure for models with 10B-1T parameters.

What you need beyond Phase 3:
- 3D parallelism: combining data, tensor, and pipeline parallelism in the same
  job. Understanding the tradeoffs between communication volumes for each
  strategy.
- Fault tolerance at scale: a 1024-GPU job will have hardware failures during
  training. You need hot-standby nodes, fast checkpoint restore, and the ability
  to detect and exclude bad GPUs without stopping the run.
- Gradient compression: when communication is the bottleneck, reduce what you
  send. Gradient quantization, top-K sparsification. Understand the convergence
  implications.

---

## The Skills That Are Not Technical

### Debugging under pressure

The ability to methodically diagnose a production failure under time pressure
is a separate skill from knowing the systems. You practice it by actually
experiencing failures, which means: do not hide from incidents, volunteer to
be on-call, write post-mortems.

The debugging method that works: form a hypothesis, write it down, test it
with the smallest possible experiment, update your model. Never run a large
experiment to test a hypothesis you could test with a small one.

### Reading papers

The field moves faster than any course or textbook. Reading primary literature
is mandatory. The ability to read a paper, extract the core contribution, and
identify what is and is not reproducible is a skill that takes years to build.

Start with the papers that everyone references:
- Attention Is All You Need (Vaswani et al., 2017)
- BERT (Devlin et al., 2018)
- GPT-2 (Radford et al., 2019)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al., 2022)
- Flash Attention (Dao et al., 2022)
- LLaMA (Touvron et al., 2023)
- ZeRO (Rajbhandari et al., 2020)

Then read the papers in whatever specialization you choose. Read them with code
open. If you cannot implement the core contribution in a weekend, you did not
understand it.

### Writing

You will design systems. You will make tradeoff decisions. You will need to
convince teammates and managers that your design is correct. The ML Systems
Engineer who cannot write a clear design document or post-mortem is limited
in their impact regardless of technical depth.

Write about what you build. Explain your decisions. Publish it. The act of
explaining something clearly forces you to find the gaps in your own understanding.

---

## What to Build (The Portfolio)

Employers hiring ML Systems Engineers are not impressed by notebooks and
fine-tuning demos. They want to see that you can build real infrastructure.

**Build 1: Pre-train an LLM from scratch on free compute**

Not fine-tuning. Pre-training. Write the tokenizer, the data pipeline, the
model, the training loop, the checkpoint system, and the evaluation. Do it
on Kaggle TPUs (free, 20hr/week). The model should be >100M parameters and
train on >1B tokens. Publish the weights, the training logs, and a technical
write-up that is honest about what worked and what did not.

This proves: you understand the full pre-training stack, you can debug JAX on
TPU, you can manage compute constraints, and you can ship something end-to-end.

**Build 2: Write a custom CUDA kernel that outperforms PyTorch**

Pick a specific operation (fused attention, RMS norm + dropout, rotary embeddings)
and write a CUDA or Triton kernel that is faster than the PyTorch default for
your specific use case. Measure it rigorously. Publish the code and the
benchmark methodology.

This proves: you can write correct parallel code, you understand the GPU memory
model, and you know how to measure performance properly.

**Build 3: Build a scalable inference service**

Serve a 7B model to handle 100+ concurrent requests. Implement: continuous
batching, KV cache management, request queuing, and SLO enforcement. Measure
and publish p50/p95/p99 latency and throughput numbers. Show what happens when
you exceed capacity and how the system degrades gracefully.

This proves: you understand inference systems, you can build production-quality
serving infrastructure, and you think about reliability.

**Build 4: Reproduce a paper result**

Pick a systems paper with a concrete benchmark claim (Flash Attention throughput,
ZeRO memory savings, speculative decoding speedup). Reproduce the result on
whatever hardware you have access to. Write up what matched, what did not, and
why.

This proves: you can engage with primary literature, you understand the
difference between claimed and measured performance, and you do not take
numbers at face value.

---

## Timeline

Honest estimate for someone starting from a CS bachelor's degree with standard
coursework (data structures, algorithms, some systems, no GPU programming):

```
Phase 0 (prerequisites):         3-6 months
Phase 1 (ML fundamentals):       2-3 months
Phase 2 (GPU programming):       2-3 months
Phase 3 (distributed training):  2-3 months
Phase 4 (inference systems):     2-3 months
Phase 5 (platform and ops):      2 months
Phase 6 (specialization):        3-6 months ongoing

Total to first job:               12-18 months
Total to senior-level depth:      3-5 years
```

These numbers assume you are working on this seriously. Not casually reading
tutorials. Seriously: working through textbooks, building projects, debugging
real systems, reading papers.

If you already have 2 years of backend/systems engineering experience, subtract
6 months. If you have no prior experience with any of Phase 0, add 6 months.

There is no fast path. The engineers who claim to have done this in 3 months
either had most of the prerequisite knowledge already or they cannot actually
do the hard parts.

---

## The Benchmark for Knowing Enough

You know enough to call yourself an ML Systems Engineer when you can do all of
the following without looking anything up:

- Explain why a transformer forward pass with batch size 1 is memory-bandwidth
  bound but with batch size 512 is compute bound
- Diagnose a training run where validation loss stops improving but training
  loss continues to decrease
- Explain the ring all-reduce algorithm and calculate its communication volume
  for N devices and a gradient of size G
- Look at a Nsight Compute roofline chart and identify whether a kernel is
  compute or memory bound and estimate how close it is to hardware ceiling
- Design a KV cache that can serve 1000 concurrent 4096-token sequences on a
  single H100 with an 80GB HBM budget, specifying your batch size, quantization,
  and GQA configuration to make it fit
- Explain what ZeRO Stage 3 does, what communication it adds, and when you would
  not use it
- Write a speculative decoding implementation from scratch given access to a
  draft and target model

If any of those feels uncertain, you know which section to go back to.

---

## What This Is Not

This roadmap does not cover prompt engineering, LangChain, RAG pipelines,
fine-tuning with LoRA on consumer GPUs, or deploying models with no-code
tools. That is not because those things have no value. It is because they are
not ML Systems Engineering. They are ML application development, which is a
different role with a different skill set and a different ceiling.

Both are legitimate. Know which one you are building toward.

---

## The Resources That Are Not Optional

These are the materials referenced throughout this roadmap that have no
acceptable substitutes:

**Books**
- *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron)
- *Operating Systems: Three Easy Pieces* (Arpaci-Dusseau)
- *Programming Massively Parallel Processors* (Kirk & Hwu)
- *Designing Data-Intensive Applications* (Kleppmann)
- *The Deep Learning Book* (Goodfellow, Bengio, Courville) -- chapters 6-9

**Courses**
- Stanford CS149: Parallel Computing (free, YouTube)
- CMU 15-418: Parallel Computer Architecture (free, course website)
- Stanford CS231n: CNNs for Visual Recognition (free, YouTube) -- for ML depth
- Fast.ai Part 2: Deep Learning from the Foundations (free)

**Papers (primary literature, not blog summaries)**
- Flash Attention 1 and 2 (Dao et al.)
- ZeRO (Rajbhandari et al.)
- Chinchilla (Hoffmann et al.)
- Megatron-LM (Shoeybi et al.)
- vLLM / PagedAttention (Kwon et al.)
- LLaMA 1/2/3 (Touvron et al.)
- Triton (Tillet et al.)

**Code to read in full**
- nanoGPT (Karpathy): 800 lines, every line matters
- vLLM source: PagedAttention implementation
- Flash Attention CUDA source: the tiling algorithm
- PyTorch FSDP implementation
- Megatron-LM: tensor and pipeline parallelism reference

---

The path is long. The material is hard. The payoff is that you can build things
that matter at scale, debug problems that most engineers cannot see, and
understand systems that most people treat as black boxes.

Start with Phase 0. Do not skip steps.
