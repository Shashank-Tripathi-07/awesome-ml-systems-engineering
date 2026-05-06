# Awesome ML Systems Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Track Awesome List](https://www.trackawesomelist.com/badge.svg)](https://www.trackawesomelist.com)

> A curated list of resources covering the full ML Systems Engineering stack -- from hardware and compilers to distributed training, inference, and production operations.

ML Systems Engineering sits at the intersection of machine learning and systems software. This list is for practitioners who build, optimize, and operate ML systems at any scale. Only resources personally vetted and considered essential are included.

⭐ = must-read / essential resource &nbsp;&nbsp; 🎓 = academic paper or course &nbsp;&nbsp; 🛠️ = hands-on tool or framework

## Contents

- [Foundations](#foundations)
- [Books](#books)
- [Courses](#courses)
- [Hardware and Accelerators](#hardware-and-accelerators)
- [ML Frameworks](#ml-frameworks)
- [Neural Network Compilers](#neural-network-compilers)
- [Kernel Programming](#kernel-programming)
- [Distributed Training](#distributed-training)
- [Collective Communication](#collective-communication)
- [Inference and Serving](#inference-and-serving)
- [Quantization and Compression](#quantization-and-compression)
- [Benchmarking and Profiling](#benchmarking-and-profiling)
- [Data Engineering](#data-engineering)
- [MLOps and Production](#mlops-and-production)
- [Fault Tolerance and Reliability](#fault-tolerance-and-reliability)
- [Security and Privacy](#security-and-privacy)
- [Sustainable AI](#sustainable-ai)
- [Networking and Interconnects](#networking-and-interconnects)
- [Storage Systems](#storage-systems)
- [Edge and Embedded ML](#edge-and-embedded-ml)
- [Landmark Papers](#landmark-papers)
- [Blogs and Newsletters](#blogs-and-newsletters)
- [Communities](#communities)
- [Conferences and Venues](#conferences-and-venues)

---

## Foundations

- [CS249r: Machine Learning Systems](https://mlsysbook.ai) ⭐ - Harvard's open textbook covering the full ML systems stack across two volumes: from neural computation to fleet-scale operations. The most comprehensive free reference in the field.
- [MLSys Conference Proceedings](https://mlsys.org) - The premier peer-reviewed venue for ML systems research. All proceedings are freely available.
- [The Deep Learning Compilation Survey](https://arxiv.org/abs/2002.08794) 🎓 - Comprehensive survey of deep learning compiler techniques; essential reading before diving into compiler backends.
- [Awesome ML Systems (GPU Mode)](https://github.com/gpu-mode/awesomeMLSys) - GPU Mode community's onboarding reading list, strong on GPU kernel and attention mechanism resources.
- [awesome-seml](https://github.com/SE-ML/awesome-seml) - Curated list on Software Engineering for ML, complementary to this list with a software process angle.

## Books

- [CS249r: Machine Learning Systems (Vol I & II)](https://mlsysbook.ai) ⭐ - Open-access two-volume textbook from Harvard covering ML workflows, hardware acceleration, distributed training, and responsible engineering.
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0) - Standard textbook for GPU programming with CUDA. Essential for anyone writing custom kernels.
- [Designing Data-Intensive Applications](https://dataintensive.net) ⭐ - Systems thinking for data pipelines, storage, and reliability. Directly applicable to ML data infrastructure.
- [Distributed Systems: Principles and Paradigms](https://www.distributed-systems.net/index.php/books/ds4/) - Foundational distributed systems theory that underpins distributed ML training. Available free online.
- [High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) - Profiling and optimization techniques for Python-based ML workflows.

## Courses

- [CS249r: TinyML and Efficient Deep Learning](https://efficientml.ai) 🎓 - MIT course on efficient ML covering quantization, pruning, and hardware-aware neural architecture design.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com) ⭐ - End-to-end course on building and deploying ML-powered products; strong on MLOps, tooling, and production.
- [GPU Mode Lectures](https://github.com/gpu-mode/lectures) 🎓 - Community lecture series on GPU programming, CUDA, and Triton kernel development. Freely available on YouTube.
- [Stanford CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall23) 🎓 - Foundational parallel computing course covering SIMD, multithreading, GPU architecture, and cache optimization.
- [CMU 15-418/618: Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/~418/) 🎓 - Deep dive into parallel hardware and programming models from CMU. Slides and assignments publicly available.
- [Deep Learning Systems (dlsyscourse.org)](https://dlsyscourse.org) 🎓 ⭐ - Course on building ML frameworks from scratch; teaches autodiff internals, operator fusion, and hardware backends.
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2023-fall-65940) 🎓 - MIT Han Lab's course on efficient neural networks, quantization, pruning, and hardware co-design.

## Hardware and Accelerators

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) ⭐ - The authoritative reference for CUDA GPU programming. Covers memory hierarchy, thread model, and execution model in depth.
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) - Deep technical dive into H100 GPU architecture including Transformer Engine, NVLink 4.0, and HBM3.
- [Google TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Official documentation on TPU pod architecture, memory hierarchy, and high-speed interconnects.
- [AMD ROCm Documentation](https://rocm.docs.amd.com/) - AMD's open GPU compute platform documentation. Increasingly relevant for large-scale training workloads.
- [Intel Gaudi Developer Documentation](https://docs.habana.ai) - Intel's ML accelerator documentation, relevant for AWS DL1/DL2 instance workloads.
- [Hot Chips Symposium](https://www.hotchips.org) - Annual symposium proceedings covering cutting-edge processor, accelerator, and memory architecture from industry and academia.

## ML Frameworks

- [PyTorch](https://pytorch.org) ⭐ 🛠️ - The dominant research and production training framework. Deep documentation on autograd, TorchScript, and distributed training primitives.
- [JAX](https://github.com/google/jax) ⭐ 🛠️ - Google's composable function transformations (grad, jit, vmap, pmap) on top of XLA. Preferred for high-performance research code.
- [TensorFlow](https://www.tensorflow.org) 🛠️ - Google's production ML framework with strong TFX ecosystem for data pipelines, serving, and model management.
- [torch.compile (PyTorch 2.x)](https://pytorch.org/docs/stable/torch.compiler.html) ⭐ - Graph capture and compilation system bridging eager execution and compiler-level optimization in PyTorch 2.0+.
- [Flax](https://github.com/google/flax) 🛠️ - Neural network library built on JAX for flexible, high-performance model development. Used internally at Google DeepMind.
- [ONNX](https://onnx.ai) 🛠️ - Open standard for representing ML models enabling interoperability across frameworks and hardware targets.

## Neural Network Compilers

- [TVM (Apache)](https://tvm.apache.org) ⭐ 🛠️ - End-to-end deep learning compiler stack supporting CPUs, GPUs, and custom accelerators. Includes Ansor for learning-based auto-tuning.
- [MLIR](https://mlir.llvm.org) ⭐ - Multi-level intermediate representation framework from LLVM. The backbone of modern ML compiler infrastructure across XLA, IREE, and Torch-MLIR.
- [XLA](https://www.tensorflow.org/xla) - Accelerated Linear Algebra compiler used by JAX and TensorFlow. Generates highly optimized GPU/TPU kernels via operation fusion.
- [Triton](https://triton-lang.org) ⭐ 🛠️ - OpenAI's Python-based language and compiler for writing GPU kernels at high productivity. Used to implement FlashAttention and PagedAttention.
- [IREE](https://iree.dev) 🛠️ - Intermediate Representation Execution Environment; a compiler and runtime for running ML models on diverse hardware targets including mobile.
- [TensorRT](https://developer.nvidia.com/tensorrt) 🛠️ - NVIDIA's high-performance inference optimizer and runtime. Supports FP8, INT8, and sparsity for CUDA GPUs.
- [OpenXLA](https://openxla.org) - Community-governed XLA project targeting broad hardware portability beyond Google's internal stack.

## Kernel Programming

- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) ⭐ - Official step-by-step tutorials for writing GPU kernels in Triton, from vector addition to fused softmax to FlashAttention.
- [CUTLASS](https://github.com/NVIDIA/cutlass) 🛠️ - NVIDIA's CUDA Templates for Linear Algebra Subroutines. High-performance GEMM and convolution primitives used inside cuDNN and TensorRT.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) ⭐ 🛠️ - IO-aware attention implementation that tiles computation to minimize HBM reads/writes. Standard for efficient Transformer training and inference.
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html) - NVIDIA's deep learning primitives library documentation; covers convolution algorithms, tensor formats, and fused operations.
- [Leet GPU](https://leetgpu.com) 🛠️ - Practice platform for GPU kernel programming challenges in CUDA and Triton, analogous to LeetCode for kernel engineers.
- [GPU Mode Resource Stream](https://github.com/gpu-mode/resource-stream) - Curated stream of GPU kernel programming resources from the GPU Mode Discord community.

## Distributed Training

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) ⭐ - Official overview of PyTorch distributed APIs: DDP, FSDP, and RPC. Best starting point before diving into framework internals.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) ⭐ 🛠️ - NVIDIA's framework for training large language models with tensor parallelism, pipeline parallelism, and data parallelism combined.
- [DeepSpeed](https://www.deepspeed.ai) 🛠️ - Microsoft's library for training massive models with ZeRO optimizer stages, pipeline parallelism, and CPU/NVMe offloading.
- [FSDP (Fully Sharded Data Parallel)](https://pytorch.org/docs/stable/fsdp.html) 🛠️ - PyTorch's native ZeRO-style model sharding. The standard approach for training large models within a single PyTorch job.
- [Alpa](https://github.com/alpa-projects/alpa) 🛠️ - System for automating parallelism strategy search across inter-op and intra-op partitioning dimensions.
- [Efficient Large-Scale LM Training on GPU Clusters](https://arxiv.org/abs/2104.04473) ⭐ 🎓 - Megatron-LM paper combining tensor, pipeline, and data parallelism. Essential reading for LLM training infrastructure.
- [GPipe](https://arxiv.org/abs/1811.06965) 🎓 - Google's pipeline parallelism paper introducing micro-batch gradient accumulation for memory-efficient model parallelism.
- [Switch Transformer](https://arxiv.org/abs/2101.03961) 🎓 - Google's paper on sparse Mixture-of-Experts training at scale with expert parallelism and load balancing.

## Collective Communication

- [NCCL](https://developer.nvidia.com/nccl) ⭐ 🛠️ - NVIDIA's Collective Communications Library for multi-GPU and multi-node AllReduce, Broadcast, and Scatter operations. The backbone of most distributed training stacks.
- [Ring-AllReduce Explained](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) ⭐ - Baidu's blog post explaining the ring-AllReduce algorithm that underpins bandwidth-optimal data parallel training.
- [Gloo](https://github.com/facebookincubator/gloo) 🛠️ - Facebook's collective communication library used as the CPU/Ethernet backend in PyTorch distributed.
- [MSCCL++](https://github.com/microsoft/mscclpp) 🛠️ - Microsoft's low-latency collective communication library with direct GPU kernel integration for fine-grained control.
- [MPI Standard](https://www.mpi-forum.org) - Message Passing Interface specification; the foundational collective communication model that NCCL and Gloo are built upon.

## Inference and Serving

- [vLLM](https://github.com/vllm-project/vllm) ⭐ 🛠️ - High-throughput LLM serving engine using PagedAttention for efficient KV cache management. Industry standard for production LLM serving.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 🛠️ - NVIDIA's optimized LLM inference library with FP8 quantization, in-flight batching, and multi-GPU tensor parallelism support.
- [SGLang](https://github.com/sgl-project/sglang) 🛠️ - Structured generation language and runtime with RadixAttention for efficient KV cache reuse across requests.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 🛠️ - Pure C/C++ LLM inference with quantization support. Runs on CPUs and Apple Silicon without framework dependencies.
- [Triton Inference Server](https://github.com/triton-inference-server/server) 🛠️ - NVIDIA's production-grade model serving platform with dynamic batching, model ensembles, and multi-framework support.
- [ONNX Runtime](https://onnxruntime.ai) 🛠️ - Cross-platform, high-performance inference engine for ONNX models across CPU, GPU, and edge targets.
- [PagedAttention](https://arxiv.org/abs/2309.06180) ⭐ 🎓 - vLLM's key paper on virtual memory-inspired KV cache management enabling high-throughput LLM serving without memory waste.
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) 🎓 - Technique for accelerating autoregressive generation using a small draft model to propose tokens verified by the full model.
- [Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) ⭐ - Anyscale's explanation of iteration-level batching, the key scheduling innovation behind modern high-throughput LLM serving.

## Quantization and Compression

- [GPTQ](https://arxiv.org/abs/2210.17323) ⭐ 🎓 - Post-training quantization for generative models enabling 4-bit inference of large language models with minimal accuracy loss.
- [AWQ (Activation-aware Weight Quantization)](https://arxiv.org/abs/2306.00978) 🎓 - Hardware-efficient 4-bit quantization that preserves salient weights identified by activation magnitude analysis.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 🛠️ - Library for 8-bit and 4-bit quantized optimizers and linear layers in PyTorch. Enables QLoRA fine-tuning on consumer GPUs.
- [SparseGPT](https://arxiv.org/abs/2301.00774) 🎓 - One-shot unstructured pruning for large language models achieving 50% sparsity without retraining.
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525) 🎓 - Comprehensive survey of knowledge distillation techniques for model compression across architectures and tasks.
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - File format used by llama.cpp for quantized model storage and memory-mapped loading on CPU and GPU.

## Benchmarking and Profiling

- [MLPerf](https://mlcommons.org/benchmarks/) ⭐ - Industry-standard ML benchmark suite covering training and inference across diverse hardware platforms. The definitive measure of ML system performance.
- [Roofline Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) ⭐ 🎓 - Performance model for identifying compute vs. memory bottleneck in GPU kernels. Essential mental model for any ML systems engineer.
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) 🛠️ - System-wide performance analysis tool for GPU workloads, essential for identifying CPU-GPU synchronization and pipeline bottlenecks.
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) 🛠️ - Kernel-level profiler for CUDA applications with roofline analysis and memory throughput breakdown.
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 🛠️ - Built-in PyTorch profiler with TensorBoard integration for operator-level CPU and GPU timing.
- [AI and Memory Wall](https://arxiv.org/abs/2403.14123) 🎓 - Analysis of how memory bandwidth limits LLM inference throughput across current and future hardware.
- [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) - Hugging Face benchmark comparing LLM inference throughput and latency across hardware configurations.

## Data Engineering

- [Apache Arrow](https://arrow.apache.org) ⭐ 🛠️ - Columnar in-memory data format enabling zero-copy data sharing between ML frameworks and data systems.
- [WebDataset](https://github.com/webdataset/webdataset) 🛠️ - Streaming dataset format for large-scale training from object storage, fully compatible with PyTorch DataLoader.
- [FFCV](https://ffcv.io) 🛠️ - Fast data loading library that reduces training data pipeline bottlenecks via a binary beton format with asynchronous loading.
- [Petastorm](https://github.com/uber/petastorm) 🛠️ - Uber's library for training deep learning models directly from Parquet datasets in HDFS or S3.
- [The Data-Centric AI Resource Hub](https://datacentricai.org) - Andrew Ng's initiative on systematic data engineering for ML with benchmarks and tooling for data quality improvement.
- [Feature Store Comparison](https://www.featurestore.org) - Community resource comparing feature store architectures including Feast, Hopsworks, and Tecton.

## MLOps and Production

- [Full Stack Deep Learning](https://fullstackdeeplearning.com) ⭐ - End-to-end course and resource hub for building and operating production ML systems. Covers tooling, testing, monitoring, and deployment.
- [Kubeflow](https://www.kubeflow.org) 🛠️ - Kubernetes-native ML platform for orchestrating training pipelines, hyperparameter tuning, and serving at scale.
- [MLflow](https://mlflow.org) 🛠️ - Open-source platform for experiment tracking, model registry, and multi-framework deployment.
- [Ray](https://www.ray.io) ⭐ 🛠️ - Distributed computing framework for scaling ML workloads with Ray Tune, Ray Train, and Ray Serve as first-class components.
- [Weights & Biases](https://wandb.ai) 🛠️ - Experiment tracking, artifact versioning, and collaborative ML development platform. Widely adopted in both research and production.
- [Evidently AI](https://www.evidentlyai.com) 🛠️ - Open-source ML monitoring library for detecting data drift, concept drift, and model performance degradation in production.
- [The ML Test Score](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/acd2df674c71b8e26e5b10f9c3e76a0f4b8a4c5.pdf) ⭐ 🎓 - Google's rubric paper defining 28 tests for production ML system readiness across data, model, infrastructure, and monitoring.

## Fault Tolerance and Reliability

- [Elastic Training with Torchelastic](https://pytorch.org/docs/stable/elastic/run.html) 🛠️ - PyTorch's elastic training framework for handling node failures and dynamic cluster resizing without restarting from scratch.
- [PyTorch Distributed Checkpointing](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) 🛠️ - PyTorch's distributed checkpointing API for asynchronous, fault-tolerant state saving during large-scale training.
- [Pathways](https://arxiv.org/abs/2203.12533) 🎓 - Google's paper on a unified ML runtime designed for reliability, heterogeneous acceleration, and multi-task training at scale.
- [Reliability at Scale (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/maeng) 🎓 - Microsoft research on failure modes and recovery strategies in large-scale ML training clusters.

## Security and Privacy

- [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) ⭐ - Google's seminal blog post introducing the federated learning paradigm for training models without centralizing user data.
- [PySyft](https://github.com/OpenMined/PySyft) 🛠️ - Framework for privacy-preserving ML via federated learning, differential privacy, and secure multi-party computation.
- [Differential Privacy in Deep Learning](https://arxiv.org/abs/1607.00133) 🎓 - Foundational paper applying differential privacy guarantees to deep learning via DP-SGD.
- [MITRE ATLAS](https://atlas.mitre.org) - MITRE's adversarial ML knowledge base documenting tactics, techniques, and real-world case studies of ML model attacks.
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) 🛠️ - IBM's open-source library for evaluating and defending ML models against adversarial inputs.

## Sustainable AI

- [Green AI](https://arxiv.org/abs/1907.10597) ⭐ 🎓 - Schwartz et al. paper calling attention to the growing energy cost of ML research and proposing efficiency as a reporting metric.
- [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02629) ⭐ 🎓 - Strubell et al. analysis of energy and carbon costs of NLP model training that sparked the field of sustainable AI.
- [CodeCarbon](https://codecarbon.io) 🛠️ - Python tool for automatically measuring and reporting CO2 emissions of ML training runs.
- [ML CO2 Impact Calculator](https://mlco2.github.io/impact/) 🛠️ - Online calculator for estimating the carbon footprint of ML training jobs across cloud providers and hardware.
- [EfficientML.ai](https://efficientml.ai) - MIT Han Lab's comprehensive resources on building energy-efficient ML models and systems.

## Networking and Interconnects

- [NVLink and NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/) ⭐ - NVIDIA's high-bandwidth GPU interconnect used in DGX and HGX systems. Critical for multi-GPU training performance.
- [InfiniBand Architecture](https://www.infinibandta.org) - High-speed networking standard used in HPC and ML clusters enabling RDMA-based collective communication at low latency.
- [RoCE (RDMA over Converged Ethernet)](https://www.rdmaconsortium.org) - Ethernet-based RDMA alternative to InfiniBand, increasingly used in hyperscale ML clusters.
- [Google Jupiter Network](https://research.google/pubs/pub43837/) 🎓 - Google's data center network architecture enabling petabit-scale bisection bandwidth for TPU pod training.
- [Ethernet for AI Clusters](https://arxiv.org/abs/2307.12229) 🎓 - Analysis of network topology and transport layer choices for large-scale ML training clusters.

## Storage Systems

- [Lustre Filesystem](https://www.lustre.org) - Parallel distributed filesystem widely deployed in HPC and ML training clusters for high-throughput checkpoint and dataset I/O.
- [Alluxio](https://www.alluxio.io) 🛠️ - Data orchestration layer enabling ML frameworks to access training data from multiple storage systems at near-memory speed.
- [AWS FSx for Lustre](https://aws.amazon.com/fsx/lustre/) - Managed Lustre filesystem tightly integrated with S3; standard choice for high-throughput ML training data access on AWS.
- [Delta Lake](https://delta.io) 🛠️ - ACID transactions and versioning for data lakes, enabling reliable ML feature pipelines and reproducible dataset snapshots.

## Edge and Embedded ML

- [TensorFlow Lite](https://www.tensorflow.org/lite) ⭐ 🛠️ - Lightweight framework for running ML models on mobile and embedded devices. Includes a converter and optimized runtime.
- [MCUNet](https://mcunet.mit.edu) ⭐ 🎓 - MIT framework for deploying neural networks on microcontrollers with kilobytes of SRAM via neural architecture search and TinyEngine.
- [Edge Impulse](https://www.edgeimpulse.com) 🛠️ - End-to-end platform for developing and deploying ML on microcontrollers and edge devices with hardware-in-the-loop testing.
- [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/) 🛠️ - ARM's optimized neural network kernels for Cortex-M microcontrollers using SIMD intrinsics.
- [Apache TVM MicroTVM](https://tvm.apache.org/docs/topic/microtvm/index.html) 🛠️ - TVM sub-project for compiling and deploying ML models on bare-metal microcontrollers without an OS.
- [TinyML Foundation](https://www.tinyml.org) - Community and resource hub for machine learning on extremely constrained devices.
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/) 🛠️ - Optimized ONNX Runtime build for mobile and embedded targets with CoreML and NNAPI execution providers.

## Landmark Papers

Seminal papers that shaped the field of ML Systems Engineering.

- [MapReduce (2004)](https://research.google/pubs/pub62/) ⭐ 🎓 - Google's foundational paper on distributed data processing. Ancestor of modern ML data pipelines and the origin of thinking about distributed computation at scale.
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) ⭐ 🎓 - Transformer architecture paper that defined the dominant computational workload in modern ML systems.
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (2018)](https://arxiv.org/abs/1802.04799) ⭐ 🎓 - Seminal ML compiler paper introducing learning-based auto-tuning for hardware-specific kernel optimization.
- [Ray: A Distributed Framework for Emerging AI Applications (2018)](https://arxiv.org/abs/1712.05889) 🎓 - OSDI paper on the Ray distributed execution framework designed for heterogeneous, dynamic ML workloads.
- [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053) ⭐ 🎓 - Efficient large-scale language model training with intra-layer model parallelism across GPU tensor cores.
- [ZeRO: Memory Optimization Towards Training Trillion Parameter Models (2020)](https://arxiv.org/abs/1910.02054) ⭐ 🎓 - Microsoft DeepSpeed's memory optimization paper partitioning optimizer state, gradients, and parameters across devices.
- [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) ⭐ 🎓 - IO-aware exact attention algorithm that became the standard for efficient Transformer training by minimizing HBM memory traffic.
- [Efficiently Scaling Transformer Inference (2022)](https://arxiv.org/abs/2211.05100) 🎓 - Google's analysis of inference cost and multi-query attention, partition strategies for low-latency production serving.
- [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691) ⭐ 🎓 - Improved IO-aware attention with better parallelism across sequence dimension and work partitioning between warps.
- [vLLM: PagedAttention (2023)](https://arxiv.org/abs/2309.06180) ⭐ 🎓 - High-throughput LLM serving via virtual memory-inspired KV cache management eliminating fragmentation and enabling sharing.
- [Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088) 🎓 - Mistral's sparse MoE model paper with systems implications for expert routing, load balancing, and expert parallelism at scale.

## Blogs and Newsletters

- [Chip Huyen's Blog](https://huyenchip.com/blog/) ⭐ - In-depth posts on ML systems, real-time ML, vector databases, and production ML engineering from a practitioner perspective.
- [Lilian Weng's Blog](https://lilianweng.github.io) ⭐ - Deep technical write-ups on ML research with strong coverage of efficiency, attention mechanisms, and systems topics.
- [Sebastian Raschka's Ahead of AI](https://magazine.sebastianraschka.com) - Practical ML engineering articles on LLM training, fine-tuning, evaluation, and research trends.
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/) - Deep technical posts on GPU computing, CUDA optimization, and ML systems from NVIDIA engineers.
- [PyTorch Blog](https://pytorch.org/blog/) - Official blog covering PyTorch internals, new features, compiler updates, and performance improvements.
- [Modal Blog](https://modal.com/blog) - Practical posts on cloud GPU infrastructure, cold start optimization, and ML deployment engineering.
- [Anyscale Blog](https://www.anyscale.com/blog) - Posts on distributed computing, Ray internals, and large-scale LLM serving infrastructure.
- [The Gradient](https://thegradient.pub) - Long-form technical writing on ML research and systems from researchers and practitioners.

## Communities

- [GPU Mode Discord](https://discord.gg/gpumode) ⭐ - Active community of GPU kernel programmers covering CUDA, Triton, and ML systems. Home of the GPU Mode lecture series.
- [MLOps Community](https://mlops.community) - Slack and podcast community focused on production ML, feature stores, model monitoring, and MLOps best practices.
- [Eleuther AI Discord](https://www.eleutherai.org) - Open-source LLM research community with deep expertise in large-scale distributed training and dataset curation.
- [Hugging Face Forums](https://discuss.huggingface.co) - Community discussions on model optimization, quantization, inference, and ML framework integration.
- [PyTorch Forums](https://discuss.pytorch.org) - Official PyTorch community forum for framework internals, distributed training questions, and debugging.
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - Research-focused ML subreddit with strong coverage of systems papers and conference proceedings.

## Conferences and Venues

Top peer-reviewed venues for ML systems research and engineering.

- [MLSys](https://mlsys.org) ⭐ - The dedicated ML systems conference covering training, inference, compilers, hardware, and data pipelines.
- [OSDI](https://www.usenix.org/conference/osdi24) ⭐ - USENIX operating systems conference. Frequently publishes landmark ML systems papers (Ray, Orca, AlpaServe).
- [SOSP](https://sosp2023.mpi-sws.org) - ACM Symposium on Operating Systems Principles. Top venue for distributed systems and infrastructure research.
- [ASPLOS](https://www.asplos-conference.org) - Architecture, Programming Languages, and Operating Systems. Covers hardware-software co-design for ML accelerators.
- [SC (Supercomputing)](https://supercomputing.org) - The HPC conference covering distributed training at scale, interconnects, and parallel I/O for ML workloads.
- [Hot Chips](https://www.hotchips.org) - Industry symposium on cutting-edge processor and accelerator design from companies and research labs.
- [ISCA](https://iscaconf.org) - International Symposium on Computer Architecture. Covers ML accelerator microarchitecture and memory system design.
- [EuroSys](https://2024.eurosys.org) - European systems conference with a strong distributed ML and storage systems track.
- [NeurIPS (Systems Track)](https://neurips.cc) - ML research conference with a systems-focused track on efficient training, serving, and deployment.

---

## Contribute

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Shashank Tripathi](https://github.com/Shashank-Tripathi-07) has waived all copyright and related or neighboring rights to this work.
