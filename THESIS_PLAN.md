# Thesis Implementation Plan

## Goal

The goal is to evaluate Rust as a source language for GPU compute by compiling Rust kernels to SPIR-V, analyzing the generated IR, and comparing performance against CUDA and Python-based baselines.

The expected contribution is not a new compiler. It is a reproducible toolchain, a small benchmark suite, an analyzer for SPIR-V output, and a written evaluation of where Rust-to-SPIR-V is strong or weak.

## Target Repository Layout

```text
rust_gpu/
|-- Cargo.toml
|-- start.md
|-- THESIS_PLAN.md
|-- crates/
|   |-- analyzer/          # SPIR-V parser and report generator
|   |-- runner/            # Benchmark launcher and result collector
|   `-- common/            # Shared data generation and validation code
|-- kernels/
|   |-- rust_spirv/        # Rust GPU kernels compiled to SPIR-V
|   |-- cuda/              # CUDA C++ versions of the same kernels
|   `-- python/            # NumPy, CuPy, or Numba benchmark baselines
|-- benchmarks/
|   |-- configs/           # Problem sizes and benchmark settings
|   |-- results/           # Raw benchmark outputs
|   `-- scripts/           # Plotting and result aggregation
|-- docs/
|   |-- methodology.md
|   |-- toolchain.md
|   `-- results.md
`-- reports/
    |-- spirv/             # Analyzer reports
    `-- figures/           # Charts used in the thesis
```

The current repository can start simple, then grow toward this layout when the first kernel and analyzer are working.

## Work Plan

### 1. Define The Research Question

Start with one precise question: how does Rust-generated SPIR-V compare with CUDA and Python GPU baselines for common compute kernels?

The first benchmark set should be vector addition, reduction, stencil, and matrix multiplication. These cover memory bandwidth, synchronization-like patterns, neighborhood access, and compute-heavy work.

### 2. Stabilize The Toolchain

Create a documented environment using Rust nightly, `rustc_codegen_spirv`, `spirv-builder`, `spirv-tools`, a GPU execution backend such as `wgpu`, and CUDA for comparison.

The repository should include exact install notes, version numbers, GPU model, driver version, and commands needed to rebuild all generated artifacts.

### 3. Build Rust-To-SPIR-V Kernels

Implement each kernel in Rust first and compile it to `.spv`. Keep inputs deterministic and validate every GPU result against a CPU reference implementation.

Store generated SPIR-V artifacts outside source directories so benchmark results can be regenerated cleanly.

### 4. Build CUDA Benchmarks

Implement CUDA C++ versions of the same kernels with the same data types, problem sizes, and correctness checks.

CUDA is the performance reference, not the thesis target. Its role is to show how far the Rust/SPIR-V path is from a mature GPU programming stack.

### 5. Build Python Benchmarks

Add Python baselines for developer productivity and ecosystem comparison. Use NumPy for CPU reference behavior, then CuPy or Numba for GPU execution if available.

Python results should be interpreted carefully because they measure both library maturity and runtime overhead, not just kernel quality.

### 6. Implement The SPIR-V Analyzer

Build a Rust analyzer that reads `.spv` files and emits structured reports. Start with instruction counts, memory operation counts, branch counts, function counts, and capability usage.

After the basic report works, add focused checks for redundant loads/stores, suspicious control flow, and differences before and after `spirv-opt`.

### 7. Build The Benchmark Runner

Create one command that runs all selected kernels across Rust/SPIR-V, CUDA, and Python implementations. It should record warmup runs, measured runs, problem size, hardware, tool versions, and output hashes.

Results should be written as machine-readable files such as JSON or CSV so charts can be regenerated.

### 8. Analyze Results

Compare runtime, throughput, instruction counts, and code complexity. Look for cases where the analyzer explains a performance result, such as extra memory operations or more branching.

The strongest thesis result is a link between generated SPIR-V structure and observed benchmark behavior.

### 9. Write The Thesis

The thesis should present the motivation, GPU compilation background, implementation, benchmark methodology, results, limitations, and future work.

The core argument should be practical: Rust-to-SPIR-V is promising, but developers need better analysis and benchmarking tools to understand generated GPU code.

## Quality Rules

Every benchmark must have a CPU correctness reference, fixed input generation, warmup runs, repeated measurements, and saved raw results.

Every chart in the thesis should be reproducible from files in the repository. Any generated SPIR-V or benchmark output should be rebuildable from source.

## First Milestone

The first milestone is a complete vector-add pipeline: Rust kernel to SPIR-V, CUDA equivalent, Python baseline, correctness validation, benchmark timing, SPIR-V disassembly, and one analyzer report.

## Optional Final Extension

If the main pipeline is complete, add a small machine learning training demo that uses the Rust GPU kernels for core numerical operations. A realistic target is linear regression, logistic regression, or a tiny multilayer perceptron with manually implemented matrix multiplication, reductions, loss computation, and parameter updates.

This extension should compare Rust GPU execution against CPU Rust, Python/NumPy, and optionally PyTorch/CUDA. The goal is not to build a full ML framework, but to show that the kernels and benchmark infrastructure can accelerate a real training-style workload.
