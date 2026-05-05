# Rust GPU + SPIR-V Thesis Overview

## Big Picture

This thesis studies how Rust GPU kernels can be compiled to SPIR-V, how good the generated GPU code is, and where the compilation pipeline can be analyzed or improved.

```text
Rust code -> compiler/backend -> SPIR-V -> GPU driver -> GPU
```

Rust is the source language, SPIR-V is the GPU intermediate representation, and the driver is responsible for final execution on the target GPU.

## What SPIR-V Is

SPIR-V is a binary intermediate representation for GPU programs. It is low-level, structured, cross-vendor, and used by ecosystems such as Vulkan and WebGPU.

SPIR-V represents parallel execution, memory operations, arithmetic instructions, and control flow. It is close enough to the hardware to expose performance problems, but portable enough to compare across GPU vendors.

## CUDA vs. SPIR-V

The CUDA pipeline is NVIDIA-specific:

```text
CUDA C++ -> NVVM IR -> PTX -> GPU
```

The SPIR-V pipeline is designed to be cross-vendor:

```text
Rust / GLSL -> SPIR-V -> GPU
```

Conceptually, PTX and SPIR-V both act as intermediate GPU representations, but CUDA is tied to NVIDIA while SPIR-V can target NVIDIA, AMD, Intel, and other Vulkan/WebGPU-capable devices.

## Where Rust Fits

Rust is used to write GPU kernels and compile them into SPIR-V. After compilation, the executable GPU artifact is SPIR-V, not Rust source code.

The relevant Rust-side components are `rustc`, `rustc_codegen_spirv`, `rust-gpu`, and helper crates such as `spirv-builder`.

## Example GPU Kernel

```rust
#[spirv(compute(threads(64)))]
pub fn vector_add(
    #[spirv(global_invocation_id)] id: UVec3,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let i = id.x as usize;
    c[i] = a[i] + b[i];
}
```

Each GPU thread computes one output element. The `global_invocation_id` gives the thread index used to select the current element.

## Execution Model

A CPU program launches GPU work by creating a GPU context, uploading buffers, loading the SPIR-V kernel, dispatching GPU threads, and reading the result back.

```text
1. Create GPU context
2. Upload input buffers
3. Load SPIR-V kernel
4. Dispatch threads
5. Read output buffers
```

## Toolchain

The minimal toolchain is Rust nightly, `rustc_codegen_spirv`, `spirv-builder`, `spirv-tools`, and `wgpu` or another Vulkan/WebGPU execution layer.

CUDA is useful for comparison, and tools such as Nsight, RenderDoc, `spirv-dis`, and `spirv-opt` can help with profiling and IR inspection.

## Project Scope

This thesis should not attempt to build a compiler frontend, reimplement Rust, or write GPU drivers. The valuable work sits between Rust kernels and the generated SPIR-V.

```text
Rust -> SPIR-V
        ^
        thesis work
```

The project should write representative GPU kernels, compile them to SPIR-V, analyze the generated IR, detect inefficiencies, benchmark runtime behavior, and propose focused improvements.

## Thesis Directions

### SPIR-V Analysis Tool

The recommended core contribution is a tool that reads `.spv` files, inspects instructions, and reports patterns that are relevant to performance.

Useful checks include redundant loads and stores, excessive branching, suspicious memory access patterns, high instruction counts, and differences between debug and optimized output.

### Performance Study

The benchmark study should compare Rust-to-SPIR-V kernels against CUDA C++ and Python-based baselines.

Relevant metrics include execution time, memory bandwidth, instruction count, dispatch overhead, compile complexity, and correctness against known CPU results.

### Source-Level Optimization

Some improvements can be tested by changing the Rust kernel source before compilation. Examples include loop unrolling, indexing changes, tiling, shared-memory-style patterns where supported, and reducing unnecessary bounds checks or branches.

### SPIR-V Optimization

Direct SPIR-V optimization is more advanced. Possible work includes dead-code cleanup, instruction simplification, and comparing `spirv-opt` output against the original generated SPIR-V.

## Code To Build

The first benchmark set should stay small and defensible:

```text
vector add
reduction
stencil
matrix multiply
```

The supporting code should include a build wrapper that produces `.spv` files, a SPIR-V analyzer that generates reports, and a benchmark harness that runs kernels and stores reproducible measurements.

## Difficulty Breakdown

| Component | Difficulty |
| --- | --- |
| Writing kernels | Easy |
| Using the toolchain | Medium |
| Understanding SPIR-V | Medium |
| Building the analyzer | Medium |
| Benchmarking correctly | Medium |
| Writing optimization passes | Hard |

## Recommended Environment

The safest setup is Linux or WSL2 with Rust nightly, the Vulkan SDK, `spirv-tools`, CUDA where available, and a fixed GPU configuration for repeatable measurements.

Native Windows is possible, but it can make low-level GPU tooling and reproducible benchmarking harder.

## Key Insight

SPIR-V is mature, but Rust GPU tooling is still relatively young. That gap makes the thesis worthwhile: the project can produce useful evidence about where Rust GPU code generation works well, where it struggles, and what kinds of analysis help developers understand the output.

## Final Summary

Rust is the frontend language, SPIR-V is the execution format, and the GPU runs the parallel kernels. The thesis contribution should be a practical analysis and benchmarking workflow for understanding and improving GPU code generated from Rust.
