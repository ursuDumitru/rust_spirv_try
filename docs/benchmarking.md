# Benchmarking

The first benchmark runner measures the CPU vector-add baseline. It exists to stabilize the result format before adding Rust/SPIR-V, CUDA, and Python backends.

Run the default Rust CPU vector-add benchmark with:

```bash
cargo run -p rust_gpu_runner -- vector-add --size 1048576 --warmups 3 --runs 10 --seed 42 --out benchmarks/results/vector_add_cpu.csv
```

For a short Rust CPU verification run, use:

```bash
cargo run -p rust_gpu_runner -- vector-add --size 1024 --warmups 1 --runs 2 --seed 42 --out benchmarks/results/vector_add_cpu.csv
```

Run the default Python NumPy vector-add benchmark with:

```bash
rust_gpu_env/bin/python kernels/python/vector_add.py --size 1048576 --warmups 3 --runs 10 --seed 42 --out benchmarks/results/vector_add_numpy.csv
```

For a short Python NumPy verification run, use:

```bash
rust_gpu_env/bin/python kernels/python/vector_add.py --size 1024 --warmups 1 --runs 2 --seed 42 --out benchmarks/results/vector_add_numpy.csv
```

Build the Rust/SPIR-V vector-add kernel and disassembly with:

```bash
cargo run -p rust_gpu_spirv_build -- vector-add
```

The workspace command runs a small Rust-GPU builder with the exact toolchain required by `spirv-builder`.

This writes:

```text
reports/spirv/vector_add.spv
reports/spirv/vector_add.spvasm
```

Validate the generated SPIR-V with:

```bash
spirv-val reports/spirv/vector_add.spv
```

The CSV schema is:

```text
backend,kernel,size,seed,warmups,runs,run_index,duration_ns,checksum,valid
```

For now, `backend` is either `cpu` or `numpy`, and `kernel` is always `vector_add`. The `valid` column must be `true`, meaning the measured output matched the reference result within the configured epsilon.
