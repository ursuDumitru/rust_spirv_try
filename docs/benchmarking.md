# Benchmarking

The first benchmark runner measures the CPU vector-add baseline. It exists to stabilize the result format before adding Rust/SPIR-V, CUDA, and Python backends.

Run the default vector-add benchmark with:

```bash
cargo run -p rust_gpu_runner -- vector-add --size 1048576 --warmups 3 --runs 10 --seed 42 --out benchmarks/results/vector_add_cpu.csv
```

For a short verification run, use:

```bash
cargo run -p rust_gpu_runner -- vector-add --size 1024 --warmups 1 --runs 2 --seed 42 --out benchmarks/results/vector_add_cpu.csv
```

The CSV schema is:

```text
backend,kernel,size,seed,warmups,runs,run_index,duration_ns,checksum,valid
```

For now, `backend` is always `cpu` and `kernel` is always `vector_add`. The `valid` column must be `true`, meaning the measured output matched the CPU reference within the configured epsilon.
