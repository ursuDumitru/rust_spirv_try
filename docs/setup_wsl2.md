# WSL2 Setup Guide

This guide prepares an Ubuntu WSL2 environment for the Rust GPU thesis project. The first project step only needs Rust and the CPU baseline tests, but the same environment will later support SPIR-V tools, Python benchmarks, and optional CUDA comparisons.

## WSL2 And Ubuntu

Install WSL2 and Ubuntu from Windows PowerShell:

```powershell
wsl --install -d Ubuntu
```

After Ubuntu starts, update the system packages:

```bash
sudo apt update
sudo apt install -y build-essential curl git pkg-config libssl-dev python3 python3-pip python3-venv
```

## Rust And Nightly

Install Rust through `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Install nightly Rust for the future Rust GPU toolchain, while keeping stable as the default toolchain:

```bash
rustup toolchain install nightly
rustup default stable
rustup override set nightly
```

Check the active toolchain:

```bash
rustc --version
cargo --version
rustup show
```

The `rustup override set nightly` command applies to the current repository directory. Run it from the repository root.

## SPIR-V Tools

Install SPIR-V command-line tools:

```bash
sudo apt install -y spirv-tools
spirv-dis --version
```

These tools are not required for the first CPU baseline, but they will be needed once the project starts generating and inspecting `.spv` files.

## Python Tools

Create a local Python virtual environment for future Python benchmarks:

```bash
python3 -m venv rust_gpu_env
source rust_gpu_env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
```

Later GPU Python baselines can use CuPy or Numba, depending on the available GPU and CUDA setup.

## Optional CUDA Setup

CUDA is optional for the first step. It becomes necessary later when comparing Rust/SPIR-V benchmarks against CUDA C++.

CUDA under WSL2 requires a compatible NVIDIA GPU, a Windows NVIDIA driver with WSL support, and CUDA tools available inside Ubuntu. Verify GPU visibility with:

```bash
nvidia-smi
```

If the CUDA compiler is installed inside WSL2, verify it with:

```bash
nvcc --version
```

Do not block the first milestone on CUDA. The CPU baseline and repository setup should work without it.

## First Repository Commands

From the repository root, verify the Rust workspace:

```bash
cargo test
```

This command should pass before adding GPU kernels, CUDA benchmarks, Python benchmarks, or SPIR-V analysis.
