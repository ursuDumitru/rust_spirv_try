use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const RUST_GPU_TOOLCHAIN: &str = "nightly-2023-05-27";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let command = env::args()
        .nth(1)
        .ok_or("missing command: expected `vector-add`")?;

    if command != "vector-add" {
        return Err(format!("unsupported command `{command}`: expected `vector-add`").into());
    }

    let root = workspace_root()?;
    let builder_manifest = root.join("kernels/rust_spirv/build_vector_add/Cargo.toml");
    let status = Command::new(rustup_command())
        .arg("run")
        .arg(RUST_GPU_TOOLCHAIN)
        .arg("cargo")
        .arg("run")
        .arg("--manifest-path")
        .arg(builder_manifest)
        .arg("--")
        .arg("vector-add")
        .status()?;

    if !status.success() {
        return Err(format!("Rust-GPU builder failed with status {status}").into());
    }

    Ok(())
}

fn rustup_command() -> PathBuf {
    if let Some(rustup) = env::var_os("RUSTUP") {
        return PathBuf::from(rustup);
    }

    if let Some(home) = env::var_os("HOME") {
        let rustup = PathBuf::from(home).join(".cargo/bin/rustup");
        if rustup.exists() {
            return rustup;
        }
    }

    PathBuf::from("rustup")
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("failed to find workspace root")?;

    Ok(root.to_path_buf())
}
