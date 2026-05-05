use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use spirv_builder::{MetadataPrintout, ModuleResult, SpirvBuilder};

const TARGET: &str = "spirv-unknown-vulkan1.2";

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
    let shader_crate = root.join("kernels/rust_spirv/vector_add");
    let report_dir = root.join("reports/spirv");
    let spv_out = report_dir.join("vector_add.spv");
    let spvasm_out = report_dir.join("vector_add.spvasm");

    fs::create_dir_all(&report_dir)?;

    let result = SpirvBuilder::new(&shader_crate, TARGET)
        .print_metadata(MetadataPrintout::None)
        .release(true)
        .build()?;
    let module_path = module_path(result.module)?;

    fs::copy(&module_path, &spv_out)?;
    disassemble(&spv_out, &spvasm_out)?;

    println!("wrote {}", spv_out.display());
    println!("wrote {}", spvasm_out.display());

    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or("failed to find workspace root")?;

    Ok(root.to_path_buf())
}

fn module_path(module: ModuleResult) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match module {
        ModuleResult::SingleModule(path) => Ok(path),
        ModuleResult::MultiModule(mut modules) => modules
            .remove("vector_add")
            .or_else(|| modules.into_values().next())
            .ok_or_else(|| "SPIR-V build did not produce any modules".into()),
    }
}

fn disassemble(spv_path: &Path, spvasm_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("spirv-dis")
        .arg(spv_path)
        .arg("-o")
        .arg(spvasm_path)
        .status()?;

    if !status.success() {
        return Err(format!("spirv-dis failed with status {status}").into());
    }

    Ok(())
}
