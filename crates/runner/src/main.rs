use std::env;
use std::fs;
use std::io::{self, Write};

use rust_gpu_runner::{format_csv, parse_args, run_vector_add};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1))?;
    let rows = run_vector_add(&config);
    let csv = format_csv(&rows);

    if let Some(path) = &config.out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, csv)?;
    } else {
        io::stdout().write_all(csv.as_bytes())?;
    }

    Ok(())
}
