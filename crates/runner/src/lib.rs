use std::fmt;
use std::path::PathBuf;
use std::time::Instant;

use rust_gpu_common::{checksum_f32_bits, generate_vector, validate_approx, vector_add_cpu};

const DEFAULT_SIZE: usize = 1_048_576;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_RUNS: usize = 10;
const DEFAULT_SEED: u64 = 42;
const VALIDATION_EPSILON: f32 = 0.0001;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerConfig {
    pub command: Command,
    pub size: usize,
    pub warmups: usize,
    pub runs: usize,
    pub seed: u64,
    pub out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    VectorAdd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvRow {
    pub backend: &'static str,
    pub kernel: &'static str,
    pub size: usize,
    pub seed: u64,
    pub warmups: usize,
    pub runs: usize,
    pub run_index: usize,
    pub duration_ns: u128,
    pub checksum: u64,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerError {
    message: String,
}

impl RunnerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RunnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RunnerError {}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            command: Command::VectorAdd,
            size: DEFAULT_SIZE,
            warmups: DEFAULT_WARMUPS,
            runs: DEFAULT_RUNS,
            seed: DEFAULT_SEED,
            out: None,
        }
    }
}

pub fn parse_args<I, S>(args: I) -> Result<RunnerConfig, RunnerError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let command = args
        .next()
        .ok_or_else(|| RunnerError::new("missing command: expected `vector-add`"))?;

    if command != "vector-add" {
        return Err(RunnerError::new(format!(
            "unsupported command `{command}`: expected `vector-add`"
        )));
    }

    let mut config = RunnerConfig::default();

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--size" => config.size = parse_value(&flag, args.next())?,
            "--warmups" => config.warmups = parse_value(&flag, args.next())?,
            "--runs" => config.runs = parse_value(&flag, args.next())?,
            "--seed" => config.seed = parse_value(&flag, args.next())?,
            "--out" => {
                let path = args
                    .next()
                    .ok_or_else(|| RunnerError::new("missing value for --out"))?;
                config.out = Some(PathBuf::from(path));
            }
            _ => return Err(RunnerError::new(format!("unknown flag `{flag}`"))),
        }
    }

    Ok(config)
}

pub fn run_vector_add(config: &RunnerConfig) -> Vec<CsvRow> {
    let a = generate_vector(config.size, config.seed);
    let b = generate_vector(config.size, config.seed.wrapping_add(1));
    let expected = vector_add_cpu(&a, &b);

    for _ in 0..config.warmups {
        let output = vector_add_cpu(&a, &b);
        std::hint::black_box(output);
    }

    (0..config.runs)
        .map(|run_index| {
            let started = Instant::now();
            let output = vector_add_cpu(&a, &b);
            let duration_ns = started.elapsed().as_nanos();
            let checksum = checksum_f32_bits(&output);
            let valid = validate_approx(&output, &expected, VALIDATION_EPSILON);

            CsvRow {
                backend: "cpu",
                kernel: "vector_add",
                size: config.size,
                seed: config.seed,
                warmups: config.warmups,
                runs: config.runs,
                run_index,
                duration_ns,
                checksum,
                valid,
            }
        })
        .collect()
}

pub fn format_csv(rows: &[CsvRow]) -> String {
    let mut output =
        String::from("backend,kernel,size,seed,warmups,runs,run_index,duration_ns,checksum,valid\n");

    for row in rows {
        output.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            row.backend,
            row.kernel,
            row.size,
            row.seed,
            row.warmups,
            row.runs,
            row.run_index,
            row.duration_ns,
            row.checksum,
            row.valid
        ));
    }

    output
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, RunnerError>
where
    T: std::str::FromStr,
{
    value
        .ok_or_else(|| RunnerError::new(format!("missing value for {flag}")))?
        .parse()
        .map_err(|_| RunnerError::new(format!("invalid value for {flag}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_uses_defaults() {
        let config = parse_args(["vector-add"]).expect("valid config");

        assert_eq!(config, RunnerConfig::default());
    }

    #[test]
    fn parse_args_accepts_custom_values() {
        let config = parse_args([
            "vector-add",
            "--size",
            "1024",
            "--warmups",
            "1",
            "--runs",
            "2",
            "--seed",
            "7",
            "--out",
            "benchmarks/results/out.csv",
        ])
        .expect("valid config");

        assert_eq!(config.size, 1024);
        assert_eq!(config.warmups, 1);
        assert_eq!(config.runs, 2);
        assert_eq!(config.seed, 7);
        assert_eq!(
            config.out,
            Some(PathBuf::from("benchmarks/results/out.csv"))
        );
    }

    #[test]
    fn parse_args_rejects_invalid_command() {
        let error = parse_args(["matrix-mul"]).expect_err("invalid command");

        assert_eq!(
            error.to_string(),
            "unsupported command `matrix-mul`: expected `vector-add`"
        );
    }

    #[test]
    fn format_csv_writes_header_and_rows() {
        let rows = vec![CsvRow {
            backend: "cpu",
            kernel: "vector_add",
            size: 1024,
            seed: 42,
            warmups: 1,
            runs: 2,
            run_index: 0,
            duration_ns: 123,
            checksum: 456,
            valid: true,
        }];

        assert_eq!(
            format_csv(&rows),
            "backend,kernel,size,seed,warmups,runs,run_index,duration_ns,checksum,valid\ncpu,vector_add,1024,42,1,2,0,123,456,true\n"
        );
    }
}
