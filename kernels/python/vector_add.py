#!/usr/bin/env python3

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_SIZE = 1_048_576
DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 10
DEFAULT_SEED = 42
VALIDATION_EPSILON = np.float32(0.0001)

U64_MASK = (1 << 64) - 1
U64_MAX = float(U64_MASK)
ZERO_SEED_FALLBACK = 0x9E37_79B9_7F4A_7C15
XORSHIFT_MULTIPLIER = 0x2545_F491_4F6C_DD1D
CHECKSUM_OFFSET = 0xCBF2_9CE4_8422_2325
CHECKSUM_MULTIPLIER = 0x0000_0100_0000_01B3

CSV_FIELDS = [
    "backend",
    "kernel",
    "size",
    "seed",
    "warmups",
    "runs",
    "run_index",
    "duration_ns",
    "checksum",
    "valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy vector-add benchmark")
    parser.add_argument("--size", type=parse_usize, default=DEFAULT_SIZE)
    parser.add_argument("--warmups", type=parse_usize, default=DEFAULT_WARMUPS)
    parser.add_argument("--runs", type=parse_usize, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=parse_u64, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def parse_usize(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def parse_u64(value: str) -> int:
    parsed = parse_usize(value)
    if parsed > U64_MASK:
        raise argparse.ArgumentTypeError("expected a value that fits in u64")
    return parsed


def generate_vector(size: int, seed: int) -> np.ndarray:
    state = ZERO_SEED_FALLBACK if seed == 0 else seed & U64_MASK
    values = np.empty(size, dtype=np.float32)

    for index in range(size):
        state = next_u64(state)
        normalized = float(state) / U64_MAX
        values[index] = np.float32(
            np.float32(normalized) * np.float32(2.0) - np.float32(1.0)
        )

    return values


def next_u64(state: int) -> int:
    state = (state ^ (state >> 12)) & U64_MASK
    state = (state ^ ((state << 25) & U64_MASK)) & U64_MASK
    state = (state ^ (state >> 27)) & U64_MASK
    return (state * XORSHIFT_MULTIPLIER) & U64_MASK


def checksum_f32_bits(values: np.ndarray) -> int:
    hash_value = CHECKSUM_OFFSET

    for bits in values.astype(np.float32, copy=False).view(np.uint32):
        mixed = hash_value ^ int(bits)
        hash_value = (mixed * CHECKSUM_MULTIPLIER) & U64_MASK

    return hash_value


def validate_approx(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return False

    return bool(np.all(np.abs(actual - expected) <= VALIDATION_EPSILON))


def run_vector_add(args: argparse.Namespace) -> list[dict[str, object]]:
    a = generate_vector(args.size, args.seed)
    b = generate_vector(args.size, (args.seed + 1) & U64_MASK)
    expected = a + b

    for _ in range(args.warmups):
        np.add(a, b)

    rows = []

    for run_index in range(args.runs):
        started = time.perf_counter_ns()
        output = np.add(a, b)
        duration_ns = time.perf_counter_ns() - started

        rows.append(
            {
                "backend": "numpy",
                "kernel": "vector_add",
                "size": args.size,
                "seed": args.seed,
                "warmups": args.warmups,
                "runs": args.runs,
                "run_index": run_index,
                "duration_ns": duration_ns,
                "checksum": checksum_f32_bits(output),
                "valid": str(validate_approx(output, expected)).lower(),
            }
        )

    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path | None) -> None:
    if output_path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = run_vector_add(args)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
