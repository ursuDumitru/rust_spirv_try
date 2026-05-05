/// Generates deterministic floating-point input data for benchmarks.
pub fn generate_vector(size: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };

    (0..size)
        .map(|_| {
            state = next_u64(state);
            let normalized = (state as f64) / (u64::MAX as f64);
            (normalized as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Computes `a + b` on the CPU.
pub fn vector_add_cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "vector_add_cpu requires inputs with the same length"
    );

    a.iter().zip(b).map(|(left, right)| left + right).collect()
}

/// Validates two floating-point vectors with an absolute epsilon.
pub fn validate_approx(actual: &[f32], expected: &[f32], epsilon: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= epsilon)
}

/// Computes a deterministic checksum from the raw bits of floating-point values.
pub fn checksum_f32_bits(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        let mixed = hash ^ u64::from(value.to_bits());
        mixed.wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn next_u64(mut state: u64) -> u64 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_vector_is_deterministic() {
        let first = generate_vector(16, 42);
        let second = generate_vector(16, 42);

        assert_eq!(first, second);
    }

    #[test]
    fn vector_add_cpu_adds_matching_vectors() {
        let a = vec![1.0, -2.5, 3.25];
        let b = vec![4.0, 2.0, -1.25];

        assert_eq!(vector_add_cpu(&a, &b), vec![5.0, -0.5, 2.0]);
    }

    #[test]
    fn validate_approx_accepts_values_within_epsilon() {
        let actual = vec![1.0, 2.0001, 2.9999];
        let expected = vec![1.0, 2.0, 3.0];

        assert!(validate_approx(&actual, &expected, 0.001));
    }

    #[test]
    fn validate_approx_rejects_values_outside_epsilon() {
        let actual = vec![1.0, 2.1, 3.0];
        let expected = vec![1.0, 2.0, 3.0];

        assert!(!validate_approx(&actual, &expected, 0.001));
    }

    #[test]
    fn validate_approx_rejects_length_mismatch() {
        let actual = vec![1.0, 2.0];
        let expected = vec![1.0];

        assert!(!validate_approx(&actual, &expected, 0.001));
    }

    #[test]
    #[should_panic(expected = "vector_add_cpu requires inputs with the same length")]
    fn vector_add_cpu_panics_on_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];

        let _ = vector_add_cpu(&a, &b);
    }

    #[test]
    fn checksum_f32_bits_is_deterministic() {
        let values = vec![1.0, -2.0, 3.5];

        assert_eq!(checksum_f32_bits(&values), checksum_f32_bits(&values));
    }

    #[test]
    fn checksum_f32_bits_changes_when_input_changes() {
        let first = vec![1.0, -2.0, 3.5];
        let second = vec![1.0, -2.0, 3.25];

        assert_ne!(checksum_f32_bits(&first), checksum_f32_bits(&second));
    }

    #[test]
    fn checksum_f32_bits_empty_input_is_stable() {
        assert_eq!(checksum_f32_bits(&[]), checksum_f32_bits(&[]));
    }
}
