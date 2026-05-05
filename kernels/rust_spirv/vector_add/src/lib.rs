#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(64)))]
pub fn vector_add(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer)] a: &[f32],
    #[spirv(storage_buffer)] b: &[f32],
    #[spirv(storage_buffer)] c: &mut [f32],
) {
    let index = id.x as usize;

    if index < a.len() && index < b.len() && index < c.len() {
        c[index] = a[index] + b[index];
    }
}
