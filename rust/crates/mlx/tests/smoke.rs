//! Phase-1 smoke gate: the C API bindings actually drive MLX — array ops,
//! a custom Metal kernel, and mx.compile — from Rust, no Python.

use mlx::compile::{CompiledFn, GraphFn};
use mlx::kernel::{MetalKernel, OutputSpec};
use mlx::{init, add, argmax, multiply, Array, Dtype, Stream};

const SAXPY_SOURCE: &str = r#"
    uint index = thread_position_in_grid.x;
    if (index >= x_shape[0]) { return; }
    out[index] = (float16_t)((float)x[index] * alpha);
"#;

#[test]
fn array_ops_roundtrip() {
    let _mlx = init();
    let s = Stream::default_stream();

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], Dtype::F32, s).unwrap();
    let two = Array::scalar_f32(2.0, s).unwrap();
    let doubled = multiply(&a, &two, s).unwrap();
    doubled.eval().unwrap();
    assert_eq!(doubled.data_f32().unwrap(), &[2.0, 4.0, 6.0, 8.0]);

    let b = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], Dtype::F32, s).unwrap();
    let sum = add(&a, &b, s).unwrap();
    sum.eval().unwrap();
    assert_eq!(sum.data_f32().unwrap(), &[11.0, 22.0, 33.0, 44.0]);

    let flat = mlx::reshape(&sum, &[2, 2], s).unwrap();
    assert_eq!(flat.shape(), vec![2, 2]);
    assert_eq!(flat.ndim(), 2);
    assert_eq!(flat.dtype(), Dtype::F32);
}

#[test]
fn metal_kernel_saxpy() {
    let _mlx = init();
    let s = Stream::default_stream();

    let kernel = MetalKernel::new(
        "saxpy_test",
        &["x", "alpha"],
        &["out"],
        SAXPY_SOURCE,
    )
    .unwrap();

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], Dtype::F32, s).unwrap();
    let alpha = Array::scalar_f32(3.0, s).unwrap();

    let out = kernel
        .apply(
            &[&x, &alpha],
            &[OutputSpec::new(&[5], Dtype::F32)],
            (5, 1, 1),
            (32, 1, 1),
            s,
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    out[0].eval().unwrap();
    assert_eq!(out[0].data_f32().unwrap(), &[3.0, 6.0, 9.0, 12.0, 15.0]);
}

#[test]
fn argmax_host_readback() {
    let _mlx = init();
    let s = Stream::default_stream();
    let a = Array::from_slice(&[1.0f32, 5.0, 3.0], &[3], Dtype::F32, s).unwrap();
    let idx = argmax(&a, s).unwrap();
    idx.eval().unwrap();
    assert_eq!(idx.item_i32().unwrap(), 1);
}

/// compiled closure: out = x*2 + y  (exercises the C closure trampoline)
struct AddMul;

impl GraphFn for AddMul {
    fn build(&self, inputs: &[Array], out: &mut Vec<Array>, s: Stream) {
        let x = &inputs[0];
        let y = &inputs[1];
        let two = Array::scalar_f32(2.0, s).unwrap();
        let x2 = multiply(x, &two, s).unwrap();
        let sum = add(&x2, y, s).unwrap();
        out.push(sum);
    }
}

#[test]
fn compile_closure() {
    let _mlx = init();
    let s = Stream::default_stream();

    let compiled = CompiledFn::compile(AddMul).unwrap();
    // apply twice with different values: compile must not bake constants in
    for (xv, yv) in [(1.0f32, 10.0), (7.0, 0.5)] {
        let x = Array::from_slice(&[xv], &[1], Dtype::F32, s).unwrap();
        let y = Array::from_slice(&[yv], &[1], Dtype::F32, s).unwrap();
        let out = compiled.apply(&[&x, &y], s).unwrap();
        assert_eq!(out.len(), 1);
        out[0].eval().unwrap();
        assert_eq!(out[0].data_f32().unwrap(), &[xv * 2.0 + yv]);
    }
}
