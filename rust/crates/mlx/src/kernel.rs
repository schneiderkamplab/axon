//! Custom Metal kernels via the MLX C API (`mx.fast.metal_kernel` parity).

use std::ffi::CString;

use mlx_c_sys as sys;

use crate::{check, take_error, Array, Dtype, Error, Result, Stream};

/// One output of a kernel application: shape + dtype.
#[derive(Clone)]
pub struct OutputSpec {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl OutputSpec {
    pub fn new(shape: &[usize], dtype: Dtype) -> OutputSpec {
        OutputSpec {
            shape: shape.to_vec(),
            dtype,
        }
    }
}

/// A registered custom Metal kernel (owning handle).
pub struct MetalKernel {
    raw: sys::mlx_fast_metal_kernel,
    _name: CString,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            unsafe {
                sys::mlx_fast_metal_kernel_free(self.raw);
            }
        }
    }
}

impl MetalKernel {
    /// Register a kernel from an MSL source string.
    ///
    /// Mirrors `mx.fast.metal_kernel(name, input_names, output_names, source,
    /// header="", ensure_row_contiguous=True, atomic_outputs=False)`.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
    ) -> Result<MetalKernel> {
        let cname = CString::new(name).map_err(|_| Error("kernel name contains NUL".into()))?;
        let csource =
            CString::new(source).map_err(|_| Error("kernel source contains NUL".into()))?;
        let empty_header = CString::new("").unwrap();
        let inputs = unsafe { sys::mlx_vector_string_new() };
        let outputs = unsafe { sys::mlx_vector_string_new() };
        unsafe {
            let append = |vec: sys::mlx_vector_string, name: &str| -> Result<()> {
                let c = CString::new(name).map_err(|_| Error("name contains NUL".into()))?;
                check(sys::mlx_vector_string_append_value(vec, c.as_ptr()))
            };
            let cleanup = |inputs: sys::mlx_vector_string, outputs: sys::mlx_vector_string| {
                sys::mlx_vector_string_free(inputs);
                sys::mlx_vector_string_free(outputs);
            };
            for n in input_names {
                if let Err(e) = append(inputs, n) {
                    cleanup(inputs, outputs);
                    return Err(e);
                }
            }
            for n in output_names {
                if let Err(e) = append(outputs, n) {
                    cleanup(inputs, outputs);
                    return Err(e);
                }
            }
            let raw = sys::mlx_fast_metal_kernel_new(
                cname.as_ptr(),
                inputs,
                outputs,
                csource.as_ptr(),
                empty_header.as_ptr(),
                true,  // ensure_row_contiguous (Python default)
                false, // atomic_outputs
            );
            cleanup(inputs, outputs);
            if raw.ctx.is_null() {
                return Err(Error(
                    take_error().unwrap_or_else(|| format!("failed to register kernel {name}")),
                ));
            }
            Ok(MetalKernel {
                raw,
                _name: cname,
            })
        }
    }

    /// Apply the kernel (lazy; eval the outputs to run).
    pub fn apply(
        &self,
        inputs: &[&Array],
        outputs: &[OutputSpec],
        grid: (i32, i32, i32),
        threadgroup: (i32, i32, i32),
        s: Stream,
    ) -> Result<Vec<Array>> {
        let config = unsafe { sys::mlx_fast_metal_kernel_config_new() };
        unsafe {
            let free_config = |config: sys::mlx_fast_metal_kernel_config| {
                sys::mlx_fast_metal_kernel_config_free(config);
            };
            for out in outputs {
                let shape: Vec<i32> = out.shape.iter().map(|&d| d as i32).collect();
                if let Err(e) = check(sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    shape.as_ptr(),
                    shape.len(),
                    out.dtype.raw(),
                )) {
                    free_config(config);
                    return Err(e);
                }
            }
            if let Err(e) = check(sys::mlx_fast_metal_kernel_config_set_grid(
                config, grid.0, grid.1, grid.2,
            )) {
                free_config(config);
                return Err(e);
            }
            if let Err(e) = check(sys::mlx_fast_metal_kernel_config_set_thread_group(
                config, threadgroup.0, threadgroup.1, threadgroup.2,
            )) {
                free_config(config);
                return Err(e);
            }

            let handles: Vec<sys::mlx_array> = inputs.iter().map(|a| a.raw()).collect();
            let in_vec = sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len());
            let mut out_vec = sys::mlx_vector_array::default();
            let status = sys::mlx_fast_metal_kernel_apply(
                &mut out_vec,
                self.raw,
                in_vec,
                config,
                s.raw(),
            );
            sys::mlx_vector_array_free(in_vec);
            sys::mlx_fast_metal_kernel_config_free(config);
            check(status)?;

            let n = sys::mlx_vector_array_size(out_vec);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let mut handle = sys::mlx_array::empty();
                check(sys::mlx_vector_array_get(&mut handle, out_vec, i))?;
                result.push(Array::from_raw(handle));
            }
            sys::mlx_vector_array_free(out_vec);
            Ok(result)
        }
    }
}
