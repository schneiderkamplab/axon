//! `mx.compile` parity: compile a Rust graph-builder closure over MLX arrays.

use std::os::raw::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use mlx_c_sys as sys;

use crate::{check, take_error, Array, Error, Result, Stream};

/// A graph builder: receives the closure inputs and appends the closure
/// outputs. Must be panic-free (a `catch_unwind` guard turns panics into MLX
/// errors, but the graph code itself never expects to unwind) and must not
/// free its inputs.
pub trait GraphFn: Send + Sync + 'static {
    fn build(&self, inputs: &[Array], out: &mut Vec<Array>, s: Stream);
}

unsafe extern "C" fn trampoline<P: GraphFn>(
    out: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> std::os::raw::c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let graph = &*(payload as *const P);
        let n = sys::mlx_vector_array_size(input);
        let mut inputs = Vec::with_capacity(n);
        for i in 0..n {
            let mut handle = sys::mlx_array::empty();
            if sys::mlx_vector_array_get(&mut handle, input, i) != 0 {
                return Err(Error("closure: failed to fetch input".into()));
            }
            inputs.push(Array::from_raw(handle));
        }
        let s = Stream::default_stream();
        let mut outputs: Vec<Array> = Vec::new();
        graph.build(&inputs, &mut outputs, s);
        let handles: Vec<sys::mlx_array> = outputs.iter().map(|a| a.raw()).collect();
        *out = sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len());
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&'static str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "compiled closure panicked".to_string());
            crate::record_error(format!("compiled closure failed: {msg}"));
            1
        }
        Ok(Err(e)) => {
            crate::record_error(format!("compiled closure failed: {e}"));
            1
        }
    }
}

unsafe extern "C" fn dtor<P>(payload: *mut c_void) {
    if !payload.is_null() {
        drop(Box::from_raw(payload as *mut P));
    }
}

/// A compiled MLX function (owning handle to the compiled closure).
pub struct CompiledFn {
    closure: sys::mlx_closure,
}

unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl Drop for CompiledFn {
    fn drop(&mut self) {
        if !self.closure.ctx.is_null() {
            unsafe {
                sys::mlx_closure_free(self.closure);
            }
        }
    }
}

impl CompiledFn {
    /// Compile `graph` (equivalent of `mx.compile(fun)`).
    pub fn compile<P: GraphFn>(graph: P) -> Result<CompiledFn> {
        let payload = Box::into_raw(Box::new(graph)) as *mut c_void;
        unsafe {
            let closure = sys::mlx_closure_new_func_payload(
                Some(trampoline::<P>),
                payload,
                Some(dtor::<P>),
            );
            let mut compiled = sys::mlx_closure::empty();
            let status = sys::mlx_compile(&mut compiled, closure, false);
            // mlx_compile copies the closure reference; our handle is ours.
            sys::mlx_closure_free(closure);
            check(status)?;
            Ok(CompiledFn { closure: compiled })
        }
    }

    /// Apply the compiled function (lazy; eval outputs to run). The stream
    /// parameter is accepted for API symmetry — the compiled closure resolves
    /// its own default stream on the calling thread.
    pub fn apply(&self, inputs: &[&Array], _s: Stream) -> Result<Vec<Array>> {
        let handles: Vec<sys::mlx_array> = inputs.iter().map(|a| a.raw()).collect();
        let in_vec =
            unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
        let mut out_vec = sys::mlx_vector_array::empty();
        let status = unsafe { sys::mlx_closure_apply(&mut out_vec, self.closure, in_vec) };
        unsafe { sys::mlx_vector_array_free(in_vec) };
        check(status)?;
        unsafe {
            let n = sys::mlx_vector_array_size(out_vec);
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let mut handle = sys::mlx_array::empty();
                if sys::mlx_vector_array_get(&mut handle, out_vec, i) != 0 {
                    sys::mlx_vector_array_free(out_vec);
                    return Err(take_error().map(Error).unwrap_or_else(|| {
                        Error("compiled apply: failed to fetch output".into())
                    }));
                }
                result.push(Array::from_raw(handle));
            }
            sys::mlx_vector_array_free(out_vec);
            Ok(result)
        }
    }
}
