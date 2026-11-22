//! Raw FFI bindings to the MLX C API (mlx-c v0.7.0, MLX v0.32.2).
//!
//! Hand-written for the surface the Axon Rust server needs: arrays, the ops
//! used by the Bifrost decode graph, fast (rms_norm / custom Metal kernels),
//! compile, safetensors loading, streams and the error handler.
//!
//! Ownership model (mirrors mlx-c): every function that *returns* an opaque
//! handle (`mlx_array_new*`, out-params like `mlx_array* res`) yields a new
//! owning reference that must be freed exactly once. Parameters marked
//! `const mlx_array` borrow. Freeing an empty handle (all-zero `ctx`) is a
//! no-op on the C side, so the safe layer can wrap zeroed handles freely.
//!
//! All functions return 0 on success and non-zero on error; the message is
//! available from the error handler installed by `mlx::init`.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// opaque handles
// ---------------------------------------------------------------------------

/// Every mlx-c object is a by-value struct wrapping a `void* ctx`.
macro_rules! handle {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name {
            pub ctx: *mut c_void,
        }
        impl Default for $name {
            fn default() -> Self {
                // The all-zero handle is the C API's "empty" value; freeing it
                // is a no-op and passing it where an optional argument is
                // documented as "may be null" means absent.
                unsafe { std::mem::zeroed() }
            }
        }
        impl $name {
            pub fn empty() -> Self {
                Self::default()
            }
            pub fn is_empty(&self) -> bool {
                self.ctx.is_null()
            }
        }
    };
}

handle!(mlx_array);
handle!(mlx_vector_array);
handle!(mlx_vector_string);
handle!(mlx_map_string_to_array);
handle!(mlx_map_string_to_string);
handle!(mlx_map_string_to_array_iterator);
handle!(mlx_closure);
handle!(mlx_stream);
handle!(mlx_stream_thread_local);
handle!(mlx_device);

/// C enum `mlx_device_type` value for CPU (see mlx/c/device.h).
pub const MLX_CPU: c_int = 0;
handle!(mlx_fast_metal_kernel);
handle!(mlx_fast_metal_kernel_config);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct mlx_optional_float {
    pub value: f32,
    pub has_value: bool,
}

// ---------------------------------------------------------------------------
// enums / constants (order fixed by the C headers)
// ---------------------------------------------------------------------------

pub type mlx_dtype = c_int;
pub const MLX_BOOL: c_int = 0;
pub const MLX_UINT8: c_int = 1;
pub const MLX_UINT16: c_int = 2;
pub const MLX_UINT32: c_int = 3;
pub const MLX_UINT64: c_int = 4;
pub const MLX_INT8: c_int = 5;
pub const MLX_INT16: c_int = 6;
pub const MLX_INT32: c_int = 7;
pub const MLX_INT64: c_int = 8;
pub const MLX_FLOAT16: c_int = 9;
pub const MLX_FLOAT32: c_int = 10;
pub const MLX_FLOAT64: c_int = 11;
pub const MLX_BFLOAT16: c_int = 12;
pub const MLX_COMPLEX64: c_int = 13;

// ---------------------------------------------------------------------------
// array
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_dtype_size(dtype: mlx_dtype) -> usize;

    pub fn mlx_array_new() -> mlx_array;
    pub fn mlx_array_new_bool(val: bool) -> mlx_array;
    pub fn mlx_array_new_int(val: c_int) -> mlx_array;
    pub fn mlx_array_new_float32(val: f32) -> mlx_array;
    pub fn mlx_array_new_data(
        data: *const c_void,
        shape: *const c_int,
        dim: c_int,
        dtype: mlx_dtype,
    ) -> mlx_array;
    pub fn mlx_array_free(arr: mlx_array) -> c_int;
    pub fn mlx_array_set(arr: *mut mlx_array, src: const_mlx_array) -> c_int;
    pub fn mlx_array_set_data(
        arr: *mut mlx_array,
        data: *const c_void,
        shape: *const c_int,
        dim: c_int,
        dtype: mlx_dtype,
    ) -> c_int;

    pub fn mlx_array_itemsize(arr: const_mlx_array) -> usize;
    pub fn mlx_array_size(arr: const_mlx_array) -> usize;
    pub fn mlx_array_nbytes(arr: const_mlx_array) -> usize;
    pub fn mlx_array_ndim(arr: const_mlx_array) -> usize;
    pub fn mlx_array_shape(arr: const_mlx_array) -> *const c_int;
    pub fn mlx_array_strides(arr: const_mlx_array) -> *const usize;
    pub fn mlx_array_dim(arr: const_mlx_array, dim: c_int) -> c_int;
    pub fn mlx_array_dtype(arr: const_mlx_array) -> mlx_dtype;

    pub fn mlx_array_eval(arr: mlx_array) -> c_int;

    pub fn mlx_array_item_int32(res: *mut i32, arr: const_mlx_array) -> c_int;
    pub fn mlx_array_item_int64(res: *mut i64, arr: const_mlx_array) -> c_int;
    pub fn mlx_array_item_float32(res: *mut f32, arr: const_mlx_array) -> c_int;

    pub fn mlx_array_data_uint8(arr: const_mlx_array) -> *const u8;
    pub fn mlx_array_data_uint16(arr: const_mlx_array) -> *const u16; // f16/bf16 bit view
    pub fn mlx_array_data_uint32(arr: const_mlx_array) -> *const u32;
    pub fn mlx_array_data_int32(arr: const_mlx_array) -> *const i32;
    pub fn mlx_array_data_int64(arr: const_mlx_array) -> *const i64;
    pub fn mlx_array_data_float32(arr: const_mlx_array) -> *const f32;
}

/// The C signatures say `const mlx_array`; on the Rust side that is the same
/// struct passed by value.
pub type const_mlx_array = mlx_array;

// ---------------------------------------------------------------------------
// vector / map containers
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_vector_array_new() -> mlx_vector_array;
    pub fn mlx_vector_array_new_data(data: *const mlx_array, size: usize) -> mlx_vector_array;
    pub fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;
    pub fn mlx_vector_array_size(vec: mlx_vector_array) -> usize;
    pub fn mlx_vector_array_get(
        res: *mut mlx_array,
        vec: mlx_vector_array,
        idx: usize,
    ) -> c_int;

    pub fn mlx_vector_string_new() -> mlx_vector_string;
    pub fn mlx_vector_string_append_value(vec: mlx_vector_string, val: *const c_char) -> c_int;
    pub fn mlx_vector_string_free(vec: mlx_vector_string) -> c_int;

    pub fn mlx_map_string_to_array_new() -> mlx_map_string_to_array;
    pub fn mlx_map_string_to_array_free(map: mlx_map_string_to_array) -> c_int;
    /// Returns 0 on success (new owning ref in `value`), 2 when the key is
    /// missing, 1 on error.
    pub fn mlx_map_string_to_array_get(
        value: *mut mlx_array,
        map: mlx_map_string_to_array,
        key: *const c_char,
    ) -> c_int;
    pub fn mlx_map_string_to_string_free(map: mlx_map_string_to_string) -> c_int;
    pub fn mlx_map_string_to_array_iterator_new(
        map: mlx_map_string_to_array,
    ) -> mlx_map_string_to_array_iterator;
    pub fn mlx_map_string_to_array_iterator_free(
        it: mlx_map_string_to_array_iterator,
    ) -> c_int;
    /// Returns 0 and fills `key`/`value` (new owning ref), 2 at the end.
    pub fn mlx_map_string_to_array_iterator_next(
        key: *mut *const c_char,
        value: *mut mlx_array,
        it: mlx_map_string_to_array_iterator,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// streams
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_device_free(dev: mlx_device) -> c_int;
    pub fn mlx_get_default_device(dev: *mut mlx_device) -> c_int;
    pub fn mlx_device_new_type(type_: c_int, index: c_int) -> mlx_device;
    pub fn mlx_stream_new_device(dev: mlx_device) -> mlx_stream;
    pub fn mlx_stream_new() -> mlx_stream;
    pub fn mlx_stream_thread_local_new(dev: mlx_device) -> mlx_stream_thread_local;
    pub fn mlx_stream_from_thread_local(
        res: *mut mlx_stream,
        tls: mlx_stream_thread_local,
    ) -> c_int;
    pub fn mlx_stream_thread_local_free(tls: mlx_stream_thread_local) -> c_int;
    pub fn mlx_stream_free(stream: mlx_stream) -> c_int;
    pub fn mlx_synchronize(stream: mlx_stream) -> c_int;
}

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

pub type mlx_error_handler_func =
    Option<unsafe extern "C" fn(msg: *const c_char, data: *mut c_void)>;

extern "C" {
    pub fn mlx_set_error_handler(
        handler: mlx_error_handler_func,
        data: *mut c_void,
        dtor: Option<unsafe extern "C" fn(data: *mut c_void)>,
    );
}

// ---------------------------------------------------------------------------
// ops (subset used by the Bifrost graph)
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_add(
        res: *mut mlx_array,
        a: const_mlx_array,
        b: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_subtract(
        res: *mut mlx_array,
        a: const_mlx_array,
        b: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_multiply(
        res: *mut mlx_array,
        a: const_mlx_array,
        b: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_matmul(
        res: *mut mlx_array,
        a: const_mlx_array,
        b: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_sigmoid(res: *mut mlx_array, a: const_mlx_array, s: mlx_stream) -> c_int;
    pub fn mlx_astype(
        res: *mut mlx_array,
        a: const_mlx_array,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_reshape(
        res: *mut mlx_array,
        a: const_mlx_array,
        shape: *const c_int,
        shape_num: usize,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_slice(
        res: *mut mlx_array,
        a: const_mlx_array,
        start: *const c_int,
        start_num: usize,
        stop: *const c_int,
        stop_num: usize,
        strides: *const c_int,
        strides_num: usize,
        s: mlx_stream,
    ) -> c_int;
    /// `start` is a device array (dynamic per-step position).
    pub fn mlx_slice_update_dynamic(
        res: *mut mlx_array,
        src: const_mlx_array,
        update: const_mlx_array,
        start: const_mlx_array,
        axes: *const c_int,
        axes_num: usize,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_concatenate(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_stack(
        res: *mut mlx_array,
        arrays: mlx_vector_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_argmax(
        res: *mut mlx_array,
        a: const_mlx_array,
        keepdims: bool,
        s: mlx_stream,
    ) -> c_int;
    /// Flattened take: `a.flat[indices]` (NOT row gathering — use
    /// `mlx_take_along_axis` for that).
    pub fn mlx_take(
        res: *mut mlx_array,
        a: const_mlx_array,
        indices: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    /// Row/axis gathering: result has `a`'s shape with `axis` replaced by
    /// `indices`' shape (indices must have the same ndim as `a`).
    pub fn mlx_take_along_axis(
        res: *mut mlx_array,
        a: const_mlx_array,
        indices: const_mlx_array,
        axis: c_int,
        s: mlx_stream,
    ) -> c_int;
    /// Elementwise remainder (Python `%` semantics; sign follows the divisor).
    pub fn mlx_remainder(
        res: *mut mlx_array,
        a: const_mlx_array,
        b: const_mlx_array,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_zeros(
        res: *mut mlx_array,
        shape: *const c_int,
        shape_num: usize,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_full(
        res: *mut mlx_array,
        shape: *const c_int,
        shape_num: usize,
        vals: const_mlx_array,
        dtype: mlx_dtype,
        s: mlx_stream,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// fast ops + custom Metal kernels
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_fast_rms_norm(
        res: *mut mlx_array,
        x: const_mlx_array,
        weight: const_mlx_array, // may be null (empty handle)
        eps: f32,
        s: mlx_stream,
    ) -> c_int;
    pub fn mlx_fast_rope(
        res: *mut mlx_array,
        x: const_mlx_array,
        dims: c_int,
        traditional: bool,
        base: mlx_optional_float,
        scale: f32,
        offset: c_int,
        freqs: const_mlx_array, // may be null
        s: mlx_stream,
    ) -> c_int;

    pub fn mlx_fast_metal_kernel_new(
        name: *const c_char,
        input_names: mlx_vector_string,
        output_names: mlx_vector_string,
        source: *const c_char,
        header: *const c_char,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> mlx_fast_metal_kernel;
    pub fn mlx_fast_metal_kernel_free(kernel: mlx_fast_metal_kernel) -> c_int;
    pub fn mlx_fast_metal_kernel_config_new() -> mlx_fast_metal_kernel_config;
    pub fn mlx_fast_metal_kernel_config_free(config: mlx_fast_metal_kernel_config) -> c_int;
    pub fn mlx_fast_metal_kernel_config_add_output_arg(
        config: mlx_fast_metal_kernel_config,
        shape: *const c_int,
        size: usize,
        dtype: mlx_dtype,
    ) -> c_int;
    pub fn mlx_fast_metal_kernel_config_set_grid(
        config: mlx_fast_metal_kernel_config,
        grid1: c_int,
        grid2: c_int,
        grid3: c_int,
    ) -> c_int;
    pub fn mlx_fast_metal_kernel_config_set_thread_group(
        config: mlx_fast_metal_kernel_config,
        t1: c_int,
        t2: c_int,
        t3: c_int,
    ) -> c_int;
    pub fn mlx_fast_metal_kernel_apply(
        outputs: *mut mlx_vector_array,
        kernel: mlx_fast_metal_kernel,
        inputs: mlx_vector_array,
        config: mlx_fast_metal_kernel_config,
        s: mlx_stream,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

pub type mlx_compile_mode = c_int;
pub const MLX_COMPILE_MODE_DISABLED: c_int = 0;
pub const MLX_COMPILE_MODE_NO_SIMPLIFY: c_int = 1;
pub const MLX_COMPILE_MODE_NO_FUSE: c_int = 2;
pub const MLX_COMPILE_MODE_ENABLED: c_int = 3;

/// `int (*)(mlx_vector_array* out, const mlx_vector_array in, void* payload)`
pub type mlx_closure_fn = Option<
    unsafe extern "C" fn(out: *mut mlx_vector_array, input: mlx_vector_array, payload: *mut c_void) -> c_int,
>;

extern "C" {
    pub fn mlx_closure_new_func_payload(
        fun: mlx_closure_fn,
        payload: *mut c_void,
        dtor: Option<unsafe extern "C" fn(payload: *mut c_void)>,
    ) -> mlx_closure;
    pub fn mlx_closure_free(cls: mlx_closure) -> c_int;
    pub fn mlx_closure_apply(
        res: *mut mlx_vector_array,
        cls: mlx_closure,
        input: mlx_vector_array,
    ) -> c_int;
    pub fn mlx_compile(res: *mut mlx_closure, fun: mlx_closure, shapeless: bool) -> c_int;
}

// ---------------------------------------------------------------------------
// io
// ---------------------------------------------------------------------------

extern "C" {
    /// Loads every tensor of a `.safetensors` file into `res` (map name →
    /// array, new owning refs) and the header metadata into `meta`.
    pub fn mlx_load_safetensors(
        res: *mut mlx_map_string_to_array,
        meta: *mut mlx_map_string_to_string,
        file: *const c_char,
        s: mlx_stream,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// memory / housekeeping
// ---------------------------------------------------------------------------

extern "C" {
    pub fn mlx_clear_cache(s: mlx_stream) -> c_int;
}
