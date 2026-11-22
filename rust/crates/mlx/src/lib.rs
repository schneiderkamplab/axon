//! Safe Rust wrapper over the MLX C API.
//!
//! Scope is deliberately narrow — exactly what the Axon Rust server needs:
//! owning `Array` handles, the op subset used by the Bifrost decode graph,
//! custom Metal kernels, `mx.compile` over a C closure, and safetensors
//! loading. Everything tensor-shaped stays inside this crate's `unsafe`
//! blocks; callers upstream (`axon-backend` and up) see only safe types.
//!
//! Stream model: one default stream per thread (mirrors Python's MLX); the
//! decode loop runs on a dedicated thread.

pub mod compile;
pub mod kernel;

use std::ffi::{CStr, CString};
use std::fmt;
use std::ops::Range;
use std::sync::Mutex;

use mlx_c_sys as sys;

/// An MLX error message captured from the C error handler.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);
}

unsafe extern "C" fn error_handler(msg: *const std::os::raw::c_char, _data: *mut std::os::raw::c_void) {
    let text = if msg.is_null() {
        "unknown MLX error".to_string()
    } else {
        CStr::from_ptr(msg).to_string_lossy().into_owned()
    };
    LAST_ERROR.with(|slot| {
        // Keep the first error of an operation; later ones are cascades.
        let mut guard = slot.lock().unwrap();
        if guard.is_none() {
            *guard = Some(text);
        }
    });
}

/// Take (and clear) the last MLX error message recorded on this thread.
pub fn take_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.lock().unwrap().take())
}

/// Record an error outside the normal `check` path (e.g. from a compiled
/// closure trampoline that cannot return one directly).
pub fn record_error(msg: String) {
    LAST_ERROR.with(|slot| {
        let mut guard = slot.lock().unwrap();
        if guard.is_none() {
            *guard = Some(msg);
        }
    });
}

fn check(status: std::os::raw::c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error(
            take_error().unwrap_or_else(|| format!("MLX call failed (status {status})")),
        ))
    }
}

/// Install the Rust error handler. Called once by [`init`]; harmless to call
/// again.
fn install_error_handler() {
    unsafe {
        sys::mlx_set_error_handler(Some(error_handler), std::ptr::null_mut(), None);
    }
}

/// One-time MLX initialization guard.
pub struct Mlx;

/// Initialize MLX (error handler + default device). Returns a guard whose drop
/// is a no-op (MLX needs no shutdown), kept for call-site clarity.
pub fn init() -> Mlx {
    install_error_handler();
    Mlx
}

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    U16,
    U32,
    I8,
    I32,
    I64,
    F16,
    F32,
    Bf16,
}

impl Dtype {
    pub fn from_raw(raw: sys::mlx_dtype) -> Dtype {
        match raw {
            sys::MLX_BOOL => Dtype::Bool,
            sys::MLX_UINT8 => Dtype::U8,
            sys::MLX_UINT16 => Dtype::U16,
            sys::MLX_UINT32 => Dtype::U32,
            sys::MLX_INT8 => Dtype::I8,
            sys::MLX_INT32 => Dtype::I32,
            sys::MLX_INT64 => Dtype::I64,
            sys::MLX_FLOAT16 => Dtype::F16,
            sys::MLX_FLOAT32 => Dtype::F32,
            sys::MLX_BFLOAT16 => Dtype::Bf16,
            other => panic!("unhandled mlx dtype code {other}"),
        }
    }

    pub fn raw(self) -> sys::mlx_dtype {
        match self {
            Dtype::Bool => sys::MLX_BOOL,
            Dtype::U8 => sys::MLX_UINT8,
            Dtype::U16 => sys::MLX_UINT16,
            Dtype::U32 => sys::MLX_UINT32,
            Dtype::I8 => sys::MLX_INT8,
            Dtype::I32 => sys::MLX_INT32,
            Dtype::I64 => sys::MLX_INT64,
            Dtype::F16 => sys::MLX_FLOAT16,
            Dtype::F32 => sys::MLX_FLOAT32,
            Dtype::Bf16 => sys::MLX_BFLOAT16,
        }
    }
}

// ---------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------

/// A compute stream. `Stream::default_stream()` is the thread's default stream
/// (like Python's `mx.default_stream()`); the decode loop should run on one
/// dedicated thread.
#[derive(Clone, Copy)]
pub struct Stream {
    raw: sys::mlx_stream,
}

impl Stream {
    /// The calling thread's default stream over the default device (the C-API
    /// equivalent of Python's `mx.default_stream()`; resolved once per thread
    /// and cached).
    pub fn default_stream() -> Stream {
        thread_local! {
            static DEFAULT: Stream = Stream::init_default();
        }
        DEFAULT.with(|s| *s)
    }

    fn init_default() -> Stream {
        unsafe {
            let mut dev = sys::mlx_device::empty();
            if sys::mlx_get_default_device(&mut dev) != 0 {
                let msg = take_error().unwrap_or_else(|| "failed to get default device".into());
                panic!("mlx: {msg}");
            }
            let tls = sys::mlx_stream_thread_local_new(dev);
            let mut raw = sys::mlx_stream::empty();
            let status = sys::mlx_stream_from_thread_local(&mut raw, tls);
            sys::mlx_stream_thread_local_free(tls);
            sys::mlx_device_free(dev);
            if status != 0 {
                let msg = take_error().unwrap_or_else(|| "failed to resolve default stream".into());
                panic!("mlx: {msg}");
            }
            Stream { raw }
        }
    }

    /// A stream on the CPU device. Safetensors `Load` nodes have no GPU eval
    /// (`Load::eval_gpu` unimplemented), so tensors are materialized host-side
    /// on this stream — mirroring the Python loader, whose arrays land on CPU
    /// and migrate lazily. (The returned stream is intentionally not freed:
    /// `Stream` is `Copy` and the leak is one small object per call.)
    pub fn cpu_stream() -> Stream {
        unsafe {
            let dev = sys::mlx_device_new_type(sys::MLX_CPU, 0);
            let raw = sys::mlx_stream_new_device(dev);
            sys::mlx_device_free(dev);
            Stream { raw }
        }
    }

    pub fn raw(&self) -> sys::mlx_stream {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(sys::mlx_synchronize(self.raw)) }
    }
}

// ---------------------------------------------------------------------------
// array
// ---------------------------------------------------------------------------

/// An owning handle to an MLX array (shared reference under the hood).
pub struct Array {
    raw: sys::mlx_array,
}

// The handle is a plain pointer to a refcounted C++ object; MLX arrays are
// safe to move between threads in the Python API the same way.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.raw.ctx.is_null() {
            unsafe {
                sys::mlx_array_free(self.raw);
            }
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut out = sys::mlx_array::empty();
        unsafe {
            sys::mlx_array_set(&mut out, self.raw);
        }
        Array { raw: out }
    }
}

impl Array {
    /// Wrap an owning raw handle. The caller must guarantee it owns the ref.
    pub(crate) unsafe fn from_raw(raw: sys::mlx_array) -> Array {
        Array { raw }
    }

    pub(crate) fn raw(&self) -> sys::mlx_array {
        self.raw
    }

    pub fn shape(&self) -> Vec<usize> {
        unsafe {
            let ndim = sys::mlx_array_ndim(self.raw);
            let ptr = sys::mlx_array_shape(self.raw);
            std::slice::from_raw_parts(ptr, ndim)
                .iter()
                .map(|&d| d as usize)
                .collect()
        }
    }

    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.raw) }
    }

    pub fn size(&self) -> usize {
        unsafe { sys::mlx_array_size(self.raw) }
    }

    pub fn dtype(&self) -> Dtype {
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.raw) })
    }

    pub fn eval(&self) -> Result<()> {
        unsafe { check(sys::mlx_array_eval(self.raw)) }
    }

    // -- constructors -------------------------------------------------------

    pub fn zeros(shape: &[usize], dtype: Dtype, s: Stream) -> Result<Array> {
        let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out = sys::mlx_array::empty();
        unsafe {
            check(sys::mlx_zeros(
                &mut out,
                shape.as_ptr(),
                shape.len(),
                dtype.raw(),
                s.raw(),
            ))?;
        }
        Ok(unsafe { Array::from_raw(out) })
    }

    pub fn scalar_i32(v: i32, s: Stream) -> Result<Array> {
        // Stream currently unused by the C scalar constructors (host-side).
        let _ = s;
        Ok(unsafe { Array::from_raw(sys::mlx_array_new_int(v)) })
    }

    pub fn scalar_f32(v: f32, s: Stream) -> Result<Array> {
        let _ = s;
        Ok(unsafe { Array::from_raw(sys::mlx_array_new_float32(v)) })
    }

    /// An array copied from host data (`T` must match `dtype`'s bit layout;
    /// use `f16`/`bf16` as raw `u16` bits via [`Array::from_u16_bits`]). The
    /// C API copies the buffer, so the host data needs no lifetime.
    pub fn from_slice<T: Copy>(data: &[T], shape: &[usize], dtype: Dtype, _s: Stream) -> Result<Array> {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "from_slice: data/shape mismatch"
        );
        let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mut out = sys::mlx_array::empty();
        unsafe {
            check(sys::mlx_array_set_data(
                &mut out,
                data.as_ptr() as *const std::os::raw::c_void,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.raw(),
            ))?;
        }
        Ok(unsafe { Array::from_raw(out) })
    }

    /// fp16 / bf16 data given as raw 16-bit patterns.
    pub fn from_u16_bits(data: &[u16], shape: &[usize], dtype: Dtype, s: Stream) -> Result<Array> {
        debug_assert!(matches!(dtype, Dtype::F16 | Dtype::Bf16));
        Self::from_slice(data, shape, dtype, s)
    }

    // -- host reads ---------------------------------------------------------

    /// Scalar readback as i32 (argmax results).
    pub fn item_i32(&self) -> Result<i32> {
        let mut out: i32 = 0;
        unsafe { check(sys::mlx_array_item_int32(&mut out, self.raw))? };
        Ok(out)
    }

    /// Scalar readback as f32.
    pub fn item_f32(&self) -> Result<f32> {
        let mut out: f32 = 0.0;
        unsafe { check(sys::mlx_array_item_float32(&mut out, self.raw))? };
        Ok(out)
    }

    /// Scalar readback as u32 (argmax results are uint32 in MLX).
    pub fn item_u32(&self) -> Result<u32> {
        unsafe {
            self.eval()?;
            if sys::mlx_array_size(self.raw) != 1 {
                return Err(Error("item_u32: array is not a scalar".into()));
            }
            let ptr = sys::mlx_array_data_uint32(self.raw);
            if ptr.is_null() {
                return Err(Error("array not available after eval".into()));
            }
            Ok(*ptr)
        }
    }

    /// Evaluated fp16/bf16 payload as raw 16-bit patterns.
    pub fn data_u16_bits(&self) -> Result<&[u16]> {
        unsafe {
            self.eval()?;
            let len = sys::mlx_array_size(self.raw);
            let ptr = sys::mlx_array_data_uint16(self.raw);
            if ptr.is_null() {
                return Err(Error("array not available after eval".into()));
            }
            Ok(std::slice::from_raw_parts(ptr, len))
        }
    }

    /// Evaluated fp32 payload.
    pub fn data_f32(&self) -> Result<&[f32]> {
        unsafe {
            self.eval()?;
            let len = sys::mlx_array_size(self.raw);
            let ptr = sys::mlx_array_data_float32(self.raw);
            if ptr.is_null() {
                return Err(Error("array not available after eval".into()));
            }
            Ok(std::slice::from_raw_parts(ptr, len))
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array{:?})", self.shape())
    }
}

// ---------------------------------------------------------------------------
// graph-building ops (free functions, all lazy)
// ---------------------------------------------------------------------------

fn out1(status: std::os::raw::c_int, raw: sys::mlx_array) -> Result<Array> {
    check(status)?;
    Ok(unsafe { Array::from_raw(raw) })
}

pub fn add(a: &Array, b: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_add(&mut out, a.raw(), b.raw(), s.raw()) };
    out1(status, out)
}

pub fn multiply(a: &Array, b: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_multiply(&mut out, a.raw(), b.raw(), s.raw()) };
    out1(status, out)
}

pub fn sigmoid(a: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_sigmoid(&mut out, a.raw(), s.raw()) };
    out1(status, out)
}

pub fn matmul(a: &Array, b: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_matmul(&mut out, a.raw(), b.raw(), s.raw()) };
    out1(status, out)
}

pub fn astype(a: &Array, dtype: Dtype, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_astype(&mut out, a.raw(), dtype.raw(), s.raw()) };
    out1(status, out)
}

pub fn reshape(a: &Array, shape: &[usize], s: Stream) -> Result<Array> {
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let mut out = sys::mlx_array::empty();
    let status = unsafe {
        sys::mlx_reshape(&mut out, a.raw(), shape.as_ptr(), shape.len(), s.raw())
    };
    out1(status, out)
}

/// Static slice with per-dimension ranges (stride 1). Dimensions beyond
/// `ranges.len()` pass through whole — MLX requires index vectors of length
/// == ndim, so trailing dims are padded with full-range/stride-1.
pub fn slice_ranges(a: &Array, ranges: &[Range<i32>], s: Stream) -> Result<Array> {
    let ndim = a.ndim();
    assert!(
        ranges.len() <= ndim,
        "slice_ranges: {} ranges for {}-d array",
        ranges.len(),
        ndim
    );
    let shape = a.shape();
    let mut start = Vec::with_capacity(ndim);
    let mut stop = Vec::with_capacity(ndim);
    let mut strides = Vec::with_capacity(ndim);
    for r in ranges {
        start.push(r.start);
        stop.push(r.end);
        strides.push(1);
    }
    for &d in shape.iter().skip(ranges.len()) {
        start.push(0);
        stop.push(d as i32);
        strides.push(1);
    }
    let mut out = sys::mlx_array::empty();
    let status = unsafe {
        sys::mlx_slice(
            &mut out,
            a.raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            s.raw(),
        )
    };
    out1(status, out)
}

/// `mx.slice_update(src, update, start, axes)` with a device-side dynamic start.
pub fn slice_update_dynamic(
    src: &Array,
    update: &Array,
    start: &Array,
    axes: &[i32],
    s: Stream,
) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe {
        sys::mlx_slice_update_dynamic(
            &mut out,
            src.raw(),
            update.raw(),
            start.raw(),
            axes.as_ptr(),
            axes.len(),
            s.raw(),
        )
    };
    out1(status, out)
}

/// `mx.stack(arrays)` along axis 0.
pub fn stack(arrays: &[&Array], s: Stream) -> Result<Array> {
    let handles: Vec<sys::mlx_array> = arrays.iter().map(|a| a.raw()).collect();
    let vec = unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_stack(&mut out, vec, s.raw()) };
    unsafe { sys::mlx_vector_array_free(vec) };
    out1(status, out)
}

/// `mx.argmax(a, keepdims=false)` — flat argmax over the whole array.
pub fn argmax(a: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_argmax(&mut out, a.raw(), false, s.raw()) };
    out1(status, out)
}

/// `a[indices]` along axis 0 (embedding lookup).
pub fn take(a: &Array, indices: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_take(&mut out, a.raw(), indices.raw(), s.raw()) };
    out1(status, out)
}

/// `mx.take_along_axis(a, indices, axis)` — gather with index arrays shaped
/// like `a` (same ndim); the result replaces `axis` with `indices`' shape.
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe {
        sys::mlx_take_along_axis(&mut out, a.raw(), indices.raw(), axis, s.raw())
    };
    out1(status, out)
}

/// Elementwise remainder (Python `%`; sign follows the divisor).
pub fn remainder(a: &Array, b: &Array, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let status = unsafe { sys::mlx_remainder(&mut out, a.raw(), b.raw(), s.raw()) };
    out1(status, out)
}

/// `mx.fast.rms_norm(x, weight=None, eps)` with an optional (borrowed) weight.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32, s: Stream) -> Result<Array> {
    let mut out = sys::mlx_array::empty();
    let w = weight.map_or(sys::mlx_array::empty(), |w| w.raw());
    let status = unsafe { sys::mlx_fast_rms_norm(&mut out, x.raw(), w, eps, s.raw()) };
    out1(status, out)
}

// ---------------------------------------------------------------------------
// memory / housekeeping
// ---------------------------------------------------------------------------

/// Release MLX's free-buffer cache: buffers freed by earlier ops are retained
/// by size class for reuse; after a one-time bulk load (safetensors + packing)
/// the retained load-time buffers are pure dead weight, so the loader calls
/// this once at the end. Live arrays are never touched.
pub fn clear_cache(s: Stream) -> Result<()> {
    unsafe { check(sys::mlx_clear_cache(s.raw)) }
}

// ---------------------------------------------------------------------------
// safetensors loading
// ---------------------------------------------------------------------------

/// Load every tensor of a `.safetensors` file.
pub fn load_safetensors(path: &std::path::Path, s: Stream) -> Result<Vec<(String, Array)>> {
    let cpath = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error("safetensors path contains NUL".into()))?;
    let mut map = sys::mlx_map_string_to_array::empty();
    let mut meta = sys::mlx_map_string_to_string::empty();
    unsafe {
        check(sys::mlx_load_safetensors(
            &mut map,
            &mut meta,
            cpath.as_ptr(),
            s.raw(),
        ))?;
    }
    // metadata is unused; drop it
    if !meta.ctx.is_null() {
        unsafe {
            sys::mlx_map_string_to_string_free(meta);
        }
    }

    let mut tensors = Vec::new();
    unsafe {
        let it = sys::mlx_map_string_to_array_iterator_new(map);
        loop {
            let mut key: *const std::os::raw::c_char = std::ptr::null();
            let mut value = sys::mlx_array::empty();
            let status = sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it);
            if status == 2 {
                break;
            }
            if status != 0 {
                sys::mlx_map_string_to_array_iterator_free(it);
                sys::mlx_map_string_to_array_free(map);
                return Err(take_error()
                    .map(Error)
                    .unwrap_or_else(|| Error("safetensors iteration failed".into())));
            }
            let name = CStr::from_ptr(key).to_string_lossy().into_owned();
            tensors.push((name, Array::from_raw(value)));
        }
        sys::mlx_map_string_to_array_iterator_free(it);
        sys::mlx_map_string_to_array_free(map);
    }
    Ok(tensors)
}
