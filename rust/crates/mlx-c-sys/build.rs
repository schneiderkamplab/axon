//! Links the MLX C API (mlx-c) against Apple's MLX.
//!
//! Layout expected under `rust/vendor/` (see `rust/README.md` for the two
//! clone/copy steps):
//! - `vendor/mlx-c/build/libmlxc.a` — the mlx-c C wrappers, compiled against
//!   the *same* MLX the Python wheel ships (v0.7.0 ↔ MLX v0.32.2).
//! - `vendor/lib/{libmlx.dylib, libjaccl.dylib, mlx.metallib}` — the runtime
//!   libraries copied out of the installed `mlx` Python wheel.
//!
//! The static wrappers resolve every MLX symbol from `libmlx.dylib` at link
//! time; the executable only needs `-lmlxc -lmlx -ljaccl -lc++` plus an
//! rpath pointing at `vendor/lib` so `@rpath/libmlx.dylib` resolves.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("crate sits directly under rust/crates/");
    let vendor = workspace.join("vendor");

    let mlxc_dir = vendor.join("mlx-c/build");
    let mlxc = mlxc_dir.join("libmlxc.a");
    if !mlxc.exists() {
        panic!(
            "libmlxc.a not found at {}\n\
             Build the C wrappers first (see rust/README.md):\n\
             cd rust/vendor/mlx-c && cmake -B build -S . \
             -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_DIR=<wheel>/share/cmake/MLX \
             -DMLX_C_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF \
             && cmake --build build",
            mlxc.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", mlxc_dir.display());
    println!("cargo:rustc-link-lib=static=mlxc");

    let lib_dir = vendor.join("lib");
    for name in ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"] {
        if !lib_dir.join(name).exists() {
            panic!(
                "{} not found at {}\n\
                 Copy the runtime libraries out of the installed mlx wheel (see rust/README.md).",
                name,
                lib_dir.display()
            );
        }
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=jaccl");
    // The C wrappers are C++ (they wrap exceptions into error codes).
    println!("cargo:rustc-link-lib=c++");

    // libmlx.dylib and libjaccl.dylib are found via rpath. The binary lands at
    // rust/target/{debug,release}/ (two levels under rust/), test and example
    // binaries one level deeper in deps/ or examples/.
    for up in ["../../", "../../../"] {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,@executable_path/{up}vendor/lib"
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", mlxc.display());
    println!("cargo:rerun-if-changed={}", lib_dir.join("libmlx.dylib").display());
}
