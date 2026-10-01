//! Give the shared library a SONAME on Linux and Android.
//!
//! rustc links a `cdylib` without one, and a library linked against a
//! SONAME-less library records the path it was handed as its dependency. CMake
//! hands the linker an absolute path when the library is an `IMPORTED` target,
//! so the Kotlin package's JNI shim came out with
//! `DT_NEEDED /home/<builder>/.../libtaladb_ffi.so` — a path no device has, so
//! the library fails to load. With a SONAME, every linker records
//! `libtaladb_ffi.so` however it was pointed at the file.
//!
//! `rustc-cdylib-link-arg` reaches only this crate's cdylib, unlike
//! `[target.*].rustflags`, which would also stamp the name onto every other
//! cdylib in the build — redb 2.x declares one.

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "linux" || os == "android" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libtaladb_ffi.so");
    }
}
