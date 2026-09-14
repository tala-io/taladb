cmake_minimum_required(VERSION 3.22)
project(taladb_jsi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the taladb-react-native package
get_filename_component(PACKAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# React Native JSI headers
# ---------------------------------------------------------------------------
find_package(ReactAndroid REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# taladb_ffi shared library (Rust pre-built)
# ---------------------------------------------------------------------------

# Add the directory containing libtaladb_ffi.so to the linker search path.
# This ensures we link by name (soname) rather than embedding a relative path.
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}")

# ---------------------------------------------------------------------------
# taladb_jsi — C++ JSI HostObject, compiled into a thin shared library that
# the Kotlin module loads via System.loadLibrary("taladb_jsi").
# ---------------------------------------------------------------------------
add_library(taladb_jsi SHARED
    "${PACKAGE_ROOT}/cpp/TalaDBHostObject.cpp"
    "${PACKAGE_ROOT}/cpp/TalaDBJni.cpp"
)

target_include_directories(taladb_jsi PRIVATE
    "${PACKAGE_ROOT}/cpp"
)

target_link_libraries(taladb_jsi
    taladb_ffi
    ReactAndroid::jsi
    android
    log
)

# ---------------------------------------------------------------------------
# 16 KB page size (Android 15+)
#
# A device with 16 KB pages cannot load a shared library whose LOAD segments are
# aligned to 4 KB, and Google Play requires 16 KB support of every app targeting
# Android 15 or later. NDK r28+ links with a 16 KB max-page-size by default, but
# r27 and earlier do not — so set it here rather than inheriting whatever the
# consuming app's NDK happens to default to. The prebuilt Rust libraries are
# already aligned; this is the one object we link ourselves.
# ---------------------------------------------------------------------------
target_link_options(taladb_jsi PRIVATE "-Wl,-z,max-page-size=16384")
