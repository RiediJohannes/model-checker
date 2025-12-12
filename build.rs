use std::path::Path;
use cxx_build::CFG;

fn main() {
    // Resolve this full path once here to allow for relative paths when importing C++ header files
    let path = Path::new("./src").canonicalize().expect("Failed to resolve 'src' directory");
    CFG.exported_header_dirs.extend(vec![path.as_path()]);

    cxx_build::bridge("src/interop.rs")
        .file("src/minisat/File.C")
        .file("src/minisat/Proof.C")
        .file("src/minisat/Solver.C")
        .file("src/minisat/Test.C")
        .std("gnu++11")
        .flag("-O3")
        .flag("-ffloat-store")
        .flag("-Wno-unused-result")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-deprecated")
        .flag("-Wno-strict-aliasing")
        .compile("cxx-minisat");

    // println!("cargo:rustc-link-search=native=clib");
    // println!("cargo:rustc-link-clib=static=minisat");

    // g++ -ffloat-store -std=gnu++98 -O3 File.C Main.C Proof.C Solver.C -lz -o minisat
}