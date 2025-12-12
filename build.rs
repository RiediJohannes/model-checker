fn main() {
    println!("cargo:rustc-link-search=native=clib");
    println!("cargo:rustc-link-clib=static=minisat");
}