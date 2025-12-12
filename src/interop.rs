#[cxx::bridge]
pub mod minisat {
    // Any shared structs, whose fields will be visible to both languages.
    // struct BlobMetadata {
    //     size: usize,
    //     tags: Vec<String>,
    // }

    // extern "Rust" {
    //     // Zero or more opaque types which both languages can pass around but
    //     // only Rust can see the fields.
    //     type MultiBuf;
    //
    //     // Functions implemented in Rust.
    //     fn next_chunk(buf: &mut MultiBuf) -> &[u8];
    // }

    unsafe extern "C++" {
        include!("minisat/Solver.h");
        include!("minisat/Test.h");

        // Zero or more opaque types which both languages can pass around but
        // only C++ can see the fields.
        type Solver;
        type FileMode;

        // Functions implemented in C++.
        // fn new_blobstore_client() -> UniquePtr<BlobstoreClient>;

        fn addInts(a: i32, b: i32) -> i32;
    }
}