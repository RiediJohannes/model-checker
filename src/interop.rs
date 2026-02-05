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
        include!("minisat/Test.h");

        fn addInts(a: i32, b: i32) -> i32;

        include!("minisat/Api.h");

        // Opaque types which both languages can pass around but only C++ can see the fields
        type SolverStub;

        // Functions implemented in C++.
        fn new_solver() -> UniquePtr<SolverStub>;
        // fn new_solver_pin() -> SolverStub;

        // fn solve(self: &BlobstoreClient) -> bool;
        fn solve(self: Pin<&mut SolverStub>) -> bool;

        // fn new_var(solver: Pin<&mut SolverStub>) -> i32;
        // fn solve(solver: Pin<&mut SolverStub>) -> bool;
        // fn model_value(solver: Pin<&mut SolverStub>, var: i32) -> i32;

        // self: &BlobstoreClient
    }
}