use std::fmt::Display;
use std::ops::Neg;

#[cxx::bridge]
pub mod minisat {
    // Shared structs, whose fields will be visible to both languages
    #[derive(Debug, Copy, Clone)]
    struct Literal {
        id: i32,
    }

    // Potential Rust types that shall be visible to C++
    // extern "Rust" {
    //     type MultiBuf;
    //
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

        fn new_var(self: Pin<&mut SolverStub>) -> Literal;
        fn add_clause(self: Pin<&mut SolverStub>, clause: &[Literal]);
        fn solve(self: Pin<&mut SolverStub>) -> bool;
    }
}


impl minisat::Literal {
    pub fn var(&self) -> i32 { self.id >> 1 }
    pub fn is_pos(&self) -> bool { self.id & 1 == 0 }
    pub fn unsign(&self) -> minisat::Literal { minisat::Literal { id: self.id & !1 } }
}

impl Neg for minisat::Literal {
    type Output = Self;
    fn neg(self) -> Self::Output { minisat::Literal { id: self.id ^ 1 } }
}

impl Display for minisat::Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", if self.is_pos() {""} else {"-"}, self.var())
    }
}
