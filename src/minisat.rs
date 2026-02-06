use std::fmt::Display;
use std::ops::Neg;

use ffi::Literal;

#[cxx::bridge]
pub mod ffi {
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

impl Literal {
    pub fn var(&self) -> i32 {
        self.id >> 1
    }
    pub fn is_pos(&self) -> bool {
        self.id & 1 == 0
    }
    pub fn unsign(&self) -> Literal {
        Literal { id: self.id & !1 }
    }
}

impl Neg for Literal {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Literal { id: self.id ^ 1 }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", if self.is_pos() { "" } else { "-" }, self.var())
    }
}

impl From<i32> for Literal {
    fn from(i: i32) -> Self {
        Literal { id: i << 1 }
    }
}

#[derive(Debug)]
pub struct Clause {
    lits: Box<[Literal]>,
}

impl Clause {
    pub fn new<I>(lits: I) -> Self
    where
        I: IntoIterator<Item = Literal>,
    {
        let v: Vec<Literal> = lits.into_iter().collect();
        Clause::from_vec(v)
    }

    pub fn from_vec(v: Vec<Literal>) -> Self {
        Self { lits: v.into_boxed_slice() }
    }
}
