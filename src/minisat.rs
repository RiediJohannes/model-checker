use std::collections::HashMap;
use std::fmt::Display;
use std::ops::Neg;
use std::pin::Pin;
use ffi::SolverStub;

pub use ffi::Literal;

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
        fn newSolver() -> UniquePtr<SolverStub>;

        fn newVar(self: Pin<&mut SolverStub>) -> Literal;
        fn addClause(self: Pin<&mut SolverStub>, clause: &[Literal]);
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

// #[derive(Debug)]
// pub struct Clause {
//     lits: Box<[Literal]>,
// }
//
// impl Clause {
//     pub fn new<I>(lits: I) -> Self
//     where
//         I: IntoIterator<Item = Literal>,
//     {
//         let v: Vec<Literal> = lits.into_iter().collect();
//         Clause::from_vec(v)
//     }
//
//     pub fn from_vec(v: Vec<Literal>) -> Self {
//         Self { lits: v.into_boxed_slice() }
//     }
// }

pub struct Solver {
    base: cxx::UniquePtr<SolverStub>,
    var_pool: HashMap<String, Literal>,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            base: ffi::newSolver(),
            var_pool: HashMap::new(),
        }
    }

    pub fn add_var(&mut self) -> Literal {
        self.remote().newVar()
    }

    pub fn add_vars(&mut self, n: usize) -> Vec<Literal> {
        (0..n).map(|_| self.add_var()).collect()
    }

    pub fn add_clause<L>(&mut self, clause: L)
    where
        L: AsRef<[Literal]>,
    {
        self.remote().addClause(clause.as_ref());
    }

    pub fn solve(&mut self) -> bool {
        self.remote().solve()
    }

    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.base.pin_mut()
    }
}
