use cxx::{CxxVector, UniquePtr};
pub use ffi::Literal;
use ffi::SolverStub;
use std::fmt::Display;
use std::ops::Neg;
use std::pin::Pin;

#[cxx::bridge]
pub mod ffi {
    // Shared structs, whose fields will be visible to both languages
    #[derive(Debug, Copy, Clone)]
    struct Literal {
        id: i32,
    }

    unsafe extern "C++" {
        include!("minisat/Api.h");

        // Opaque types which both languages can pass around but only C++ can see the fields
        /// Proxy handle of the minisat SAT solver accessed through the Rust<>C++ interop.
        type SolverStub;

        // Functions implemented in C++.
        fn newSolver(proof_store: &mut ResolutionProof) -> UniquePtr<SolverStub>;
        fn newVar(self: Pin<&mut SolverStub>) -> Literal;
        fn addClause(self: Pin<&mut SolverStub>, clause: &[Literal]);
        fn solve(self: Pin<&mut SolverStub>) -> bool;
        #[rust_name = "solve_with_assumptions"]
        fn solve(self: Pin<&mut SolverStub>, assumptions: &[Literal]) -> bool;
        fn getModel(self: Pin<&mut SolverStub>) -> UniquePtr<CxxVector<i8>>;
    }

    // Rust functions visible in C++
    extern "Rust" {
        type ResolutionProof;
        // fn notify_clause(self: &mut ResolutionProof, id: u32, lits: &[i32]);
        fn notify_clause(self: &mut ResolutionProof, id: u32, lits: UniquePtr<CxxVector<i32>>);
        fn notify_resolution(self: &mut ResolutionProof, resolution_id: i32, left: i32, right: i32, pivot: i32, resolvent: &[i32]);
    }
}

impl Literal {
    pub fn from_var(var: i32) -> Self {
        Self { id: var << 1 }
    }

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
        Literal { id: i }
    }
}


/// Thin wrapper around [SolverStub] to offer a more developer-friendly interface.
pub struct Solver {
    stub: UniquePtr<SolverStub>,
    resolution: Box<ResolutionProof>,  // IMPORTANT to box this member, otherwise passing its reference to C++ will lead to memory-issues!
}

impl Solver {
    pub fn new() -> Self {
        let mut resolution = Box::new(ResolutionProof::new());

        Self {
            stub: ffi::newSolver(&mut resolution),
            resolution,
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

    pub fn solve_assuming<L>(&mut self, assumptions: L) -> bool
    where
        L: AsRef<[Literal]>
    {
        self.remote().solve_with_assumptions(assumptions.as_ref())
    }

    pub fn get_model(&mut self) -> UniquePtr<CxxVector<i8>>
    {
        self.remote().getModel()
    }

    /// Quick and idiomatic access to a pinned mutable reference of the underlying solver.
    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.stub.pin_mut()
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


struct ResolutionStep {
    left: i32,
    right: i32,
    pivot: Literal,
    resolvent: i32,
}
impl ResolutionStep {
    pub fn new<L>(left: i32, right: i32, pivot: L, resolvent_id: i32) -> Self
    where L: Into<Literal>
    {
        Self {
            left,
            right,
            pivot: pivot.into(),
            resolvent: resolvent_id
        }
    }
}

pub struct ResolutionProof {
    root_clauses: Vec<Clause>,
    intermediate_clauses: Vec<Clause>,
    resolution_steps: Vec<ResolutionStep>,
}

impl ResolutionProof {
    pub fn new() -> Self {
        Self {
            root_clauses: Vec::new(),
            intermediate_clauses: Vec::new(),
            resolution_steps: Vec::new(),
        }
    }

    // pub fn notify_clause(self: &mut ResolutionProof, id: u32, lits: &[i32]) {
    //     let clause = Clause::new(lits.iter().map(|&lit| Literal::from(lit.clone())));
    //     self.root_clauses.push(clause);
    //     dbg!(id, self.root_clauses.len() - 1);
    // }

    pub fn notify_clause(&mut self, id: u32, lits: UniquePtr<CxxVector<i32>>) {
        let clause = Clause::new(
            lits.iter().map(|&lit| Literal::from(lit))
        );

        self.root_clauses.push(clause);
        assert_eq!(id as usize, self.root_clauses.len() - 1);
    }

    pub fn notify_resolution(self: &mut ResolutionProof, resolvent_id: i32, left: i32, right: i32, pivot: i32, resolvent: &[i32]) {
        let resolved_clause = Clause::new(
            resolvent.iter().map(|&lit| Literal::from(lit))
        );

        self.resolution_steps.push(ResolutionStep::new(left, right, pivot, resolvent_id));

        if resolvent_id >= 0 {
            self.root_clauses.push(resolved_clause);
            assert_eq!(resolvent_id as usize, self.root_clauses.len() - 1);
        } else {
            self.intermediate_clauses.push(resolved_clause);
            assert_eq!((-resolvent_id) as usize, self.intermediate_clauses.len());
        }
    }
}