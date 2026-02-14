pub use ffi::Literal;

use crate::logic::resolution::{Partition, ResolutionProof};
use crate::logic::types::{Clause, CNF, XCNF};

use cxx::{CxxVector, UniquePtr};
use ffi::SolverStub;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Neg;
use std::pin::Pin;


// Fixed IDs to use for SAT variables representing boolean constants
pub const BOTTOM: i32 = 0;
pub const TOP: i32 = BOTTOM + 1;
pub const VAR_OFFSET: usize = 1;

pub const TRUE: Literal = Literal::raw(TOP);
pub const FALSE: Literal = Literal::raw(BOTTOM);


#[cxx::bridge]
pub mod ffi {
    // Shared structs, whose fields will be visible to both languages
    #[derive(Copy, Clone, Hash, PartialEq, Eq)]
    struct Literal {
        id: i32,
    }

    unsafe extern "C++" {
        include!("logic/minisat/Stub.h");

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
        fn notify_clause(self: &mut ResolutionProof, id: u32, lits: &[i32]);
        fn notify_resolution(self: &mut ResolutionProof, resolution_id: i32, left: i32, right: i32, pivot_var: i32, resolvent: &[i32]);
    }
}

impl Literal {
    pub const fn from_var(var: i32) -> Self {
        Self { id: var << 1 }
    }
    pub const fn raw(lit_id: i32) -> Self {
        Self { id: lit_id }
    }

    pub const fn var(&self) -> i32 {
        self.id >> 1
    }
    pub const fn is_pos(&self) -> bool {
        self.id & 1 == 0
    }
    pub const fn unsign(&self) -> Literal {
        Literal { id: self.id & !1 }
    }
}
impl Neg for Literal {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Literal { id: self.id ^ 1 }
    }
}
impl Debug for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Literal: {}", self.id)
    }
}
impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.id {
            0 => write!(f, "⊥"),
            1 => write!(f, "⊤"),
            _ => write!(f, "{}{}", if self.is_pos() { "" } else { "-" }, self.var()),
        }
    }
}
impl From<i32> for Literal {
    fn from(i: i32) -> Self {
        Literal { id: i }
    }
}


/// Thin wrapper around [SolverStub] to offer a more developer-friendly interface, plus some additional
/// methods for logic formula transformations.
pub struct Solver {
    stub: UniquePtr<SolverStub>,
    pub resolution: Option<Box<ResolutionProof>>,  // IMPORTANT to box this member, otherwise passing its reference to C++ will lead to memory-issues!
}

impl Solver {
    pub fn new() -> Self {
        let mut resolution = Box::new(ResolutionProof::new());

        let mut solver = Self {
            stub: ffi::newSolver(&mut resolution),
            resolution: Some(resolution),
        };

        // Add the constant false literal to the solver
        solver.clear_partition();

        let bottom = solver.add_var();
        assert_eq!(bottom.var(), BOTTOM);

        solver.add_clause([-bottom]);

        solver
    }

    pub fn add_var(&mut self) -> Literal {
        self.remote().newVar()
    }

    pub fn add_vars(&mut self, n: usize) -> Vec<Literal> {
        (0..n).map(|_| self.add_var()).collect()
    }

    pub fn add_clause<L>(&mut self, clause: L)
    where L: AsRef<[Literal]>
    {
        self.remote().addClause(clause.as_ref());
    }

    pub fn set_partition(&mut self, partition: Partition) {
        if let Some(proof) = self.resolution.as_mut() {
            proof.partition = Some(partition);
        }
    }

    pub fn clear_partition(&mut self) {
        if let Some(proof) = self.resolution.as_mut() {
            proof.partition = None;
        }
    }

    pub fn solve(&mut self) -> bool {
        self.remote().solve()
    }

    pub fn solve_assuming<L>(&mut self, assumptions: L) -> bool
    where L: AsRef<[Literal]> {
        self.remote().solve_with_assumptions(assumptions.as_ref())
    }

    pub fn get_model(&mut self) -> UniquePtr<CxxVector<i8>>
    {
        self.remote().getModel()
    }

    #[allow(non_snake_case)]
    pub fn tseitin_or(&mut self, left: &XCNF, right: &XCNF) -> XCNF {
        // Detect trivial cases
        if left == TRUE || right == TRUE {
            return XCNF::from(TRUE);
        } else if left == FALSE {
            return (*right).clone();
        } else if right == FALSE {
            return (*left).clone();
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = {
            let c1 = &Clause::new([-left.out_lit, tseitin_lit]);
            let c2 = &Clause::new( [-right.out_lit, tseitin_lit]);
            let c3 = &Clause::new([-tseitin_lit, left.out_lit, right.out_lit]);
            c1 & c2 & c3
        };

        let clauses = &left.clauses & &right.clauses & &tseitin_clauses;
        XCNF::new(clauses, tseitin_lit)
    }

    pub fn tseitin_and(&mut self, left: &XCNF, right: &XCNF) -> XCNF {
        // Detect trivial cases
        if left == FALSE || right == FALSE {
            return XCNF::from(FALSE);
        } else if left == TRUE {
            return (*right).clone();
        } else if right == TRUE {
            return (*left).clone();
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = {
            let c1 = &Clause::new([-tseitin_lit, left.out_lit]);
            let c2 = &Clause::new( [-tseitin_lit, right.out_lit]);
            let c3 = &Clause::new([-left.out_lit, -right.out_lit, tseitin_lit]);
            c1 & c2 & c3
        };

        let clauses = &left.clauses & &right.clauses & &tseitin_clauses;
        XCNF::new(clauses, tseitin_lit)
    }

    /// Quick and idiomatic access to a pinned mutable reference of the underlying solver.
    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.stub.pin_mut()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tseitin_or() {

    }

    #[test]
    fn tseitin_and() {

    }
}