pub use ffi::Literal;

use crate::logic::resolution::{Partition, ResolutionProof};
use crate::logic::types::{Clause, CNF, XCNF};

use cxx::{CxxVector, UniquePtr};
use ffi::SolverStub;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Neg;
use std::pin::Pin;


// Fixed IDs to use for SAT variables representing boolean constants
const BOTTOM: i32 = 0;
const TOP: i32 = BOTTOM + 1;

pub const VAR_OFFSET: usize = 1;
pub const TRUE: Literal = Literal::raw(TOP);
pub const FALSE: Literal = Literal::raw(BOTTOM);


/// This submodule defines the contract for functions and types shared across the foreign function interface (FFI)
/// between Rust and C++.
#[cxx::bridge]
pub mod ffi {
    // Shared structs, whose fields will be visible to both languages
    /// A single literal in a SAT formula.
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
        #[rust_name = "newSolverWithProof"]
        fn newSolver(proof_store: &mut ResolutionProof) -> UniquePtr<SolverStub>;
        fn newSolver() -> UniquePtr<SolverStub>;

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
    /// Creates a [Literal] from a given variable ID.
    pub const fn from_var(var: i32) -> Self {
        Self { id: var << 1 }
    }
    /// Converts a raw integer to a [Literal] struct without additional parsing logic.
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

    /// Modifies the variable ID represented by this literal. The current variable ID is moved by the
    /// given shift, which can be both positive or negative. A shift of 0 is effectively a NOP.
    pub fn shift_var(&mut self, shift: i32) {
        let sign_bit = self.id & 1;
        let shifted_var = self.var() + shift;
        self.id = (shifted_var << 1) | sign_bit;
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
        Literal::raw(i)
    }
}


/// Wrapper around [SolverStub] to offer a more developer-friendly interface and enable proof-logging,
/// plus some additional methods for logic formula transformations (tseitin transformations).
pub struct Solver {
    stub: UniquePtr<SolverStub>,
    pub top_var: i32,   // ID of the highest variable instantiated so far
    pub resolution: Option<Box<ResolutionProof>>,  // IMPORTANT to box this member, otherwise passing its reference to C++ will lead to memory-issues!
}

impl Solver {
    pub fn new() -> Self {
        let mut resolution = Box::new(ResolutionProof::new());

        let mut solver = Self {
            stub: ffi::newSolverWithProof(&mut resolution),
            top_var: -1,
            resolution: Some(resolution),
        };


        // Add the constant false literal to the solver (in partition A)
        solver.set_partition(Partition::A);

        let bottom = solver.add_var();
        assert_eq!(bottom.var(), BOTTOM);
        solver.add_clause([-bottom]);

        solver.clear_partition();

        solver
    }

    pub fn add_var(&mut self) -> Literal {
        self.top_var += 1;
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

    /// Connects two given propositional formulas in [XCNF] with an **OR-gate** using the tseitin transformation.
    /// The returned [XCNF] object represents a CNF formula that is satisfiable iff the disjunction
    /// of the given parent clauses is satisfiable.
    pub fn tseitin_or(&mut self, left: &XCNF, right: &XCNF) -> XCNF {
        // Detect trivial cases
        if left == &TRUE || right == &TRUE {
            return XCNF::from(TRUE);
        } else if left == &FALSE {
            return (*right).clone();
        } else if right == &FALSE {
            return (*left).clone();
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = {
            let c1 = &Clause::new([-left.out_lit, tseitin_lit]);
            let c2 = &Clause::new( [-right.out_lit, tseitin_lit]);
            let c3 = &Clause::new([-tseitin_lit, left.out_lit, right.out_lit]);
            c1 & c2 & c3
        };

        let clauses = &left.formula & &right.formula & &tseitin_clauses;
        XCNF::new(clauses, tseitin_lit)
    }

    /// Connects two given propositional formulas in [XCNF] with an **AND-gate** using the tseitin transformation.
    /// The returned [XCNF] object represents a CNF formula that is satisfiable iff the conjunction
    /// of the given parent clauses is satisfiable.
    pub fn tseitin_and(&mut self, left: &XCNF, right: &XCNF) -> XCNF {
        // Detect trivial cases
        if left == &FALSE || right == &FALSE {
            return XCNF::from(FALSE);
        } else if left == &TRUE {
            return (*right).clone();
        } else if right == &TRUE {
            return (*left).clone();
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = {
            let c1 = &Clause::new([-tseitin_lit, left.out_lit]);
            let c2 = &Clause::new( [-tseitin_lit, right.out_lit]);
            let c3 = &Clause::new([-left.out_lit, -right.out_lit, tseitin_lit]);
            c1 & c2 & c3
        };

        let clauses = &left.formula & &right.formula & &tseitin_clauses;
        XCNF::new(clauses, tseitin_lit)
    }

    /// Quick and idiomatic access to a pinned mutable reference of the underlying solver.
    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.stub.pin_mut()
    }
}


/// Thin wrapper around [SolverStub] to offer a more developer-friendly interface.
/// This struct uses the solver without any proof logging capabilities for increased performance.
pub struct SimpleSolver {
    stub: UniquePtr<SolverStub>,
    pub top_var: i32,   // ID of the highest variable instantiated so far
    pub assumptions: Vec<Literal>,
}

impl SimpleSolver {
    pub fn new() -> Self {
        let mut solver = Self {
            stub: ffi::newSolver(),
            top_var: -1,
            assumptions: Vec::new(),
        };

        // Add the constant false literal to the solver
        let bottom = solver.add_var();
        assert_eq!(bottom.var(), BOTTOM);
        solver.add_clause([-bottom]);

        solver
    }

    pub fn add_var(&mut self) -> Literal {
        self.top_var += 1;
        self.remote().newVar()
    }

    pub fn add_clause<L>(&mut self, clause: L)
    where L: AsRef<[Literal]>
    {
        self.remote().addClause(clause.as_ref());
    }

    pub fn solve(&mut self) -> bool {
        let assumptions = self.assumptions.clone();
        self.remote().solve_with_assumptions(&assumptions)
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


// ============== Unit Tests ================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::cnf;
    use crate::logic::resolution::{Partition, VariableLocality};
    use crate::logic::solving::{Solver, FALSE, TRUE, VAR_OFFSET};
    use crate::logic::{Clause, Literal, XCNF};

    impl Solver {
        /// Helper function to collect the added clause in a Clause struct
        fn add_and_get_clause<L>(&mut self, literals: L) -> Clause
        where L: AsRef<[Literal]>, Clause: From<L> {
            self.add_clause(literals.as_ref());
            Clause::from(literals)
        }
    }

    impl XCNF {
        fn contains_all(&self, clauses: &[Clause]) -> bool {
            clauses.iter().all(|c| self.formula.clauses.contains(c))
        }
    }

    fn setup_interpolants() -> (Solver, XCNF, XCNF) {
        const N_VARS: usize = 4;
        let mut solver = Solver::new();
        let vars = solver.add_vars(N_VARS);
        let (x, y, z1, z2) = (vars[0], vars[1], vars[2], vars[3]);

        let I1 = XCNF::new(cnf![[-z1, x], [-z1, y], [z1, -x, -y]], z1);
        let I2 = XCNF::new(cnf![[z2, -x], [z2, -y], [-z2, x, y]], z2);

        assert_eq!(z2.var() as usize, VAR_OFFSET - 1 + N_VARS);
        (solver, I1, I2)
    }

    /// Check if the public `top_var` counter of the solver accurately tracks the highest variable ID
    /// currently instantiated in the solver.
    #[test]
    fn top_var_counter() {
        let mut solver: Solver = Solver::new();

        assert_eq!(solver.top_var, 0);
        assert_eq!(solver.top_var, FALSE.var());

        let x1 = solver.add_var();
        assert_eq!(solver.top_var, 1);
        assert_eq!(solver.top_var, x1.var());
        assert_eq!(x1.var(), VAR_OFFSET as i32);

        let new_vars = solver.add_vars(9);
        assert_eq!(solver.top_var, 10);
        assert_eq!(solver.top_var, new_vars.last().unwrap().var());

        // Clauses do not change the var counter
        solver.add_clause([x1, -new_vars[1]]);
        solver.add_clause([-new_vars[7]]);
        assert_eq!(solver.top_var, 10);

        let x11 = solver.add_var();
        assert_eq!(solver.top_var, 11);
        assert_eq!(solver.top_var, x11.var());
    }

    #[test]
    fn variable_partition() {
        let mut solver = Solver::new();

        let a1 = solver.add_var();
        let a2 = solver.add_var();
        let s  = solver.add_var();
        let b1 = solver.add_var();
        let b2 = solver.add_var();

        solver.set_partition(Partition::A);
        solver.add_clause([-a1, a2]);
        solver.add_clause([a2, s]);

        solver.set_partition(Partition::B);
        solver.add_clause([b1, -b2]);
        solver.add_clause([b1, -s]);
        // Force unsatisfiability
        solver.add_clause([-b1]);
        solver.add_clause([s]);

        assert!(!solver.solve());  // formula should be unsat

        let proof = solver.resolution.as_ref().unwrap();
        assert_eq!(proof.var_locality(a1), Some(VariableLocality::Local(Partition::A)));
        assert_eq!(proof.var_locality(a2), Some(VariableLocality::Local(Partition::A)));
        assert_eq!(proof.var_locality(s), Some(VariableLocality::Shared));
        assert_eq!(proof.var_locality(b1), Some(VariableLocality::Local(Partition::B)));
        assert_eq!(proof.var_locality(b2), Some(VariableLocality::Local(Partition::B)));
    }

    #[test]
    fn clause_partition() {
        let mut solver = Solver::new();

        let x = solver.add_var();
        let y = solver.add_var();
        let z  = solver.add_var();

        solver.set_partition(Partition::A);
        let C_A1 = solver.add_and_get_clause([-x, y]);
        let C_A2 = solver.add_and_get_clause([-y, z]);

        solver.set_partition(Partition::B);
        let C_B1 = solver.add_and_get_clause([x]);
        let C_B2 = solver.add_and_get_clause([-x, y]);  // duplicate from partition A
        let C_B3 = solver.add_and_get_clause([-y, -z]);

        assert!(!solver.solve());  // formula should be unsat

        let proof = solver.resolution.as_ref().unwrap();
        let A_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::A).map(|c| proof.get_clause(*c).unwrap().clone()).collect();
        let B_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::B).map(|c| proof.get_clause(*c).unwrap().clone()).collect();

        assert!(A_clauses.contains(&C_A1));
        assert!(A_clauses.contains(&C_A2));
        assert!(B_clauses.contains(&C_B1));
        assert!(B_clauses.contains(&C_B2));
        assert!(B_clauses.contains(&C_B3));
    }

    #[test]
    fn tseitin_or_trivial() {
        let mut solver = Solver::new();
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        // Case 1: Trivial TRUE
        let T_or_T = solver.tseitin_or(&T, &T);
        assert_eq!(&T_or_T, &T);

        // Case 1: Trivial mixed TRUE/FALSE
        let T_or_F = solver.tseitin_or(&T, &F);
        assert_eq!(&T_or_F, &T);
        let F_or_T = solver.tseitin_or(&F, &T);
        assert_eq!(&F_or_T, &T);

        // Case 3: Trivial FALSE
        let F_or_F = solver.tseitin_or(&F, &F);
        assert_eq!(&F_or_F, &F);

        // Assert that we did not need any auxiliary tseitin variables for these cases
        assert_eq!(solver.top_var, 0);
    }

    #[test]
    fn tseitin_or_arbitrary() {
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        let (mut solver, I1, I2) = setup_interpolants();
        let initial_top = solver.top_var;

        // Case 1: I or TRUE
        let I1_or_T = solver.tseitin_or(&I1, &T);
        assert_eq!(&I1_or_T, &T);
        let I2_or_T = solver.tseitin_or(&I2, &T);
        assert_eq!(&I2_or_T, &T);

        // Case 2: I or FALSE
        let I1_or_F = solver.tseitin_or(&I1, &F);
        assert_eq!(&I1_or_F, &I1);
        let I2_or_F = solver.tseitin_or(&I2, &F);
        assert_eq!(&I2_or_F, &I2);

        // Assert that we did not need any additional tseitin variables for these cases
        assert_eq!(solver.top_var, initial_top);

        // Case 3: I1 or I2
        let I1_or_I2 = solver.tseitin_or(&I1, &I2);
        assert!(I1_or_I2.contains_all(&I1.formula.clauses));
        assert!(I1_or_I2.contains_all(&I2.formula.clauses));

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_or_I2.formula.len(), I1.formula.len() + I2.formula.len() + 3);
        // The now top variable should have become the output literal
        let t = Literal::from_var(solver.top_var);
        assert_eq!(I1_or_I2.out_lit, t);

        let tseitin_clauses = vec![
            Clause::from([t, -I1.out_lit]),
            Clause::from([t, -I2.out_lit]),
            Clause::from([-t, I1.out_lit, I2.out_lit])
        ];
        assert!(I1_or_I2.contains_all(&tseitin_clauses));

        // Check: We needed exactly one additional variable, no more
        assert_eq!(solver.top_var, initial_top + 1);
    }

    #[test]
    fn tseitin_and_trivial() {
        let mut solver = Solver::new();
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        // Case 1: Trivial TRUE
        let T_and_T = solver.tseitin_and(&T, &T);
        assert_eq!(&T_and_T, &T);

        // Case 1: Trivial mixed TRUE/FALSE
        let T_and_F = solver.tseitin_and(&T, &F);
        assert_eq!(&T_and_F, &F);
        let F_and_T = solver.tseitin_and(&F, &T);
        assert_eq!(&F_and_T, &F);

        // Case 3: Trivial FALSE
        let F_and_F = solver.tseitin_and(&F, &F);
        assert_eq!(&F_and_F, &F);

        // Assert that we did not need any auxiliary tseitin variables for these cases
        assert_eq!(solver.top_var, 0);
    }

    #[test]
    fn tseitin_and_arbitrary() {
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        let (mut solver, I1, I2) = setup_interpolants();
        let initial_top = solver.top_var;

        // Case 1: I or TRUE
        let I1_and_T = solver.tseitin_and(&I1, &T);
        assert_eq!(&I1_and_T, &I1);
        let I2_and_T = solver.tseitin_and(&I2, &T);
        assert_eq!(&I2_and_T, &I2);

        // Case 2: I or FALSE
        let I1_and_F = solver.tseitin_and(&I1, &F);
        assert_eq!(&I1_and_F, &F);
        let I2_and_F = solver.tseitin_and(&I2, &F);
        assert_eq!(&I2_and_F, &F);

        // Assert that we did not need any additional tseitin variables for these cases
        assert_eq!(solver.top_var, initial_top);

        // Case 3: I1 or I2
        let I1_and_I2 = solver.tseitin_and(&I1, &I2);
        assert!(I1_and_I2.contains_all(&I1.formula.clauses));
        assert!(I1_and_I2.contains_all(&I2.formula.clauses));

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_and_I2.formula.len(), I1.formula.len() + I2.formula.len() + 3);
        // The now top variable should have become the output literal
        let t = Literal::from_var(solver.top_var);
        assert_eq!(I1_and_I2.out_lit, t);

        let tseitin_clauses = vec![
            Clause::from([-t, I1.out_lit]),
            Clause::from([-t, I2.out_lit]),
            Clause::from([t, -I1.out_lit, -I2.out_lit])
        ];
        assert!(I1_and_I2.contains_all(&tseitin_clauses));

        // Check: We needed exactly one additional variable, no more
        assert_eq!(solver.top_var, initial_top + 1);
    }
}