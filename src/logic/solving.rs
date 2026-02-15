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

        solver.clear_partition();

        // Add the constant false literal to the solver
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


// ============== Unit Tests ================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::cnf;
    use crate::logic::resolution::Partition;
    use crate::logic::solving::{Solver, FALSE, VAR_OFFSET};
    use crate::logic::{types, Clause, Literal, TRUE, XCNF};

    impl Solver {
        /// Helper function to collect the added clause in a Clause struct
        fn add_and_get_clause<L>(&mut self, literals: L) -> Clause
        where L: AsRef<[Literal]>, types::Clause: From<L> {
            self.add_clause(literals.as_ref());
            Clause::from(literals)
        }
    }

    /// Checks if the solver meets the expected preconditions upon construction via the new function.
    #[test]
    fn solver_preconditions() {
        let mut solver: Solver = Solver::new();

        let proof = solver.resolution.as_ref().expect("Solver should have a resolution proof object");
        assert_eq!(proof.partition, None);

        let x = solver.add_var();
        assert_eq!(x.var(), VAR_OFFSET as i32);

        let sat_if_unchanged = solver.solve();
        assert!(sat_if_unchanged);

        // Check if forcing the (supposedly) constant false literal to true leads to an inconsistency
        solver.add_clause([FALSE]);
        let unsat_if_bottom_asserted_as_true = solver.solve();
        assert!(!unsat_if_bottom_asserted_as_true);
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
        assert_eq!(proof.var_partition(a1), Some(Partition::A));
        assert_eq!(proof.var_partition(a2), Some(Partition::A));
        assert_eq!(proof.var_partition(s),  Some(Partition::AB));
        assert_eq!(proof.var_partition(b1), Some(Partition::B));
        assert_eq!(proof.var_partition(b2), Some(Partition::B));
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
        let x_next = solver.add_var();
        assert_eq!(x_next.var(), VAR_OFFSET as i32);
    }

    #[test]
    fn tseitin_or_arbitrary() {
        let mut solver = Solver::new();
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        const N_VARS: usize = 4;
        let mut id_counter = VAR_OFFSET - 1;
        let vars = solver.add_vars(N_VARS); id_counter += N_VARS;
        let x = vars[0];
        let y = vars[1];
        let z1 = vars[2];
        let z2 = vars[3];
        assert_eq!(z2.var() as usize, id_counter);
        let I1 = XCNF::new(cnf![[-z1, x], [-z1, y], [z1, -x, -y]], z1);
        let I2 = XCNF::new(cnf![[z2, -x], [z2, -y], [-z2, x, y]], z2);

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
        let x_next = solver.add_var(); id_counter += 1;
        assert_eq!(x_next.var() as usize, id_counter);

        // Case 3: I1 or I2
        let I1_or_I2 = solver.tseitin_or(&I1, &I2);
        id_counter += 1;  // this time, the tseitin transformation needs an auxiliary variable
        for C1 in &I1.formula {
            assert!(&I1_or_I2.formula.clauses.contains(C1));
        }
        for C2 in &I2.formula {
            assert!(&I1_or_I2.formula.clauses.contains(C2));
        }

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_or_I2.formula.len(), I1.formula.len() + I2.formula.len() + 3);
        // The now top variable should have become the output literal
        let t = Literal::from_var(id_counter as i32);
        assert_eq!(I1_or_I2.out_lit, t);

        let tseitin_clauses = vec![
            Clause::from([t, -I1.out_lit]),
            Clause::from([t, -I2.out_lit]),
            Clause::from([-t, I1.out_lit, I2.out_lit])
        ];
        for C_T in tseitin_clauses {
            assert!(&I1_or_I2.formula.clauses.contains(&C_T));
        }

        // Check: We needed exactly one additional variable, no more
        let x_next = solver.add_var(); id_counter += 1;
        assert_eq!(x_next.var() as usize, id_counter);
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
        let x_next = solver.add_var();
        assert_eq!(x_next.var(), VAR_OFFSET as i32);
    }

    fn tseitin_and_arbitrary() {
        let mut solver = Solver::new();
        let T = XCNF::from(TRUE);
        let F = XCNF::from(FALSE);

        const N_VARS: usize = 4;
        let mut id_counter = VAR_OFFSET - 1;
        let vars = solver.add_vars(N_VARS); id_counter += N_VARS;
        let x = vars[0];
        let y = vars[1];
        let z1 = vars[2];
        let z2 = vars[3];
        assert_eq!(z2.var() as usize, id_counter);
        let I1 = XCNF::new(cnf![[-z1, x], [-z1, y], [z1, -x, -y]], z1);
        let I2 = XCNF::new(cnf![[z2, -x], [z2, -y], [-z2, x, y]], z2);

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
        let x_next = solver.add_var(); id_counter += 1;
        assert_eq!(x_next.var() as usize, id_counter);

        // Case 3: I1 or I2
        let I1_and_I2 = solver.tseitin_and(&I1, &I2);
        id_counter += 1;  // this time, the tseitin transformation needs an auxiliary variable
        for C1 in &I1.formula {
            assert!(&I1_and_I2.formula.clauses.contains(C1));
        }
        for C2 in &I2.formula {
            assert!(&I1_and_I2.formula.clauses.contains(C2));
        }

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_and_I2.formula.len(), I1.formula.len() + I2.formula.len() + 3);
        // The now top variable should have become the output literal
        let t = Literal::from_var(id_counter as i32);
        assert_eq!(I1_and_I2.out_lit, t);

        let tseitin_clauses = vec![
            Clause::from([-t, I1.out_lit]),
            Clause::from([-t, I2.out_lit]),
            Clause::from([t, -I1.out_lit, -I2.out_lit])
        ];
        for C_T in tseitin_clauses {
            assert!(&I1_and_I2.formula.clauses.contains(&C_T));
        }

        // Check: We needed exactly one additional variable, no more
        let x_next = solver.add_var(); id_counter += 1;
        assert_eq!(x_next.var() as usize, id_counter);
    }
}