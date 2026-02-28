pub use ffi::Literal;
use std::collections::{HashMap, TryReserveError};

use crate::logic::resolution::{Partition, ResolutionProof};
use crate::logic::types::{Clause, CNF, XCNF};

use crate::cnf;
use crate::logic::resolution::VariableLocality::{Local, Shared};
use cxx::{CxxVector, UniquePtr};
use ffi::SolverStub;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, Neg};
use std::pin::Pin;
use thiserror::Error;


// Fixed IDs to use for SAT variables representing boolean constants
const BOTTOM: i32 = 0;
const TOP: i32 = BOTTOM + 1;

/// The ID that the first user-added SAT variable is assigned (smaller IDs are reserved for fixed constants).
pub const VAR_OFFSET: usize = 1;
/// A [Literal] representing the constant **true** value (verum).
pub const TRUE: Literal = Literal::raw(TOP);
/// A [Literal] representing the constant **false** value (falsum).
pub const FALSE: Literal = Literal::raw(BOTTOM);


/// This submodule defines the contract for functions and types shared across the foreign function interface (FFI)
/// between Rust and C++. The CXX-Bridge library ensures that all types and functions defined within
/// this module are correctly cross-compiled and linked between the Rust and C++ part of this application.
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

    // Rust type names and functions visible to C++
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

    #[allow(dead_code)]
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

    /// Computes an interpolant for the two inconsistent partitions `(A, B)` of clauses in the [BmcModel]
    /// according to the Huang-Krajíček-Pudlák interpolation system.
    /// The resulting interpolant will over-approximate the states that satisfy the clauses in [Partition::A].
    #[allow(non_snake_case)]
    pub fn compute_interpolant(&mut self) -> Result<XCNF, InterpolationError> {
        // Temporarily take ownership of the resolution proof to simultaneously allow for mutable references
        // to the solver during this function
        let proof_box = self.resolution.take().ok_or(InterpolationError::MissingProof)?;
        let proof = proof_box.deref();

        let mut interpolation = InterpolationStorage::new(proof.len())?;

        // Base case: Annotate root clauses in partition A/B with Bottom/Top
        for A_clause_id in proof.clauses_in_partition(Partition::A) {
            interpolation.mapping.insert(*A_clause_id, FALSE);
        }

        for B_clause_id in proof.clauses_in_partition(Partition::B) {
            interpolation.mapping.insert(*B_clause_id, TRUE);
        }

        // Inductive case: Compute new part. interpolant from previous part. interpolants
        for step in proof.resolutions() {
            let I_L: Literal = *interpolation.mapping.get(&step.left)
                .ok_or(InterpolationError::MissingInterpolant(step.left))?;
            let I_R: Literal = *interpolation.mapping.get(&step.right)
                .ok_or(InterpolationError::MissingInterpolant(step.right))?;
            debug_assert!(!interpolation.mapping.contains_key(&step.resolvent));

            let pivot_locality = proof.var_locality(step.pivot)
                .ok_or(InterpolationError::MissingVariablePartition(step.pivot.var()))?;

            let I_resolvent: Interpolant = match pivot_locality {
                Local(Partition::A) => {  // I_L OR I_R
                    self.tseitin_or(I_L, I_R)
                },
                Local(Partition::B) => {  // I_L AND I_R
                    self.tseitin_and(I_L, I_R)
                },
                Shared => { // (I_L OR x) AND (I_R OR ~x)
                    let x = step.pivot;
                    let left_conjunct = self.tseitin_or(I_L, x);
                    let right_conjunct = self.tseitin_or(I_R, -x);

                    self.tseitin_and_interpolants(left_conjunct, right_conjunct)
                }
            };

            // println!(
            //     "C_L = {}\t -> I_L: {}\n\
            //     C_R = {}\t -> I_R: {}\n\
            //     -- Resolving on literal {:} (partition: {:?})\n\
            //     => {}\n",
            //         proof.get_clause(step.left).unwrap(), &I_L,
            //         proof.get_clause(step.right).unwrap(), &I_R,
            //         &step.pivot, &pivot_locality, &I_resolvent
            // );

            interpolation.add_interpolant(step.resolvent, I_resolvent);
        }

        self.resolution = Some(proof_box);   // return proof ownership

        let final_interpolant: XCNF = interpolation.into();
        Ok(final_interpolant)
    }

    /// Connects two given propositional formulas in [XCNF] with an **OR-gate** using the tseitin transformation.
    /// The returned [XCNF] object represents a CNF formula that is satisfiable iff the disjunction
    /// of the given parent clauses is satisfiable.
    pub fn tseitin_or(&mut self, left: Literal, right: Literal) -> Interpolant {
        // Detect trivial cases
        if left == TRUE || right == TRUE {
            return Interpolant::Literal(TRUE);
        } else if left == FALSE {
            return Interpolant::Literal(right);
        } else if right == FALSE || left == right {
            return Interpolant::Literal(left);
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = cnf![
            [-left, tseitin_lit],
            [-right, tseitin_lit],
            [-tseitin_lit, left, right]
        ];

        Interpolant::Formula(XCNF::new(tseitin_clauses, tseitin_lit))
    }

    /// Connects two given propositional formulas in [XCNF] with an **AND-gate** using the tseitin transformation.
    /// The returned [XCNF] object represents a CNF formula that is satisfiable iff the conjunction
    /// of the given parent clauses is satisfiable.
    pub fn tseitin_and(&mut self, left: Literal, right: Literal) -> Interpolant {
        // Detect trivial cases
        if left == FALSE || right == FALSE {
            return Interpolant::Literal(FALSE);
        } else if left == TRUE {
            return Interpolant::Literal(right);
        } else if right == TRUE || left == right {
            return Interpolant::Literal(left);
        }

        let tseitin_lit: Literal = self.add_var();
        let tseitin_clauses: CNF = cnf![
            [-tseitin_lit, left],
            [-tseitin_lit, right],
            [-left, -right, tseitin_lit]
        ];

        Interpolant::Formula(XCNF::new(tseitin_clauses, tseitin_lit))
    }

    pub fn tseitin_and_interpolants(&mut self, left_itp: Interpolant, right_itp: Interpolant) -> Interpolant {
        let conjunction_itp = self.tseitin_and(left_itp.literal(), right_itp.literal());
        if let Interpolant::Literal(lit) = conjunction_itp && (lit == TRUE || lit == FALSE) {
            return conjunction_itp;
        }

        let conjunction_lit = conjunction_itp.literal();
        let combined_clauses = left_itp.into_cnf() & right_itp.into_cnf() & conjunction_itp.into_cnf();

        Interpolant::Formula(XCNF::new(combined_clauses, conjunction_lit))
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

    #[expect(unused)]
    pub fn get_model(&mut self) -> UniquePtr<CxxVector<i8>>
    {
        self.remote().getModel()
    }

    /// Quick and idiomatic access to a pinned mutable reference of the underlying solver.
    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.stub.pin_mut()
    }
}


// --------------- Interpolation --------------

#[derive(Error, Debug)]
pub enum InterpolationError {
    #[error("SAT solver had no resolution proof attached")]
    MissingProof,

    #[error("Clause {0} was missing an interpolant")]
    MissingInterpolant(i32),

    #[error("Variable {0} has not been assigned to any variable partition (A/B)")]
    MissingVariablePartition(i32),

    #[error("Program ran out of available memory during interpolation")]
    OutOfMemory(#[from] TryReserveError),
}

#[derive(Debug)]
pub enum Interpolant {
    Literal(Literal),
    Formula(XCNF),
}
impl Interpolant {
    /// Converts the interpolant into a [CNF], thereby consuming the original interpolant.
    pub fn into_cnf(self) -> CNF {
        match self {
            Interpolant::Literal(_) => CNF::from(vec![]),
            Interpolant::Formula(xcnf) => xcnf.formula,
        }
    }

    pub fn literal(&self) -> Literal {
        match self {
            Interpolant::Literal(lit) => *lit,
            Interpolant::Formula(xcnf) => xcnf.out_lit,
        }
    }
}
impl From<Literal> for Interpolant {
    fn from(lit: Literal) -> Self {
        Interpolant::Literal(lit)
    }
}
impl Display for Interpolant {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Interpolant::Literal(lit) => write!(f, "{}", lit),
            Interpolant::Formula(xcnf) => write!(f, "{:?}", xcnf),
        }
    }
}

/// Stores the tseitin [Literal] uniquely representing a partial interpolant for each clause id as
/// well as all the clauses making up the interpolant.
struct InterpolationStorage {
    pub mapping: HashMap<i32, Literal>,
    clause_database: Vec<Clause>,
    output_literal: Literal,
}

impl InterpolationStorage {
    /// Creates a new storage for partial interpolants during algorithmic interpolation with
    /// sufficient pre-allocated capacities to avoid relocation.
    /// Note that the construction of this object might fail if the requested memory exceeds
    /// the memory currently available on the system.
    /// ## Arguments
    /// - `proof_size` - The size of the [ResolutionProof] for which an interpolant is computed in terms
    ///   of the number of resolution steps in the proof tree.
    pub fn new(proof_size: usize) -> Result<Self, TryReserveError> {
        let mut interpolant_mapping = HashMap::new();
        interpolant_mapping.try_reserve(2 * proof_size + 1)?;

        // Except three tseitin clauses per resolution step
        let mut clause_vector = Vec::new();
        clause_vector.try_reserve(3 * proof_size)?;

        Ok(Self {
            mapping: interpolant_mapping,
            clause_database: clause_vector,
            output_literal: TRUE,
        })
    }

    pub fn add_interpolant(&mut self, clause_id: i32, interpolant: Interpolant) {
        let itp_lit = interpolant.literal();

        self.mapping.insert(clause_id, itp_lit);
        self.clause_database.extend(interpolant.into_cnf().clauses);
        self.output_literal = itp_lit;
    }
}
impl From<InterpolationStorage> for XCNF {
    fn from(storage: InterpolationStorage) -> Self {
        XCNF::new(storage.clause_database.into(), storage.output_literal)
    }
}


// ============== Unit Tests ================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::cnf;
    use crate::logic::resolution::{Partition, VariableLocality};
    use crate::logic::solving::{Interpolant, Solver, FALSE, TRUE, VAR_OFFSET};
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

    impl PartialEq<Literal> for Interpolant {
        fn eq(&self, literal: &Literal) -> bool {
            match self {
                Interpolant::Literal(lit) => lit == literal,
                Interpolant::Formula(xcnf) => xcnf == literal,
            }
        }
    }
    impl PartialEq<Interpolant> for Interpolant {
        fn eq(&self, other: &Interpolant) -> bool {
            match (self, other) {
                (Interpolant::Literal(lit1), Interpolant::Literal(lit2)) => lit1 == lit2,
                (Interpolant::Formula(xcnf1), Interpolant::Formula(xcnf2)) => xcnf1 == xcnf2,
                _ => false,
            }
        }
    }
    impl Clone for Interpolant {
        fn clone(&self) -> Self {
            match self {
                Interpolant::Literal(lit) => Interpolant::Literal(*lit),
                Interpolant::Formula(xcnf) => Interpolant::Formula(xcnf.clone()),
            }
        }
    }
    #[allow(clippy::from_over_into)]
    impl Into<XCNF> for Interpolant {
        fn into(self) -> XCNF {
            match self {
                Interpolant::Literal(lit) => XCNF::from(lit),
                Interpolant::Formula(xcnf) => xcnf,
            }
        }
    }

    fn setup_interpolants() -> (Solver, Interpolant, Interpolant) {
        const N_VARS: usize = 4;
        let mut solver = Solver::new();
        let vars = solver.add_vars(N_VARS);
        let (x, y, z1, z2) = (vars[0], vars[1], vars[2], vars[3]);

        let I1 = XCNF::new(cnf![[-z1, x], [-z1, y], [z1, -x, -y]], z1);
        let I2 = XCNF::new(cnf![[z2, -x], [z2, -y], [-z2, x, y]], z2);

        assert_eq!(z2.var() as usize, VAR_OFFSET - 1 + N_VARS);
        (solver, Interpolant::Formula(I1), Interpolant::Formula(I2))
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

        // Case 1: Trivial TRUE
        let T_or_T = solver.tseitin_or(TRUE, TRUE);
        assert_eq!(T_or_T, TRUE);

        // Case 1: Trivial mixed TRUE/FALSE
        let T_or_F = solver.tseitin_or(TRUE, FALSE);
        assert_eq!(T_or_F, TRUE);
        let F_or_T = solver.tseitin_or(FALSE, TRUE);
        assert_eq!(F_or_T, TRUE);

        // Case 3: Trivial FALSE
        let F_or_F = solver.tseitin_or(FALSE, FALSE);
        assert_eq!(F_or_F, FALSE);

        // Assert that we did not need any auxiliary tseitin variables for these cases
        assert_eq!(solver.top_var, 0);
    }

    #[test]
    fn tseitin_and_trivial() {
        let mut solver = Solver::new();

        // Case 1: Trivial TRUE
        let T_and_T = solver.tseitin_and(TRUE, TRUE);
        assert_eq!(T_and_T, TRUE);

        // Case 1: Trivial mixed TRUE/FALSE
        let T_and_F = solver.tseitin_and(TRUE, FALSE);
        assert_eq!(T_and_F, FALSE);
        let F_and_T = solver.tseitin_and(FALSE, TRUE);
        assert_eq!(F_and_T, FALSE);

        // Case 3: Trivial FALSE
        let F_and_F = solver.tseitin_and(FALSE, FALSE);
        assert_eq!(F_and_F, FALSE);

        // Assert that we did not need any auxiliary tseitin variables for these cases
        assert_eq!(solver.top_var, 0);
    }

    #[test]
    fn tseitin_or_literals() {
        let mut solver = Solver::new();
        let I1 = Interpolant::Literal(solver.add_var());
        let I2 = Interpolant::Literal(solver.add_var());

        let initial_top = solver.top_var;

        // Case 1: I or TRUE
        let I1_or_T = solver.tseitin_or(I1.literal(), TRUE);
        assert_eq!(I1_or_T, TRUE);
        let I2_or_T = solver.tseitin_or(I2.literal(), TRUE);
        assert_eq!(I2_or_T, TRUE);

        // Case 2: I or FALSE
        let I1_or_F = solver.tseitin_or(I1.literal(), FALSE);
        assert_eq!(I1_or_F, I1);
        let I2_or_F = solver.tseitin_or(I2.literal(), FALSE);
        assert_eq!(I2_or_F, I2);

        // Case 3: I and I
        let I1_or_I1 = solver.tseitin_and(I1.literal(), I1.literal());
        assert_eq!(I1_or_I1, I1);
        let I2_or_I2 = solver.tseitin_and(I2.literal(), I2.literal());
        assert_eq!(I2_or_I2, I2);

        // Assert that we did not need any additional tseitin variables for these cases
        assert_eq!(solver.top_var, initial_top);

        // Case 4: I1 or I2
        let I1_or_I2 = solver.tseitin_or(I1.literal(), I2.literal());
        let I1_or_I2_lit = I1_or_I2.literal();
        let I1_or_I2_xcnf = match I1_or_I2 {
            Interpolant::Literal(lit) => panic!("Expected XCNF, got Literal: {:?}", lit),
            Interpolant::Formula(xcnf) => xcnf,
        };

        // The now top variable should have become the output literal
        let t = Literal::from_var(solver.top_var);
        assert_eq!(I1_or_I2_lit, t);

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_or_I2_xcnf.formula.len(), 3);
        let tseitin_clauses = vec![
            Clause::from([t, -I1.literal()]),
            Clause::from([t, -I2.literal()]),
            Clause::from([-t, I1.literal(), I2.literal()]),
        ];
        assert!(I1_or_I2_xcnf.contains_all(&tseitin_clauses));

        // Check: We needed exactly one additional variable, no more
        assert_eq!(solver.top_var, initial_top + 1);
    }

    #[test]
    fn tseitin_and_literals() {
        let mut solver = Solver::new();
        let I1 = Interpolant::Literal(solver.add_var());
        let I2 = Interpolant::Literal(solver.add_var());

        let initial_top = solver.top_var;

        // Case 1: I and TRUE
        let I1_and_T = solver.tseitin_and(I1.literal(), TRUE);
        assert_eq!(I1_and_T, I1);
        let I2_and_T = solver.tseitin_and(I2.literal(), TRUE);
        assert_eq!(I2_and_T, I2);

        // Case 2: I and FALSE
        let I1_and_F = solver.tseitin_and(I1.literal(), FALSE);
        assert_eq!(I1_and_F, FALSE);
        let I2_and_F = solver.tseitin_and(I2.literal(), FALSE);
        assert_eq!(I2_and_F, FALSE);

        // Case 3: I and I
        let I1_and_I1 = solver.tseitin_and(I1.literal(), I1.literal());
        assert_eq!(I1_and_I1, I1);
        let I2_and_I2 = solver.tseitin_and(I2.literal(), I2.literal());
        assert_eq!(I2_and_I2, I2);

        // Assert that we did not need any additional tseitin variables for these cases
        assert_eq!(solver.top_var, initial_top);

        // Case 4: I1 and I2
        let I1_and_I2 = solver.tseitin_and(I1.literal(), I2.literal());
        let I1_and_I2_lit = I1_and_I2.literal();
        let I1_and_I2_xcnf = match I1_and_I2 {
            Interpolant::Literal(lit) => panic!("Expected XCNF, got Literal: {:?}", lit),
            Interpolant::Formula(xcnf) => xcnf,
        };

        // The now top variable should have become the output literal
        let t = Literal::from_var(solver.top_var);
        assert_eq!(I1_and_I2_lit, t);

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_and_I2_xcnf.formula.len(), 3);
        let tseitin_clauses = vec![
            Clause::from([-t, I1.literal()]),
            Clause::from([-t, I2.literal()]),
            Clause::from([-I1.literal(), -I2.literal(), t]),
        ];
        assert!(I1_and_I2_xcnf.contains_all(&tseitin_clauses));

        // Check: We needed exactly one additional variable, no more
        assert_eq!(solver.top_var, initial_top + 1);
    }

    #[test]
    fn tseitin_with_interpolants() {
        let (mut solver, I1, I2) = setup_interpolants();
        let I1_xcnf: XCNF = I1.clone().into();
        let I2_xcnf: XCNF = I2.clone().into();
        let initial_top = solver.top_var;

        // Case 1: I or TRUE
        let I1_and_T = solver.tseitin_and_interpolants(I1.clone(), TRUE.into());
        assert_eq!(I1_and_T, I1);
        let I2_and_T = solver.tseitin_and_interpolants(I2.clone(), TRUE.into());
        assert_eq!(I2_and_T, I2);

        // Case 2: I or FALSE
        let I1_and_F = solver.tseitin_and_interpolants(I1.clone(), FALSE.into());
        assert_eq!(I1_and_F, FALSE);
        let I2_and_F = solver.tseitin_and_interpolants(I2.clone(), FALSE.into());
        assert_eq!(I2_and_F, FALSE);

        // Assert that we did not need any additional tseitin variables for these cases
        assert_eq!(solver.top_var, initial_top);

        // Case 3: I1 or I2
        let I1_and_I2 = solver.tseitin_and_interpolants(I1, I2);
        let I1_and_I2_lit = I1_and_I2.literal();
        let I1_and_I2_xcnf = match I1_and_I2 {
            Interpolant::Literal(lit) => panic!("Expected XCNF, got Literal: {:?}", lit),
            Interpolant::Formula(xcnf) => xcnf,
        };
        assert!(I1_and_I2_xcnf.contains_all(&I1_xcnf.formula.clauses));
        assert!(I1_and_I2_xcnf.contains_all(&I2_xcnf.formula.clauses));

        // Check if the tseitin clauses and variable were added correctly
        assert_eq!(I1_and_I2_xcnf.formula.len(), I1_xcnf.formula.len() + I2_xcnf.formula.len() + 3);
        // The now top variable should have become the output literal
        let t = Literal::from_var(solver.top_var);
        assert_eq!(I1_and_I2_xcnf.out_lit, t);
        assert_eq!(I1_and_I2_lit, t);

        let tseitin_clauses = vec![
            Clause::from([-t, I1_xcnf.out_lit]),
            Clause::from([-t, I2_xcnf.out_lit]),
            Clause::from([t, -I1_xcnf.out_lit, -I2_xcnf.out_lit])
        ];
        assert!(I1_and_I2_xcnf.contains_all(&tseitin_clauses));

        // Check: We needed exactly one additional variable, no more
        assert_eq!(solver.top_var, initial_top + 1);
    }
}