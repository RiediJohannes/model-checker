use cxx::{CxxVector, UniquePtr};
pub use ffi::Literal;
use ffi::SolverStub;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::ops::Neg;
use std::pin::Pin;

#[cxx::bridge]
pub mod ffi {
    // Shared structs, whose fields will be visible to both languages
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
        fn notify_clause(self: &mut ResolutionProof, id: u32, lits: &[i32]);
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
    solved: bool,
}

impl Solver {
    pub fn new() -> Self {
        let mut resolution = Box::new(ResolutionProof::new());

        Self {
            stub: ffi::newSolver(&mut resolution),
            resolution,
            solved: false,
        }
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
        self.resolution.partition = Some(partition);
    }

    pub fn clear_partition(&mut self) {
        self.resolution.partition = None;
    }

    pub fn solve(&mut self) -> bool {
        self.solved = false;
        let is_sat =self.remote().solve();
        self.solved = true;

        is_sat
    }

    pub fn solve_assuming<L>(&mut self, assumptions: L) -> bool
    where L: AsRef<[Literal]> {
        self.solved = false;
        let is_sat = self.remote().solve_with_assumptions(assumptions.as_ref());
        self.solved = true;

        is_sat
    }

    pub fn get_model(&mut self) -> UniquePtr<CxxVector<i8>>
    {
        self.remote().getModel()
    }

    pub fn get_proof(&self) -> Option<&ResolutionProof> {
        match self.solved {
            true => Some(self.resolution.as_ref()),
            false => None
        }
    }

    /// Quick and idiomatic access to a pinned mutable reference of the underlying solver.
    fn remote(&mut self) -> Pin<&mut SolverStub> {
        self.stub.pin_mut()
    }
}


#[derive(Debug,Clone)]
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

    #[inline]
    pub fn from_vec(v: Vec<Literal>) -> Self {
        Self { lits: v.into_boxed_slice() }
    }
}

impl<'a> IntoIterator for &'a Clause {
    type Item = Literal;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Literal>>;

    fn into_iter(self) -> Self::IntoIter {
        self.lits.iter().copied()
    }
}

pub type CNF = Vec<Clause>;

impl PartialEq<Literal> for CNF {
    fn eq(&self, lit: &Literal) -> bool {
        self.len() == 1 && self[0].lits.len() == 1 && self[0].lits[0] == *lit
    }
}
impl PartialEq<Literal> for &CNF {
    fn eq(&self, lit: &Literal) -> bool {
        self.len() == 1 && self[0].lits.len() == 1 && self[0].lits[0] == *lit
    }
}
impl From<Literal> for CNF {
    fn from(lit: Literal) -> Self {
        vec![Clause::new([lit])]
    }
}
impl From<Clause> for CNF {
    fn from(clause: Clause) -> Self {
        vec![clause]
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Partition {
    A,
    B,
    AB
}


pub struct ResolutionStep {
    pub left: i32,
    pub right: i32,
    pub pivot: Literal,
    pub resolvent: i32,
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

    clauses_per_partition: HashMap<Partition, HashSet<i32>>,
    vars_per_partition: HashMap<Partition, HashSet<i32>>,
    partition: Option<Partition>,
}

impl ResolutionProof {
    pub fn new() -> Self {
        let mut clauses_dict = HashMap::new();
        let mut vars_dict = HashMap::new();
        for p in [Partition::A, Partition::B] {
            clauses_dict.insert(p, HashSet::new());
            vars_dict.insert(p, HashSet::new());
        }

        Self {
            root_clauses: Vec::new(),
            intermediate_clauses: Vec::new(),
            resolution_steps: Vec::new(),

            clauses_per_partition: clauses_dict,
            vars_per_partition: vars_dict,
            partition: None,
        }
    }

    pub fn get_clause(&self, clause_id: i32) -> Option<&Clause> {
        if clause_id >= 0 {
            self.root_clauses.get(clause_id as usize)
        } else {
            self.intermediate_clauses.get((-clause_id) as usize)
        }
    }

    pub fn clauses_in_partition(&self, partition: Partition) -> impl Iterator<Item = &i32> {
        self.clauses_per_partition
            .get(&partition)
            .into_iter()    // Option -> Iterator (0 or 1 item)
            .flat_map(|ids| ids.iter())
            // Optionally query the clause also
            // .filter_map(move |&id| {
            //     self.get_clause(id).map(|clause| (id, clause))
            // })
    }

    pub fn var_partition(&self, lit: Literal) -> Option<Partition> {
        let var = lit.var();
        let in_a = self.vars_per_partition[&Partition::A].contains(&var);
        let in_b = self.vars_per_partition[&Partition::B].contains(&var);

        if in_a && in_b {
            return Some(Partition::AB);
        } else if in_a {
            return Some(Partition::A);
        } else if in_b {
            return Some(Partition::B);
        }

        None
    }

    pub fn resolutions(&self) -> impl Iterator<Item = &ResolutionStep> {
        self.resolution_steps.iter()
    }

    pub fn notify_clause(&mut self, id: u32, lits: &[i32]) {
        let clause = Clause::new(
            lits.iter().map(|&lit| Literal::from(lit))
        );

        if let Some(partition) = self.partition {
            self.clauses_per_partition.entry(partition).or_default().insert(id as i32);
            for lit in &clause {
                self.vars_per_partition.entry(partition).or_default().insert(lit.var());
            }
        }

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

    pub fn clear(&mut self) {
        self.root_clauses.clear();
        self.intermediate_clauses.clear();
        self.resolution_steps.clear();

        for p in [Partition::A, Partition::B] {
            self.clauses_per_partition.entry(p).or_default().clear();
            self.vars_per_partition.entry(p).or_default().clear();
        }
    }
}