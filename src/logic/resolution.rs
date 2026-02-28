use super::Literal;
use crate::logic::types::Clause;
use std::collections::{HashMap, HashSet};


#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Partition {
    A,
    B
}
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum VariableLocality {
    Local(Partition),
    Shared
}

#[derive(Debug)]
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
            pivot: pivot.into().unsign(),
            resolvent: resolvent_id
        }
    }
}

/// A big container representing the proof trace of a resolution proof. Apart from all [Clause]s and
/// all [ResolutionStep]s (in order), this also holds information about the [Partition] assigned to
/// each root clause and variable occurring in the proof.
pub struct ResolutionProof {
    root_clauses: Vec<Clause>,
    intermediate_clauses: Vec<Clause>,
    resolution_steps: Vec<ResolutionStep>,

    clauses_per_partition: HashMap<Partition, HashSet<i32>>,
    vars_per_partition: HashMap<Partition, HashSet<i32>>,
    pub partition: Option<Partition>,
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

    /// Length of the proof as in the number of resolution steps.
    pub fn len(&self) -> usize {
        self.resolution_steps.len()
    }

    /// Iterates over all resolution steps in the order they were executed.
    pub fn resolutions(&self) -> impl Iterator<Item = &ResolutionStep> {
        self.resolution_steps.iter()
    }

    /// Get a reference to the [Clause] represented by a given clause ID.
    pub fn get_clause(&self, clause_id: i32) -> Option<&Clause> {
        // IDs for root clauses are non-negative integers 0, 1, 2,..., whereas IDs for intermediate clauses
        // obtained through resolutions count towards negative infinity -1, -2,...
        if clause_id >= 0 {
            self.root_clauses.get(clause_id as usize)
        } else {
            // The first (index 0) intermediate clause has ID -1
            self.intermediate_clauses.get((-(clause_id + 1)) as usize)
        }
    }

    /// Iterates over all root clauses in a given [Partition].
    pub fn clauses_in_partition(&self, partition: Partition) -> impl Iterator<Item = &i32> {
        self.clauses_per_partition
            .get(&partition)
            .into_iter()    // Option -> Iterator (0 or 1 item)
            .flat_map(|ids| ids.iter())
    }

    /// Retrieves the locality of a SAT variable ([Literal]), i.e. if the variable only occurs in
    /// root clauses of specific [Partition] or is shared across both partitions.
    pub fn var_locality(&self, lit: Literal) -> Option<VariableLocality> {
        let var = lit.var();
        let in_a = self.vars_per_partition[&Partition::A].contains(&var);
        let in_b = self.vars_per_partition[&Partition::B].contains(&var);

        if in_a && in_b {
            return Some(VariableLocality::Shared);
        } else if in_a {
            return Some(VariableLocality::Local(Partition::A));
        } else if in_b {
            return Some(VariableLocality::Local(Partition::B));
        }

        None
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
        debug_assert_eq!(id as usize, self.root_clauses.len() - 1);
    }

    pub fn notify_resolution(self: &mut ResolutionProof, resolvent_id: i32, left: i32, right: i32, pivot_var: i32, resolvent: &[i32]) {
        let resolved_clause = Clause::new(
            resolvent.iter().map(|&lit| Literal::from(lit))
        );

        let pivot = Literal::from_var(pivot_var);
        self.resolution_steps.push(ResolutionStep::new(left, right, pivot, resolvent_id));

        if resolvent_id >= 0 {
            self.root_clauses.push(resolved_clause);
            debug_assert_eq!(resolvent_id as usize, self.root_clauses.len() - 1);
        } else {
            self.intermediate_clauses.push(resolved_clause);
            debug_assert_eq!((-resolvent_id) as usize, self.intermediate_clauses.len());
        }
    }
}