use super::aiger::AIG;
use crate::logic::solving::SimpleSolver;
use crate::logic::VAR_OFFSET;
use std::fmt::Display;
use std::ops::Deref;

#[cfg(debug_assertions)] use crate::logic::{CNF, XCNF};
#[cfg(debug_assertions)] use std::collections::HashSet;


#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bool3 {
    True,
    False,
    Undef,
}
impl From<i8> for Bool3 {
    fn from(v: i8) -> Self {
        match v {
            1  => Bool3::True,
            -1 => Bool3::False,
            _  => Bool3::Undef,
        }
    }
}
impl Display for Bool3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bool3::True  => write!(f, "1"),
            Bool3::False => write!(f, "0"),
            Bool3::Undef => write!(f, "x"),
        }
    }
}
impl PartialEq<i32> for Bool3 {
    fn eq(&self, num: &i32) -> bool {
        match self {
            Bool3::True  => *num == 1,
            Bool3::False => *num == 0,
            Bool3::Undef => *num == -1,
        }
    }
}
impl PartialEq<Bool3> for i32 {
    fn eq(&self, other: &Bool3) -> bool {
        other == self
    }
}

#[derive(Debug)]
pub struct InputTrace {
    trace: Vec<Vec<Bool3>>,
}
impl Deref for InputTrace {
    type Target = Vec<Vec<Bool3>>;
    fn deref(&self) -> &Self::Target {
        &self.trace
    }
}
impl<const M: usize, const N: usize> PartialEq<[[i32; N]; M]> for InputTrace {
    fn eq(&self, reference_matrix: &[[i32; N]; M]) -> bool {
        if self.len() != M {
            return false;
        }

        for (bool_row, num_row) in self.iter().zip(reference_matrix.iter()) {
            if bool_row.len() != N {
                return false;
            }

            for (bool3, num) in bool_row.iter().zip(num_row.iter()) {
                if *num == Bool3::Undef {
                    continue;
                }

                if bool3 != num {
                    return false;
                }
            }
        }

        true
    }
}

pub fn extract_input_trace(graph: &AIG, model: &[i8]) -> InputTrace {
    let num_inputs = graph.inputs.len();
    let vars_per_frame = graph.variables().count();

    // The first variable is the constant bottom
    let total_vars = model.len().saturating_sub(1);
    let num_steps = total_vars / vars_per_frame;

    let trace = (0..num_steps)
        .map(|t| {
            (0..num_inputs)
                .map(|i| {
                    let idx = VAR_OFFSET + t * vars_per_frame + i;
                    Bool3::from(model.get(idx).copied().unwrap_or(0))
                })
                .collect()
        })
        .collect();

    InputTrace { trace }
}

/// Prints the series of inputs to the circuit that lead to the given satisfying assignment.
pub fn print_input_trace(graph: &AIG, model: &[i8]) {
    let trace = extract_input_trace(graph, model);
    let num_i = graph.inputs.len();

    // Header
    println!("\n=== Input Trace (Counter-Example) ===");
    print!("{:>4} | ", "t");
    for i in 0..num_i {
        print!("In_{:<2} ", i);
    }
    println!("\n{:-<5}+{:-<}", "", "-".repeat(num_i * 5));

    // Body
    for (k, step) in trace.iter().enumerate() {
        print!("{:>4} | ", k);

        for val in step {
            // center-align inside 3 spaces to match old formatting
            print!(" {:^1}  ", val);
        }

        println!();
    }

    println!("=====================================\n");
}


#[cfg(debug_assertions)]
pub fn vars_in_cnf(cnf: &CNF) -> HashSet<i32> {
    let mut vars = HashSet::new();
    for clause in cnf {
        for lit in clause {
            vars.insert(lit.var());
        }
    }
    vars
}

/// Verifies that the interpolant `I` satisfies the following properties w.r.t. the clause partition `(A, B)`:
/// - `A => I`
/// - `B => ~I`
/// - `I` only contains variables shared between A and B
#[allow(non_snake_case)]
#[cfg(debug_assertions)]
pub fn verify_interpolant_properties(interpolant: &XCNF, A_cnf: CNF, B_cnf: CNF, top_var: i32) {
    // Property 1: A => I, or equivalently (A and ~I) is UNSAT
    let mut solver_a = SimpleSolver::new();
    for _ in 1..=top_var { solver_a.add_var(); }

    for clause in &A_cnf { solver_a.add_clause(clause); }
    for clause in &interpolant.formula.clauses { solver_a.add_clause(clause); }
    solver_a.add_clause([-interpolant.out_lit]);
    assert!(!solver_a.solve(), "Property A => I failed: A and ~I is SAT!");

    // Property 2: B => ~I, or equivalently (I and B) is UNSAT
    let mut solver_b = SimpleSolver::new();
    for _ in 1..=top_var { solver_b.add_var(); }

    for clause in &B_cnf { solver_b.add_clause(clause); }
    for clause in &interpolant.formula.clauses { solver_b.add_clause(clause); }
    solver_b.add_clause([interpolant.out_lit]);
    assert!(!solver_b.solve(), "Property B => ~I failed: I and B is SAT!");

    // Property 3: Interpolant I only contains variables shared between A and B
    let A_vars = vars_in_cnf(&A_cnf);
    let B_vars = vars_in_cnf(&B_cnf);
    let I_vars = vars_in_cnf(&interpolant.formula);

    // Obtain all variables local to some partition (either A or B)
    let local_vars = A_vars.symmetric_difference(&B_vars).cloned().collect();
    assert!(I_vars.is_disjoint(&local_vars), "Interpolant I contained some variables local to either partition A or B!");
}
