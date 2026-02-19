use super::aiger::AIG;
use crate::logic::solving::Solver;
use crate::logic::{CNF, VAR_OFFSET, XCNF};
use std::collections::HashSet;


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
/// - A => I
/// - B => ~I
/// - I only contains variables shared between A and B
#[allow(non_snake_case)]
pub fn verify_interpolant_properties(interpolant: &XCNF, A_cnf: CNF, B_cnf: CNF, top_var: i32) {
    // Property 1: A => I, or equivalently (A and ~I) is UNSAT
    let mut solver_a = Solver::new();
    for _ in 1..=top_var { solver_a.add_var(); }

    for clause in &A_cnf { solver_a.add_clause(clause); }
    for clause in &interpolant.formula.clauses { solver_a.add_clause(clause); }
    solver_a.add_clause([-interpolant.out_lit]);
    assert!(!solver_a.solve(), "Property A => I failed: A and ~I is SAT!");

    // Property 2: B => ~I, or equivalently (I and B) is UNSAT
    let mut solver_b = Solver::new();
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

pub fn print_sat_model(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();
    let num_l = graph.latches.len();
    let num_a = graph.and_gates.len();
    let frame_size = num_i + num_l + num_a;

    println!("#--- SAT Model Assignment (True Variables) ---");

    for (idx, &val) in model.iter().enumerate().skip(VAR_OFFSET) {
        // Only print variables that were set to true
        if val != 1 {
            continue;
        }

        let normalized = idx - VAR_OFFSET;
        let t = normalized / frame_size; // Time step
        let f = normalized % frame_size; // Offset in frame

        // Determine the semantic meaning of the variable within the current frame
        let (label, local_idx) = if f < num_i {
            ("Input", f)
        } else if f < num_i + num_l {
            ("Latch", f - num_i)
        } else {
            ("AND", f - (num_i + num_l))
        };

        println!("{:>4}: {}_{}@{}", idx, label, local_idx, t);
    }
    println!("--------------------------------------------");
}

pub fn print_input_trace(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();

    // The first variable is the constant bottom
    let total_vars = model.len().saturating_sub(1);
    let vars_per_frame = graph.variables().count();
    let num_steps = total_vars / vars_per_frame;

    // Header
    println!("\n=== Input Trace (Counter-Example) ===");
    print!("{:>4} | ", "t");
    for i in 0..num_i {
        print!("In_{:<2} ", i);
    }
    println!("\n{:-<5}+{:-<}", "", "-".repeat(num_i * 5));

    for k in 0..num_steps {
        print!("{:>4} | ", k);
        for i in 0..num_i {
            // Use frame_size derived from variables().len()
            let idx = VAR_OFFSET + (k * vars_per_frame) + i;
            if let Some(&val) = model.get(idx) {
                let display = match val {
                    1  => " T  ", // True
                    -1 => " F  ", // False
                    0  => " x  ", // Undefined
                    _  => " !  ", // Should not happen
                };
                print!("{}", display);
            }
        }
        println!();
    }
    println!("=====================================\n");
}