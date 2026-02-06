#![allow(dead_code)]

mod minisat;
mod bmc;

use std::process;


fn main() {
    // TODO: Parse CLI parameters (file name and k)

    println!("Hello, world!");

    // Test SAT solver calls across FFI
    let mut solver= minisat::Solver::new();

    let a = solver.add_var();
    let b = solver.add_var();
    let c = solver.add_var();

    solver.add_clause([a]);
    solver.add_clause([-a, b]);
    solver.add_clause([-b, c]);
    solver.add_clause([-c, -a]);

    let result = solver.solve();
    dbg!(result);


    let instance = bmc::load_model("data/combination.aag").unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.unwind(10).check_bounded();
    println!("{:?}", checking_result)
}