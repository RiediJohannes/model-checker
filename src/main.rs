#![allow(dead_code)]

mod minisat;
mod bmc;

use minisat::ffi as sat;
use std::process;


fn main() {
    // TODO: Parse CLI parameters (file name and k)

    println!("Hello, world!");

    // Test SAT solver calls across FFI
    let mut solver= sat::new_solver();

    let a = solver.pin_mut().new_var();
    let b = solver.pin_mut().new_var();
    let c = solver.pin_mut().new_var();

    solver.pin_mut().add_clause(&[a]);
    solver.pin_mut().add_clause(&[-a, b]);
    solver.pin_mut().add_clause(&[-b, c]);
    solver.pin_mut().add_clause(&[-c, -a]);

    let result = solver.pin_mut().solve();
    dbg!(result);


    let instance = bmc::load_model("data/combination.aag").unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.unwind(10).check_bounded();
    println!("{:?}", checking_result)
}