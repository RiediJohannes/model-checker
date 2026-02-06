#![allow(dead_code)]

mod interop;
mod bmc;

use interop::minisat as minisat;
use std::process;

unsafe extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn main() {
    // TODO: Parse CLI parameters (file name and k)

    println!("Hello, world!");

    unsafe {
        // Call the square root function from the C standard library
        dbg!(sqrt(2.0));
    }

    // Call a function of the minisat module through the C++ foreign function interface (FFI)
    dbg!(minisat::addInts(1, 2));


    // Test SAT solver calls across FFI
    let mut solver= minisat::new_solver();

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