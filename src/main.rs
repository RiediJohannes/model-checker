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
    let mut unpinned_solver= minisat::new_solver();
    let solver = unpinned_solver.pin_mut();
    let result = solver.solve();

    dbg!(result);


    let instance = bmc::load_instance("data/combination.aag").unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.check_bounded(10);
    println!("{:?}", checking_result)
}