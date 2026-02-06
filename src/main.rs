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
    let result = solver.pin_mut().solve();
    let result = solver.pin_mut().solve();

    dbg!(result);

    let lit = solver.pin_mut().new_var();
    dbg!(&lit);
    dbg!(lit.var());
    let lit2 = solver.pin_mut().new_var();
    dbg!(&lit2);
    dbg!(lit2.var());
    println!("{}", &lit);
    println!("{}", -lit2);
    println!("{}", lit2);

    let instance = bmc::load_model("data/combination.aag").unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.check_bounded(10);
    println!("{:?}", checking_result)
}