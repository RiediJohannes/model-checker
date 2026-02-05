mod interop;
mod bmc;

use interop::minisat as minisat;


unsafe extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn main() {
    println!("Hello, world!");

    unsafe {
        // Call the square root function from the C standard library
        println!("{}", sqrt(2.0));
    }

    // Call a function of the minisat module through the C++ FFI
    println!("{}", minisat::addInts(1, 2));

    let aiger = bmc::load_instance("data/combination.aag");
    println!("{:?}", aiger);
}