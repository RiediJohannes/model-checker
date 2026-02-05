#![allow(dead_code)]

mod interop;
mod bmc;

use std::process;
use interop::minisat as minisat;

unsafe extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn main() {
    // TODO: Parse CLI parameters (file name and k)

    println!("Hello, world!");

    unsafe {
        // Call the square root function from the C standard library
        println!("{}", sqrt(2.0));
    }

    // Call a function of the minisat module through the C++ FFI
    println!("{}", minisat::addInts(1, 2));

    if let Ok(mut instance) = bmc::load_instance("data/combination.aag") {
        instance.check_bounded(10);
    }

    let mut instance = bmc::load_instance("data/combination.aag").unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.check_bounded(10);
    println!("{:?}", checking_result)
}