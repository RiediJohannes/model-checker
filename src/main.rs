#![allow(dead_code)]

mod minisat;
mod bmc;

use std::process;
use clap::Parser;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of unwinding steps to the transition relation
    k: u32,
    /// Path to an AIGER file in ASCII format (*.aag)
    file_path: String,
}

fn main() {
    let _args = Args::parse();
    
    let aiger_file = "data/combination.aag";  // args.file_path
    let k: u32 = 2; // args.k

    let instance = bmc::load_model(aiger_file).unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = instance.unwind(k).unwrap().check_bounded();
    println!("{:?}", checking_result)
}