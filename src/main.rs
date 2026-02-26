#![allow(dead_code)]

mod logic;
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
    #[clap(long, short, action)]
    interpolate: bool,
}


fn main() {
    let mut args = Args::parse();

    let aiger_file = "data/count10.aag";  // args.file_path
    // let aiger_file = "./../ascii/texas_ifetch_3.aag";  // args.file_path
    let k: u32 = 3; // args.k
    args.interpolate = true;

    let instance = bmc::load_model(aiger_file).unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = match args.interpolate {
        false => bmc::check_bounded(&instance, k),
        true => bmc::check_interpolated(&instance, k)
    };

    match checking_result {
        Ok(conclusion) => println!("{:}", conclusion),
        Err(e) => eprintln!("ERROR: {e}")
    }
}