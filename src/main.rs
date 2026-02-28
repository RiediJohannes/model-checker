mod logic;
mod bmc;

use clap::Parser;
use std::path::PathBuf;
use std::process;

#[derive(Parser, Debug)]
#[command(author, about, long_about = None)]
struct Args {
    /// Number of unwinding steps to the transition relation
    k: u32,

    /// Path to an AIGER file in ASCII format (*.aag)
    #[clap(value_name = "AAG_FILE", value_parser = clap::value_parser!(PathBuf))]
    file_path: PathBuf,

    /// Use this flag to perform bounded model checking incrementally (automatically increasing
    /// unwinding depth k) with interpolation-based fixpoint detection.
    #[clap(long, short = 'i', action)]
    interpolate: bool,

    /// Add some additional print statements during checking.
    #[clap(long, short = 'v', action)]
    verbose: bool,
}


fn main() {
    let args = Args::parse();

    let instance = bmc::load_model(&args.file_path).unwrap_or_else(|e| {
        eprintln!("Parsing error: {e}");
        process::exit(1);
    });

    let checking_result = match args.interpolate {
        false => bmc::check_bounded(&instance, args.k, args.verbose),
        true => bmc::check_interpolated(&instance, args.k, args.verbose)
    };

    match checking_result {
        Ok(conclusion) => println!("{:}", conclusion),
        Err(e) => eprintln!("ERROR: {e}")
    }
}