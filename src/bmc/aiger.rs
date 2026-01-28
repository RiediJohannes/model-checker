use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
pub struct Aiger {
    pub max_var: u32,
    pub inputs: Vec<u32>,
    pub latches: Vec<(u32, u32)>,
    pub outputs: Vec<u32>,
    pub and_gates: Vec<AndGate>,
}

#[derive(Debug)]
pub struct AndGate {
    pub lhs: u32,
    pub rhs0: u32,
    pub rhs1: u32,
}

pub fn parse_aiger_ascii(path: &str) -> Aiger {
    let file = File::open(path).unwrap();
    let mut lines = BufReader::new(file).lines();

    // Header: aag M I L O A
    let header = lines.next().unwrap().unwrap();
    let parts: Vec<u32> = header
        .split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();

    let max_var = parts[0];
    let num_inputs = parts[1];
    let num_latches = parts[2];
    let num_outputs = parts[3];
    let num_ands = parts[4];

    // Inputs
    let mut inputs = Vec::with_capacity(num_inputs as usize);
    for _ in 0..num_inputs {
        let lit = lines.next().unwrap().unwrap().parse().unwrap();
        inputs.push(lit);
    }

    // Latches
    let mut latches = Vec::with_capacity(num_latches as usize);
    for _ in 0..num_latches {
        let line = lines.next().unwrap().unwrap();
        let mut it = line.split_whitespace();
        let lhs = it.next().unwrap().parse().unwrap();
        let rhs = it.next().unwrap().parse().unwrap();
        latches.push((lhs, rhs));
    }

    // Outputs
    let mut outputs = Vec::with_capacity(num_outputs as usize);
    for _ in 0..num_outputs {
        let lit = lines.next().unwrap().unwrap().parse().unwrap();
        outputs.push(lit);
    }

    // AND gates
    let mut and_gates = Vec::with_capacity(num_ands as usize);
    for _ in 0..num_ands {
        let line = lines.next().unwrap().unwrap();
        let mut it = line.split_whitespace();
        let lhs = it.next().unwrap().parse().unwrap();
        let rhs0 = it.next().unwrap().parse().unwrap();
        let rhs1 = it.next().unwrap().parse().unwrap();
        and_gates.push(AndGate { lhs, rhs0, rhs1 });
    }

    // Symbols and comments can be parsed here if needed
    // Lines starting with 'i', 'l', 'o' are symbols
    // Line starting with 'c' begins comments

    Aiger {
        max_var,
        inputs,
        latches,
        outputs,
        and_gates,
    }
}