use std::fs::File;
use std::io::{BufRead, BufReader};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Failed to parse integer: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("Missing field in input")]
    MissingField,

    #[error("Invalid file format: Expected header tag 'aag', got '{0}'")]
    InvalidHeaderTag(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}


#[derive(Debug)]
pub struct AIG {
    pub max_idx: u32,
    pub inputs: Vec<u32>,
    pub latches: Vec<Latch>,
    pub outputs: Vec<u32>,
    pub and_gates: Vec<AndGate>,
}

#[derive(Debug)]
pub struct AndGate {
    pub out: u32,
    pub in1: u32,
    pub in2: u32,
}

#[derive(Debug)]
pub struct Latch {
    pub out: u32,
    pub next: u32,
}

pub fn parse_aiger_ascii(path: &str) -> Result<AIG, ParseError> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    // Header: aag M I L O A
    let header = lines.next().unwrap()?;
    let mut fields = header.split_whitespace();

    // Assert the first field is "aag"
    match fields.next() {
        Some("aag") => {}
        Some(tag) => return Err(ParseError::InvalidHeaderTag(tag.to_string())),
        None => return Err(ParseError::MissingField),
    }

    // M I L O A
    let max_idx: u32 = fields.next().ok_or(ParseError::MissingField)?.parse()?;
    let num_inputs: u32 = fields.next().ok_or(ParseError::MissingField)?.parse()?;
    let num_latches: u32 = fields.next().ok_or(ParseError::MissingField)?.parse()?;
    let num_outputs: u32 = fields.next().ok_or(ParseError::MissingField)?.parse()?;
    let num_ands: u32 = fields.next().ok_or(ParseError::MissingField)?.parse()?;

    // Inputs
    // Parse inputs
    let inputs: Vec<u32> = (&mut lines)
        .take(num_inputs as usize)
        .map(|line| line?.parse().map_err(ParseError::from))
        .collect::<Result<_, _>>()?;

    // Parse latches
    let latches: Vec<Latch> = (&mut lines)
        .take(num_latches as usize)
        .map(|line| {
            let line = line?;
            let mut signals = line.split_whitespace();
            Ok::<Latch, ParseError>(Latch {
                out: signals.next().ok_or(ParseError::MissingField)?.parse()?,
                next: signals.next().ok_or(ParseError::MissingField)?.parse()?,
            })
        })
        .collect::<Result<_, _>>()?;

    // Parse outputs
    let outputs: Vec<u32> = (&mut lines)
        .take(num_outputs as usize)
        .map(|line| line?.parse().map_err(ParseError::from))
        .collect::<Result<_, _>>()?;

    // Parse AND gates
    let and_gates: Vec<AndGate> = (&mut lines)
        .take(num_ands as usize)
        .map(|line| {
            let line = line?;
            let mut signals = line.split_whitespace();
            Ok::<AndGate, ParseError>(AndGate {
                out: signals.next().ok_or(ParseError::MissingField)?.parse()?,
                in1: signals.next().ok_or(ParseError::MissingField)?.parse()?,
                in2: signals.next().ok_or(ParseError::MissingField)?.parse()?,
            })
        })
        .collect::<Result<_, _>>()?;

    // Gate names and comments in the AIGER file are ignored

    Ok(AIG {
        max_idx,
        inputs,
        latches,
        outputs,
        and_gates,
    })
}