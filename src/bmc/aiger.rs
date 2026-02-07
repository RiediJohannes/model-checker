use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Failed to parse integer: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("Constant signal '0' or '1' used in illegal place!")]
    IllegalConstant,

    #[error("Missing field in input file")]
    MissingField,

    #[error("Invalid file format: Expected header tag 'aag', got '{0}'")]
    InvalidHeaderTag(String),

    #[error("Error while reading input file: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug)]
pub enum Signal {
    Constant(bool),
    Var(AigVar)
}
impl FromStr for Signal {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let num = s.parse::<u32>()?;
        match num {
            0 => Ok(Signal::Constant(false)),
            1 => Ok(Signal::Constant(true)),
            _ => Ok(Signal::Var(AigVar { val: num << 1 }))
        }
    }
}

#[derive(Debug)]
pub struct AigVar {
    val: u32
}
impl AigVar {
    pub fn idx(&self) -> u32 {
        self.val >> 1
    }
    pub fn is_neg(&self) -> bool {
        self.val & 1 == 1
    }
}
impl FromStr for AigVar {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let num = s.parse::<u32>()?;
        if num < 2 {
            Err(ParseError::IllegalConstant)
        } else {
            Ok(Self { val: num })
        }
    }
}


#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct AIG {
    pub max_idx: u32,
    pub inputs: Vec<AigVar>,
    pub latches: Vec<Latch>,
    pub outputs: Vec<Signal>,
    pub and_gates: Vec<AndGate>,
}
impl AIG {
    pub fn variables(&self) -> impl Iterator<Item = Signal> {
        (2..=self.max_idx)
            .step_by(2)
            .map(|val| Signal::Var(AigVar { val }))
    }
}

#[derive(Debug)]
pub struct AndGate {
    pub out: AigVar,
    pub in1: Signal,
    pub in2: Signal,
}

#[derive(Debug)]
pub struct Latch {
    pub out: AigVar,
    pub next: Signal,
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
    let inputs: Vec<AigVar> = (&mut lines)
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
    let outputs: Vec<Signal> = (&mut lines)
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