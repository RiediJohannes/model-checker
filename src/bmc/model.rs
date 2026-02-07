use std::collections::HashMap;
use thiserror::Error;
use crate::bmc::aiger;
use crate::bmc::aiger::{ParseError, AIG};
use crate::minisat::{Solver, Literal};

pub fn load_model(name: &str) -> Result<BmcModel, ParseError> {
    let aig = aiger::parse_aiger_ascii(name)?;

    Ok(BmcModel {
        name: name.to_string(),
        graph: aig,
    })
}

#[derive(Error, Debug)]
pub enum ModelCheckingError {
    #[error("Tried to access a signal at time step {0}, but current model is only unrolled until t = {1}")]
    InvalidTimeStep(u32, u32),
}

#[derive(Debug)]
pub enum ModelCheckingConclusion {
    Ok,
    Fail,
}

pub struct BmcModel {
    name: String,
    graph: AIG,
}

impl BmcModel {
    pub fn unwind(&'_ self, k: u32) -> Result<UnwoundBmcModel<'_>, ModelCheckingError> {
        let mut unwound = UnwoundBmcModel::new(self);

        // Special handling of time step 0

        // We need variables for each input, latch, gate and output at each time step
        for t in 1..(k+1) {
            unwound.add_step();

            // Initialize variables for all signals at the current time step
            for s in 1..self.graph.max_idx {
                unwound.signal_at_time(s, t)?;
            }

            // Add SAT clauses

            // AND-gates
            // for (i, gate) in self.graph.and_gates.iter().enumerate() {
            //     // gate.in1
            // }
            // let _c = solver.add_clause([-inputs[0], outputs[0]]);
        }

        dbg!(unwound.time_steps.len() - 1);

        Ok(unwound)
    }
}


pub struct UnwoundBmcModel<'a> {
    base: &'a BmcModel,
    time_steps: Vec<HashMap<u32, Literal>>,
    solver: Solver,
}

impl UnwoundBmcModel<'_> {
    pub fn new(base_model: &'_ BmcModel) -> UnwoundBmcModel<'_> {
        let mut time_steps = Vec::new();
        time_steps.push(HashMap::with_capacity(base_model.graph.max_idx as usize));

        UnwoundBmcModel {
            base: base_model,
            time_steps,
            solver: Solver::new()
        }
    }

    pub fn add_step(&mut self) -> usize {
        self.time_steps.push(HashMap::with_capacity(self.base.graph.max_idx as usize));
        self.time_steps.len() - 1  // return new time step id
    }

    pub fn signal_at_time(&mut self, signal_idx: u32, time: u32) -> Result<Literal,ModelCheckingError> {
        if let Some(step_vars) = self.time_steps.get_mut(time as usize) {
            Ok(*step_vars.entry(signal_idx).or_insert(self.solver.add_var()))
        } else {
            Err(ModelCheckingError::InvalidTimeStep(time, (self.time_steps.len() + 1) as u32))
        }
    }

    pub fn check_bounded(&mut self) -> ModelCheckingConclusion {
        ModelCheckingConclusion::Ok
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }
}