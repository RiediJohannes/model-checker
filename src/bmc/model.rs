use std::collections::{HashMap};
use thiserror::Error;
use crate::bmc::aiger;
use crate::bmc::aiger::{ParseError, Signal, AIG};
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

        // We need variables for each input, latch, gate and output at each time step
        for t in 0..=k {
            unwound.add_step();

            // Initialize variables for all signals at the current time step (for reasonable order)
            for sig in self.graph.variables() {
                unwound.signal_at_time(&sig, t)?;
            }

            // Add SAT clauses
            // AND-gates
            for gate in self.graph.and_gates.iter() {
                let out = unwound.signal_at_time(&gate.out, t)?;
                let in1 = unwound.signal_at_time(&gate.in1, t)?;
                let in2 = unwound.signal_at_time(&gate.in2, t)?;

                unwound.solver.add_clause([-in1, -in2, out]);
                unwound.solver.add_clause([-out, in1]);
                unwound.solver.add_clause([-out, in2]);
            }

            // Latches
            for latch in self.graph.latches.iter() {
                let curr = unwound.signal_at_time(&latch.out, t)?;

                // Special handling of time step 0
                if t < 1 {
                    unwound.solver.add_clause([-curr]);
                } else {
                    let prev = unwound.signal_at_time(&latch.next, t-1)?;

                    unwound.solver.add_clause([-prev, curr]);
                    unwound.solver.add_clause([-curr, prev]);
                }

                // if t < 1 {
                //     unwound.solver.add_clause([-curr]);
                // }
                //
                // let curr_plus_1 = unwound.signal_at_time(&latch.out, t+1)?;
                // let next = unwound.signal_at_time(&latch.next, t)?;
                //
                // unwound.solver.add_clause([-curr_plus_1, next]);
                // unwound.solver.add_clause([-next, curr_plus_1]);
            }

            // Debug: Add this clause to make the formula UNSAT
            // let bottom = unwound.signal_at_time(&Signal::Constant(false), t)?;
            // unwound.solver.add_clause([bottom]);
        }

        // Assert that the output is true at SOME time step -> property violation
        let property_violation_clause: Vec<Literal> = (0..=k)
            .map(|t| unwound.signal_at_time(&self.graph.outputs[0], t))
            .collect::<Result<_, _>>()?;

        unwound.solver.add_clause(property_violation_clause);

        Ok(unwound)
    }
}


pub struct UnwoundBmcModel<'a> {
    base: &'a BmcModel,
    time_steps: Vec<HashMap<u32, Literal>>,
    solver: Solver,
}

impl UnwoundBmcModel<'_> {
    const CONSTANT_LOW: i32 = 0;
    const CONSTANT_HIGH: i32 = 1;

    pub fn new(base_model: &'_ BmcModel) -> UnwoundBmcModel<'_> {
        let time_steps = vec![HashMap::with_capacity(base_model.graph.max_idx as usize)];

        // Add the constant zero and constat one literal to the solver
        let mut solver = Solver::new();
        let bottom = solver.add_var();
        let top = solver.add_var();

        assert_eq!(bottom.var(), Self::CONSTANT_LOW);
        assert_eq!(top.var(), Self::CONSTANT_HIGH);

        solver.add_clause([-bottom]);
        solver.add_clause([top]);

        UnwoundBmcModel {
            base: base_model,
            time_steps,
            solver
        }
    }

    pub fn add_step(&mut self) -> usize {
        self.time_steps.push(HashMap::with_capacity(self.base.graph.max_idx as usize));
        self.time_steps.len() - 1  // return new time step id
    }

    pub fn signal_at_time(&mut self, signal: &Signal, time: u32) -> Result<Literal, ModelCheckingError> {
        match signal {
            // For the constant signals, always return the same fixed literal (irrespective of the time step)
            Signal::Constant(true) => Ok(Self::CONSTANT_HIGH.into()),
            Signal::Constant(false) => Ok(Self::CONSTANT_LOW.into()),
            Signal::Var(aig_var) => {
                if let Some(step_vars) = self.time_steps.get_mut(time as usize) {
                    let lit = *step_vars.entry(aig_var.idx()).or_insert(self.solver.add_var());
                    Ok(if aig_var.is_neg() { -lit } else { lit })
                } else {
                    Err(ModelCheckingError::InvalidTimeStep(time, (self.time_steps.len() + 1) as u32))
                }
            }
        }
    }

    pub fn check_bounded(&mut self) -> ModelCheckingConclusion {
        // If the formula is SAT, the model's property was violated
        match self.solver.solve() {
            true => {
                let model = self.solver.get_model();
                // self.pretty_print_model(model.as_slice());
                ModelCheckingConclusion::Fail
            },
            false => ModelCheckingConclusion::Ok,
        }
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }
}
