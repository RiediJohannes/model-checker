use crate::bmc::aiger::{AndGate, Latch, ParseError, Signal, AIG};
use crate::bmc::{aiger, debug};
use crate::logic::resolution::Partition;
use crate::logic::solving::Solver;
use crate::logic::{Clause, Literal, FALSE, TRUE, XCNF};

use std::collections::HashMap;
use std::ops::Deref;
use thiserror::Error;


#[derive(Error, Debug)]
pub enum ModelCheckingError {
    #[error("Tried to access a signal at time step {0}, but current model is only unrolled until t = {1}")]
    InvalidTimeStep(u32, u32),

    #[error("Failed to compute interpolant for k = {0} at iteration {1}.")]
    FailedInterpolation(u32, usize)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ModelConclusion {
    Safe,
    CounterExample,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PropertyCheck {
    Ok,
    Fail,
}


pub fn load_model(name: &str) -> Result<AIG, ParseError> {
    aiger::parse_aiger_ascii(name)
}

pub fn check_bounded(graph: &AIG, k: u32) -> Result<PropertyCheck, ModelCheckingError> {
    // Single iteration of bounded model checking
    let mut bmc = BmcModel::from_aig(graph)?;
    bmc.unwind(k)?;

    let output = match bmc.check() {
        ModelConclusion::Safe => PropertyCheck::Ok,
        ModelConclusion::CounterExample => PropertyCheck::Fail
    };
    Ok(output)
}

pub fn check_interpolated(graph: &AIG, initial_bound: u32) -> Result<PropertyCheck, ModelCheckingError> {
    let mut interpolants: Vec<XCNF> = Vec::new();
    let mut k = initial_bound;

    // Q <- Compute exact initial states Q
    // F <- Unroll transition relation k times

    loop {
        // Q' <- Add all interpolants in I to initial states Q
        // Seed BMC with initial states Q' and F from the transition relation unrolls

        // These two lines are just a placeholder
        let mut bmc = BmcModel::from_aig(graph)?;
        bmc.unwind(initial_bound)?;

        if bmc.check() == ModelConclusion::Safe {
            let interpolant = bmc.compute_interpolant()
                .ok_or(ModelCheckingError::FailedInterpolation(k, interpolants.len()))?;
            dbg!(&interpolant);

            // TODO Rename interpolant
            // I(s0) <- Rename interpolant I(s1) to I(s0) (talk about states in time step t = 0)

            // TODO Fixpoint Check
            // if (I(s0) or Q') => Q') {
            //     return Ok(PropertyCheck::Ok)
            // }

            interpolants.push(interpolant);
        } else {
            if interpolants.is_empty() {
                return Ok(PropertyCheck::Fail);
            }

            k += 1;
            interpolants.clear();
        }
    }
}

pub struct BmcModel<'a> {
    graph: &'a AIG,
    time_steps: Vec<HashMap<u32, Literal>>,
    solver: Solver,
}

impl BmcModel<'_> {
    pub fn from_aig(graph: &'_ AIG) -> Result<BmcModel<'_>, ModelCheckingError> {
        let mut model = BmcModel {
            graph,
            time_steps: Vec::new(),
            solver: Solver::new()
        };

        model.solver.set_partition(Partition::A);
        // Initialize variables for all signals at time step 0
        model.add_step();
        for var in graph.variables() {
            model.signal_at_time(&Signal::Var(var), 0)?;
        }

        // AND-gates
        for gate in graph.and_gates.iter() {
            model.encode_and_gate(gate, 0)?;
        }

        // Latches
        for latch in model.graph.latches.iter() {
            let curr = model.signal_at_time(&latch.out, 0)?;
            model.solver.add_clause([-curr]);
        }

        Ok(model)
    }

    fn add_step(&mut self) -> usize {
        self.time_steps.push(HashMap::with_capacity(self.graph.max_idx as usize));
        self.time_steps.len() - 1  // return new time step id
    }

    fn signal_at_time(&mut self, signal: &Signal, time: u32) -> Result<Literal, ModelCheckingError> {
        match signal {
            // For the constant signals, always return the same fixed literal (irrespective of the time step)
            Signal::Constant(true) => Ok(TRUE),
            Signal::Constant(false) => Ok(FALSE),
            Signal::Var(aig_var) => {
                if let Some(step_vars) = self.time_steps.get_mut(time as usize) {
                    let lit = *step_vars.entry(aig_var.idx()).or_insert_with(||self.solver.add_var());
                    Ok(if aig_var.is_neg() { -lit } else { lit })
                } else {
                    Err(ModelCheckingError::InvalidTimeStep(time, (self.time_steps.len() + 1) as u32))
                }
            }
        }
    }

    pub fn unwind(&'_ mut self, k: u32) -> Result<(), ModelCheckingError> {
        // We need variables for each input, latch, gate and output at each time step
        for t in 1..=k {
            self.add_step();

            if t == 2 {
                self.solver.set_partition(Partition::B);
            }

            // Initialize variables for all signals at the current time step (for reasonable order)
            for var in self.graph.variables() {
                self.signal_at_time(&Signal::Var(var), t)?;
            }

            // Add SAT clauses
            // AND-gates
            for gate in self.graph.and_gates.iter() {
                self.encode_and_gate(gate, t)?;
            }
            // Latches
            for latch in self.graph.latches.iter() {
                self.encode_latch(latch, t)?;
            }
        }

        // Assert that the output is true at SOME time step -> property violation
        let property_violation_clause: Vec<Literal> = (0..=k)
            .map(|t| self.signal_at_time(&self.graph.outputs[0], t))
            .collect::<Result<_, _>>()?;

        self.solver.add_clause(property_violation_clause);

        Ok(())
    }

    pub fn check(&mut self) -> ModelConclusion {
        // If the formula is SAT, the model's property was violated
        match self.solver.solve() {
            true => {
                let model = self.solver.get_model();
                // debug::print_sat_model(self.graph, model.as_slice());
                debug::print_input_trace(self.graph, model.as_slice());
                ModelConclusion::CounterExample
            },
            false => ModelConclusion::Safe
        }
    }

    #[allow(non_snake_case)]
    pub fn compute_interpolant(&mut self) -> Option<XCNF> {
        let mut interpolants: HashMap<i32, XCNF> = HashMap::new();
        // Temporarily take ownership of the resolution proof to allow for mutable references to solver during this function
        let proof_box = self.solver.resolution.take()?;
        let proof = proof_box.deref();

        // Base case: Annotate root clauses in partition A/B with Bottom/Top
        for A_clause_id in proof.clauses_in_partition(Partition::A) {
            interpolants.insert(*A_clause_id, FALSE.into());
        }

        for B_clause_id in proof.clauses_in_partition(Partition::B) {
            interpolants.insert(*B_clause_id, TRUE.into());
        }

        // Inductive case: Compute new part. interpolant from previous part. interpolants
        let mut last = 0;
        for step in proof.resolutions() {
            let I_L: &XCNF = interpolants.get(&step.left).unwrap();
            let I_R: &XCNF = interpolants.get(&step.right).unwrap();
            assert!(!interpolants.contains_key(&step.resolvent));

            let pivot_partition = proof.var_partition(step.pivot)?;
            let I_resolvent: XCNF = match pivot_partition {
                Partition::A => {  // I_L OR I_R
                    self.solver.tseitin_or(I_L, I_R)
                },
                Partition::B => {  // I_L AND I_R
                    self.solver.tseitin_and(I_L, I_R)
                },
                Partition::AB => { // (I_L OR x) AND (I_R OR ~x)
                    let x = step.pivot;
                    let left_conjunct = self.solver.tseitin_or(I_L, &XCNF::from(x));
                    let right_conjunct = self.solver.tseitin_or(I_R, &XCNF::from(-x));

                    self.solver.tseitin_and(&left_conjunct, &right_conjunct)
                }
            };

            // println!(
            //     "I_L: {:?},\n\
            //     I_R: {:?},\n\
            //     -- Resolving on literal {:} (partition: {:?})\n\
            //     => {:?}\n",
            //     &I_L, &I_R, &step.pivot, &pivot_partition, &I_resolvent
            // );

            interpolants.insert(step.resolvent, I_resolvent);
            last = step.resolvent;
        }

        let last_clause = proof.get_clause(last)?;
        assert_eq!(last_clause, &Clause::from_vec(vec![]));  // the last clause must be empty

        self.solver.resolution = Some(proof_box);   // return proof ownership

        let final_interpolant = interpolants.remove(&last)?;
        // let final_interpolant = interpolants.get(&last)?.clone();
        Some(final_interpolant)
    }

    fn encode_and_gate(&mut self, gate: &AndGate, t: u32) -> Result<(), ModelCheckingError> {
        let out = self.signal_at_time(&gate.out, t)?;
        let in1 = self.signal_at_time(&gate.in1, t)?;
        let in2 = self.signal_at_time(&gate.in2, t)?;

        self.solver.add_clause([-in1, -in2, out]);
        self.solver.add_clause([-out, in1]);
        self.solver.add_clause([-out, in2]);

        Ok(())
    }

    fn encode_latch(&mut self, latch: &Latch, t: u32) -> Result<(), ModelCheckingError> {
        let curr = self.signal_at_time(&latch.out, t)?;
        let prev = self.signal_at_time(&latch.next, t-1)?;

        self.solver.add_clause([-prev, curr]);
        self.solver.add_clause([-curr, prev]);

        Ok(())
    }
}