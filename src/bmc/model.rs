use crate::bmc::aiger;
use crate::bmc::aiger::{AndGate, Latch, ParseError, Signal, AIG};
use crate::bmc::debug;
use crate::logic::resolution::Partition;
use crate::logic::solving::{InterpolationError, SimpleSolver, Solver};
use crate::logic::{Literal, FALSE, TRUE, XCNF};

use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use thiserror::Error;

#[cfg(test)] use crate::bmc::debug::InputTrace;
#[cfg(debug_assertions)] use crate::logic::{Clause, CNF};


#[derive(Error, Debug)]
pub enum ModelCheckingError {
    #[error("Tried to access a signal at time step {0}, but current model is only unrolled until t = {1}")]
    InvalidTimeStep(u32, u32),

    #[error("Failed to compute interpolant for k = {0} at iteration {1}.")]
    FailedInterpolation(u32, usize),

    #[error("Program ran out of available memory")]
    OutOfMemory
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
impl Display for PropertyCheck {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{:?}", self).to_uppercase())
    }
}


pub fn load_model(name: &PathBuf) -> Result<AIG, ParseError> {
    aiger::parse_aiger_ascii(name)
}

pub fn check_bounded(graph: &AIG, k: u32, verbose: bool) -> Result<PropertyCheck, ModelCheckingError> {
    // Single iteration of bounded model checking
    let mut bmc = BmcModel::from_aig(graph, k, true)?;

    let output = match bmc.check() {
        ModelConclusion::Safe => PropertyCheck::Ok,
        ModelConclusion::CounterExample => PropertyCheck::Fail
    };

    if verbose && output == PropertyCheck::Fail {
        let model = bmc.solver.get_model();
        debug::print_input_trace(graph, model.as_slice());
    }

    Ok(output)
}

pub fn check_interpolated(graph: &AIG, initial_bound: u32, verbose: bool) -> Result<PropertyCheck, ModelCheckingError> {
    // First, check if the property can be violated within the initial state s0
    if let PropertyCheck::Fail = check_bounded(graph, 0, verbose)? {
        return Ok(PropertyCheck::Fail);
    }

    let mut k = max(initial_bound, 1);
    let mut bmc = BmcModel::from_aig(graph, k, false)?;

    loop {
        if verbose { println!("k = {} | interpolants: {}", k, bmc.interpolation_count); }

        if bmc.check() == ModelConclusion::Safe {
            let itp_s1 = match bmc.solver.compute_interpolant() {
                Ok(itp) => itp,
                Err(InterpolationError::OutOfMemory(_)) => return Err(ModelCheckingError::OutOfMemory),
                _ => return Err(ModelCheckingError::FailedInterpolation(k, bmc.interpolation_count))
            };

            #[cfg(debug_assertions)]
            check_interpolant(&mut bmc, &itp_s1);

            // I(s0) <- Rename interpolant I(s1) to I(s0) (talk about states in time step t = 0)
            let itp_s0 = bmc.rename_interpolant(itp_s1);

            // Fixpoint Check
            if bmc.check_fixpoint(&itp_s0, bmc.solver.top_var) {
                return Ok(PropertyCheck::Ok)
            }

            // Merge new interpolant I^{i+1} with current initial states Q or I^1 or I^2 ... or I^i
            bmc.add_interpolant(itp_s0);
        } else {
            if bmc.interpolation_count == 0 {
                if verbose {
                    let model = bmc.solver.get_model();
                    debug::print_input_trace(graph, model.as_slice());
                }

                return Ok(PropertyCheck::Fail);
            }

            // Increase k by one step and start BMC from scratch (discard all interpolants)
            k += 1;
            bmc = BmcModel::from_aig(graph, k, false)?;

            if verbose { println!("---"); }
        }
    }
}

pub struct BmcModel<'a> {
    graph: &'a AIG,
    time_steps: Vec<HashMap<u32, Literal>>,
    solver: Solver,
    assumption_lit: Option<Literal>,
    interpolation_count: usize,
    fixpoint_solver: SimpleSolver,
}

impl BmcModel<'_> {

    /// Creates a new bounded model checking (BMC) instance from an And-Inverter Graph (AIG).
    /// The model immediately initializes all SAT variables needed to express the various circuit
    /// signals in all time steps t in [0, k] and unrolls the transition relation [k] times.
    pub fn from_aig(graph: &'_ AIG, k: u32, include_p0: bool) -> Result<BmcModel<'_>, ModelCheckingError> {
        let mut model = BmcModel {
            graph,
            time_steps: Vec::new(),
            interpolation_count: 0,
            assumption_lit: None,
            solver: Solver::new(),
            fixpoint_solver: SimpleSolver::new()
        };

        // Initialize variables for all signals at time steps 0..=k
        for t in 0..=k {
            model.add_step();
            for var in graph.variables() {
                let _lit = model.signal_at_time(&Signal::Var(var), t)?;
                // println!("Lit {} = AigVar {}@{}", _lit.var(), 2*var.idx(), t);
            }
        }

        // Add the same number of variables to the fixpoint solver
        for _ in 1..=model.solver.top_var { model.fixpoint_solver.add_var(); }

        // Compute and enforce the exact initial states Q
        model.add_initial_states()?;
        // Unroll the transition relation k times
        model.unwind(k, include_p0)?;

        Ok(model)
    }

    /// Adds the given interpolant to the solver state in an incremental manner.
    pub fn add_interpolant(&mut self, interpolant: XCNF) {
        self.interpolation_count += 1;
        self.solver.set_partition(Partition::A);

        // Add new variables to the solver
        for _ in (self.solver.top_var+1)..=interpolant.out_lit.var() { self.solver.add_var(); }

        // Add all clauses to the solver
        for clause in interpolant.formula.clauses {
            self.solver.add_clause(&clause);
        }

        // Add interpolant to the current solver clauses in an incremental manner
        // Add clauses to express a <-> (I or b), where b is a new assumption literal
        if let Some(a) = self.assumption_lit {
            let b = self.solver.add_var();

            self.solver.add_clause([-a, interpolant.out_lit, b]);
            self.solver.add_clause([-interpolant.out_lit, a]);
            self.solver.add_clause([-b, a]);

            self.assumption_lit = Some(b);
        }
    }

    /// Check the property encoded by the current state of the [BmcModel].
    /// ## Returns
    /// - [ModelConclusion::CounterExample]: If a property violation could be found within the `k` unwinding steps.
    /// - [ModelConclusion::Safe]: If the model is safe w.r.t. the first `k` unwinding steps.
    pub fn check(&mut self) -> ModelConclusion {
        let is_sat = match self.assumption_lit {
            Some(a) => self.solver.solve_assuming([-a]),
            None => self.solver.solve(),
        };

        // If the formula is SAT, the model's property was violated
        match is_sat {
            true => {
                #[cfg(debug_assertions)]
                {
                    let model = self.solver.get_model();
                    debug::print_input_trace(self.graph, model.as_slice());
                }

                ModelConclusion::CounterExample
            },
            false => ModelConclusion::Safe
        }
    }

    /// Given an extended CNF interpolant over the state at time step `t = 1`, selectively renames
    /// its literals such that the interpolant talks about the state at `t = 0` instead.
    pub fn rename_interpolant(&self, mut interpolant: XCNF) -> XCNF {
        let m = self.graph.max_idx as i32;
        interpolant.shift_literals((m+1)..=(2*m), -m);

        interpolant
    }

    pub fn check_fixpoint(&mut self, interpolant: &XCNF, top_var: i32) -> bool {
        for _ in (self.fixpoint_solver.top_var+1)..=top_var {
            self.fixpoint_solver.add_var();
        }

        // Add all clauses of the new interpolant to the solver
        for clause in interpolant.formula.clauses.iter() {
            self.fixpoint_solver.add_clause(clause);
        }

        // Reached fixpoint if I and (~Q and ~I1 and ~I2...) is UNSAT
        if self.fixpoint_solver.assumptions.len() > 1 {
            let last_itp_lit = self.fixpoint_solver.assumptions.pop().unwrap();
            self.fixpoint_solver.assumptions.push(-last_itp_lit);
        }

        self.fixpoint_solver.assumptions.push(interpolant.out_lit);

        let not_implied_is_sat = self.fixpoint_solver.solve();
        !not_implied_is_sat   // ~implication is unsat <=> implication is valid <=> reached fixpoint
    }


    /* --- Private methods --- */

    /// Adds clauses for the exact (as opposed to over-approximated) initial states Q_0 to the solver.
    /// Returns a **[Literal] q_0**, which enforces these initial states when set to true.
    fn add_initial_states(&mut self) -> Result<Literal, ModelCheckingError> {
        self.solver.set_partition(Partition::A);

        // Single literal to enforce (or deactivate) the initial latch state
        let q0 = self.solver.add_var();
        let _q0_fp = self.fixpoint_solver.add_var();
        debug_assert_eq!(q0, _q0_fp);

        // AND-gates
        for gate in self.graph.and_gates.iter() {
            self.encode_and_gate(gate, 0)?;
        }

        // Latches
        let mut latch_lits = Vec::new();
        for latch in self.graph.latches.iter() {
            let init_val = self.signal_at_time(&latch.out, 0)?;
            self.solver.add_clause([-q0, -init_val]);  // q_0 implies L_i@0 = 0
            self.fixpoint_solver.add_clause([-q0, -init_val]);

            latch_lits.push(init_val);
        }

        // The other direction: All latches are 0 implies q0
        latch_lits.push(q0);
        self.solver.add_clause(latch_lits.as_slice());
        self.fixpoint_solver.add_clause(latch_lits.as_slice());

        // Add extendable clause to force initial states
        let a = self.solver.add_var();
        self.assumption_lit = Some(a);
        self.solver.add_clause([q0, a]);

        self.fixpoint_solver.assumptions.push(-q0);

        Ok(q0)
    }

    /// Unwinds the transition relation as described by the model's [AIG] graph `k` times
    /// and adds the resulting clauses to the model's state.
    fn unwind(&'_ mut self, k: u32, include_p0: bool) -> Result<(), ModelCheckingError> {
        self.solver.set_partition(Partition::A);

        // We need variables for each input, latch, gate and output at each time step
        for t in 1..=k {
            if t == 2 {
                self.solver.set_partition(Partition::B);
            }

            // AND-gates
            for gate in self.graph.and_gates.iter() {
                self.encode_and_gate(gate, t)?;
            }
            // Latches
            for latch in self.graph.latches.iter() {
                self.encode_latch(latch, t)?;
            }
        }

        self.solver.set_partition(Partition::B);

        let property_range = match include_p0 {
            true => 0..=k,
            false => 1..=k,
        };

        // Assert that the output is true at SOME time step -> property violation
        let property_violation_clause: Vec<Literal> = property_range
            .map(|t| self.signal_at_time(&self.graph.outputs[0], t))
            .collect::<Result<_, _>>()?;

        self.solver.add_clause(property_violation_clause);

        Ok(())
    }

    /// Explicitly adds a new time step to the model. [Literal]s for [Signal]s can only be instantiated
    /// for time steps in the current range of unrolled time steps. Returns the new time step id.
    fn add_step(&mut self) -> usize {
        self.time_steps.push(HashMap::with_capacity(self.graph.max_idx as usize));
        self.time_steps.len() - 1  // return new time step id
    }

    /// Retrieves the SAT [Literal] for the requested [Signal] at the given time step `t`.
    /// If no literal exists for the given (Signal, `t`) pair, a new SAT variable is created and returned.
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

    fn encode_and_gate(&mut self, gate: &AndGate, t: u32) -> Result<(), ModelCheckingError> {
        let out = self.signal_at_time(&gate.out, t)?;
        let in1 = self.signal_at_time(&gate.in1, t)?;
        let in2 = self.signal_at_time(&gate.in2, t)?;

        self.solver.add_clause([-in1, -in2, out]);
        self.solver.add_clause([-out, in1]);
        self.solver.add_clause([-out, in2]);

        if t == 0 {
            self.fixpoint_solver.add_clause([-in1, -in2, out]);
            self.fixpoint_solver.add_clause([-out, in1]);
            self.fixpoint_solver.add_clause([-out, in2]);
        }

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


// ================= Unit Tests ================
#[cfg(debug_assertions)]
pub fn check_interpolant(bmc: &mut BmcModel, interpolant: &XCNF) {
    let proof = bmc.solver.resolution.take().unwrap();

    let mut a_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::A).map(|c| proof.get_clause(*c).unwrap().clone()).collect();
    let b_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::B).map(|c| proof.get_clause(*c).unwrap().clone()).collect();

    if let Some(a) = bmc.assumption_lit {
        a_clauses.push(Clause::from(-a));   // add the assumption as an A-partition unit clause
    }

    debug::verify_interpolant_properties(interpolant, CNF::from(a_clauses), CNF::from(b_clauses), bmc.solver.top_var);

    bmc.solver.resolution = Some(proof);
}

#[cfg(test)]
/// Helper function to extract a witness for a given AIG and unwinding budget `k` (if there is one).
pub fn bmc_find_witness(graph: &AIG, k: u32) -> Option<InputTrace> {
    let mut bmc = BmcModel::from_aig(graph, k, true).unwrap();

    match bmc.check() {
        ModelConclusion::Safe => None,
        ModelConclusion::CounterExample => {
            let model = bmc.solver.get_model();
            let trace = debug::extract_input_trace(graph, model.as_slice());
            Some(trace)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnf;

    impl Default for AIG {
        fn default() -> Self {
            // Obtain a dummy AIG object needed for construction of a BmcModel
            AIG {
                max_idx: 1,
                inputs: Vec::new(),
                latches: Vec::new(),
                outputs: Vec::new(),
                and_gates: Vec::new(),
            }
        }
    }
    impl BmcModel<'_> {
        fn dummy(graph: &'_ AIG) -> BmcModel<'_> {
            BmcModel {
                graph,
                time_steps: Vec::new(),
                interpolation_count: 0,
                assumption_lit: None,
                solver: Solver::new(),
                fixpoint_solver: SimpleSolver::new(),
            }
        }
    }


    #[test]
    fn test_interpolant_basic_identity() {
        let dummy_aig = AIG::default();
        let mut bmc = BmcModel::dummy(&dummy_aig);

        let x = bmc.solver.add_var();

        bmc.solver.set_partition(Partition::A);
        let clauses_a = cnf!([x]);
        bmc.solver.add_clause(&clauses_a[0]);

        bmc.solver.set_partition(Partition::B);
        let clauses_b = cnf!([-x]);
        bmc.solver.add_clause(&clauses_b[0]);

        assert!(!bmc.solver.solve(), "Can only compute an interpolant if the partitions (A, B) are inconsistent");

        let itp = bmc.solver.compute_interpolant().expect("Failed to compute interpolant");
        debug::verify_interpolant_properties(&itp, clauses_a, clauses_b, bmc.solver.top_var);
    }

    #[test]
    fn test_interpolant_transitive_chain() {
        let dummy_aig = AIG::default();
        let mut bmc = BmcModel::dummy(&dummy_aig);

        let a = bmc.solver.add_var(); // Local to A
        let b = bmc.solver.add_var(); // Shared
        let c = bmc.solver.add_var(); // Local to B

        bmc.solver.set_partition(Partition::A);
        let clauses_a = cnf!([a], [-a, b]);
        for c in &clauses_a { bmc.solver.add_clause(c); }

        bmc.solver.set_partition(Partition::B);
        let clauses_b = cnf!([-b, c], [-c]);
        for c in &clauses_b { bmc.solver.add_clause(c); }

        assert!(!bmc.solver.solve(), "Can only compute an interpolant if the partitions (A, B) are inconsistent");

        let itp = bmc.solver.compute_interpolant().expect("Failed to compute interpolant");
        debug::verify_interpolant_properties(&itp, clauses_a, clauses_b, bmc.solver.top_var);
    }

    #[test]
    fn test_interpolant_disjunction() {
        let dummy_aig = AIG::default();
        let mut bmc = BmcModel::dummy(&dummy_aig);

        let x = bmc.solver.add_var();
        let y = bmc.solver.add_var();

        bmc.solver.set_partition(Partition::A);
        let clauses_a = cnf!([x, y]);
        bmc.solver.add_clause(&clauses_a[0]);

        bmc.solver.set_partition(Partition::B);
        let clauses_b = cnf!([-x], [-y]);
        for c in &clauses_b { bmc.solver.add_clause(c); }

        assert!(!bmc.solver.solve(), "Can only compute an interpolant if the partitions (A, B) are inconsistent");

        let itp = bmc.solver.compute_interpolant().expect("Failed to compute interpolant");
        debug::verify_interpolant_properties(&itp, clauses_a, clauses_b, bmc.solver.top_var);
    }
}