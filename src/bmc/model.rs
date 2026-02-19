use crate::bmc::aiger::{AndGate, Latch, ParseError, Signal, AIG};
use crate::bmc::{aiger, debug};
use crate::logic::resolution::Partition;
use crate::logic::solving::{SimpleSolver, Solver};
use crate::logic::{Clause, Literal, CNF, FALSE, TRUE, XCNF};

use crate::cnf;
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
    let mut bmc = BmcModel::from_aig(graph, k)?;

    bmc.add_initial_states()?;
    bmc.unwind(k)?;

    let output = match bmc.check() {
        ModelConclusion::Safe => PropertyCheck::Ok,
        ModelConclusion::CounterExample => PropertyCheck::Fail
    };
    Ok(output)
}

pub fn check_interpolated(graph: &AIG, initial_bound: u32) -> Result<PropertyCheck, ModelCheckingError> {
    let mut k = initial_bound;

    // Q <- Compute exact initial states Q
    // F <- Unroll transition relation k times
    let mut bmc = BmcModel::from_aig(graph, k)?;

    // These two lines are just a placeholder
    // TODO Maybe merge this with the from_aig method and keep q_0 as a private member
    bmc.add_initial_states()?;
    bmc.unwind(k)?;

    loop {
        // Q' <- Add all interpolants in I to initial states Q
        // Seed BMC with initial states Q' and F from the transition relation unrolls

        if bmc.check() == ModelConclusion::Safe {
            let itp_s1 = bmc.compute_interpolant()
                .ok_or(ModelCheckingError::FailedInterpolation(k, bmc.interpolation_count()))?;

            // TODO Remove these debug checks
                let proof = bmc.solver.resolution.take().unwrap();
                // let a_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::A).map(|c| proof.get_clause(*c).unwrap().clone()).collect();
                let mut a_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::A).map(|c| proof.get_clause(*c).unwrap().clone()).collect();
                a_clauses.push(Clause::from(-bmc.assumption_lit.unwrap()));
                let b_clauses: Vec<Clause> = proof.clauses_in_partition(Partition::B).map(|c| proof.get_clause(*c).unwrap().clone()).collect();
                debug::verify_interpolant_properties(&itp_s1, CNF::from(a_clauses), CNF::from(b_clauses), bmc.solver.top_var);

                bmc.solver.resolution = Some(proof);
            // End of debug checks

            // I(s0) <- Rename interpolant I(s1) to I(s0) (talk about states in time step t = 0)
            let itp_s0 = bmc.rename_interpolant(itp_s1);

            // Fixpoint Check
            if bmc.check_fixpoint(&itp_s0) {
                return Ok(PropertyCheck::Ok)
            }

            bmc.add_interpolant(itp_s0);
        } else {
            if bmc.interpolation_count() == 0 {
                return Ok(PropertyCheck::Fail);
            }

            k += 1;
            bmc = BmcModel::from_aig(graph, k)?;
            bmc.add_initial_states()?;
            bmc.unwind(k)?;
        }
    }
}

pub struct BmcModel<'a> {
    graph: &'a AIG,
    time_steps: Vec<HashMap<u32, Literal>>,
    interpolants: Vec<XCNF>,
    assumption_lit: Option<Literal>,
    solver: Solver,
    fixpoint_solver: SimpleSolver
}

impl BmcModel<'_> {

    /// Creates a new bounded model checking (BMC) instance from an And-Inverter Graph (AIG).
    /// The model immediately initializes all SAT variables needed to express the various circuit
    /// signals in all time steps t in [0, k].
    pub fn from_aig(graph: &'_ AIG, k: u32) -> Result<BmcModel<'_>, ModelCheckingError> {
        let mut model = BmcModel {
            graph,
            time_steps: Vec::new(),
            interpolants: Vec::new(),
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

        Ok(model)
    }

    /// Adds clauses for the exact (as opposed to over-approximated) initial states Q_0 to the solver.
    /// Returns a **[Literal] q_0**, which enforces these initial states when set to true.
    pub fn add_initial_states(&mut self) -> Result<Literal, ModelCheckingError> {
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

        // TODO Is this direction of the implication actually necessary?
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
    pub fn unwind(&'_ mut self, k: u32) -> Result<(), ModelCheckingError> {
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

        // Assert that the output is true at SOME time step -> property violation
        let property_violation_clause: Vec<Literal> = (1..=k)
            .map(|t| self.signal_at_time(&self.graph.outputs[0], t))
            .collect::<Result<_, _>>()?;

        self.solver.add_clause(property_violation_clause);

        Ok(())
    }

    pub fn add_interpolant(&mut self, interpolant: XCNF) {
        self.interpolants.push(interpolant.clone());  // TODO Maybe we don't even need this list?

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

    pub fn interpolation_count(&self) -> usize {
        self.interpolants.len()
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
                let model = self.solver.get_model();
                // debug::print_sat_model(self.graph, model.as_slice());
                debug::print_input_trace(self.graph, model.as_slice());
                ModelConclusion::CounterExample
            },
            false => ModelConclusion::Safe
        }
    }

    /// Computes an interpolant for the two inconsistent partitions `(A, B)` of clauses in the [BmcModel]
    /// according to the Huang-Krajíček-Pudlák interpolation system.
    /// The resulting interpolant will over-approximate the states that satisfy the clauses in [Partition::A].
    #[allow(non_snake_case)]
    pub fn compute_interpolant(&mut self) -> Option<XCNF> {
        // Temporarily take ownership of the resolution proof to allow for mutable references to solver during this function
        let proof_box = self.solver.resolution.take()?;
        let proof = proof_box.deref();

        let mut interpolants: HashMap<i32, XCNF> = HashMap::with_capacity(2 * proof.len() + 1);

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
            debug_assert!(!interpolants.contains_key(&step.resolvent));

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
            //     "C_L = {}\t -> I_L: {:?}\n\
            //     C_R = {}\t -> I_R: {:?}\n\
            //     -- Resolving on literal {:} (partition: {:?})\n\
            //     => {:?}\n",
            //     proof.get_clause(step.left).unwrap(), &I_L,
            //     proof.get_clause(step.right).unwrap(), &I_R,
            //     &step.pivot, &pivot_partition, &I_resolvent
            // );

            interpolants.insert(step.resolvent, I_resolvent);
            last = step.resolvent;
        }

        self.solver.resolution = Some(proof_box);   // return proof ownership

        let final_interpolant = interpolants.remove(&last)?;
        Some(final_interpolant)
    }

    /// Given an extended CNF interpolant over the state at time step `t = 1`, selectively renames
    /// its literals such that the interpolant talks about the state at `t = 0` instead.
    pub fn rename_interpolant(&self, mut interpolant: XCNF) -> XCNF {
        let m = self.graph.max_idx as i32;
        interpolant.shift_literals((m+1)..=(2*m), -m);

        interpolant
    }

    pub fn check_fixpoint(&mut self, interpolant: &XCNF) -> bool {
        for _ in (self.fixpoint_solver.top_var+1)..=interpolant.out_lit.var() {
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
#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

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
                interpolants: Vec::new(),
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

        let itp = bmc.compute_interpolant().expect("Failed to compute interpolant");
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

        let itp = bmc.compute_interpolant().expect("Failed to compute interpolant");
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

        let itp = bmc.compute_interpolant().expect("Failed to compute interpolant");
        debug::verify_interpolant_properties(&itp, clauses_a, clauses_b, bmc.solver.top_var);
    }
}