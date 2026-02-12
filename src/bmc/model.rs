use std::collections::{HashMap};
use thiserror::Error;
use crate::bmc::aiger;
use crate::bmc::aiger::{AndGate, Latch, ParseError, Signal, AIG};
use crate::minisat::{Solver, Literal, Partition, CNF, cnf_from_unit};


#[derive(Error, Debug)]
pub enum ModelCheckingError {
    #[error("Tried to access a signal at time step {0}, but current model is only unrolled until t = {1}")]
    InvalidTimeStep(u32, u32),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Conclusion {
    Ok,
    Fail,
}

// Fixed IDs to use for SAT variables representing boolean constants
const BOTTOM: i32 = 0;
const TOP: i32 = 1;


pub fn load_model(name: &str) -> Result<AIG, ParseError> {
    aiger::parse_aiger_ascii(name)
}

pub fn check_bounded(graph: &AIG, k: u32) -> Result<Conclusion,ModelCheckingError> {
    // Single iteration of bounded model checking
    let mut bmc = BmcModel::from_aig(graph)?;
    bmc.unwind(k)?;

    Ok(bmc.check())
}

pub fn check_interpolated(graph: &AIG, initial_k: u32) -> Result<Conclusion,ModelCheckingError> {
    // TODO Implement interpolation loop until fixpoint
    // unwound_model <- unwind initial k with initial states Q^0(s0)
    // I(s1) <- split formula into partitions (A,B) and compute interpolant I(s1)
    // I(s0) <- rename variables in I(s1) to I(s0)
    // Q^{i+1}(s0) <- extend initial states (Q^i(s0) OR I(s0))
    // if (I(s0) or Q^{i+1}(so)) => Q^{i+1}(s0) {
    //      return FIXPOINT -- SAFE
    // } else {
    //      /* Restart BMC with initial states Q^{i+1}(s0) */
    //      - from new initial states, unwind k times
    //      - This means, we need to be able to initialize a BmcModel with the respective initial states
    // }

    // Single iteration of bounded model checking
    let mut bmc = BmcModel::from_aig(graph)?;
    bmc.unwind(initial_k)?;

    if bmc.check() == Conclusion::Ok {
        return Ok(Conclusion::Ok);
    }

    let _interpolant = bmc.compute_interpolant();

    Ok(Conclusion::Ok)
}

pub struct BmcModel<'a> {
    graph: &'a AIG,
    time_steps: Vec<HashMap<u32, Literal>>,
    solver: Solver,
}

impl BmcModel<'_> {
    pub fn from_aig(graph: &'_ AIG) -> Result<BmcModel<'_>, ModelCheckingError> {
        // Add the constant zero and constant one literal to the solver
        let mut solver = Solver::new();
        solver.clear_partition();

        let bottom = solver.add_var();
        let top = solver.add_var();
        assert_eq!(bottom.var(), BOTTOM);
        assert_eq!(top.var(), TOP);

        solver.add_clause([-bottom]);
        solver.add_clause([top]);

        let mut model = BmcModel {
            graph,
            time_steps: Vec::new(),
            solver
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
            Signal::Constant(true) => Ok(Literal::from_var(TOP)),
            Signal::Constant(false) => Ok(Literal::from_var(BOTTOM)),
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

    pub fn check(&mut self) -> Conclusion {
        // If the formula is SAT, the model's property was violated
        match self.solver.solve() {
            true => {
                let model = self.solver.get_model();
                // print_sat_model(self.graph, model.as_slice());
                print_input_trace(self.graph, model.as_slice());
                Conclusion::Fail
            },
            false => Conclusion::Ok
        }
    }

    #[allow(non_snake_case)]
    pub fn compute_interpolant(&self) -> Option<CNF> {
        let mut interpolants: HashMap<i32,CNF> = HashMap::new();
        let proof = self.solver.get_proof()?;

        // Base case: Annotate root clauses in partition A/B with Bottom/Top
        for A_clause_id in proof.clauses_in_partition(Partition::A) {
            interpolants.insert(*A_clause_id, cnf_from_unit(Literal::from_var(BOTTOM)));
        }

        for B_clause_id in proof.clauses_in_partition(Partition::B) {
            interpolants.insert(*B_clause_id, cnf_from_unit(Literal::from_var(TOP)));
        }

        // Inductive case: Compute new part. interpolant from previous part. interpolants
        for step in proof.resolutions() {
            let left_itp = interpolants.get(&step.left).unwrap();
            let right_itp = interpolants.get(&step.right).unwrap();
            assert!(!interpolants.contains_key(&step.resolvent));

            let pivot_partition = proof.var_partition(step.pivot)?;
            // TODO Merge interpolants (three case split)
            let resolvent_itp: CNF = match pivot_partition {
                Partition::A => {
                    // I_L OR I_R

                    cnf_from_unit(BOTTOM.into())
                },
                Partition::B => {
                    // I_L AND I_R

                    cnf_from_unit(BOTTOM.into())
                },
                Partition::AB => {
                    // (I_L OR x) AND (I_R OR ~x)

                    cnf_from_unit(BOTTOM.into())
                }
            };

            // TODO Optionally simplify resolvent interpolant


            interpolants.insert(step.resolvent, resolvent_itp);
        }

        None
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

fn print_sat_model(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();
    let num_l = graph.latches.len();
    let num_a = graph.and_gates.len();
    let frame_size = num_i + num_l + num_a;

    println!("--- SAT Model Assignment (True Variables) ---");

    for (idx, &val) in model.iter().enumerate().skip(2) {
        // Only print variables that were set to true
        if val != 1 {
            continue;
        }

        let normalized = idx - 2;
        let t = normalized / frame_size; // Time step
        let f = normalized % frame_size; // Offset in frame

        // Determine the semantic meaning of the variable within the current frame
        let (label, local_idx) = if f < num_i {
            ("Input", f)
        } else if f < num_i + num_l {
            ("Latch", f - num_i)
        } else {
            ("AND", f - (num_i + num_l))
        };

        println!("{:>4}: {}_{}@{}", idx, label, local_idx, t);
    }
    println!("--------------------------------------------");
}

fn print_input_trace(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();

    // The first two variables are constants (top/bottom)
    let total_vars = model.len().saturating_sub(2);
    let vars_per_frame = graph.variables().count();
    let num_steps = total_vars / vars_per_frame;

    // Header
    println!("\n=== Input Trace (Counter-Example) ===");
    print!("{:>4} | ", "t");
    for i in 0..num_i {
        print!("In_{:<2} ", i);
    }
    println!("\n{:-<5}+{:-<}", "", "-".repeat(num_i * 5));

    for k in 0..num_steps {
        print!("{:>4} | ", k);
        for i in 0..num_i {
            // Use frame_size derived from variables().len()
            let idx = 2 + (k * vars_per_frame) + i;
            if let Some(&val) = model.get(idx) {
                let display = match val {
                    1  => " T  ", // True
                    -1 => " F  ", // False
                    0  => " x  ", // Undefined
                    _  => " !  ", // Should not happen
                };
                print!("{}", display);
            }
        }
        println!();
    }
    println!("=====================================\n");
}

