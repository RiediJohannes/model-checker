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
    pub fn unwind(&'_ self, k: u32) -> UnwoundBmcModel<'_> {
        let mut solver = Solver::new();

        // Special handling of time step 0

        // We need variables for each input, latch, gate and output at each time step
        for t in 1..k {
            // Create SAT vars
            let inputs: Vec<Literal> = solver.add_vars(self.graph.inputs.len());
            let outputs: Vec<Literal> = solver.add_vars(self.graph.outputs.len());
            let gates: Vec<Literal> = solver.add_vars(self.graph.and_gates.len());
            let latches: Vec<Literal> = solver.add_vars(self.graph.latches.len());

            // Add SAT clauses

            // AND-gates
            // for (i, gate) in self.graph.and_gates.iter().enumerate() {
            //     // gate.in1
            // }
            // let _c = solver.add_clause([-inputs[0], outputs[0]]);
        }

        UnwoundBmcModel {
            base: self,
            k,
            solver,
        }
    }
}

pub struct UnwoundBmcModel<'a> {
    base: &'a BmcModel,
    k: u32,
    solver: Solver,
}

impl UnwoundBmcModel<'_> {
    pub fn check_bounded(&mut self) -> ModelCheckingConclusion {
        ModelCheckingConclusion::Ok
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }
}