use crate::bmc::aiger;
use crate::bmc::aiger::{ParseError, AIG};
use crate::minisat::Clause;

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
        let clauses = Vec::with_capacity((3 * k * self.graph.max_idx) as usize);

        // Special handling of time step 0

        // We need variables for each input, latch, gate and output at each time step
        for t in 1..k {
            // Create SAT vars
            // let inputs = self.graph.inputs.iter().map(|i| Literal {id: i}).collect::<Vec<_>>();
            let outputs = self.graph.outputs.iter().map(|o| 2*o).collect::<Vec<_>>();
            let gates = self.graph.and_gates.iter().map(|g| g.out).collect::<Vec<_>>();
            let latches = self.graph.latches.iter().map(|l| l.out).collect::<Vec<_>>();

            // Add SAT clauses
            // let c = Clause::new([a,b,c])
        }

        UnwoundBmcModel {
            base: self,
            k,
            cnf: clauses,
        }
    }
}

pub struct UnwoundBmcModel<'a> {
    base: &'a BmcModel,
    k: u32,
    cnf: Vec<Clause>,
}

impl UnwoundBmcModel<'_> {
    pub fn check_bounded(&mut self) -> ModelCheckingConclusion {
        ModelCheckingConclusion::Ok
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }
}