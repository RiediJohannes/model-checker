use crate::bmc::aiger;
use crate::bmc::aiger::{ParseError, AIG};

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
    pub fn check_bounded(self, bound: u64) -> ModelCheckingConclusion {
        let mut unwound_model = self.unwind(bound);
        unwound_model.check()
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }

    fn unwind(self: BmcModel, k: u64) -> UnwoundBmcModel {
        UnwoundBmcModel {
            base: self,
            k,
            cnf: String::new(),
        }
    }
}

pub struct UnwoundBmcModel {
    base: BmcModel,
    k: u64,
    cnf: String,
}

impl UnwoundBmcModel {
    pub fn check(&mut self) -> ModelCheckingConclusion {
        ModelCheckingConclusion::Ok
    }
}