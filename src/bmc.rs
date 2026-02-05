mod aiger;

use aiger::{AIG, ParseError};

pub fn load_instance(name: &str) -> Result<BmcInstance, ParseError> {
    let aig = aiger::parse_aiger_ascii(name)?;

    Ok(BmcInstance {
        name: name.to_string(),
        graph: aig,
    })
}

#[derive(Debug)]
pub enum ModelCheckingConclusion {
    Ok,
    Fail,
}

pub struct BmcInstance {
    name: String,
    graph: AIG,
}

impl BmcInstance {
    pub fn check_bounded(self, bound: u64) -> ModelCheckingConclusion {
        let mut unwound_instance = self.unwind(bound);
        unwound_instance.check()
    }

    pub fn check_interpolated(&mut self) -> ModelCheckingConclusion {
        todo!("Interpolated model checking not yet implemented.");
    }

    fn unwind(self: BmcInstance, k: u64) -> UnwoundBmcInstance {
        UnwoundBmcInstance {
            base: self,
            k,
            cnf: String::new(),
        }
    }
}

pub struct UnwoundBmcInstance {
    base: BmcInstance,
    k: u64,
    cnf: String,
}

impl UnwoundBmcInstance {
    pub fn check(&mut self) -> ModelCheckingConclusion {
        ModelCheckingConclusion::Ok
    }
}
