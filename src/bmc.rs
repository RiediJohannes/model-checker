mod aiger;

pub fn load_instance(name: &str) -> aiger::AIG {
    aiger::parse_aiger_ascii(name).unwrap()
}