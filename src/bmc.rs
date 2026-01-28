mod aiger;

pub fn load_instance(name: &str) -> aiger::Aiger {
    aiger::parse_aiger_ascii(name)
}