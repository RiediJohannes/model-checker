mod aiger;
mod model;
mod debug;

pub use model::*;


#[cfg(test)]
mod tests {
    use super::*;

    fn execute_bmc(aiger_file: &str, k: u32, interpolate: bool, expected_result: PropertyCheck) {
        let instance = load_model(aiger_file).expect("Failed to parse test data file");

        let result = match interpolate {
            false => check_bounded(&instance, k),
            true => check_interpolated(&instance, k)
        };

        assert!(result.is_ok(), "Encountered error during model checking: {:?}", result.err());
        assert_eq!(result.unwrap(), expected_result);
    }

    #[test]
    fn bounded_counter_ok() {
        const AIGER_FILE: &str = "data/count10.aag";
        const BOUND: u32 = 3;

        execute_bmc(AIGER_FILE, BOUND, false, PropertyCheck::Ok);
    }

    #[test]
    fn bounded_counter_fail() {
        const AIGER_FILE: &str = "data/count10.aag";
        const BOUND: u32 = 9;

        execute_bmc(AIGER_FILE, BOUND, false, PropertyCheck::Fail);
    }

    #[test]
    fn bounded_lock_ok() {
        const AIGER_FILE: &str = "data/combination.aag";
        const BOUND: u32 = 1;

        execute_bmc(AIGER_FILE, BOUND, false, PropertyCheck::Ok);
    }

    #[test]
    fn bounded_lock_fail() {
        const AIGER_FILE: &str = "data/combination.aag";
        const BOUND: u32 = 3;

        execute_bmc(AIGER_FILE, BOUND, false, PropertyCheck::Fail);
    }
}