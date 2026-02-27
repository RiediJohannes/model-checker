mod aiger;
mod model;
mod debug;


pub use model::*;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::bmc::debug::InputTrace;
    use std::path::PathBuf;

    fn execute_bmc(aiger_file: &str, k: u32, interpolate: bool, expected_result: PropertyCheck) {
        let aiger_path: PathBuf = aiger_file.into();
        let instance = load_model(&aiger_path).expect("Failed to parse test data file");

        let result = match interpolate {
            false => check_bounded(&instance, k, false),
            true => check_interpolated(&instance, k, false)
        };

        assert!(result.is_ok(), "Encountered error during model checking: {:?}", result.err());
        assert_eq!(result.unwrap(), expected_result);
    }

    fn find_witness(aiger_file: &str, k: u32) -> InputTrace {
        let aiger_path: PathBuf = aiger_file.into();
        let instance = load_model(&aiger_path).expect("Failed to parse test data file");

        bmc_find_witness(&instance, k).expect(&format!("Failed to find witness for AIG = {} with k = {}", aiger_path.display(), k))
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

        let witness = find_witness(AIGER_FILE, BOUND);
        let expected_inputs = [
            [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]
        ];
        assert_eq!(witness, expected_inputs);
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

        let witness = find_witness(AIGER_FILE, BOUND);
        let expected_inputs = [
            [1], [0], [1], [-1]
        ];
        assert_eq!(witness, expected_inputs);
    }

    #[test]
    fn interpolated_counter_fail() {
        const AIGER_FILE: &str = "data/count10.aag";
        const BOUND: u32 = 2;

        execute_bmc(AIGER_FILE, BOUND, true, PropertyCheck::Fail);
    }

    #[test]
    fn interpolated_lock_fail() {
        const AIGER_FILE: &str = "data/lock_5.aag";
        const BOUND: u32 = 1;

        execute_bmc(AIGER_FILE, BOUND, true, PropertyCheck::Fail);
    }

    #[test]
    fn interpolated_safe_lock_ok() {
        const AIGER_FILE: &str = "data/safe.aag";
        const BOUND: u32 = 2;

        execute_bmc(AIGER_FILE, BOUND, true, PropertyCheck::Ok);
    }

    #[test]
    /// Again the 4-bit counter counting to a number >= 10, but this time
    /// the third bit is stuck on forever true => we reach a counterexample more quickly
    fn counter_3rd_bit_on() {
        const AIGER_FILE: &str = "data/count10_3rd_bit_on.aag";

        // We already fail in 4 steps
        execute_bmc(AIGER_FILE, 3, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE, 4, false, PropertyCheck::Fail);

        execute_bmc(AIGER_FILE, 2, true, PropertyCheck::Fail);
        execute_bmc(AIGER_FILE, 20, true, PropertyCheck::Fail);

        let witness = find_witness(AIGER_FILE, 4);
        let expected_inputs = [
            [1], [1], [1], [1], [-1]
        ];
        assert_eq!(witness, expected_inputs);
    }

    #[test]
    /// Again the 4-bit counter counting to a number >= 10, but this time
    /// the third bit is stuck on forever false => we can never count to 10 or higher
    fn counter_3rd_bit_off() {
        const AIGER_FILE: &str = "data/count10_3rd_bit_off.aag";

        // We never fail
        execute_bmc(AIGER_FILE, 10, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE, 100, false, PropertyCheck::Ok);

        execute_bmc(AIGER_FILE, 2, true, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE, 20, true, PropertyCheck::Ok);
    }

    #[test]
    fn multi_input_alternating() {
        const AIGER_FILE: &str = "data/multi_input_alternating.aag";

        execute_bmc(AIGER_FILE, 6, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE, 7, false, PropertyCheck::Fail);

        execute_bmc(AIGER_FILE, 2, true, PropertyCheck::Fail);

        let witness = find_witness(AIGER_FILE, 7);
        let expected_inputs = [
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [-1, -1, -1, -1],
        ];
        assert_eq!(witness, expected_inputs);
    }

    #[test]
    fn multi_input_puzzle() {
        const AIGER_FILE: &str = "data/multi_input_puzzle.aag";

        // Only after 31 time steps and the correct input combination in each step will the property
        // be violated
        execute_bmc(AIGER_FILE, 30, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE, 31, false, PropertyCheck::Fail);

        execute_bmc(AIGER_FILE, 2, true, PropertyCheck::Fail);
    }

    #[test]
    /// Special circuit that can only ever be violated in the initial state s0
    fn violated_in_initial_state() {
        const AIGER_FILE: &str = "data/one_shot.aag";

        execute_bmc(AIGER_FILE, 0, false, PropertyCheck::Fail);
        execute_bmc(AIGER_FILE, 0, true, PropertyCheck::Fail);

        // This is still detected for higher k
        execute_bmc(AIGER_FILE, 4, false, PropertyCheck::Fail);
        execute_bmc(AIGER_FILE, 4, true, PropertyCheck::Fail);
    }

    #[test]
    /// Check if the model checker correctly handles the special case of 0 unwindings
    /// of the transition relation
    fn no_unwinding() {
        const AIGER_FILE_FAIL: &str = "data/combination.aag";
        execute_bmc(AIGER_FILE_FAIL, 0, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE_FAIL, 0, true, PropertyCheck::Fail);

        const AIGER_FILE_SAFE: &str = "data/safe.aag";
        execute_bmc(AIGER_FILE_SAFE, 0, false, PropertyCheck::Ok);
        execute_bmc(AIGER_FILE_SAFE, 0, true, PropertyCheck::Ok);
    }

    #[test]
    /// Check if the model checker correctly handles purely combinatorial circuits (no latches)
    fn combinatorial_circuits() {
        const AIGER_COMB_SAT: &str = "data/combinatorial_sat.aag";
        execute_bmc(AIGER_COMB_SAT, 0, false, PropertyCheck::Fail);
        execute_bmc(AIGER_COMB_SAT, 2, false, PropertyCheck::Fail);
        execute_bmc(AIGER_COMB_SAT, 0, true, PropertyCheck::Fail);
        execute_bmc(AIGER_COMB_SAT, 2, true, PropertyCheck::Fail);

        let witness = find_witness(AIGER_COMB_SAT, 0);
        let expected_inputs = [
            [1, 1, 0, 1],
        ];
        assert_eq!(witness, expected_inputs);

        const AIGER_COMB_UNSAT: &str = "data/combinatorial_unsat.aag";
        execute_bmc(AIGER_COMB_UNSAT, 0, false, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_UNSAT, 0, false, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_UNSAT, 2, true, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_UNSAT, 2, true, PropertyCheck::Ok);
    }

    #[test]
    /// 6-bit counter that violates the property if it overflows (counts to 64)
    fn counter_6bit() {
        const AIGER_COMB_SAT: &str = "data/counter_6bit.aag";

        execute_bmc(AIGER_COMB_SAT, 62, false, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_SAT, 63, false, PropertyCheck::Fail);

        execute_bmc(AIGER_COMB_SAT, 55, true, PropertyCheck::Fail);
    }

    #[test]
    /// 6-bit counter that violates the property if it overflows (counts to 64),
    /// BUT it can never reach this state (saturates).
    fn saturating_counter_6bit() {
        const AIGER_COMB_SAT: &str = "data/saturating_counter_6bit.aag";

        execute_bmc(AIGER_COMB_SAT, 60, false, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_SAT, 70, false, PropertyCheck::Ok);

        execute_bmc(AIGER_COMB_SAT, 5, true, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_SAT, 55, true, PropertyCheck::Ok);
        execute_bmc(AIGER_COMB_SAT, 70, true, PropertyCheck::Ok);
    }
}