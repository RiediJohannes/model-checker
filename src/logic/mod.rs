pub mod solving;
pub mod resolution;

mod types;

pub use solving::Literal;
pub use solving::{FALSE, TRUE, VAR_OFFSET};
pub use types::*;


#[cfg(test)]
mod tests {
    use crate::logic::solving::Solver;
    use crate::logic::{FALSE, TRUE, VAR_OFFSET};

    /// Checks if the solver meets the expected preconditions upon construction via the new function.
    #[test]
    fn solver_preconditions() {
        let mut solver: Solver = Solver::new();

        let proof = solver.resolution.as_ref().expect("Solver should have a resolution proof object");
        assert_eq!(proof.partition, None);

        let x = solver.add_var();
        assert_eq!(x.var(), VAR_OFFSET as i32);

        // Check if the solver initially returns SAT if no clause has been added
        let sat_if_unchanged = solver.solve();
        assert!(sat_if_unchanged);

        // Check if forcing the (supposedly) constant false literal to true leads to an inconsistency
        solver.add_clause([FALSE]);
        let unsat_if_bottom_asserted_as_true = solver.solve();
        assert!(!unsat_if_bottom_asserted_as_true);
    }

    #[test]
    fn solve_sat() {
        let mut solver: Solver = Solver::new();

        let x = solver.add_var();
        let y = solver.add_var();
        let z = solver.add_var();
        let w = solver.add_var();

        solver.add_clause([x]);
        solver.add_clause([-x, y]);
        solver.add_clause([-y, z]);
        solver.add_clause([-z, x]);
        solver.add_clause([-w]);

        assert!(solver.solve());

        let model = solver.get_model();
        assert_eq!(model.as_slice(), vec![-1, 1, 1, 1, -1].as_slice());
    }

    #[test]
    fn solve_unsat() {
        let mut solver: Solver = Solver::new();

        let x = solver.add_var();
        let y = solver.add_var();
        let z = solver.add_var();

        solver.add_clause([x]);
        solver.add_clause([-x, y]);
        solver.add_clause([-y, z]);
        solver.add_clause([-z, -x]);

        assert!(!solver.solve());
    }

    #[test]
    fn solve_constants() {
        // Check if FALSE works as expected
        let mut solver: Solver = Solver::new();
        let x = solver.add_var();

        solver.add_clause([x, FALSE]);
        solver.add_clause([-x, FALSE]);

        assert!(!solver.solve());

        // Check if TRUE works as expected
        let mut solver: Solver = Solver::new();
        let y = solver.add_var();

        solver.add_clause([y, TRUE]);
        solver.add_clause([-y]);

        assert!(solver.solve());
        let model = solver.get_model();
        assert_eq!(model.as_slice(), vec![-1, -1].as_slice());
    }

    #[test]
    fn solve_with_assumptions() {
        let mut solver: Solver = Solver::new();

        let x = solver.add_var();
        let y = solver.add_var();
        let z = solver.add_var();

        // Formula only becomes unsat if the unit clause [x] is added
        solver.add_clause([-x, y]);
        solver.add_clause([-y, z]);
        solver.add_clause([-z, -x]);

        assert!(!solver.solve_assuming([x]));
    }
}
