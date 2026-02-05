#include "Api.h"

SolverStub::SolverStub() = default;

int SolverStub::new_var() {
    return solver.newVar();
}

// void SolverStub::add_clause_1(int lit) {
//     solver.addClause(mkLit(lit));
// }

bool SolverStub::solve() {
    return solver.solve();
}

std::unique_ptr<SolverStub> new_solver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}