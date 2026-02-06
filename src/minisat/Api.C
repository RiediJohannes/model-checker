#include "Api.h"
#include "model-checker/src/interop.rs.h"  // import shared types

SolverStub::SolverStub() = default;

Literal SolverStub::new_var() {
    const int nextVar = solver.newVar();
    const Literal lit{ nextVar };
    return lit;
}

// void SolverStub::add_clause(int lit) {
//     solver.addClause(mkLit(lit));
// }

bool SolverStub::solve() {
    return solver.solve();
}

std::unique_ptr<SolverStub> new_solver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}