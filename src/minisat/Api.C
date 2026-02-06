#include "Api.h"
#include "model-checker/src/interop.rs.h"  // import shared types

SolverStub::SolverStub() = default;

Literal SolverStub::new_var() {
    const int nextVar = solver.newVar();
    const Literal lit{ 2*nextVar };
    return lit;
}

bool SolverStub::solve() {
    return solver.solve();
}
//
// void SolverStub::add_clause(Slice<const Literal> clause) {
//     // todo
// }

std::unique_ptr<SolverStub> new_solver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}