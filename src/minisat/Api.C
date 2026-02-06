#include "Api.h"
#include "model-checker/src/interop.rs.h"  // import shared types


inline Lit literalToLit(const Literal& l) {
    return toLit(l.id);
}

SolverStub::SolverStub() = default;

Literal SolverStub::new_var() {
    const int nextVar = solver.newVar();
    const Literal lit{ 2*nextVar };
    return lit;
}

bool SolverStub::solve() {
    return solver.solve();
}

void SolverStub::add_clause(rust::Slice<const Literal> rustClause) {
    vec<Lit> clause;
    for (const auto& rustLit : rustClause) {
        const Lit lit = literalToLit(rustLit);
        clause.push(lit);
    }

    solver.addClause(clause);
}

std::unique_ptr<SolverStub> new_solver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}