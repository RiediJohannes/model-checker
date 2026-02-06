#include "Api.h"
#include "model-checker/src/minisat.rs.h"  // import shared types


inline Lit literalToLit(const Literal& l) {
    return toLit(l.id);
}

SolverStub::SolverStub() = default;

Literal SolverStub::newVar() {
    const int nextVar = solver.newVar();
    const Literal lit{ 2*nextVar };
    return lit;
}

bool SolverStub::solve() {
    return solver.solve();
}

void SolverStub::addClause(rust::Slice<const Literal> rustClause) {
    vec<Lit> clause;
    for (const auto& rustLit : rustClause) {
        const Lit lit = literalToLit(rustLit);
        clause.push(lit);
    }

    solver.addClause(clause);
}

std::unique_ptr<SolverStub> newSolver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}