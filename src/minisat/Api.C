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

void SolverStub::addClause(rust::Slice<const Literal> const rustClause) {
    vec<Lit> clause;
    for (const auto& rustLit : rustClause) {
        const Lit lit = literalToLit(rustLit);
        clause.push(lit);
    }

    solver.addClause(clause);
}

bool SolverStub::solve() {
    return solver.solve();
}

std::unique_ptr<std::vector<int8_t>> SolverStub::getModel() {
    // Allocate a new vector because we can only pass a std::vector across the FFI
    auto result = std::unique_ptr<std::vector<int8_t>>(new std::vector<int8_t>());
    result->reserve(solver.model.size());

    for (int i = 0; i < solver.model.size(); i++) {
        auto val = solver.model[i];
        result->push_back(val == l_True ? 1 : (val == l_False ? -1 : 0));
    }
    return result;
}

std::unique_ptr<SolverStub> newSolver() {
    return std::unique_ptr<SolverStub>(new SolverStub());
}