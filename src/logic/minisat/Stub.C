#include "Stub.h"
#include "model-checker/src/logic/solving.rs.h"  // import shared types


inline Lit literalToLit(const Literal& l) {
    return toLit(l.id);
}

SolverStub::SolverStub(ResolutionProof& proofStore)
    : traverser(proofStore)
{
    solver.proof = new Proof(traverser);
}

Literal SolverStub::newVar() {
    const int nextVar = solver.newVar();
    const Literal lit{ 2*nextVar };
    return lit;
}

void SolverStub::addClause(const rust::Slice<const Literal> rustClause) {
    vec<Lit> clause;
    for (const auto& rustLit : rustClause) {
        const Lit lit = literalToLit(rustLit);
        clause.push(lit);
    }

    solver.addClause(clause);
}

bool SolverStub::solve() {
    return solver.solve();
    // if (!isSat && solver.proof != nullptr) solver.proof->save("proof.txt");
}

bool SolverStub::solve(const rust::Slice<const Literal> assumptions) {
    vec<Lit> mapped_assumptions;

    for (const auto& l : assumptions) {
        mapped_assumptions.push(literalToLit(l));
    }

    return solver.solve(mapped_assumptions);
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

std::unique_ptr<SolverStub> newSolver(ResolutionProof& proofStore) {
    return std::unique_ptr<SolverStub>(new SolverStub(proofStore));
}

SolverStub::~SolverStub() {
    if (solver.proof != nullptr) {
        delete solver.proof;
        solver.proof = nullptr;
    }
}