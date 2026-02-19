#pragma once

#include <memory>
#include <vector>

#include "Solver.h"
#include "rust/cxx.h"  // needed to define shared types

struct Literal;          // full definition in solving.rs
struct ResolutionProof;  // full definition in solving.rs

class SolverStub {
public:
    explicit SolverStub(ResolutionProof& proofStore);  // Solver with proof logging
    SolverStub();   // Solver without proof logging
    ~SolverStub();

    Literal newVar();
    void addClause(rust::Slice<const Literal> clause);  // Linter complains about unknown namespace but header is (and must be) included in Stub.C
    bool solve();
    bool solve(rust::Slice<const Literal> assumptions);
    std::unique_ptr<std::vector<int8_t>> getModel();

private:
    Solver solver;
    CallbackTraverser* traverser;
};

std::unique_ptr<SolverStub> newSolver();
std::unique_ptr<SolverStub> newSolver(ResolutionProof& proofStore);
