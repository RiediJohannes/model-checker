#pragma once

#include <memory>
#include <vector>

#include "Solver.h"
#include "rust/cxx.h"  // needed to define shared types

struct Literal;        // full definition in minisat.rs

class SolverStub {
public:
    SolverStub();

    Literal newVar();
    void addClause(rust::Slice<const Literal> clause);  // Linter complains about unknown namespace but header is (and must be) included in Api.C
    bool solve();
    std::unique_ptr<std::vector<int8_t>> getModel();

private:
    Solver solver;
};

std::unique_ptr<SolverStub> newSolver();
