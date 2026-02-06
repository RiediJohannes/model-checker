#pragma once

#include <memory>

#include "Solver.h"
#include "rust/cxx.h"  // needed to define shared types
// #include "model-checker/src/interop.rs.h"  // import shared types

struct Literal;        // full definition in interop.rs

class SolverStub {
public:
    SolverStub();

    Literal new_var();
    // void add_clause(rust::Slice<const Literal> clause);
    bool solve();

private:
    Solver solver;
};

std::unique_ptr<SolverStub> new_solver();
