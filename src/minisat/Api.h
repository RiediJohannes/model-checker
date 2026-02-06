#ifndef MODEL_CHECKER_API_H
#define MODEL_CHECKER_API_H

#include <memory>

#include "Solver.h"
#include "rust/cxx.h"  // needed to define shared types

struct Literal;        // full definition in interop.rs

class SolverStub {
public:
    SolverStub();

    Literal new_var();
    bool solve();

private:
    Solver solver;
};

std::unique_ptr<SolverStub> new_solver();

#endif //MODEL_CHECKER_API_H