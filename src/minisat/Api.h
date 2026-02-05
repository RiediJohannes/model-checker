#ifndef MODEL_CHECKER_API_H
#define MODEL_CHECKER_API_H

#include <memory>

#include "Solver.h"

class SolverStub {
public:
    SolverStub();

    int new_var();
    bool solve();

    // int model_value(int var) const;

private:
    Solver solver;
};

std::unique_ptr<SolverStub> new_solver();

#endif //MODEL_CHECKER_API_H