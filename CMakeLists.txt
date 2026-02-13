cmake_minimum_required(VERSION 3.28)
project(model_checker)

set(CMAKE_CXX_STANDARD 11)

include_directories(src/minisat)
include_directories(target/cxxbridge)

add_executable(model_checker
        src/logic/minisat/File.C
        src/logic/minisat/File.h
        src/logic/minisat/Global.h
        src/logic/minisat/Heap.h
        src/logic/minisat/Main.C
        src/logic/minisat/Proof.C
        src/logic/minisat/Proof.h
        src/logic/minisat/Solver.C
        src/logic/minisat/Solver.h
        src/logic/minisat/SolverTypes.h
        src/logic/minisat/Sort.h
        src/logic/minisat/VarOrder.h
        src/logic/minisat/Stub.C
        src/logic/minisat/Stub.h)

add_compile_options(-Wno-deprecated)

find_package(ZLIB REQUIRED)
target_link_libraries(model_checker PRIVATE ZLIB::ZLIB)
